"""
Disk-budget-aware job supervisor for SpaGCN. Submits one
`sbatch run_spagcn_range.sbatch --samples <sample> ...` job per sample in
--samples, capped by BOTH --max-concurrent (how many may run at once) and
--storage-budget-gb (the sum of currently-running jobs' adjacency-matrix disk
usage may never exceed this). Always submits the next queued sample as soon
as a running one finishes and frees enough budget -- same spirit as SLURM's
array `%K` throttle, but admitting by disk size, not just job count.

Why size-aware admission matters here: SpaGCN's adjacency matrix is a dense
n_nuclei x n_nuclei float32 array, and nuclei counts vary by >30x across
samples (e.g. CRC_P6_5mm ~75k vs CRC_P2 ~225k). "3 jobs at a time" by count
alone can still blow a fixed storage budget depending on which 3 happen to
overlap -- e.g. CRC_P2's matrix alone is ~202GB, already most of a 250GB
budget on its own.

Per-sample nuclei counts (and therefore disk estimates) are read directly
from the combined adata's obs['Slide_ID'] counts at startup (backed mode, no
full load), not hardcoded, so this stays correct as that file changes.

Robustness: a job that dies mid-run (before run_spagcn_range.py's own
cleanup executes) leaves its adj.npy on disk. This script checks for that
directly on every poll -- if a "finished" job's adj.npy still exists, its
budget is NOT released and a warning is printed, rather than silently
over-committing disk for the next job.

This script itself is meant to run for the whole multi-day batch -- submit it
as its own tiny long-running SLURM job (schedule_spagcn_jobs.sbatch) rather
than running it in a terminal you might close.

Usage:
    python schedule_spagcn_jobs.py \\
        --samples CRC_P6_5mm CRC_P11mm_2_cut CRC_P4_SMAP CRC_P5 CRC_P2 \\
        --n-clusters-min 7 --n-clusters-max 10 \\
        --max-concurrent 3 --storage-budget-gb 220
"""
import argparse
import subprocess
import time
from pathlib import Path

import anndata as ad


def get_nuclei_counts(dir_combined_adata, samples):
    print(f"Reading nuclei counts from {dir_combined_adata} (backed, metadata only)...", flush=True)
    adata = ad.read_h5ad(dir_combined_adata, backed="r")
    counts = adata.obs["Slide_ID"].value_counts()
    missing = [s for s in samples if s not in counts.index]
    if missing:
        raise ValueError(f"No Slide_ID rows for: {missing} in {dir_combined_adata}")
    return {s: int(counts[s]) for s in samples}


def estimate_disk_gb(n_nuclei):
    # Dense n x n float32 adjacency matrix -- the dominant disk cost by far.
    # nuclei_centroids_svg.h5ad and the transient per-n_clusters results.h5ad
    # are each under ~200MB, folded into the +2GB fixed overhead below.
    return (n_nuclei ** 2 * 4) / 1e9 + 2


def mem_and_time_for(gb):
    # ~3x the adjacency matrix as peak RAM (see run_all_samples.sh's sizing
    # rationale); time is a generous, unmeasured guess scaled the same way.
    mem_gb = max(32, round(gb * 3))
    hours = max(6, round(gb / 6))
    return f"{mem_gb}G", f"{hours:02d}:00:00"


def squeue_alive(job_ids):
    if not job_ids:
        return set()
    out = subprocess.run(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
                          capture_output=True, text=True)
    return set(out.stdout.split())


def adj_still_present(sample):
    return (Path("../../data") / sample / "02_baselines/spagcn/data/adj.npy").exists()


def submit(sample, n_clusters_min, n_clusters_max, mem, time_, outputs_dir, extra_args):
    cmd = [
        "sbatch", "--parsable", f"--mem={mem}", f"--time={time_}",
        "run_spagcn_range.sbatch",
        "--samples", sample,
        "--n-clusters-min", str(n_clusters_min),
        "--n-clusters-max", str(n_clusters_max),
        "--outputs-dir", outputs_dir,
        *extra_args,
    ]
    print("Submitting:", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--n-clusters-min", type=int, required=True)
    parser.add_argument("--n-clusters-max", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--storage-budget-gb", type=float, default=220,
                         help="Sum of currently-running jobs' estimated adjacency-matrix "
                              "disk usage may never exceed this (leave margin below your "
                              "actual free space)")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--combined-adata",
                         default="../../data/combined/01_preprocessing/cell_seg_stardist/"
                                 "002um/custom_binning/count_area_filtered_adata.h5ad")
    args, extra_args = parser.parse_known_args()

    sizes_gb = {s: estimate_disk_gb(n) for s, n in
                get_nuclei_counts(Path(args.combined_adata), args.samples).items()}

    queue = []
    for s in args.samples:
        if sizes_gb[s] > args.storage_budget_gb:
            print(f"SKIPPING {s}: needs ~{sizes_gb[s]:.0f}GB alone, more than "
                  f"--storage-budget-gb={args.storage_budget_gb:.0f} -- it can never be admitted.",
                  flush=True)
        else:
            queue.append(s)
    # Largest-first (best-fit-decreasing): packs the awkward big samples while
    # budget is fully free, rather than stranding them behind small ones.
    queue.sort(key=lambda s: sizes_gb[s], reverse=True)

    running = {}  # job_id -> (sample, reserved_gb)
    stuck = {}    # sample -> gb, for jobs whose adj.npy outlived the SLURM job (never released)

    def reserved_total():
        return sum(gb for _, gb in running.values()) + sum(stuck.values())

    while queue or running:
        alive = squeue_alive(list(running.keys()))
        for jid in [j for j in running if j not in alive]:
            sample, gb = running.pop(jid)
            if adj_still_present(sample):
                print(f"WARNING: job {jid} ({sample}) is no longer in squeue but "
                      f"data/{sample}/02_baselines/spagcn/data/adj.npy still exists -- "
                      f"it likely died before its own cleanup ran. NOT releasing its "
                      f"~{gb:.0f}GB budget; please check this job's logs and clean up "
                      f"manually.", flush=True)
                stuck[sample] = gb
            else:
                print(f"Job {jid} ({sample}) finished cleanly, freeing ~{gb:.0f}GB "
                      f"({reserved_total():.0f}/{args.storage_budget_gb:.0f}GB now reserved)",
                      flush=True)

        progressed = True
        while progressed and queue:
            progressed = False
            for i, sample in enumerate(queue):
                gb = sizes_gb[sample]
                if len(running) >= args.max_concurrent:
                    break
                if reserved_total() + gb > args.storage_budget_gb:
                    continue
                mem, time_ = mem_and_time_for(gb)
                job_id = submit(sample, args.n_clusters_min, args.n_clusters_max, mem, time_,
                                 args.outputs_dir, extra_args)
                running[job_id] = (sample, gb)
                queue.pop(i)
                print(f"Started {sample} as job {job_id} (~{gb:.0f}GB, mem={mem}, time={time_}); "
                      f"{reserved_total():.0f}/{args.storage_budget_gb:.0f}GB reserved, "
                      f"{len(running)}/{args.max_concurrent} slots used", flush=True)
                progressed = True
                break

        if queue or running:
            time.sleep(args.poll_seconds)

    if stuck:
        print(f"\nDone, but {len(stuck)} sample(s) left stuck/unreleased: {list(stuck)} -- "
              f"check these manually.", flush=True)
    else:
        print("\nAll samples submitted and finished cleanly.", flush=True)


if __name__ == "__main__":
    main()
