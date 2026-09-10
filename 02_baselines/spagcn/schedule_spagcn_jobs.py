"""
RAM-budget-aware job supervisor for SpaGCN. Submits one
`sbatch run_spagcn_range.sbatch --samples <sample> ...` job per sample in
--samples, capped by BOTH --max-concurrent (how many may run at once) and
--mem-budget-gb (the sum of currently-running jobs' estimated peak RAM usage
may never exceed this). Always submits the next queued sample as soon as a
running one finishes and frees enough budget -- same spirit as SLURM's array
`%K` throttle, but admitting by size, not just job count.

Gates by RAM, not disk: run_spagcn_range.py (as of the skip_adj_disk change)
keeps the adjacency matrix purely in memory and never writes adj.npy to disk
at all, so disk is no longer the resource that needs rationing across
concurrent jobs -- RAM is. Earlier versions of this script tracked disk usage
instead; if you're pointing this at a version of run_spagcn_range.py that
still writes adj.npy to disk, this script's admission logic no longer applies
to that constraint.

--mem-budget-gb is a courtesy ceiling, not the actual safety mechanism, now
that the cluster's real topology is known: `sinfo -N -o "%N %m"` (2026-09-10)
shows 6 compute nodes, each with its OWN ~1008GB of physical RAM -- not one
800GB pool shared across all your jobs. SLURM's own scheduler already refuses
to place a job on a node that can't satisfy its --mem request, so it cannot
actually oversubscribe a node's real memory regardless of what this script
does. What --mem-budget-gb/--max-concurrent still do is cap how much you ask
for/how many jobs you keep in flight at once, so as not to try to claim every
node on a cluster other people also use (we've directly observed another
user's job running alongside ours). Defaults below assume ~5 of the 6 nodes
as a reasonable share, not a hard technical ceiling -- raise both if you want
to use more of the cluster and know it's actually free (`squeue`).

Why size-aware admission still matters even per-node: SpaGCN's adjacency
matrix is a dense n_nuclei x n_nuclei float32 array, and nuclei counts vary by
>30x across samples (e.g. CRC_P6_5mm ~75k vs CRC_P2 ~225k) -- "N jobs at a
time" by count alone could still ask for far more total memory than intended
if several large samples happen to queue up together, even though no single
one would ever oversubscribe an actual node on its own.

Per-sample nuclei counts (and therefore RAM estimates) are read directly from
the combined adata's obs['Slide_ID'] counts at startup (backed mode, no full
load), not hardcoded, so this stays correct as that file changes.

No "stuck budget" handling is needed here, unlike a disk-based version: RAM
is reclaimed unconditionally by the OS the instant a process exits, clean
finish or crash alike -- there's no equivalent of a leftover file on disk to
check for.

This script itself is meant to run for the whole multi-day batch -- submit it
as its own tiny long-running SLURM job (schedule_spagcn_jobs.sbatch) rather
than running it in a terminal you might close.

Usage:
    python schedule_spagcn_jobs.py \\
        --samples CRC_P6_5mm CRC_P11mm_2_cut CRC_P4_SMAP CRC_P5 CRC_P2 \\
        --n-clusters-min 7 --n-clusters-max 10 \\
        --max-concurrent 5 --mem-budget-gb 4750
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


def estimate_matrix_gb(n_nuclei):
    # Dense n x n float32 adjacency matrix.
    return (n_nuclei ** 2 * 4) / 1e9


def estimate_peak_ram_gb(n_nuclei):
    # ~3x the adjacency matrix (raw adj + adj_exp + a torch copy of it, all
    # resident at once at points in the pipeline -- traced for CRC_P1 in an
    # earlier debugging session), +2GB fixed overhead for
    # nuclei_centroids_svg.h5ad / the transient per-n_clusters results.h5ad.
    return estimate_matrix_gb(n_nuclei) * 3 + 2


def mem_and_time_for(peak_ram_gb):
    # time is a generous, unmeasured guess -- /18 rather than the raw matrix
    # size's old /6 keeps the same hour estimates as before now that the
    # input here is already the 3x-scaled peak RAM figure, not the raw matrix.
    mem_gb = max(32, round(peak_ram_gb))
    hours = max(6, round(peak_ram_gb / 18))
    return f"{mem_gb}G", f"{hours:02d}:00:00"


def squeue_alive(job_ids):
    if not job_ids:
        return set()
    out = subprocess.run(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
                          capture_output=True, text=True)
    return set(out.stdout.split())


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
    parser.add_argument("--max-concurrent", type=int, default=5,
                         help="Cluster has 6 compute nodes (`sinfo -N -o \"%%N %%m\"`), each with "
                              "its own independent RAM -- this isn't a shared pool. Default reserves "
                              "~1 node's worth of headroom for other users; raise toward 6 if you "
                              "know the cluster is free (`squeue`)")
    parser.add_argument("--mem-budget-gb", type=float, default=4750,
                         help="Sum of currently-running jobs' estimated peak RAM usage may never "
                              "exceed this. A courtesy ceiling on how much you claim at once, not "
                              "the real safety mechanism -- SLURM won't place a job on a node that "
                              "can't satisfy its --mem regardless. Default is ~5 nodes' worth "
                              "(~950GB usable each, leaving OS/other-process headroom per node)")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--combined-adata",
                         default="../../data/combined/01_preprocessing/cell_seg_stardist/"
                                 "002um/custom_binning/count_area_filtered_adata.h5ad")
    args, extra_args = parser.parse_known_args()

    ram_gb = {s: estimate_peak_ram_gb(n) for s, n in
              get_nuclei_counts(Path(args.combined_adata), args.samples).items()}

    queue = []
    for s in args.samples:
        if ram_gb[s] > args.mem_budget_gb:
            print(f"SKIPPING {s}: needs ~{ram_gb[s]:.0f}GB peak RAM alone, more than "
                  f"--mem-budget-gb={args.mem_budget_gb:.0f} -- it can never be admitted.",
                  flush=True)
        else:
            queue.append(s)
    # Largest-first (best-fit-decreasing): packs the awkward big samples while
    # budget is fully free, rather than stranding them behind small ones.
    queue.sort(key=lambda s: ram_gb[s], reverse=True)

    running = {}  # job_id -> (sample, reserved_gb)

    def reserved_total():
        return sum(gb for _, gb in running.values())

    while queue or running:
        alive = squeue_alive(list(running.keys()))
        for jid in [j for j in running if j not in alive]:
            sample, gb = running.pop(jid)
            # RAM is reclaimed unconditionally by the OS the instant a process
            # exits -- clean finish or crash alike -- so unlike a disk-based
            # budget, there's no "did it actually get released" check needed.
            print(f"Job {jid} ({sample}) finished, freeing ~{gb:.0f}GB "
                  f"({reserved_total():.0f}/{args.mem_budget_gb:.0f}GB now reserved)", flush=True)

        progressed = True
        while progressed and queue:
            progressed = False
            for i, sample in enumerate(queue):
                gb = ram_gb[sample]
                if len(running) >= args.max_concurrent:
                    break
                if reserved_total() + gb > args.mem_budget_gb:
                    continue
                mem, time_ = mem_and_time_for(gb)
                job_id = submit(sample, args.n_clusters_min, args.n_clusters_max, mem, time_,
                                 args.outputs_dir, extra_args)
                running[job_id] = (sample, gb)
                queue.pop(i)
                print(f"Started {sample} as job {job_id} (~{gb:.0f}GB peak RAM, mem={mem}, time={time_}); "
                      f"{reserved_total():.0f}/{args.mem_budget_gb:.0f}GB reserved, "
                      f"{len(running)}/{args.max_concurrent} slots used", flush=True)
                progressed = True
                break

        if queue or running:
            time.sleep(args.poll_seconds)

    print("\nAll samples submitted and finished.", flush=True)


if __name__ == "__main__":
    main()
