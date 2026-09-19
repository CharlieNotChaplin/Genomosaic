"""
Sequential batch runner: for each --sample, in order,
  1. build the adjacency matrix + prepped centroids once (build_adj_and_centroids),
     keeping the matrix itself purely in memory (adj.npy is never written to
     disk -- see skip_adj_disk in build_adj_and_centroids)
  2. find l once (spatial-only, independent of n_clusters)
  3. for n_clusters in [--n-clusters-min, --n-clusters-max]: find its resolution,
     train, and write one compact {sample}_cluster_{n}.parquet (id, sample,
     n_clusters, pred, refined_pred) to --outputs-dir
  4. delete every intermediate/heavy file actually written to disk for that
     sample (nuclei_centroids_svg.h5ad, l.txt, res_cluster_*.txt,
     results_cluster_*.h5ad, plots) -- only the parquet/csv files in
     --outputs-dir survive
  5. only then move to the next sample

Samples are processed strictly one at a time within this single process --
a sample's cleanup always finishes before the next sample's adjacency build
starts (this is the whole point of one sequential script rather than
run_all_samples.sh's parallel per-sample sbatch jobs). Reuses
build_adj_and_centroids/find_l/find_res/run_train from run_spagcn_svg.py rather
than duplicating that logic, so it also inherits the NaN-adjacency-matrix fix
and the scipy `.A` compat shim defined there.

Keeping the adjacency matrix in memory only (never on disk) trades disk for
RAM: it does NOT reduce how much RAM a sample needs -- the matrix always had
to be resident to be used for anything -- it only removes the redundant disk
copy. So --mem on the cluster still needs to cover the LARGEST sample in
--samples (not the average, since samples run sequentially, not in parallel,
within this one process), same as before; it's --storage that this change
actually relieves. See run_all_samples.sh's per-sample nuclei-count table for
sizing guidance either way.

Usage:
    python run_spagcn_range.py --samples CRC_P1 CRC_P6_5mm \\
        --n-clusters-min 7 --n-clusters-max 10
"""
import argparse
from pathlib import Path

import run_spagcn_svg as core


def process_sample(sample, n_clusters_min, n_clusters_max, outputs_dir, args):
    basedir = Path("../../data")
    dir_sample = basedir / sample
    dir_base_input = dir_sample / "input"
    dir_base_output = dir_sample / f"output/binned_outputs/square_{args.bin_size}"
    dir_custom_binning = dir_sample / "01_preprocessing" / args.cellseg_dir / args.bin_size / "custom_binning"
    dir_combined_adata = (basedir / "combined" / "01_preprocessing" / args.cellseg_dir / args.bin_size
                           / "custom_binning" / "count_area_filtered_adata.h5ad")
    # Scratch dirs are keyed by this job's cluster-count range so several jobs
    # for the SAME sample (e.g. --n-clusters-min 9 --n-clusters-max 9 and
    # 10 10 submitted side by side) never share files. Without this they
    # collide: every job writes nuclei_centroids_svg.h5ad (an HDF5 write lock
    # error if two open it at once) and l.txt (which run_train re-reads from
    # disk, and which each job truncates on write and unlinks at the end -- so
    # one job finishing deletes it out from under the others). The final
    # per-(sample, n_clusters) parquet files in --outputs-dir are already
    # distinct per count, so those need no change.
    run_tag = f"_n{n_clusters_min}-{n_clusters_max}"
    dir_spagcn_data = dir_sample / f"02_baselines/spagcn/data{run_tag}"
    dir_spagcn_results = dir_sample / f"02_baselines/spagcn/results{run_tag}"
    suffix = "_svggenes" if args.svg_genes_only else ""

    print(f"\n{'=' * 70}\nSample: {sample}\n{'=' * 70}", flush=True)

    # Always rebuild fresh: this script deletes adj.npy/nuclei_centroids_svg.h5ad
    # at the end of every sample regardless, so there is never a legitimate cache
    # to reuse between invocations -- only stale/pre-patch leftovers from outside
    # this script, which we don't want silently reused.
    #
    # skip_adj_disk=True: adj.npy is never written at all -- it's kept purely in
    # memory for this sample's entire lifetime (build -> search -> every
    # n_clusters' train -> cleanup), since this whole sequence runs in one
    # continuous process here. This trades disk for RAM: RAM usage is unchanged
    # (the matrix always had to be resident to be used at all), but disk usage
    # for the dominant multi-GB/TB item drops to zero. Only valid because this
    # script never splits search/train across separate processes/SLURM jobs --
    # run_spagcn_svg.py's own --stage search/--stage train CLI still needs it on
    # disk to hand off between those.
    raw_adata, adj = core.build_adj_and_centroids(
        dir_custom_binning, dir_combined_adata, dir_base_input, dir_base_output,
        dir_spagcn_data, dir_spagcn_results, sample,
        args.beta, args.alpha, force_recompute=True, skip_adj_disk=True,
    )

    # Separate prepped copy for search (find_l/find_res reuse this one object
    # across every n_clusters value, same as before) -- raw_adata itself is kept
    # untouched so run_train can take its own fresh copy from it every
    # iteration, exactly matching its original disk-reload-every-call behavior.
    search_adata = core.prep_adata_for_search(raw_adata.copy())
    core.log_rss(f"{sample}: after raw_adata.copy()+prep_adata_for_search")
    l = core.find_l(dir_spagcn_data, adj, args.p, args.l_start, args.l_end, args.l_tol, args.l_max_run)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n_clusters in range(n_clusters_min, n_clusters_max + 1):
        print(f"\n--- {sample}: n_clusters={n_clusters} ---", flush=True)
        core.find_res(dir_spagcn_data, search_adata, adj, l, n_clusters, args.seed)
        core.run_train(dir_spagcn_data, dir_spagcn_results, n_clusters, args.seed, args.svg_genes_only,
                        adata=raw_adata, adj=adj)

        result_h5ad = dir_spagcn_results / f"results_cluster_{n_clusters}{suffix}.h5ad"
        result_adata = core.sc.read_h5ad(result_h5ad)

        out_df = result_adata.obs[["pred", "refined_pred"]].copy()
        out_df.insert(0, "id", out_df.index)
        out_df.insert(1, "sample", sample)
        out_df.insert(2, "n_clusters", n_clusters)
        out_df.reset_index(drop=True, inplace=True)

        out_path = outputs_dir / f"{sample}_cluster_{n_clusters}{suffix}.{args.format}"
        if args.format == "csv":
            out_df.to_csv(out_path, index=False)
        else:
            out_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)", flush=True)
        written.append(out_path)

        # Per-n_clusters intermediates -- safe to delete now, the next n_clusters
        # iteration only needs adata/adj/l (already in memory), not these files.
        (dir_spagcn_data / f"res_cluster_{n_clusters}.txt").unlink(missing_ok=True)
        result_h5ad.unlink(missing_ok=True)
        (dir_spagcn_results / f"pred_cluster_{n_clusters}{suffix}.png").unlink(missing_ok=True)
        (dir_spagcn_results / f"refined_pred_cluster_{n_clusters}{suffix}.png").unlink(missing_ok=True)

    # Shared prep files -- only removable now that every n_clusters is done.
    # adj.npy was never written to disk in the first place (skip_adj_disk=True
    # above), so that unlink is a no-op; kept for symmetry/defensiveness in
    # case that ever changes.
    del raw_adata, search_adata, adj
    (dir_spagcn_data / "nuclei_centroids_svg.h5ad").unlink(missing_ok=True)
    (dir_spagcn_data / "adj.npy").unlink(missing_ok=True)
    (dir_spagcn_data / "l.txt").unlink(missing_ok=True)
    print(f"Cleaned up intermediate files for {sample}. Kept: {[str(p) for p in written]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True,
                         help="Sample names, processed strictly in this order -- "
                              "one sample's cleanup always finishes before the next starts")
    parser.add_argument("--n-clusters-min", type=int, required=True)
    parser.add_argument("--n-clusters-max", type=int, required=True)
    parser.add_argument("--outputs-dir", default="outputs",
                         help="Where the per-(sample, n_clusters) result files land (default: ./outputs)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                         help="parquet is smaller/faster and already used elsewhere in this repo "
                              "(tissue_positions.parquet, nuclei_geometry.parquet); csv if you need plain text")
    parser.add_argument("--bin-size", default="002um")
    parser.add_argument("--cellseg-dir", default="cell_seg_stardist")
    parser.add_argument("--svg-genes-only", action="store_true",
                         help="Restrict training to var['spatially_variable'] genes (does not affect adj/l/res search)")
    parser.add_argument("--beta", type=int, default=49)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--l-start", type=float, default=0.01, help="spg.search_l lower bound")
    parser.add_argument("--l-end", type=float, default=1000, help="spg.search_l upper bound")
    parser.add_argument("--l-tol", type=float, default=0.01, help="spg.search_l tolerance on p")
    parser.add_argument("--l-max-run", type=int, default=100, help="spg.search_l bisection step cap")
    args = parser.parse_args()

    if args.n_clusters_min > args.n_clusters_max:
        raise ValueError("--n-clusters-min must be <= --n-clusters-max")

    outputs_dir = Path(args.outputs_dir)
    for sample in args.samples:
        process_sample(sample, args.n_clusters_min, args.n_clusters_max, outputs_dir, args)


if __name__ == "__main__":
    main()
