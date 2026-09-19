"""
Run STAGATE (graph attention auto-encoder for spatial domains, Dong & Zhang
2022) for one sample at full nucleus resolution, producing domain labels in
the same output shape as the SpaGCN/ConGI/SpatialFusion baselines
({sample}_cluster_{n}.parquet with id/sample/n_clusters/pred).

What STAGATE uses: log-normalized expression of highly variable genes + the
spatial coordinates (via a spatial neighbor graph). It does NOT use the
histology image at all -- unlike SpaGCN/ConGI/SpatialFusion -- so it's the
expression+space-only reference point among the baselines.

Input, used as-is (no re-normalization): count_area_filtered_adata.h5ad is
already normalize_total(1e4)+log1p'd by cell_preprocessing.py, which is
exactly the input STAGATE's own tutorials feed it, and it already carries a
var['highly_variable'] flag (2,000 genes) that STAGATE's train_STAGATE picks
up automatically. Re-normalizing would repeat the double-transform bug found
and fixed in run_spagcn_svg.py.

Unlike SpaGCN, nothing here is O(n^2): the spatial graph is sparse (~k edges
per nucleus) and the attention layers are sparse ops. Cost is linear in the
number of nuclei, dominated by the dense n x n_genes reconstruction matmuls
each epoch (full-batch training, 500 epochs by default).

Adaptations of the vendored code (done from this script, not by editing it):
  - TensorFlow: the vendored code targets TF 1.15 / Python 3.7, but already
    imports `tensorflow.compat.v1`; calling tf.disable_v2_behavior() first
    lets it run unchanged on modern TF 2.x (CPU or GPU builds), so no ancient
    TF1/Python 3.7 environment is needed.
  - Cal_Spatial_Net builds one tiny pandas DataFrame PER NUCLEUS in a Python
    loop and concatenates them. Measured: 26s at 150k nuclei (so ~2.5 min at
    the largest sample) vs 0.7s for build_spatial_net below -- not a blocker,
    just needlessly slow and allocation-heavy at this scale. build_spatial_net
    produces the identical edge table (verified against the original for both
    KNN and radius graphs, including a duplicated-coordinate pair) from a
    sparse graph in one vectorized step.
  - Clustering: the paper's mclust (R) isn't used -- it isn't built for
    six-figure cell counts and would add an R/rpy2 dependency. Instead a
    Leiden resolution bisection on the STAGATE embedding hits each target
    cluster count, same approach as run_spatialfusion_range.py.

Like ConGI/SpatialFusion, clustering runs on the already-trained embedding,
so training happens once per sample no matter how many --n-clusters values are
swept. The embedding is cached at
data/{sample}/02_baselines/stagate/data/STAGATE_embedding.parquet, so a
re-run (e.g. after a Leiden miss or timeout during clustering) reuses it
instead of retraining; pass --force-retrain to redo it.

Run from this directory (paths are relative to it):
    python run_stagate_range.py --samples CRC_P2 --n-clusters-min 7 --n-clusters-max 10
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # must run before STAGATE builds its graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from STAGATE.Train_STAGATE import train_STAGATE  # noqa: E402


def build_spatial_net(adata, k_cutoff=None, rad_cutoff=None):
    """Vectorized equivalent of STAGATE.utils.Cal_Spatial_Net: same
    adata.uns['Spatial_Net'] edge table (Cell1, Cell2, Distance; self-pairs and
    zero-distance pairs dropped), built from a sparse neighbor graph instead of
    one DataFrame per node."""
    coor = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if rad_cutoff is not None:
        graph = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor) \
            .radius_neighbors_graph(coor, mode="distance")
    else:
        graph = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor) \
            .kneighbors_graph(coor, mode="distance")
    graph = graph.tocoo()
    keep = graph.data > 0
    names = np.asarray(adata.obs_names)
    adata.uns["Spatial_Net"] = pd.DataFrame({
        "Cell1": names[graph.row[keep]],
        "Cell2": names[graph.col[keep]],
        "Distance": graph.data[keep],
    })
    print(f"The graph contains {keep.sum()} edges, {adata.n_obs} cells "
          f"({keep.sum() / adata.n_obs:.2f} neighbors per cell).", flush=True)


def load_sample_adata(dir_custom_binning, dir_combined_adata, sample):
    """Same fallback logic as run_spagcn_svg.py's load_sample_adata."""
    per_sample_path = dir_custom_binning / "count_area_filtered_adata.h5ad"
    if per_sample_path.exists():
        return sc.read_h5ad(per_sample_path)
    if not dir_combined_adata.exists():
        raise FileNotFoundError(f"Neither {per_sample_path} nor {dir_combined_adata} exist")
    print(f"{per_sample_path} not found, loading {sample} out of {dir_combined_adata} instead", flush=True)
    combined = sc.read_h5ad(dir_combined_adata, backed="r")
    mask = (combined.obs["Slide_ID"] == sample).values
    if mask.sum() == 0:
        raise ValueError(f"No rows with Slide_ID == {sample!r} in {dir_combined_adata}")
    return combined[mask].to_memory()


def merge_to_k(labels, emb, k):
    """Merge the closest clusters (Ward criterion on size-weighted centroids in
    embedding space) until exactly k remain. Used only when no Leiden
    resolution yields exactly k: Leiden's cluster count can skip values, and
    a missing cluster count would leave a hole in the cross-method comparison."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    size, cen = {}, {}
    for c in np.unique(labels):
        m = labels == c
        size[c], cen[c] = int(m.sum()), emb[m].mean(axis=0)
    while len(size) > k:
        keys = list(size)
        cost, a, b = min(
            (size[x] * size[y] / (size[x] + size[y]) * float(((cen[x] - cen[y]) ** 2).sum()), x, y)
            for i, x in enumerate(keys) for y in keys[i + 1:]
        )
        cen[a] = (cen[a] * size[a] + cen[b] * size[b]) / (size[a] + size[b])
        size[a] += size[b]
        del size[b], cen[b]
        labels[labels == b] = a
    return pd.Series(pd.factorize(labels)[0]).astype(str).values


def res_search_fixed_clus(adata, fixed_clus_count, emb, lo=0.005, hi=5.0, max_run=20,
                          min_rel_width=0.01, random_state=0):
    """Leiden resolution bisection to an exact cluster count (same search as
    run_spatialfusion_range.py; see its docstring for why not a linear sweep),
    stopping early once the bracket is narrower than min_rel_width -- past that
    the count just flips between two neighbors at ~identical resolutions (seen
    in testing: 12 wasted runs alternating 6/8). If no resolution gives exactly
    k, the smallest partition found with MORE than k clusters is merged down to
    k (merge_to_k). Returns (resolution or None, labels or None, note)."""
    best_above = None  # (n_clusters, labels) with the fewest clusters still > k
    for _ in range(max_run):
        res = (lo + hi) / 2
        sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added="_tmp_leiden",
                     flavor="igraph", n_iterations=2, directed=False)
        n = adata.obs["_tmp_leiden"].nunique()
        print(f"    resolution {res:.4f} -> {n} clusters", flush=True)
        if n == fixed_clus_count:
            return res, adata.obs["_tmp_leiden"].astype(int).values.astype(str), "exact Leiden hit"
        if n < fixed_clus_count:
            lo = res
        else:
            hi = res
            if best_above is None or n < best_above[0]:
                best_above = (n, adata.obs["_tmp_leiden"].astype(int).values)
        if (hi - lo) / hi < min_rel_width:
            break
    if best_above is None:
        return None, None, "no partition with >= k clusters found"
    n_above, labels = best_above
    return None, merge_to_k(labels, emb, fixed_clus_count), \
        f"no exact Leiden hit; merged a {n_above}-cluster partition down to {fixed_clus_count}"


def process_sample(sample, args):
    basedir = Path("../../data")
    dir_custom_binning = basedir / sample / "01_preprocessing" / args.cellseg_dir / args.bin_size / "custom_binning"
    dir_combined_adata = (basedir / "combined" / "01_preprocessing" / args.cellseg_dir / args.bin_size
                           / "custom_binning" / "count_area_filtered_adata.h5ad")
    dir_scratch = basedir / sample / "02_baselines" / "stagate" / "data"
    emb_path = dir_scratch / "STAGATE_embedding.parquet"

    print(f"\n{'=' * 70}\nSample: {sample}\n{'=' * 70}", flush=True)

    adata = load_sample_adata(dir_custom_binning, dir_combined_adata, sample)
    adata.var_names_make_unique()
    if not adata.obs["id"].is_unique:
        raise ValueError(f"Nucleus IDs in {sample}'s adata are not unique.")
    adata.obs = adata.obs.assign(id=adata.obs["id"].astype(str)).set_index("id", drop=False)
    if args.max_cells:
        adata = adata[:args.max_cells].copy()

    use_cache = emb_path.exists() and not args.force_retrain and not args.max_cells
    if use_cache:
        print(f"Reusing cached embedding {emb_path} (pass --force-retrain to retrain)", flush=True)
        emb = pd.read_parquet(emb_path)
        emb = emb.loc[adata.obs_names]
    else:
        if args.hvg_key not in adata.var:
            raise ValueError(f"var[{args.hvg_key!r}] not found -- train_STAGATE would silently fall back "
                             f"to ALL {adata.n_vars} genes instead of highly variable ones")
        # float32 halves the dense (n_nuclei x n_HVG) matrix STAGATE builds and
        # feeds to TensorFlow every epoch; the source h5ad stores float64.
        adata.X = adata.X.astype(np.float32)
        adata.var["highly_variable"] = adata.var[args.hvg_key].fillna(False).astype(bool)
        print(f"{sample}: {adata.n_obs} nuclei, training on {int(adata.var['highly_variable'].sum())} "
              f"highly variable genes (of {adata.n_vars})", flush=True)

        build_spatial_net(adata, k_cutoff=args.k_cutoff, rad_cutoff=args.rad_cutoff)
        adata = train_STAGATE(
            adata, hidden_dims=args.hidden_dims, alpha=args.alpha, n_epochs=args.n_epochs, lr=args.lr,
            key_added="STAGATE", random_seed=args.seed, save_loss=True,
        )
        loss = adata.uns["STAGATE_loss"]
        print(f"Training loss: first {loss[0]:.2f} -> last {loss[-1]:.2f} over {len(loss)} epochs", flush=True)
        emb = pd.DataFrame(adata.obsm["STAGATE"], index=adata.obs_names)
        emb.columns = emb.columns.astype(str)
        if not args.max_cells:
            dir_scratch.mkdir(parents=True, exist_ok=True)
            emb.to_parquet(emb_path)
            print(f"Wrote {emb_path} {emb.shape}", flush=True)

    emb_adata = ad.AnnData(emb.values.astype(np.float32))
    emb_adata.obs_names = emb.index.astype(str)
    sc.pp.neighbors(emb_adata, use_rep="X", random_state=args.seed)

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for n_clusters in range(args.n_clusters_min, args.n_clusters_max + 1):
        print(f"\n--- {sample}: n_clusters={n_clusters} ---", flush=True)
        res, labels, note = res_search_fixed_clus(emb_adata, n_clusters, emb_adata.X,
                                                   random_state=args.seed)
        if labels is None:
            print(f"  Could not produce {n_clusters} clusters ({note}), skipping.", flush=True)
            continue
        print(f"  {n_clusters} clusters: {note}", flush=True)
        out_df = pd.DataFrame({"id": emb_adata.obs_names, "sample": sample,
                                "n_clusters": n_clusters, "pred": labels})
        out_path = outputs_dir / f"{sample}_cluster_{n_clusters}.{args.format}"
        if args.format == "csv":
            out_df.to_csv(out_path, index=False)
        else:
            out_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--n-clusters-min", type=int, required=True)
    parser.add_argument("--n-clusters-max", type=int, required=True)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--bin-size", default="002um")
    parser.add_argument("--cellseg-dir", default="cell_seg_stardist")

    parser.add_argument("--hvg-key", default="highly_variable",
                         help="var column flagging the genes STAGATE trains on (default: the existing "
                              "2,000-gene 'highly_variable' flag)")
    parser.add_argument("--k-cutoff", type=int, default=10,
                         help="Spatial graph: connect each nucleus to its k nearest neighbors. KNN rather "
                              "than a radius because nucleus density varies a lot across a section; the "
                              "STAGATE paper's Visium radius yields ~6 neighbors per spot.")
    parser.add_argument("--rad-cutoff", type=float, default=None,
                         help="Use a radius graph (in WSI pixels) instead of KNN; overrides --k-cutoff")
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[512, 30],
                         help="Encoder sizes (STAGATE default 512 30)")
    parser.add_argument("--alpha", type=float, default=0,
                         help="Cell-type-aware graph pruning weight; 0 (STAGATE's default) disables it")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2020, help="STAGATE's own default seed")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None,
                         help="Smoke-test only: use the first N nuclei. Requires a non-default "
                              "--outputs-dir and never reads/writes the cached embedding, so a partial "
                              "run can't be mistaken for a full one.")
    args = parser.parse_args()
    if args.n_clusters_min > args.n_clusters_max:
        raise ValueError("--n-clusters-min must be <= --n-clusters-max")
    if args.max_cells and args.outputs_dir == "outputs":
        parser.error("--max-cells requires a non-default --outputs-dir")

    for sample in args.samples:
        process_sample(sample, args)


if __name__ == "__main__":
    main()
