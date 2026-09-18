"""
Run SpatialFusion (full paired H&E + RNA mode) for one sample, producing
spatial-niche cluster labels in the same output shape as the SpaGCN/ConGI
baselines ({sample}_cluster_{n}.parquet with id/sample/n_clusters/pred), for
direct downstream comparison.

Two-stage pipeline, unlike SpaGCN/ConGI (see this project's own exploration
notes): SpatialFusion itself is a lightweight, PRETRAINED (no training here,
pure inference) AE+GCN that fuses two already-computed foundation-model
embeddings per cell:
  - z_he (H&E/UNI, 1536-dim): already computed for this project by an
    earlier, unrelated pipeline step -- reused as-is from
    data/{sample}/01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad.
  - z_rna (scGPT, 512-dim): NOT previously computed anywhere in this
    project -- run run_scgpt_embed.py --samples {sample} first (writes
    data/{sample}/02_baselines/spatialfusion/data/scGPT.parquet).

Both dimensions were confirmed directly against the packaged pretrained AE
checkpoint's own weight shapes (encoder1.model.0.weight: (64, 1536),
encoder2.model.0.weight: (64, 512)) before writing this script, not assumed.

Unlike SpaGCN, and same as ConGI: clustering (here, a Leiden resolution
bisection search for each target cluster count) runs on the
ALREADY-COMPUTED joint embedding, so run_full_embedding (the actual AE+GCN
inference) only runs ONCE per sample regardless of how many --n-clusters
values are swept.

GPU note: this script itself (SpatialFusion's own AE+GCN forward pass) does
NOT need a GPU -- both explicitly support --device cpu, and are small
networks (hidden_dim=10 for the GCN) doing pure inference, no training. Only
run_scgpt_embed.py (the scGPT extraction step that has to run first) is
GPU-bound.

Usage:
    python run_scgpt_embed.py --samples CRC_P2          # RNA embeddings, once
    python run_spatialfusion_range.py --samples CRC_P2 --n-clusters-min 7 --n-clusters-max 10
"""
import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from spatialfusion.embed.embed import AEInputs, run_full_embedding


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


def res_search_fixed_clus(adata, fixed_clus_count, lo=0.005, hi=5.0, max_run=20, random_state=0):
    """
    Bisection on Leiden resolution until the cluster count equals
    fixed_clus_count exactly (cluster count is close to monotonic in
    resolution). Deliberately NOT congi/metrics.py's 0.02-step sweep from
    high to low: that's ~100 Leiden runs, fine at ConGI's ~3.6k pooled bins
    but far too slow at this baseline's nucleus-level 144k-785k cells, where
    one Leiden run alone takes a meaningful fraction of a minute or more.
    Operates on the already-computed joint embedding's neighbor graph, never
    recomputing the embedding -- see module docstring.

    Returns (resolution, labels) on an exact hit, else (None, None) after
    max_run tries (the exact count isn't always reachable -- Leiden's count
    can skip values as resolution changes).
    """
    for _ in range(max_run):
        res = (lo + hi) / 2
        sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added="_tmp_leiden",
                     flavor="igraph", n_iterations=2, directed=False)
        n = adata.obs["_tmp_leiden"].nunique()
        print(f"    resolution {res:.4f} -> {n} clusters", flush=True)
        if n == fixed_clus_count:
            return res, adata.obs["_tmp_leiden"].copy()
        if n < fixed_clus_count:
            lo = res
        else:
            hi = res
    return None, None


def process_sample(sample, args):
    basedir = Path("../../data")
    dir_custom_binning = basedir / sample / "01_preprocessing" / args.cellseg_dir / args.bin_size / "custom_binning"
    dir_combined_adata = (basedir / "combined" / "01_preprocessing" / args.cellseg_dir / args.bin_size
                           / "custom_binning" / "count_area_filtered_adata.h5ad")
    dir_uni = basedir / sample / "01_preprocessing" / "unimodel" / "metadata" / "cell_token_embeddings.h5ad"
    dir_scgpt = (Path(args.scgpt_parquet) if args.scgpt_parquet else
                 basedir / sample / "02_baselines" / "spatialfusion" / "data" / "scGPT.parquet")

    print(f"\n{'=' * 70}\nSample: {sample}\n{'=' * 70}", flush=True)

    adata = load_sample_adata(dir_custom_binning, dir_combined_adata, sample)
    adata.var_names_make_unique()
    if not adata.obs["id"].is_unique:
        raise ValueError(f"Nucleus IDs in {sample}'s adata are not unique.")
    adata.obs = adata.obs.assign(id=adata.obs["id"].astype(str)).set_index("id", drop=False)
    adata.obsm["spatial_px"] = adata.obsm["spatial"]

    if not dir_uni.exists():
        raise FileNotFoundError(
            f"{dir_uni} not found -- this project's existing UNI extraction step "
            f"(01_preprocessing/histology/unimodel) hasn't been run for {sample}."
        )
    if not dir_scgpt.exists():
        raise FileNotFoundError(
            f"{dir_scgpt} not found -- run `python run_scgpt_embed.py --samples {sample}` first."
        )

    uni_adata = ad.read_h5ad(dir_uni)
    # X_token_footprint (the nucleus's own crop) vs X_token_halo (a larger
    # surrounding context window) -- footprint chosen as the more direct
    # per-nucleus morphology signal; halo is the alternative if a niche-scale
    # (rather than single-cell-scale) H&E signal is wanted instead.
    z_he = pd.DataFrame(uni_adata.obsm["X_token_footprint"], index=uni_adata.obs["id"].values)
    z_rna = pd.read_parquet(dir_scgpt)

    # SpatialFusion intersects cell IDs across adata/z_he/z_rna and silently
    # drops the rest, so the output covers only cells present in ALL three.
    # Confirmed on CRC_P2: 224,825 nuclei in the adata but UNI embeddings for
    # only 144,296 of them (the existing UNI step only embeds nuclei inside
    # its filtered patches), so ~36% of cells get no niche label here even
    # though SpaGCN labels every nucleus. Compare baselines on the shared
    # subset of cell IDs, not the raw outputs.
    n_kept = len(adata.obs_names.intersection(z_he.index).intersection(z_rna.index))
    print(f"{sample}: {n_kept}/{adata.n_obs} cells have both UNI and scGPT embeddings "
          f"({adata.n_obs - n_kept} will have no label in the output)", flush=True)

    inputs = AEInputs(adata=adata, z_he=z_he, z_rna=z_rna)
    emb_df = run_full_embedding(
        ae_inputs_by_sample={sample: inputs},
        device=args.device,
        spatial_key="spatial_px",
        combine_mode=args.combine_mode,
        k=args.k,
    )
    print(f"{sample}: joint embedding computed for {len(emb_df)} cells, "
          f"{emb_df.shape[1]} dims (+ metadata)", flush=True)

    feat_cols = [c for c in emb_df.columns if c not in
                 ("sample_id", "cell_id", "celltype", "X_coord", "Y_coord")]
    emb_adata = ad.AnnData(emb_df[feat_cols].values, obs=emb_df[["cell_id"]].copy())
    emb_adata.obs_names = emb_df["cell_id"].astype(str).values
    sc.pp.neighbors(emb_adata, use_rep="X", random_state=args.seed)

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for n_clusters in range(args.n_clusters_min, args.n_clusters_max + 1):
        print(f"\n--- {sample}: n_clusters={n_clusters} ---", flush=True)
        res, labels = res_search_fixed_clus(emb_adata, n_clusters, random_state=args.seed)
        if labels is None:
            print(f"  Could not find a resolution giving exactly {n_clusters} clusters, skipping.", flush=True)
            continue
        print(f"  Found resolution {res} -> {n_clusters} clusters", flush=True)

        out_df = pd.DataFrame({"id": emb_adata.obs_names, "sample": sample,
                                "n_clusters": n_clusters, "pred": labels.values})
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
    parser.add_argument("--combine-mode", default="average",
                         choices=["average", "concat", "z1", "z2", "gated"],
                         help="How to fuse z_he/z_rna -- 'average' (default) is the full paired "
                              "mode; 'z1'/'z2' would give H&E-only/RNA-only instead.")
    parser.add_argument("--k", type=int, default=30, help="KNN graph neighbors for the GCN stage")
    parser.add_argument("--scgpt-parquet", default=None,
                         help="Override the scGPT embeddings path (default: "
                              "data/{sample}/02_baselines/spatialfusion/data/scGPT.parquet, written "
                              "by run_scgpt_embed.py). Only meaningful with a single --samples entry.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu",
                         help="SpatialFusion's own AE+GCN inference supports 'cpu' natively "
                              "(default here) -- only run_scgpt_embed.py needs a GPU")
    args = parser.parse_args()
    if args.n_clusters_min > args.n_clusters_max:
        raise ValueError("--n-clusters-min must be <= --n-clusters-max")

    for sample in args.samples:
        process_sample(sample, args)


if __name__ == "__main__":
    main()
