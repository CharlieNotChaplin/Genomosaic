"""
Standalone port of evaluate_final_loso.ipynb's sections 3 ("Clustering-quality
metrics (ARI / silhouette / NMI) vs. bin-level label transfer") and 5+6
("Spatial smoothing" + "ARI / silhouette / NMI again, post-smoothing").

Unlike the notebook, "raw" (pre-smoothing) and "hard"/"spagcn" (post-smoothing)
rows live in ONE long-format table (a `label_kind` column) instead of two
separate cell outputs with a redundant re-clustering pass in between --
embeddings are loaded and Leiden-clustered exactly once per (unit, variant,
slide, embedding).

Iterates every (unit, variant, slide) combination and silently skips whatever
isn't finished yet (no saved embeddings .npz, or no label-transfer grid
predictions for that slide) -- e.g. as of writing, bins has no finished runs
for the "svg"/"colon_xenium_panel" variants, and nothing has a finished
"colon_xenium_panel" run yet at all, so those rows just won't appear until the
corresponding LOSO training fold lands on disk. Safe to re-run at any point;
each run reflects whatever's finished at that moment.

Usage:
    python evaluate_clustering_quality.py
    python evaluate_clustering_quality.py --units nuclei --variants svg colon_xenium_panel
    python evaluate_clustering_quality.py --slides CRC_P1 CRC_P11mm --smoothing-modes hard
    python evaluate_clustering_quality.py --smooth-reference

--smooth-reference additionally scores every (embedding, label_kind) against a
spatially-smoothed copy of the label-transfer reference itself (majority-vote
only -- see build's docstring), doubling the row count via a `reference_kind`
column ("raw"/"smoothed"). This is a secondary/diagnostic number, not a
replacement for the raw-reference ARI: smoothing the reference with the same
spatial-neighbor notion used to smooth the clusters mechanically inflates
agreement for any spatially-coherent clustering, not just ones that learned
real biology -- it mainly helps tell "my embedding disagrees with tissue
architecture" apart from "the label-transfer classifier itself has
salt-and-pepper noise at the bin/patch level."
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import geopandas as gpd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from functions.dual_autoencoder import build_pairing, extract_cell_xy, build_spatial_graph, refine_domains
from functions import eval_metrics as em
from functions import label_transfer as lt

BASEDIR = REPO_ROOT / "data"

# Mirrors submit_run_loso_final_nuclei.sbatch / submit_run_loso_final_bins.sbatch
# EXACTLY, same as evaluate_final_loso.ipynb cell 3 -- if those scripts' VARIANT_FLAGS
# ever change, update here too (deliberately not auto-derived from the sbatch files).
SLIDES = ["CRC_P1", "CRC_P11mm", "CRC_P11mm_2_cut", "CRC_P2", "CRC_P3_SMAP",
          "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5"]

UNITS = {
    "nuclei": dict(
        bin_size="002um", dir_cellseg="cell_seg_stardist",
        token_filename="cell_token_embeddings.h5ad",
        run_tag_prefix="",  # nuclei's RUN_TAG has no unit-name component
        cap_tag="cap300000",
        spatial_kwargs=dict(delaunay=True, grid=False, n_neighs=6),
    ),
    "bins": dict(
        bin_size="008um", dir_cellseg="grid_bin",
        token_filename="bin_token_embeddings.h5ad",
        run_tag_prefix="_bins",
        cap_tag="cap610000",
        spatial_kwargs=dict(delaunay=False, grid=True, n_neighs=8),
    ),
}

VARIANTS = {
    "svg": dict(middle_tag="", gene_panel=None),
    "colon_xenium_panel": dict(middle_tag="_panel_colon_xenium", gene_panel="colon_xenium"),
    "svg_semantic_excl_raw_0.2": dict(middle_tag="_anchorsim0.2_simsrc_raw", gene_panel=None),
}

LEIDEN_RESOLUTION = 0.5  # matches new_single_omics.ipynb's / evaluate_final_loso.ipynb's own precedent
LABEL_COL = "predicted_tissuetype_grid_image"
SMOOTHING_MODES = ["hard", "spagcn"]  # see refine_domains' docstring for what each means
EMBEDDINGS = ["z_trans", "z_token", "z_joint"]


def run_tag(unit, variant):
    u, v = UNITS[unit], VARIANTS[variant]
    anchorsim_part = v["middle_tag"] if variant == "svg_semantic_excl_raw_0.2" else ""
    panel_part = v["middle_tag"] if variant == "colon_xenium_panel" else ""
    return (f"{u['run_tag_prefix']}_edrop0.1_gdrop0.1{anchorsim_part}_wd1e-05"
            f"{panel_part}_{u['cap_tag']}_oversample_normal")


def checkpoint_dir(unit):
    return BASEDIR / "combined/03_integration/autoencoders" / UNITS[unit]["bin_size"] / "loso_token_checkpoints"


def reconstruct_held_out_cells(unit, held_out_slide):
    """
    Cheap reconstruction (no model forward pass) of one held-out slide's
    cell/bin ordering and coordinates -- mirrors plot_loso_held_out.py,
    generalized over unit (nuclei uses nuclei_geometry.parquet for coords;
    bins carry obsm['spatial'] directly -- see extract_cell_xy's own
    None-safe priority order).

    Returns (cell_xy, trans_sub) or None if this slide has no token data for
    this unit (skipped, matching run_loso.py's/_bins.py's own token-
    availability restriction).
    """
    u = UNITS[unit]
    dir_transcriptomics = BASEDIR / "combined/01_preprocessing" / u["dir_cellseg"] / u["bin_size"] / "custom_binning"

    # backed="r" + to_memory(): the merged file holds every slide -- only this slide's rows
    # are materialized into RAM (see evaluate_final_loso.ipynb's reconstruct_held_out_cells).
    f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad", backed="r")
    f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"].astype(str) == held_out_slide].to_memory()
    trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
    f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)

    geom_path = dir_transcriptomics / "nuclei_geometry.parquet"
    f_trans_coord = gpd.read_parquet(geom_path) if geom_path.exists() else None
    if f_trans_coord is not None:
        f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"] == held_out_slide].copy()

    token_path = BASEDIR / held_out_slide / "01_preprocessing/unimodel/metadata" / u["token_filename"]
    if not token_path.exists():
        print(f"  [{unit}] no {u['token_filename']} for {held_out_slide} -- skipping")
        return None
    f_token_adata = sc.read_h5ad(token_path)
    token_key = f_token_adata.obs["Slide_ID"].astype(str) + "_" + f_token_adata.obs["id"].astype(str)
    f_token_adata.obs = f_token_adata.obs.set_index(token_key)
    f_token_adata = f_token_adata[f_token_adata.obs["matched"].astype(bool)].copy()

    if unit == "nuclei":
        dir_histology = BASEDIR / "combined/01_preprocessing/unimodel/metadata"
        f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad", backed="r")
        f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"] == held_out_slide].to_memory()
        hist_global_id = f_hist_adata.obs["Slide_ID"].astype(str) + "_" + f_hist_adata.obs["Patch_ID"].astype(str)
        f_hist_adata.obs = f_hist_adata.obs.set_index(hist_global_id)
        keep_mask, _, _ = build_pairing(f_trans_adata, f_hist_adata)
    else:
        # bins have no histology-patch pairing step in the bin-level pipeline
        # (see run_loso_bins.py) -- every bin with token data is kept.
        keep_mask = np.ones(f_trans_adata.n_obs, dtype=bool)

    has_token = f_trans_adata.obs_names.isin(f_token_adata.obs_names)
    keep_mask = keep_mask & has_token
    trans_sub = f_trans_adata[keep_mask].copy()

    cell_xy = extract_cell_xy(trans_sub, f_trans_coord)
    return cell_xy, trans_sub


def load_embeddings(unit, variant, held_out_slide):
    """Loads a finished fold's saved npz -- returns None (with a printed
    reason) if it doesn't exist yet or the row-count reconstruction fails."""
    tag = run_tag(unit, variant)
    npz_path = checkpoint_dir(unit) / f"loso_token_{held_out_slide}{tag}_embeddings.npz"
    if not npz_path.exists():
        return None

    recon = reconstruct_held_out_cells(unit, held_out_slide)
    if recon is None:
        return None
    cell_xy, trans_sub = recon

    npz = np.load(npz_path)
    z_trans, z_token = npz["held_out_z_trans"], npz["held_out_z_token"]
    if z_trans.shape[0] != cell_xy.shape[0]:
        print(f"  [{unit}/{variant}/{held_out_slide}] row mismatch: "
              f"{z_trans.shape[0]} embeddings vs {cell_xy.shape[0]} reconstructed cells -- skipping "
              f"(see reconstruct_held_out_cells' docstring before trusting a fix)")
        return None

    return dict(
        cell_xy=cell_xy, trans_sub=trans_sub,
        z_trans=z_trans, z_token=z_token,
        z_joint=np.concatenate([z_trans, z_token], axis=1),  # same convention as plot_loso_held_out.py
    )


def load_grid_reference(slide, label_col=LABEL_COL):
    """Bin-level label-transfer output for one slide, or None if it hasn't
    been produced (see functions/label_transfer.py's main_grid)."""
    path = BASEDIR / slide / "02_baselines/label_transfer" / f"{slide}_grid_predictions.parquet"
    if not path.exists():
        print(f"  no label-transfer grid predictions for {slide} ({path})")
        return None
    return pd.read_parquet(path, columns=["pxl_col", "pxl_row", label_col])


def cluster_embedding(z, resolution=LEIDEN_RESOLUTION, n_neighbors=15, seed=0):
    """Leiden community detection on a kNN graph built directly in embedding
    space (replaces a fixed-k method like KMeans -- Leiden finds its own
    cluster count via `resolution`). Wraps z in a throwaway AnnData purely so
    sc.pp.neighbors/sc.tl.leiden have something to operate on."""
    adata = ad.AnnData(np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    # flavor="igraph" -- confirmed python-igraph is installed in this project's own
    # conda env (both locally and on the cluster). Switched from "leidenalg" after
    # clustering_quality_62769's own log showed it spending ~25h stuck on a single
    # ~786k-nuclei slide (CRC_P11mm) alone -- scanpy's own runtime warning that
    # igraph is "orders of magnitude faster" was not an exaggeration at this scale.
    # n_iterations=2 per scanpy's migration guidance, for parity with leidenalg.
    sc.tl.leiden(adata, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2)
    return adata.obs["leiden"].to_numpy()


def build_slide_graph(unit, cell_xy):
    """This slide's own spatial connectivity graph (unit-appropriate:
    Delaunay for nuclei, grid/Moore-neighborhood for bins -- see UNITS'
    spatial_kwargs). build_spatial_graph needs an AnnData with obsm['spatial']
    + a library_key; a throwaway single-slide AnnData is the cheapest way to
    reuse it as-is."""
    tmp = ad.AnnData(np.zeros((cell_xy.shape[0], 1)))
    tmp.obsm["spatial"] = cell_xy
    # squidpy's spatial-neighbor builders require a categorical library_key column --
    # a plain string column (this notebook's own build_slide_graph cell uses one) raises
    # TypeError at call time, so it's cast explicitly here.
    tmp.obs["Slide_ID"] = pd.Categorical(["single_slide"] * cell_xy.shape[0])
    return build_spatial_graph(tmp, library_key="Slide_ID", **UNITS[unit]["spatial_kwargs"])


def evaluate_fold(unit, variant, slide, smoothing_modes=SMOOTHING_MODES, label_col=LABEL_COL,
                   smooth_reference=False):
    """
    Loads one (unit, variant, slide)'s embeddings once, clusters each of
    z_trans/z_token/z_joint once ("raw"), spatially smooths that same
    clustering once per mode in `smoothing_modes`, and scores ARI/NMI (vs.
    the nearest-grid label-transfer reference) + silhouette/Calinski-Harabasz/
    Davies-Bouldin for every (embedding, label_kind) combination.

    smooth_reference: also scores against a majority-vote-smoothed copy of the
    reference itself (reference_kind "smoothed", alongside "raw") -- see the
    module docstring for why this is a diagnostic add-on, not a replacement.
    Uses "hard" mode only, regardless of `smoothing_modes`: this smooths a
    noisy classifier's own output (a denoising step), not a clustering result
    where spagcn's "protect genuine minority regions" rule would matter.

    Returns a list of row dicts (label_kind "raw" plus one per smoothing
    mode, x1 or x2 for reference_kind), or [] if this fold's embeddings or
    reference aren't available yet.
    """
    data = load_embeddings(unit, variant, slide)
    if data is None:
        return []
    grid_ref = load_grid_reference(slide, label_col)
    if grid_ref is None:
        return []

    reference = lt.assign_nearest_grid_label(data["cell_xy"], grid_ref, label_col=label_col)
    conn = build_slide_graph(unit, data["cell_xy"])

    reference_variants = {"raw": reference}
    if smooth_reference:
        reference_variants["smoothed"] = refine_domains(reference, conn, mode="hard")

    rows = []
    for emb_name in EMBEDDINGS:
        z = data[emb_name]
        raw = cluster_embedding(z)
        labelsets = {"raw": raw}
        for mode in smoothing_modes:
            labelsets[mode] = refine_domains(raw, conn, mode=mode)

        for label_kind, labels in labelsets.items():
            quality = em.cluster_quality_metrics(z, labels)
            for reference_kind, ref_labels in reference_variants.items():
                rows.append({
                    "unit": unit, "variant": variant, "held_out_slide": slide, "embedding": emb_name,
                    "label_kind": label_kind,  # "raw" = pre-smoothing, for a direct before/after comparison
                    "reference_kind": reference_kind,  # "raw"/"smoothed" label-transfer reference
                    "n_units": z.shape[0], "n_clusters_found": len(np.unique(labels)),
                    "ari_vs_label_transfer": em.ari_between_labelings(labels, ref_labels),
                    "nmi_vs_label_transfer": em.nmi_between_labelings(labels, ref_labels),
                    **quality,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", nargs="+", choices=list(UNITS), default=list(UNITS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--slides", nargs="+", choices=SLIDES, default=SLIDES)
    parser.add_argument("--smoothing-modes", nargs="+", choices=["hard", "spagcn"], default=SMOOTHING_MODES)
    parser.add_argument("--label-col", default=LABEL_COL,
                         help="Column in *_grid_predictions.parquet to use as the reference labeling.")
    parser.add_argument("--smooth-reference", action="store_true",
                         help="Also score against a majority-vote-smoothed copy of the label-transfer "
                              "reference (adds a reference_kind column). Diagnostic, not a replacement "
                              "for the raw-reference ARI -- see the module docstring.")
    parser.add_argument("--output-dir", type=Path, default=BASEDIR / "combined/03_integration/autoencoders",
                         help="Where to write clustering_quality.csv / clustering_quality_summary.csv.")
    args = parser.parse_args()

    rows = []
    for unit in args.units:
        for variant in args.variants:
            for slide in args.slides:
                rows.extend(evaluate_fold(unit, variant, slide, args.smoothing_modes, args.label_col,
                                           args.smooth_reference))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = args.output_dir / "clustering_quality.csv"
    summary_path = args.output_dir / "clustering_quality_summary.csv"

    if not rows:
        print("Nothing to report -- no (unit, variant, slide) has both a finished embedding "
              "and a label-transfer reference on disk yet.")
        return

    quality_df = pd.DataFrame(rows)
    quality_df.to_csv(quality_path, index=False)
    print(f"Wrote {len(quality_df)} rows to {quality_path}")

    # before/after pivot: one row per (unit, variant, embedding[, reference_kind]),
    # columns = label_kind -- the quickest way to eyeball "did smoothing help" for
    # ARI/NMI/silhouette at a glance. reference_kind only appears as a grouping level
    # when --smooth-reference was passed (quality_df always has the column, but it's
    # constant ("raw") otherwise, so grouping by it is harmless either way).
    metric_cols = ["ari_vs_label_transfer", "nmi_vs_label_transfer", "silhouette"]
    group_cols = ["unit", "variant", "embedding", "reference_kind", "label_kind"]
    summary_df = (
        quality_df.groupby(group_cols)[metric_cols]
        .mean()
        .unstack("label_kind")
    )
    summary_df.to_csv(summary_path)
    print(f"Wrote summary to {summary_path}")

    print(f"\nMean ARI vs. label transfer, by ({', '.join(group_cols)}):")
    print(
        quality_df.groupby(group_cols)["ari_vs_label_transfer"]
        .agg(["mean", "count"])
        .to_string()
    )


if __name__ == "__main__":
    main()
