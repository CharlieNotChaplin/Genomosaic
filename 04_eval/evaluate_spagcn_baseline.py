"""
ARI/NMI for the SpaGCN spatial-domain baseline (02_baselines/spagcn/outputs/
*_cluster_*.parquet) against the SAME nearest-grid label-transfer reference
evaluate_clustering_quality.py scores the autoencoder's Leiden clusters
against -- lets the two be compared apples-to-apples.

Scores BOTH columns each output file already carries:
  - pred: SpaGCN's raw GCN clustering, no smoothing.
  - refined_pred: SpaGCN's OWN built-in majority-vote smoothing (the
    package's spg.refine() -- a different implementation from this project's
    refine_domains, not to be confused with it).
against the label-transfer reference (each nucleus's nearest 8um grid cell's
predicted_tissuetype_grid_image).

--smooth-reference additionally scores against a SMOOTHED copy of that same
reference, read from the "<label-col>_smoothed" column that
smooth_label_transfer_reference.py adds to each *_grid_predictions.parquet
file (run that script first -- this script does NOT compute the smoothing
itself, it only reads the column if present). That script smooths on the
GRID's own regular-lattice neighbor graph, once, shared across every
consumer -- a different (and simpler) smoothing order than
evaluate_clustering_quality.py's own --smooth-reference, which instead
smooths AFTER assigning each nucleus its nearest grid cell's label, using the
nuclei's own Delaunay graph. Both are defensible; they are not guaranteed to
agree.

Coordinates for the nearest-grid lookup come from the COMBINED
count_area_filtered_adata.h5ad (obsm['spatial'], joined by the same
obs['id'] + obs['Slide_ID'] each output parquet's own "id"/"sample" columns
were built from -- see 02_baselines/spagcn/run_spagcn_svg.py's
build_adj_and_centroids/load_sample_adata), NOT anything under
data/<slide>/02_baselines/spagcn/ (those intermediates are deleted after a
run; only the compact outputs/*.parquet files persist).

Usage:
    python evaluate_spagcn_baseline.py
    python evaluate_spagcn_baseline.py --samples CRC_P5 CRC_P2
    python evaluate_spagcn_baseline.py --smooth-reference
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from functions import eval_metrics as em
from functions import label_transfer as lt

BASEDIR = REPO_ROOT / "data"
SPAGCN_OUTPUTS_DIR = REPO_ROOT / "02_baselines/spagcn/outputs"
DIR_COMBINED_ADATA = (
    BASEDIR / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning/count_area_filtered_adata.h5ad"
)
LABEL_COL = "predicted_tissuetype_grid_image"
LABEL_KINDS = ["pred", "refined_pred"]


def load_grid_reference(slide, columns):
    """Bin-level label-transfer output for one slide (pxl_col/pxl_row plus
    whichever of `columns` actually exist in the file), or None if the file
    itself hasn't been produced (see functions/label_transfer.py's
    main_grid). Missing individual columns (e.g. a "*_smoothed" one that
    smooth_label_transfer_reference.py hasn't been run for yet) are dropped
    with a printed warning rather than raising -- callers must check for
    their presence in the returned DataFrame."""
    path = BASEDIR / slide / "02_baselines/label_transfer" / f"{slide}_grid_predictions.parquet"
    if not path.exists():
        print(f"  no label-transfer grid predictions for {slide} ({path})")
        return None
    available = set(pq.ParquetFile(path).schema.names)
    missing = [c for c in columns if c not in available]
    if missing:
        print(f"  {slide}: {missing} not in {path.name} -- run smooth_label_transfer_reference.py first "
              f"if you need a '*_smoothed' column")
    cols_to_read = ["pxl_col", "pxl_row"] + [c for c in columns if c in available]
    return pd.read_parquet(path, columns=cols_to_read)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", nargs="+", default=None,
                         help="Restrict to these slides (default: every sample with output files present).")
    parser.add_argument("--label-col", default=LABEL_COL,
                         help="Column in *_grid_predictions.parquet to use as the reference labeling.")
    parser.add_argument("--smooth-reference", action="store_true",
                         help="Also score against '<label-col>_smoothed' (added by "
                              "smooth_label_transfer_reference.py -- run that first). Adds a "
                              "reference_kind column ('raw'/'smoothed').")
    parser.add_argument("--outputs-dir", type=Path, default=SPAGCN_OUTPUTS_DIR,
                         help="Directory of <sample>_cluster_<k>.parquet files (id/sample/n_clusters/pred[, "
                              "refined_pred]). Any baseline with that layout works, e.g. "
                              "02_baselines/stagate/outputs -- the same scoring, so results are comparable.")
    parser.add_argument("--label-kinds", nargs="+", default=LABEL_KINDS,
                         help="Label columns to score (STAGATE/SpatialFusion only have 'pred').")
    parser.add_argument("--output", type=Path, default=None,
                         help="Default: <outputs-dir>/<baseline>_clustering_quality.csv, baseline = the "
                              "outputs dir's parent folder name.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.outputs_dir / f"{args.outputs_dir.parent.name}_clustering_quality.csv"

    smoothed_col = f"{args.label_col}_smoothed"
    columns_needed = [args.label_col, smoothed_col] if args.smooth_reference else [args.label_col]

    files = sorted(args.outputs_dir.glob("*_cluster_*.parquet"))
    if args.samples:
        files = [f for f in files if any(f.name.startswith(f"{s}_cluster_") for s in args.samples)]
    if not files:
        print("No matching SpaGCN output files found.")
        return

    print(f"Loading combined transcriptomics adata for spatial coords: {DIR_COMBINED_ADATA}")
    trans_combined = sc.read_h5ad(DIR_COMBINED_ADATA, backed="r")
    obs_id = trans_combined.obs["id"].astype(str)
    obs_slide = trans_combined.obs["Slide_ID"].astype(str)
    spatial = np.asarray(trans_combined.obsm["spatial"])

    reference_cache = {}
    rows = []
    for f in files:
        spagcn_df = pd.read_parquet(f)
        sample = str(spagcn_df["sample"].iloc[0])
        n_clusters = int(spagcn_df["n_clusters"].iloc[0])
        print(f"{f.name}: sample={sample} n_clusters={n_clusters} n_rows={len(spagcn_df)}")

        if sample not in reference_cache:
            reference_cache[sample] = load_grid_reference(sample, columns_needed)
        grid_ref = reference_cache[sample]
        if grid_ref is None:
            continue

        reference_cols = {"raw": args.label_col}
        if args.smooth_reference:
            if smoothed_col in grid_ref.columns:
                reference_cols["smoothed"] = smoothed_col
            else:
                print(f"  {sample}: skipping smoothed reference (column not available)")

        mask = (obs_slide == sample).to_numpy()
        coord_lookup = pd.DataFrame(spatial[mask], index=obs_id[mask].to_numpy(), columns=["x", "y"])
        coords = coord_lookup.reindex(spagcn_df["id"].astype(str).to_numpy())
        missing = coords["x"].isna()
        if missing.any():
            print(f"  {missing.sum()} / {len(spagcn_df)} SpaGCN ids not found in the combined adata for "
                  f"{sample} -- dropping them")
        keep = ~missing.to_numpy()
        spagcn_df = spagcn_df.loc[keep].reset_index(drop=True)
        coords = coords.loc[keep].to_numpy()
        if len(spagcn_df) == 0:
            continue

        for reference_kind, ref_col in reference_cols.items():
            reference = lt.assign_nearest_grid_label(coords, grid_ref, label_col=ref_col)
            for label_kind in args.label_kinds:
                labels = spagcn_df[label_kind].to_numpy()
                rows.append({
                    "sample": sample, "n_clusters": n_clusters, "label_kind": label_kind,
                    "reference_kind": reference_kind,
                    "n_units": len(labels), "n_clusters_found": len(np.unique(labels)),
                    "ari_vs_label_transfer": em.ari_between_labelings(labels, reference),
                    "nmi_vs_label_transfer": em.nmi_between_labelings(labels, reference),
                })

    if not rows:
        print("Nothing scored -- no sample had both a SpaGCN output file and a label-transfer reference.")
        return

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(result_df)} rows to {args.output}")
    print(result_df.sort_values(["sample", "n_clusters", "reference_kind", "label_kind"]).to_string(index=False))


if __name__ == "__main__":
    main()
