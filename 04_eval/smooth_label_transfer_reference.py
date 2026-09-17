"""
Persists a spatially-smoothed copy of each slide's label-transfer grid
predictions as new columns in that SAME *_grid_predictions.parquet file --
computed once so every downstream ARI evaluation (evaluate_clustering_quality
.py's --smooth-reference, evaluate_spagcn_baseline.py's --smooth-reference)
can read the same precomputed answer instead of re-deriving its own.

Smoothing happens ON THE GRID ITSELF: each 8um grid cell is a node in its own
regular-lattice neighbor graph (build_spatial_graph(..., grid=True,
n_neighs=8) -- the same convention this project already uses for "bins"-unit
data, since a label-transfer grid cell IS exactly that kind of unit), then
majority-vote smoothed (refine_domains(..., mode="hard") -- a denoising step
on the classifier's own output, not a clustering result where spagcn's
"protect genuine minority regions" rule would matter; same reasoning
evaluate_clustering_quality.py already uses for its own reference smoothing).

This is a DIFFERENT smoothing order from evaluate_clustering_quality.py's
existing --smooth-reference, which smooths the reference AFTER assigning each
nucleus its nearest grid cell's label (using the nuclei's own Delaunay graph,
not the grid's). Both are defensible; this one is arguably the more natural
"denoise the reference itself, once, independent of whatever's being scored
against it" version -- ask before assuming evaluate_clustering_quality.py
should switch over to reading this column instead of its own inline version.

For each label column present, adds one new column "<col>_smoothed" and
rewrites the parquet file in place (all existing columns kept as-is).
Columns handled: predicted_tissuetype_grid_image,
predicted_tissuetype_grid_transcriptomics (whichever are actually present --
CRC_P2, the reference sample, has neither and is skipped).

Usage:
    python smooth_label_transfer_reference.py
    python smooth_label_transfer_reference.py --slides CRC_P1 CRC_P5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from functions.dual_autoencoder import build_spatial_graph, refine_domains

BASEDIR = REPO_ROOT / "data"
# Full 10-slide roster (eval_preprocessing.ipynb's `samples`) -- NOT
# evaluate_clustering_quality.py's 9-slide LOSO roster, which deliberately
# excludes CRC_P6_5mm as an unfair LOSO target; that exclusion has nothing to
# do with whether CRC_P6_5mm has a label-transfer reference worth smoothing
# (it does, and has a SpaGCN baseline output too).
SLIDES = ["CRC_P1", "CRC_P2", "CRC_P3_SMAP", "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5",
          "CRC_P6_5mm", "CRC_P11mm", "CRC_P11mm_2_cut"]
LABEL_COLS = ["predicted_tissuetype_grid_image", "predicted_tissuetype_grid_transcriptomics"]


def build_grid_graph(pxl_col, pxl_row):
    """Throwaway single-slide AnnData purely so build_spatial_graph (which
    needs obsm['spatial'] + a categorical library_key) has something to
    operate on -- same trick evaluate_clustering_quality.py's
    build_slide_graph uses."""
    tmp = ad.AnnData(np.zeros((len(pxl_col), 1)))
    tmp.obsm["spatial"] = np.column_stack([pxl_col, pxl_row])
    tmp.obs["Slide_ID"] = pd.Categorical(["single_slide"] * len(pxl_col))
    return build_spatial_graph(tmp, library_key="Slide_ID", grid=True, delaunay=False, n_neighs=8)


def smooth_slide(slide, label_cols=LABEL_COLS):
    """Reads, smooths, and rewrites one slide's *_grid_predictions.parquet.
    Returns True if it wrote anything, False if skipped (no file, or none of
    label_cols present)."""
    path = BASEDIR / slide / "02_baselines/label_transfer" / f"{slide}_grid_predictions.parquet"
    if not path.exists():
        print(f"  no label-transfer grid predictions for {slide} ({path}) -- skipping")
        return False

    df = pd.read_parquet(path)
    present_cols = [c for c in label_cols if c in df.columns]
    if not present_cols:
        print(f"  {slide}: none of {label_cols} present in {path.name} -- skipping")
        return False

    conn = build_grid_graph(df["pxl_col"].to_numpy(), df["pxl_row"].to_numpy())

    for col in present_cols:
        raw_labels = df[col].to_numpy()
        smoothed = refine_domains(raw_labels, conn, mode="hard")
        n_changed = int((smoothed != raw_labels).sum())
        df[f"{col}_smoothed"] = smoothed
        print(f"  {slide}/{col}: {n_changed} / {len(raw_labels)} grid cells reassigned by smoothing "
              f"({n_changed / len(raw_labels):.1%})")

    df.to_parquet(path, index=False)
    print(f"  wrote {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slides", nargs="+", choices=SLIDES, default=SLIDES)
    parser.add_argument("--label-cols", nargs="+", default=LABEL_COLS)
    args = parser.parse_args()

    for slide in args.slides:
        print(f"{slide}:")
        smooth_slide(slide, args.label_cols)


if __name__ == "__main__":
    main()
