"""
Per-sample, SVG-free alternative to get_svg.py: takes the existing count_area_filtered_adata.h5ad
(already normalized, log1p'd, HVG-flagged and clustered by preprocess.py's normalize_and_cluster --
see functions/transcriptomics.py) and adds the one thing it's still missing for anything that reads
a spatial graph: obsm['spatial'] and the pruned spatial neighbor graph, WITHOUT touching Moran's
I/SVGs (get_svg.py's own, separate, much more expensive step).

Two things this deliberately does NOT reuse from get_svg.py:
- obsm['spatial'] is computed here the same way (nucleus centroids from nuclei_geometry.parquet,
  reindexed onto count_area_filtered_adata's own id/row order, with the same "did every id
  match" assertion get_svg.py has -- this is exactly the join that was silently wrong before the
  nuclei_geometry.parquet/count_area_filtered_adata.h5ad rebuild, so the assertion is not optional).
- The spatial graph itself is NOT get_svg.py's own sq.gr.spatial_neighbors call (plain Delaunay,
  no pruning -- built only to feed Moran's I, never used elsewhere). It is
  functions.dual_autoencoder.build_spatial_graph, the exact function build_pipeline_data calls at
  training time (Delaunay + 95th-percentile edge-length pruning for nuclei), via UNIT_PRESETS["nuclei"]
  from functions/training_pipeline.py -- so these edges are the ones the model actually trains on,
  not a different, unpruned graph that happens to also be Delaunay-based.

Output, written next to (not replacing) count_area_filtered_adata.h5ad, and not merged into
data/combined/...:
    data/<sample>/01_preprocessing/cell_seg_stardist/002um/custom_binning/simplified_normalised_adata.h5ad
X: full normalized+log1p expression (every gene, not SVG- or HVG-restricted -- var['highly_variable']
   is still there to subset by later, same column normalize_and_cluster already wrote).
obsm['spatial']: nucleus centroids.
obsp['spatial_connectivities'] / obsp['spatial_distances']: the pruned graph above.

Usage (from 01_preprocessing/transcriptomics/scripts/):
    python build_simplified_normalised_adata.py --sample CRC_P1
"""
import argparse
import gc
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "functions").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from functions.dual_autoencoder import build_spatial_graph
from functions.merge import read_h5ad_with_null_uns_healing, sanitize_uns
from functions.training_pipeline import UNIT_PRESETS

PRESET = UNIT_PRESETS["nuclei"]  # bin_size=002um, dir_cellseg=cell_seg_stardist, n_neighs=6, delaunay, no grid
OUT_NAME = "simplified_normalised_adata.h5ad"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="Sample name, e.g. CRC_P1")
    ap.add_argument("--basedir", type=Path, default=None)
    args = ap.parse_args()
    basedir = args.basedir or (REPO_ROOT / "data")
    dir_custom_binning = basedir / args.sample / "01_preprocessing" / PRESET["dir_cellseg"] / PRESET["bin_size"] / "custom_binning"

    print(f"[{args.sample}] loading count_area_filtered_adata.h5ad")
    adata = read_h5ad_with_null_uns_healing(dir_custom_binning / "count_area_filtered_adata.h5ad")
    adata.var_names_make_unique()

    print(f"[{args.sample}] loading nuclei_geometry.parquet, aligning centroids onto adata's own id order")
    nuclei_geometry = gpd.read_parquet(dir_custom_binning / "nuclei_geometry.parquet")
    if "centroid" not in nuclei_geometry.columns:
        nuclei_geometry["centroid"] = nuclei_geometry.geometry.centroid
    geometry_aligned = nuclei_geometry.set_index("id").reindex(adata.obs["id"])
    n_missing = geometry_aligned["centroid"].isnull().sum()
    assert n_missing == 0, (
        f"{n_missing} nucleus id(s) in count_area_filtered_adata.h5ad have no matching row in "
        f"nuclei_geometry.parquet -- this is exactly the drift bug the recent rebuild was meant to "
        f"fix; re-run preprocess.py --sample {args.sample} --unit cell before this script."
    )
    adata.obsm["spatial"] = np.column_stack([geometry_aligned["centroid"].x.values, geometry_aligned["centroid"].y.values])
    del nuclei_geometry, geometry_aligned
    gc.collect()

    print(f"[{args.sample}] building the model's spatial graph (Delaunay + 95th-pct pruning, n_neighs={PRESET['n_neighs']})")
    if "Slide_ID" not in adata.obs or adata.obs["Slide_ID"].astype(str).nunique() != 1:
        adata.obs["Slide_ID"] = args.sample
    conn = build_spatial_graph(
        adata, library_key="Slide_ID",
        n_neighs=PRESET["n_neighs"], delaunay=PRESET["spatial_graph_delaunay"], grid=PRESET["spatial_graph_grid"],
    )

    stray_nones = sanitize_uns(adata)
    if stray_nones:
        print(f"[{args.sample}] sanitized stray null uns entries before saving: {stray_nones}")

    out_path = dir_custom_binning / OUT_NAME
    adata.write(out_path)
    print(f"[{args.sample}] {adata.n_obs:,} nuclei, {adata.n_vars:,} genes "
          f"({int(adata.var['highly_variable'].sum()):,} HVGs), {conn.nnz:,} edges -> {out_path}")


if __name__ == "__main__":
    main()
