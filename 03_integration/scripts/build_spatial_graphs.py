"""
Precomputes and caches each slide's spatial graph -- the same one build_pipeline_data builds on the
fly for training/LOSO (functions.dual_autoencoder.build_spatial_graph: Delaunay + 95th-percentile
edge pruning for nuclei, an 8-neighbor Moore grid for bins). NOT get_svg.py's own graph, which uses
different parameters (no pruning for cells, only 4 orthogonal neighbors for bins) and exists only for
its own Moran's I call.

Not wired into training -- build_pipeline_data still recomputes its own graph every run, cheaply,
from obsm['spatial']. This script exists to answer "can this be computed one slide at a time, and is
a dense matrix the right way to store it": no dense storage needed (the graph is sparse to begin
with, scipy.sparse.csr_matrix -- a dense (n, n) matrix would be roughly 616GB for CRC_P11mm's ~785k
nuclei alone), and slides are processed strictly one at a time here, loading only that slide's own
coordinates (never its expression matrix) before freeing it and moving on.

Output, one file per (unit, slide), sibling to that slide's own count_area_filtered_adata.h5ad:
    data/<slide>/01_preprocessing/<dir_cellseg>/<bin_size>/custom_binning/spatial_graph.h5ad
An AnnData with no real expression (X is a (n, 1) placeholder): obs['id'] is the row identity that
matches nuclei_geometry.parquet's own "id" column (NOT obs_names, which is a synthetic per-slide
position -- verified these are different directly: obs_names gives e.g. "Normal_P3_0", obs['id']
and nuclei_geometry.parquet's "id" both give "ID_51773" for that same row). obsm['spatial'] is the
coordinates the graph was built from; obsp['spatial_connectivities'] / obsp['spatial_distances'] is
the sparse graph itself -- anndata's h5ad writer stores a sparse obsp as its CSR components, not as
a dense block, so no manual sparse handling is needed here or on reload.
Reload with: anndata.read_h5ad(path).obsp['spatial_connectivities']

Usage (from 03_integration/scripts/):
    python build_spatial_graphs.py --unit nuclei
    python build_spatial_graphs.py --unit bins --slides CRC_P1 CRC_P2
"""
import argparse
import gc
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break
from functions.dual_autoencoder import build_spatial_graph
from functions.training_pipeline import DATA_DIR, UNIT_PRESETS


def trans_path_for(unit, basedir):
    preset = UNIT_PRESETS[unit]
    return (basedir / "combined/01_preprocessing" / preset["dir_cellseg"] / preset["bin_size"]
            / "custom_binning" / "count_area_filtered_adata.h5ad")


def slide_coords(unit, slide, basedir):
    """This slide's own (obs_names, ids, xy) from the merged transcriptomics file. Only obs and
    obsm['spatial'] are ever read -- X (the actual expression matrix) is never touched, so this
    stays light regardless of how many genes that file has.

    obs_names (the AnnData index, e.g. "Normal_P3_0") is a synthetic per-slide position, NOT the
    same id nuclei_geometry.parquet's own "id" column uses -- obs['id'] (e.g. "ID_51773") is the
    one that matches it, value for value, same row order (verified directly against both files).
    Both are returned; obs['id'] is what you want to join back to a nucleus polygon."""
    f = ad.read_h5ad(trans_path_for(unit, basedir), backed="r")
    mask = (f.obs["Slide_ID"].astype(str) == slide).to_numpy()
    obs_names = f.obs_names[mask].to_numpy()
    ids = f.obs["id"].astype(str).to_numpy()[mask] if "id" in f.obs.columns else None
    xy = np.asarray(f.obsm["spatial"])[mask]
    f.file.close()
    return obs_names, ids, xy


def build_and_save(unit, slide, basedir, out_name):
    preset = UNIT_PRESETS[unit]
    obs_names, ids, xy = slide_coords(unit, slide, basedir)
    if len(obs_names) == 0:
        print(f"[{unit}/{slide}] no units found in the merged transcriptomics file -- skipping")
        return

    tmp = ad.AnnData(np.zeros((len(xy), 1), dtype=np.float32), obs=pd.DataFrame(index=obs_names))
    tmp.obsm["spatial"] = xy
    tmp.obs["Slide_ID"] = pd.Categorical([slide] * len(xy))  # build_spatial_graph's own library_key requirement
    if ids is not None:
        tmp.obs["id"] = ids  # the id that matches nuclei_geometry.parquet / the bin grid id, not obs_names
    conn = build_spatial_graph(
        tmp, library_key="Slide_ID",
        n_neighs=preset["n_neighs"], delaunay=preset["spatial_graph_delaunay"], grid=preset["spatial_graph_grid"],
    )
    # build_spatial_graph already wrote obsp['spatial_connectivities']/['spatial_distances'] onto
    # tmp (and returned the connectivities matrix directly too) -- tmp already has everything needed.

    out_dir = basedir / slide / "01_preprocessing" / preset["dir_cellseg"] / preset["bin_size"] / "custom_binning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    tmp.write_h5ad(out_path)
    print(f"[{unit}/{slide}] {len(obs_names):,} units, {conn.nnz:,} edges -> {out_path}")

    del tmp, conn, obs_names, xy
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", required=True, choices=["nuclei", "bins"])
    ap.add_argument("--slides", nargs="+", default=None,
                     help="defaults to every slide present in the merged transcriptomics file; "
                          "processed one at a time regardless of how many are given")
    ap.add_argument("--basedir", type=Path, default=None)
    ap.add_argument("--out-name", default="spatial_graph.h5ad")
    args = ap.parse_args()
    basedir = args.basedir or DATA_DIR

    slides = args.slides
    if slides is None:
        f = ad.read_h5ad(trans_path_for(args.unit, basedir), backed="r")
        slides = sorted(f.obs["Slide_ID"].astype(str).unique())
        f.file.close()
        print(f"no --slides given, discovered {len(slides)} slides in the merged file: {slides}")

    for slide in slides:
        build_and_save(args.unit, slide, basedir, args.out_name)


if __name__ == "__main__":
    main()
