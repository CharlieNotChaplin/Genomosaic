"""
Script version of NucleusPreprocessing.ipynb's merge cells (34-36), nuclei-only -- merging itself
has no script otherwise ("merging is only available in this notebook", per that notebook's own
markdown), which is fine on a workstation but not on a cluster compute node, where this needs to
run as its own batch job after submit_preprocess_cell.sbatch's array has written every sample's
per-sample files.

Always merges with merge_mode="full": every sample listed is re-read from its current per-sample
files and the combined output is rebuilt from scratch, never "incremental" (which skips any sample
whose Slide_ID is already in the existing combined file, however stale that sample's contribution
actually is -- this is what caused nuclei_geometry.parquet to drift out of sync with
count_area_filtered_adata.h5ad in the first place: an earlier merge of the parquet ran
incrementally after a later full-mode merge of the adata had already moved on).

Does NOT run NucleusPreprocessing.ipynb's cell 37 (recompute_global_gene_selection): that recomputes
HVG/SVG/PCA selection from each sample's per-sample Moran's I (uns["moranI"]), which only exists if
get_svg.py was run first. Skipping SVGs means there is no moranI to recompute from -- this merges
counts and geometry only, same as this project's preprocess.py (which also deliberately skips SVGs).
Run get_svg.py per sample and this notebook's cell 37 separately, afterwards, if/when SVGs are needed.

Usage (from 01_preprocessing/transcriptomics/scripts/):
    python merge_combined_nuclei.py
    python merge_combined_nuclei.py --samples CRC_P1 CRC_P2 Normal_P3
"""
import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "functions").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from functions.merge import merge_anndatas, merge_parquet

BASEDIR = REPO_ROOT / "data"
BIN_SIZE = "002um"
PATH_TEMPLATE = "{basedir}/{sample}/01_preprocessing/cell_seg_stardist/{bin_size}/custom_binning/{filename}"
# var columns identical across samples, excluded from the per-sample varm bookkeeping
REFERENCE_ONLY_VAR_COLS = {"gene_ids", "feature_types", "genome"}


def discover_samples(basedir):
    exclude_dirs = {"new", "archived", "combined"}
    return sorted(p.name for p in basedir.iterdir() if p.is_dir() and p.name not in exclude_dirs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", default=None,
                     help="defaults to every sample directory under data/ (excluding new/archived/combined)")
    ap.add_argument("--basedir", type=Path, default=None)
    args = ap.parse_args()
    basedir = args.basedir or BASEDIR
    samples = args.samples or discover_samples(basedir)
    print(f"Merging {len(samples)} samples: {samples}")

    print("merging nuclei (pre-QC, grouped counts)")
    merged = merge_anndatas(
        samples, PATH_TEMPLATE, "nuclei_grouped_UMI_count.h5ad", basedir,
        auto_detect_var_cols=True, reference_only_var_cols=REFERENCE_ONLY_VAR_COLS,
        bin_size=BIN_SIZE, merge_mode="full",
    )
    del merged
    gc.collect()

    print("merging custom bins (2um bins assigned to a nucleus)")
    merged = merge_anndatas(
        samples, PATH_TEMPLATE, "filtered_custom_bins.h5ad", basedir,
        auto_detect_var_cols=True, reference_only_var_cols=REFERENCE_ONLY_VAR_COLS,
        bin_size=BIN_SIZE, merge_mode="full",
    )
    del merged
    gc.collect()

    print("merging count_area_filtered_adata (QC'd, clustered nuclei)")
    merged = merge_anndatas(
        samples, PATH_TEMPLATE, "count_area_filtered_adata.h5ad", basedir,
        auto_detect_var_cols=True, reference_only_var_cols=REFERENCE_ONLY_VAR_COLS,
        bin_size=BIN_SIZE, per_sample_uns_moran_key="moranI", merge_mode="full",
    )
    del merged
    gc.collect()

    print("merging nuclei_geometry (polygons)")
    merged_parquet = merge_parquet(
        samples, PATH_TEMPLATE, "nuclei_geometry.parquet", basedir,
        bin_size=BIN_SIZE, merge_mode="full",
    )
    del merged_parquet
    gc.collect()

    print("done.")


if __name__ == "__main__":
    main()
