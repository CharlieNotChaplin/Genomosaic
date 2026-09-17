"""
Aggregates each slide's raw 2um Visium HD bins into 8x8um grid cells (summed
per-gene UMI counts) -- the grid-bin analogue of CustomBinning.ipynb's /
regenerate_grouping_files.py's nuclei_grouped_UMI_count.h5ad step, but
grouping by a regular 8um grid cell instead of by StarDist nucleus polygon.
No QC/count-area filtering is applied here (that's bin_preprocessing.py's
job, producing count_area_filtered_adata.h5ad downstream) -- this is the RAW,
pre-filter population, exactly parallel to what nuclei_grouped_UMI_count.h5ad
already is on the nuclei side.

The actual aggregation is NOT reimplemented here -- it reuses
functions/label_transfer.py's loaders, the same ones bin_preprocessing.py's
own "load bin counts" step already uses: load_grid_bin_counts (SpaceRanger's
native square_008um output, when present) or aggregate_expression_to_grid
(exact 4x4-block summation from square_002um otherwise -- not an
approximation, see that function's own docstring).

Writes one per-sample file:
    data/{sample}/01_preprocessing/grid_bin/008um/custom_binning/nuclei_grouped_UMI_count.h5ad
(same filename the nuclei path uses, for path-template parallelism with
functions/merge.py's merge_anndatas), then immediately folds it into

    data/combined/01_preprocessing/grid_bin/008um/custom_binning/nuclei_grouped_UMI_count.h5ad

via merge_anndatas(..., merge_mode="incremental") -- the same mechanism
CustomBinning.ipynb's own merge cell uses for the nuclei version, which reads
the existing combined file, appends only the samples not already in it, and
writes the result back. Samples are processed strictly one at a time
(aggregate -> write per-sample -> merge into combined -> free everything)
so peak memory is one slide's raw bin counts plus the current combined file,
never every slide's raw data at once.

Must be run from THIS directory (01_preprocessing/transcriptomics/) --
same "../../data" convention bin_preprocessing.py/regenerate_grouping_files.py
already use.

Usage:
    python aggregate_grid_bin_UMI_count.py
    python aggregate_grid_bin_UMI_count.py --samples CRC_P1 CRC_P2
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("../..").resolve()))
from functions.label_transfer import load_grid_bin_counts, aggregate_expression_to_grid
from functions.merge import merge_anndatas

SAMPLES = ["CRC_P1", "CRC_P2", "CRC_P3_SMAP", "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5",
           "CRC_P6_5mm", "CRC_P11mm", "CRC_P11mm_2_cut"]

BASEDIR = Path("../../data")
BIN_SIZE = "008um"
DIR_BINNING = "grid_bin"
FILENAME = "nuclei_grouped_UMI_count.h5ad"
PATH_TEMPLATE = "{basedir}/{sample}/01_preprocessing/" + DIR_BINNING + "/{bin_size}/custom_binning/{filename}"
REFERENCE_ONLY_VAR_COLS = {"gene_ids", "feature_types", "genome"}


def native_grid_is_consistent(sample, tol=(12, 20)):
    """
    Sanity check before trusting a native square_008um output: its in-tissue
    bin count should be ~16x FEWER than square_002um's (4x4 aggregation) --
    checked directly rather than assumed, after finding CRC_P11mm_2_cut's
    ratio come out at 0.32 instead (every other sample checked out at
    15.8-16.0). Not stale/mismatched data: square_002um for this sample was
    deliberately reduced to only the tissue section actually being used, so
    it's SMALLER than the native square_008um grid, which still covers the
    original, uncropped capture area. Aggregating from that native file
    would pull in bins outside the region the rest of this sample's pipeline
    (nuclei, SVGs, ...) is scoped to -- aggregate_expression_to_grid, run
    directly on the (already-cropped) square_002um data, stays consistent
    with that scope instead. This is the same native-vs-fallback check
    bin_preprocessing.py's own "load bin counts" step uses, so that script
    is equally exposed to this for any sample it hasn't been run on yet.
    """
    p8 = BASEDIR / sample / "output/binned_outputs/square_008um/spatial/tissue_positions.parquet"
    p2 = BASEDIR / sample / "output/binned_outputs/square_002um/spatial/tissue_positions.parquet"
    n8 = (pd.read_parquet(p8, columns=["in_tissue"])["in_tissue"] == 1).sum()
    n2 = (pd.read_parquet(p2, columns=["in_tissue"])["in_tissue"] == 1).sum()
    ratio = n2 / n8 if n8 else float("nan")
    ok = tol[0] <= ratio <= tol[1]
    if not ok:
        print(f"{sample}: native square_008um covers a different region than square_002um "
              f"(n_002um/n_008um={ratio:.2f}, expected ~16) -- falling back to "
              f"aggregate_expression_to_grid (from square_002um) to stay scoped to the same region")
    return ok


def aggregate_sample(sample, force=False):
    """Writes this sample's own raw 8um-grid-grouped UMI count file (no
    QC/filtering) and returns its path. Skips the (potentially large) load
    if that file already exists and force=False -- makes reruns after a
    partial failure (e.g. an OOM partway through --samples) cheap, since
    they won't re-read every already-finished sample's raw counts."""
    dir_custom_binning = BASEDIR / sample / "01_preprocessing" / DIR_BINNING / BIN_SIZE / "custom_binning"
    dir_custom_binning.mkdir(parents=True, exist_ok=True)
    out_path = dir_custom_binning / FILENAME

    if out_path.exists() and not force:
        print(f"{sample}: {out_path} already exists -- skipping (pass --force to redo)")
        return out_path

    grid_h5_path = BASEDIR / sample / "output/binned_outputs/square_008um/filtered_feature_bc_matrix.h5"
    if grid_h5_path.exists() and native_grid_is_consistent(sample):
        print(f"{sample}: native square_008um output found")
        adata = load_grid_bin_counts(sample, data_dir=BASEDIR)
    else:
        print(f"{sample}: aggregating from square_002um instead")
        adata = aggregate_expression_to_grid(sample, data_dir=BASEDIR)

    # Same obs normalization bin_preprocessing.py applies right after its own
    # identical load step, so this file's obs schema matches what the rest of
    # the bin pipeline (and merge_anndatas' obs_name_mode="prefix" default,
    # which builds obs_names as f"{sample}_{obs_name}") expects.
    adata.obs["Slide_ID"] = sample
    adata.obs["id"] = [f"grid_{r}_{c}" for r, c in zip(adata.obs["array_row"], adata.obs["array_col"])]
    adata.obs_names = adata.obs["id"].values
    adata.var_names_make_unique()

    adata.write(out_path)
    print(f"{sample}: wrote {out_path} ({adata.n_obs} grid bins, {adata.n_vars} genes)")
    del adata
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=SAMPLES)
    parser.add_argument("--force", action="store_true",
                         help="Redo a sample's aggregation even if its per-sample file already exists.")
    args = parser.parse_args()

    for sample in args.samples:
        print(f"\n=== {sample} ===")
        aggregate_sample(sample, force=args.force)
        gc.collect()

        merged = merge_anndatas(
            [sample],  # incremental mode: only this one gets read (already-merged ones are skipped anyway)
            PATH_TEMPLATE,
            FILENAME,
            BASEDIR,
            auto_detect_var_cols=True,
            reference_only_var_cols=REFERENCE_ONLY_VAR_COLS,
            bin_size=BIN_SIZE,
            merge_mode="incremental",
        )
        del merged
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
