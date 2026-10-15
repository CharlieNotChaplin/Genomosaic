"""
Aggregates each slide's raw 2um Visium HD bins into 8x8um grid cells (summed
per-gene UMI counts) -- the grid-bin analogue of CustomBinning.ipynb's
nuclei_grouped_UMI_count.h5ad step, but
grouping by a regular 8um grid cell instead of by StarDist nucleus polygon.
No QC/count-area filtering is applied here (that's preprocess.py --unit bin's
job, producing count_area_filtered_adata.h5ad downstream) -- this is the RAW,
pre-filter population, exactly parallel to what nuclei_grouped_UMI_count.h5ad
already is on the nuclei side.

The actual aggregation is NOT reimplemented here -- it reuses
functions/transcriptomics.py's load_grid_bins, the same loader preprocess.py
--unit bin uses: SpaceRanger's native square_008um output when present, else
exact 4x4-block summation from square_002um (see label_transfer.py's
aggregate_expression_to_grid). Unlike preprocess.py, it also falls back to
square_002um when the native grid covers a different region (check_region=True).

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
The incremental merge loads the whole combined file, so peak RAM grows with it
and is worst for the last sample (about 220 GB for all ten slides, CRC_P11mm
last). Reruns are safe, finished samples are skipped (--force redoes them).

Usage:
    python aggregate_grid_bin_UMI_count.py
    python aggregate_grid_bin_UMI_count.py --samples CRC_P1 CRC_P2
"""

import argparse
import gc
import sys
from pathlib import Path

# repo root = first parent containing functions/ and data/
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "functions").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from functions.merge import merge_anndatas
from functions.transcriptomics import load_grid_bins

SAMPLES = ["CRC_P1", "CRC_P2", "CRC_P3_SMAP", "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5",
           "CRC_P6_5mm", "CRC_P11mm", "CRC_P11mm_2_cut"]

BASEDIR = REPO_ROOT / "data"
BIN_SIZE = "008um"
DIR_BINNING = "grid_bin"
FILENAME = "nuclei_grouped_UMI_count.h5ad"
PATH_TEMPLATE = "{basedir}/{sample}/01_preprocessing/" + DIR_BINNING + "/{bin_size}/custom_binning/{filename}"
REFERENCE_ONLY_VAR_COLS = {"gene_ids", "feature_types", "genome"}


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

    # check_region: native square_008um can cover a larger region than the cropped square_002um
    # (CRC_P11mm_2_cut), in that case aggregate from square_002um
    adata = load_grid_bins(sample, BASEDIR, check_region=True)

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
