#!/bin/bash
#SBATCH --job-name=simplified_adata
#SBATCH --output=logs/simplified_adata_%a.out
#SBATCH --error=logs/simplified_adata_%a.err
#SBATCH --partition=gpu_node
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# One task per slide -- build_simplified_normalised_adata.py is CPU-only (geopandas centroid
# join, squidpy Delaunay graph), no GPU needed despite the gpu_node partition, same as the other
# submit_*.sbatch scripts here. Much lighter than preprocess.py (no raw bin loading/spatial join,
# just reading the already-QC'd count_area_filtered_adata.h5ad + nuclei_geometry.parquet and
# building one graph), hence the smaller --mem/--time than submit_preprocess_cell.sbatch.
#
# Requires count_area_filtered_adata.h5ad and nuclei_geometry.parquet to already be current for
# every slide -- i.e. run after submit_preprocess_cell.sbatch's array has finished. Does NOT need
# the combined merge (submit_merge_combined_nuclei.sbatch) -- this reads and writes per-sample
# files only, nothing from data/combined/.
#
# Concurrency is capped in the --array range itself ("%3" = at most 3 tasks running at once,
# the rest queue and start as slots free up) rather than via a separate throttle script:
#   sbatch --array=0-9%3 submit_build_simplified_normalised_adata.sbatch
# Raise/lower the %3 to allow more/fewer concurrent tasks.

SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 CRC_P6_5mm Normal_P3 Normal_P5)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# see submit_preprocess_cell.sbatch for why both of these are needed on this cluster
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "Processing slide: $SLIDE"
python -u build_simplified_normalised_adata.py --sample "$SLIDE"
