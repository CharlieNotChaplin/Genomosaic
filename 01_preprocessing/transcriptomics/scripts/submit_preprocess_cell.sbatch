#!/bin/bash
#SBATCH --job-name=preprocess_cell
#SBATCH --output=logs/preprocess_cell_%a.out
#SBATCH --error=logs/preprocess_cell_%a.err
#SBATCH --partition=gpu_node
#SBATCH -c 16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# One task per slide -- preprocess.py --unit cell is CPU-only (geopandas sjoin,
# scanpy QC/clustering), no GPU needed despite the gpu_node partition, same as
# submit_token_preprocessing.sbatch.
#
# Rebuilds the per-sample nuclei outputs from scratch (filtered_custom_bins.h5ad,
# nuclei_grouped_UMI_count.h5ad, count_area_filtered_adata.h5ad, nuclei_geometry.parquet)
# WITHOUT spatially variable genes -- that is get_svg.py, run separately, not called here.
# CRC_P1 already finished locally before this was moved to the cluster -- not included
# below, rerun it too if it needs to be regenerated for consistency.
#
# The local machine (30G RAM) ran out of memory on CRC_P11mm, the next slide after
# CRC_P1 (more StarDist nuclei / 2um bins than CRC_P1): the spatial join (sjoin) and the
# per-nucleus count summation both hold the full per-slide bin table and a growing lil_matrix
# in RAM at once, so this needs considerably more than the local box has. 180G is a guess
# sized for CRC_P11mm/CRC_P5 (the two largest slides, ~900-950MB StarDist polygon pickles);
# lower it for a lighter rerun of just the smaller slides if needed.
#
# Usage:
#   sbatch --array=0-8 submit_preprocess_cell.sbatch

SLIDES=(CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 CRC_P6_5mm Normal_P3 Normal_P5)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# numba (a scanpy dependency) decorates a function with @njit(cache=True) at import time and
# tries to write its compile cache next to the source file in site-packages -- which lives on
# beegfs, a network filesystem numba's cache locator can't handle, crashing `import scanpy`
# immediately with "RuntimeError: cannot cache function ...: no locator available". Point the
# cache at node-local storage instead.
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "Processing slide: $SLIDE (unit=cell)"
python -u preprocess.py --sample "$SLIDE" --unit cell
