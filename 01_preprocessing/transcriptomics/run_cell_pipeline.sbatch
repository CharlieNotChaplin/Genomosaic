#!/bin/bash
#SBATCH --job-name=cell_pipeline
#SBATCH --output=logs/cell_pipeline_%a.out
#SBATCH --error=logs/cell_pipeline_%a.err
#SBATCH --partition=gpu_node
#SBATCH -c 60
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Throwaway orchestration script (delete after use): runs cell_preprocessing.py
# (load StarDist nuclei -> spatial-join -> sum counts -> QC -> cluster ->
# geometry parquet -> patch-assign, see CustomBinning.ipynb) then get_svg.py
# (--unit cell, the default) for the same sample, so the whole
# transcriptomics-side per-cell pipeline for one sample is a single job.
# Assumes NucleiSegmentation.py has already produced polys_FullSection.pckl
# for this sample -- segmentation itself is not part of this script. Submit
# for every sample either as an array (`sbatch --array=0-9
# run_cell_pipeline.sbatch`, uses SLIDES below) or one at a time (`sbatch
# run_cell_pipeline.sbatch CRC_P1`).
#
# N_JOBS overrides get_svg.py's --n-jobs (default 20 there) for one
# submission, without editing this file, same as run_bin_pipeline.sbatch:
#   sbatch --export=ALL,N_JOBS=5 run_cell_pipeline.sbatch CRC_P11mm
#
# Run CustomBinning.ipynb's merge cells (49-51) by hand afterwards, once
# every sample has both steps done -- merging needs every sample present at
# once, so it isn't part of this per-sample script.
SLIDES=(CRC_P1 CRC_P2 CRC_P5 CRC_P6_5mm CRC_P11mm CRC_P11mm_2_cut CRC_P3_SMAP CRC_P4_SMAP Normal_P3 Normal_P5)
SAMPLE="${1:-${SLIDES[$SLURM_ARRAY_TASK_ID]}}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=== $SAMPLE: cell preprocessing (load/spatial-join/QC/cluster/patch-assign) ==="
python -u cell_preprocessing.py --sample "$SAMPLE"

echo "=== $SAMPLE: get_svg.py --unit cell (n-jobs=${N_JOBS:-20}) ==="
python -u get_svg.py --sample "$SAMPLE" --unit cell --n-jobs "${N_JOBS:-20}"
