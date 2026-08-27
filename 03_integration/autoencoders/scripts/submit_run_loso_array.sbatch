#!/bin/bash
#SBATCH --job-name=loso_array
#SBATCH --output=logs/loso_%a.out
#SBATCH --error=logs/loso_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full N-fold leave-one-slide-out (token branch): one array task per slide, each
# holding out a different one -- run_leave_one_slide_out_token retrains Phase 1b +
# Phase 3 from scratch per fold, so this runs all 8 folds genuinely in parallel
# instead of 8x the wall-clock of the single CRC_P1 sanity check (which took ~46
# min end to end -- --time above is padded to 2h for margin, not a rerun of that
# job's own timing math).
#
# Requires: run_loso.py, autoencoder_functions_new.py, and this .sbatch file all in
# the SAME flat directory (no scripts/ subdirectory) when the job actually runs --
# that's what makes `from autoencoder_functions_new import *` resolve without any
# PYTHONPATH tricks. Deploy accordingly before submitting.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P5 CRC_P6_5mm Normal_P3 Normal_P5)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

# Load Conda environment
source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# Working directory = wherever this script itself lives (the flat deployment
# folder), regardless of where `sbatch` was invoked from.
cd "$(dirname "$0")"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
