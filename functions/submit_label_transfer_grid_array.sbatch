#!/bin/bash
#SBATCH --job-name=label_transfer_grid
#SBATCH --output=logs/label_transfer_grid_%a.out
#SBATCH --error=logs/label_transfer_grid_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# 8um bin-level label transfer (main_grid) for the 3 slides that repeatedly OOM'd
# on a local 30GB desktop: CRC_P11mm (1.1GB native square_008um, ~10x every other
# slide that succeeded locally) and CRC_P3_SMAP/CRC_P4_SMAP (no native square_008um
# at all -- main_grid falls back to aggregate_expression_to_grid, which loads the
# full square_002um matrix first, an even bigger footprint than CRC_P11mm's). The
# other 6 slides (CRC_P1, CRC_P5, CRC_P6_5mm, CRC_P11mm_2_cut, Normal_P3, Normal_P5)
# already completed locally one-at-a-time; their *_grid_predictions.h5ad already
# exist under data/{slide}/02_baselines/label_transfer/.
#
# One task per slide (not one job looping over all 3) so a crash on one doesn't
# lose the others' progress -- same reasoning that made the local runs go one
# slide at a time after the original all-8-in-one-process main_grid() call got
# OOM-killed with nothing saved.
#
# Requires label_transfer.py in the SAME directory as this file (functions/) and
# `sbatch` run from there too, so the plain `python -u label_transfer.py` below
# resolves.
SLIDES=(CRC_P11mm CRC_P3_SMAP CRC_P4_SMAP)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> slide: $SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python -u label_transfer.py --mode grid --targets "$SLIDE"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
