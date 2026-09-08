#!/bin/bash
#SBATCH --job-name=loso_bins_array
#SBATCH --output=logs/loso_bins_%a.out
#SBATCH --error=logs/loso_bins_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Bin-level LOSO array -- mirrors submit_run_loso_array.sbatch exactly, just calling
# run_loso_bins.py instead. --mem bumped to 128G (vs. the cell-level script's 64G):
# CRC_P11mm alone is ~1.54M bins, and the combined bin file (4.79M bins total) is
# larger on disk than the cell-level combined file, so this needs more headroom.
# --time bumped to 4h as a first guess -- check actual wall-clock on the first array
# task and adjust once real numbers are in (bin epochs may cost more per step than
# cell epochs simply from more nodes per slide, even with the same cells_per_batch).
#
# CRC_P6_5mm dropped from this array, same rationale as the cell-level script.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso_bins.py --held_out_slide "$HELD_OUT_SLIDE"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
