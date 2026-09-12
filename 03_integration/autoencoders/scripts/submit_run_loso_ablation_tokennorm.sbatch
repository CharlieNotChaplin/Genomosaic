#!/bin/bash
#SBATCH --job-name=loso_tokennorm
#SBATCH --output=logs/loso_tokennorm_%a.out
#SBATCH --error=logs/loso_tokennorm_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# token_enc normalization test: --token-norm layer switches MLPEncoder's
# BatchNorm1d to LayerNorm (see MLPEncoder's own docstring for the rationale --
# same single-slide-batch concern that motivated trans_enc's own BatchNorm->
# LayerNorm switch, minus the attention-specific driver, but z_token is the
# thing directly measured for Slide_ID leakage, which has stayed persistently
# high (~0.8-0.88) across every other factor tested so far).
#
# Compare against submit_run_loso_ablation_baseline.sbatch's output (today's
# code, token_norm=batch), NOT the stale recalibrated_slidestratified_newcap
# baseline -- same reasoning as the weighting on/off test: that baseline predates
# multiple architecture changes since, so it's not a clean single-variable
# comparison anymore.
#
# Nuclei only for now, same scope as the rest of the wave-1/2 sub-ablations --
# not mirrored to run_loso_bins.py unless this shows something worth checking
# on bins too.
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "Config: --token-norm layer"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --token-norm layer

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
