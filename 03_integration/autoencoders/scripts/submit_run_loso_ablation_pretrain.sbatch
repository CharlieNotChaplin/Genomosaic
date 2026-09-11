#!/bin/bash
#SBATCH --job-name=loso_pretrain
#SBATCH --output=logs/loso_pretrain_%a.out
#SBATCH --error=logs/loso_pretrain_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Wave-2 sub-ablation: is Phase-1 token-branch pretraining useful at all?
# skip_token_pretrain=True (token_enc/token_dec start from the same random init
# as the rest of the model, straight into Phase 3) vs the current default
# (pretraining kept -- every existing run to date). freeze_token stays False
# here (its own default) -- freezing a never-pretrained branch isn't meaningful,
# see submit_run_loso_ablation_freeze.sbatch's own note.
#
# Same representative 3-slide subset as wave 1 (CRC_P2, CRC_P4_SMAP, Normal_P3).
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
echo "Config: skip_token_pretrain=True"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --skip-token-pretrain

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
