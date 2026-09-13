#!/bin/bash
#SBATCH --job-name=loso_weightdecay
#SBATCH --output=logs/loso_weightdecay_%a.out
#SBATCH --error=logs/loso_weightdecay_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Phase 3 (joint training) weight_decay test: --weight-decay 1e-5, the same value
# Phase 1 token pretraining already uses successfully (token_pretrain_weight_decay).
# Phase 3 has run with weight_decay=0.0 in every experiment to date, despite
# MLPDecoder's own docstring describing weight_decay as the actual mechanism
# behind slide-conditioning's benefit (the incentive to drop slide identity from
# the bottleneck once conditioning makes it unnecessary for reconstruction).
# train()'s optimizer was also switched from plain Adam to AdamW alongside this
# flag being added, since Adam's L2-style decay doesn't behave like AdamW's
# decoupled decay once this is actually nonzero -- a no-op at weight_decay=0.0,
# so every other run is unaffected by that switch.
#
# Same representative 3-slide subset as the rest of this ablation campaign
# (CRC_P2, CRC_P4_SMAP, Normal_P3). Compare against
# submit_run_loso_ablation_baseline.sbatch's output.
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
echo "Config: --weight-decay 1e-5"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --weight-decay 1e-5

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
