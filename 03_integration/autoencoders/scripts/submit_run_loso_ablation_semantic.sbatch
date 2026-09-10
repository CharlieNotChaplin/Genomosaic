#!/bin/bash
#SBATCH --job-name=loso_semantic_excl
#SBATCH --output=logs/loso_semantic_excl_%a.out
#SBATCH --error=logs/loso_semantic_excl_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Wave-1 sub-ablation: contrastive loss's semantic exclusion (local_anchor_sim_threshold).
# None (the current default, "off") already has full 9-fold data from every existing
# run -- not rerun here. Also has prior signal from train_autoenc_token.py's own
# 18-config sweep (None beat 0.9 there, though co-varied with contrast weight/warmup,
# so not a clean isolated read). This tests two threshold values in isolation, each
# across the same representative 3-slide subset as the other wave-1 scripts
# (CRC_P2, CRC_P4_SMAP, Normal_P3): 0.9 (re-confirming the old sweep's tested value,
# this time isolated) and 0.7 (a looser cutoff, masking more candidates as
# near-duplicates).
#
# 6 tasks = 3 slides x 2 threshold values, explicit parallel arrays.
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3 CRC_P2 CRC_P4_SMAP Normal_P3)
THRESHOLDS=(0.9  0.9         0.9       0.7    0.7         0.7)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
THRESHOLD="${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE, local_anchor_sim_threshold=$THRESHOLD"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --local-anchor-sim-threshold "$THRESHOLD"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
