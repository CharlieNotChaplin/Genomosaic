#!/bin/bash
#SBATCH --job-name=loso_spatial_excl
#SBATCH --output=logs/loso_spatial_excl_%a.out
#SBATCH --error=logs/loso_spatial_excl_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Wave-1 sub-ablation: contrastive loss's spatial exclusion (local_neighbor_radius).
# local_neighbor_radius=100 (the current default, "on") already has full 9-fold
# data from every existing run -- not rerun here. This tests the two NEW comparison
# points: None (spatial exclusion off entirely) and 200 (roughly double the
# current radius), each across the same representative 3-slide subset as
# submit_run_loso_ablation_corruption.sbatch (CRC_P2, CRC_P4_SMAP, Normal_P3).
#
# 6 tasks = 3 slides x 2 radius values, explicit parallel arrays (not
# div/mod-encoded) to keep the slide<->value pairing obvious at a glance.
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3 CRC_P2 CRC_P4_SMAP Normal_P3)
RADII=(none    none        none      200    200         200)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
RADIUS="${RADII[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE, local_neighbor_radius=$RADIUS"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --local-neighbor-radius "$RADIUS"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
