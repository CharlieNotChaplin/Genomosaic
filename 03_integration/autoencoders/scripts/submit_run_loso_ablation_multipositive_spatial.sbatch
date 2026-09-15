#!/bin/bash
#SBATCH --job-name=loso_multipos_spatial
#SBATCH --output=logs/loso_multipos_spatial_%a.out
#SBATCH --error=logs/loso_multipos_spatial_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Tests --local-positive-mode attract: multi-positive InfoNCE (SupCon-style)
# instead of the current false-negative EXCLUSION. Spatial neighbors
# (--local-neighbor-radius, default 100, already active in every run) are
# promoted to ADDITIONAL positives -- actively pulled together -- instead of
# just being removed from the negative set. anchor_sim_threshold stays at its
# off/None default, so this only exercises the spatial criterion, not the
# semantic one (chosen deliberately: spatial_off/spatial_200 already showed
# the neighbor-radius criterion carries a real signal as a mask, unlike the
# semantic/raw criterion which has been mostly inert at every threshold
# tried so far -- see ablation_studies_summary.txt).
#
# Compare against submit_run_loso_ablation_baseline.sbatch (--local-positive-
# mode exclude, i.e. today's default, same neighbor_radius=100).
#
# CAVEAT deliberately not addressed by this run (per explicit instruction):
# multi-positive training changes what one-to-one exact-cell retrieval
# (token_retrieval_accuracy) MEANS -- a spatial neighbor becomes a second
# "correct" answer by design, so held_out_cell_top1_retrieval may drop even
# if the representations are more useful. Read this round's numbers with
# that in mind; not corrected for here.
#
# Check each fold's .out log for local_frac_extra_positive (printed every
# epoch by train(), analogous to local_frac_masked) before drawing
# conclusions -- confirms how many pairs are actually being pulled together.
#
# Same representative 3-slide subset as the rest of this ablation campaign
# (CRC_P2, CRC_P4_SMAP, Normal_P3).
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
echo "Config: --local-positive-mode attract (spatial neighbors, radius=100 default)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --local-positive-mode attract

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
