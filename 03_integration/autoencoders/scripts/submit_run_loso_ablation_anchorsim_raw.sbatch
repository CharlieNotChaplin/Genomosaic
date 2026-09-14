#!/bin/bash
#SBATCH --job-name=loso_anchorsim_raw
#SBATCH --output=logs/loso_anchorsim_raw_%a.out
#SBATCH --error=logs/loso_anchorsim_raw_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Tests --local-anchor-sim-source raw: the semantic-exclusion mask (see
# local_infonce's docstring) computed on fixed, pre-encoder feature vectors
# (raw SVG expression / raw histology token) instead of the model's own live,
# currently-training projections ("projected", the only mode tested so far --
# both thresholds tried there, 0.7 and 0.9, clearly UNDERPERFORMED leaving
# masking off entirely; see ablation_studies_summary.txt). "raw" can't feed
# into that same collapse feedback loop since the comparison vectors never
# change during training -- untested, this is that test.
#
# IMPORTANT: thresholds tuned for "projected" do NOT carry over -- raw
# z-scored gene-expression cosine similarity has a different distribution
# than a 32-dim contrastively-trained projection space. This sweeps a
# deliberately WIDE bracket (0.5/0.7/0.9) rather than assuming 0.7-0.9 is
# still the right neighborhood, since we have no prior measurement of the
# raw-feature similarity distribution to calibrate against.
#
# Before trusting ANY of these runs: check each fold's .out log for
# `local_frac_masked` (printed every epoch by train()) -- near 0.0 means the
# threshold is too strict to matter (equivalent to off), near 1.0 means it's
# masking almost every negative (would starve the contrastive loss of
# negatives entirely, likely collapse). A useful setting should land clearly
# between those, and should NOT climb toward 1.0 over epochs (that would be
# the SAME collapse signature "projected" showed, just via a different route
# if the raw threshold turns out too permissive for this data).
#
# 3 thresholds x the usual representative 3-slide subset (CRC_P2, CRC_P4_SMAP,
# Normal_P3) = 9 array tasks. threshold index = task_id / 3, slide index = task_id % 3.
THRESHOLDS=(0.5 0.7 0.9)
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3)
N_SLIDES=${#SLIDES[@]}
THRESH="${THRESHOLDS[$((SLURM_ARRAY_TASK_ID / N_SLIDES))]}"
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> anchor_sim_threshold=$THRESH (raw), held-out slide: $HELD_OUT_SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --local-anchor-sim-threshold "$THRESH" --local-anchor-sim-source raw

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
