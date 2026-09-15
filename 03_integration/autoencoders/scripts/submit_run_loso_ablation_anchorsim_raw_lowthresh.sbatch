#!/bin/bash
#SBATCH --job-name=loso_anchorsim_raw_low
#SBATCH --output=logs/loso_anchorsim_raw_low_%a.out
#SBATCH --error=logs/loso_anchorsim_raw_low_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Follow-up to submit_run_loso_ablation_anchorsim_raw.sbatch (0.5/0.7/0.9): that
# round confirmed --local-anchor-sim-source raw structurally avoids the
# collapse-feedback loop "projected" is prone to (final_local_frac_masked was
# perfectly FLAT across all 100 epochs at every threshold, zero drift) -- but
# the bracket itself (reused from "projected") turned out miscalibrated: 0.7
# and 0.9 masked under 0.06% of pairs (mechanically ~= off), and even 0.5 only
# masked 1.7%. Raw z-scored SVG expression cosine similarity apparently sits
# on a much lower scale than the 32-dim contrastively-trained projection space
# "projected" checks. This round goes lower to find where frac_masked
# actually lands in a meaningful 10-40% range where an effect would be
# visible either way -- see ablation_studies_summary.txt's "SEMANTIC
# EXCLUSION, RAW-FEATURE SOURCE" section for the full first-round writeup and
# numbers.
#
# Compare against layernorm_full_temp0.07_baseline (off, no masking) and the
# 0.5/0.7/0.9 round already in ablation_studies_summary.csv.
#
# 3 thresholds x the usual representative 3-slide subset (CRC_P2, CRC_P4_SMAP,
# Normal_P3) = 9 array tasks. threshold index = task_id / 3, slide index = task_id % 3.
THRESHOLDS=(0.1 0.2 0.3)
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
