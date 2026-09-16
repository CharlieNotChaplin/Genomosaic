#!/bin/bash
#SBATCH --job-name=loso_final_bins
#SBATCH --output=logs/loso_final_bins_%a.out
#SBATCH --error=logs/loso_final_bins_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-26%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full 9-fold LOSO, BIN level, 3 variants: SVG gene selection (tasks 0-8), the
# 10x Xenium Human Colon gene panel (tasks 9-17), and SVG + raw-source semantic
# exclusion (tasks 18-26) -- mirrors submit_run_loso_final_nuclei.sbatch's
# config and 3-variant structure exactly (see that script's own comment for
# the full rationale on each choice, and ablation_studies_summary.txt for the
# numbers): edge_dropout_p=gene_dropout_p=0.1, weight_decay=1e-5,
# temperature=0.05. NONE of these were independently re-validated for bins
# specifically -- carried over on the same assumption-transfer basis
# run_loso_bins.py's own module docstring already uses for
# local_neighbor_radius. `%3` caps this at 3 concurrent GPUs.
#
# IMPORTANT -- semantic-exclusion threshold (0.2, raw source) is carried over
# UNCHECKED from nuclei: run submit_run_loso_ablation_bins_semantic_probe.sbatch
# FIRST (a separate, standalone 3-slide/3-task probe) and check its
# local_frac_masked before trusting tasks 18-26 here -- nuclei's own 0.7/0.9
# bracket turned out to barely mask anything and needed a second, lower-
# threshold round before 0.2 became a sensible value; bins' raw feature space
# (different SVG count, different token aggregation) has never been checked
# at all. If the probe shows 0.2 is similarly miscalibrated for bins, edit
# the --local-anchor-sim-threshold value below before submitting this array.
#
# --mem/--time unchanged from submit_run_loso_bins_final_array.sbatch (128G/10h,
# ~453 min mean observed bins fold duration from the original bin-level run).
#
# Intended to run AFTER submit_run_loso_final_nuclei.sbatch fully completes, not
# concurrently with it -- see submit_all_final_loso.sh, which chains the two via
# a SLURM dependency instead of requiring you to submit/watch/resubmit each by hand.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
N_SLIDES=${#SLIDES[@]}
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"
VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / N_SLIDES))

if [ "$VARIANT_IDX" -eq 0 ]; then
    VARIANT_FLAGS=""
    VARIANT_LABEL="svg"
elif [ "$VARIANT_IDX" -eq 1 ]; then
    VARIANT_FLAGS="--gene-panel colon_xenium"
    VARIANT_LABEL="colon_xenium_panel"
else
    VARIANT_FLAGS="--local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw --local-positive-mode exclude"
    VARIANT_LABEL="svg_semantic_excl_raw_0.2"
fi

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> variant: $VARIANT_LABEL, held-out slide: $HELD_OUT_SLIDE"
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1 weight_decay=1e-5 temperature=0.05 $VARIANT_FLAGS"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso_bins.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1 --weight-decay 1e-5 --temperature 0.05 $VARIANT_FLAGS

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
