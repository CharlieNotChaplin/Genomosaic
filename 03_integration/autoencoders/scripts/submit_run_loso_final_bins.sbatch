#!/bin/bash
#SBATCH --job-name=loso_final_bins
#SBATCH --output=logs/loso_final_bins_%a.out
#SBATCH --error=logs/loso_final_bins_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-17%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full 9-fold LOSO, BIN level, SVG gene selection (tasks 0-8) vs. the 10x Xenium
# Human Colon gene panel (tasks 9-17) -- mirrors submit_run_loso_final_nuclei.sbatch's
# config exactly (see that script's own comment for the full rationale on each
# choice, and ablation_studies_summary.txt for the numbers): edge_dropout_p=
# gene_dropout_p=0.1, weight_decay=1e-5, temperature=0.05, local_anchor_sim_threshold
# left off. NONE of these were independently re-validated for bins specifically --
# this is the same nuclei-validated config carried over on the same assumption-
# transfer basis run_loso_bins.py's own module docstring already uses for
# local_neighbor_radius. `%3` caps this at 3 concurrent GPUs.
#
# --mem/--time unchanged from submit_run_loso_bins_final_array.sbatch (128G/10h).
#
# Intended to run AFTER submit_run_loso_final_nuclei.sbatch fully completes, not
# concurrently with it -- see submit_all_final_loso.sh, which chains the two via
# a SLURM dependency instead of requiring you to submit/watch/resubmit each by hand.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
N_SLIDES=${#SLIDES[@]}
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"

if [ "$((SLURM_ARRAY_TASK_ID / N_SLIDES))" -eq 0 ]; then
    GENE_FLAG=""
    GENE_LABEL="svg"
else
    GENE_FLAG="--gene-panel colon_xenium"
    GENE_LABEL="colon_xenium_panel"
fi

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> gene set: $GENE_LABEL, held-out slide: $HELD_OUT_SLIDE"
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1 weight_decay=1e-5 temperature=0.05 $GENE_FLAG"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso_bins.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1 --weight-decay 1e-5 --temperature 0.05 $GENE_FLAG

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
