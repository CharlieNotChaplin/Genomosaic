#!/bin/bash
#SBATCH --job-name=loso_final_bins
#SBATCH --output=logs/loso_final_bins_%a.out
#SBATCH --error=logs/loso_final_bins_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-26%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Final leave-one-slide-out runs at bin level: 9 held-out slides x 3 variants (27 tasks, at most 3 concurrent).
#   tasks 0-8    SVG genes
#   tasks 9-17   10x Xenium colon gene panel
#   tasks 18-26  SVG genes with raw-source semantic exclusion (threshold 0.2, taken over from nuclei and not calibrated for bins)
# The unit-specific settings come from --unit bins, all others from config/training_config.json.
# Folds take 8-9 h, hence --time=24:00:00. Run after submit_run_loso_final_nuclei.sbatch (see submit_all_final_loso.sbatch).

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
echo "Variant flags: $VARIANT_FLAGS"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --unit bins $VARIANT_FLAGS
status=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
exit $status
