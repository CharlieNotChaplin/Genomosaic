#!/bin/bash
#SBATCH --job-name=loso_bins_semantic_probe
#SBATCH --output=logs/loso_bins_semantic_probe_%a.out
#SBATCH --error=logs/loso_bins_semantic_probe_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Calibration probe: does nuclei's own productive raw-source semantic-exclusion
# threshold (0.2, the middle of nuclei's 0.1-0.3 range) mean anything similar
# for bins? Bins' raw SVG-expression/token feature space was never checked --
# different SVG count, different token aggregation (footprint+halo over a
# larger fixed area than a nucleus), different unit density. Reusing 0.7/0.9
# blind for NUCLEI already burned us once (see ablation_studies_summary.txt's
# first anchorsim round) -- don't repeat that mistake for bins by assuming
# 0.2 transfers untested.
#
# Single threshold (0.2), 3-slide subset, --local-anchor-sim-source raw,
# --local-positive-mode exclude (not attract -- that mode failed outright for
# nuclei's semantic criterion, not worth probing for bins too). Before reading
# results: check each fold's .out log for `local_frac_masked` -- if it's near
# 0 (too strict, inert, like nuclei's own first 0.5/0.7/0.9 round) or masking
# nearly everything, 0.2 isn't the right value for bins and a bins-specific
# threshold sweep is needed before this can be added to the final bins run at
# all, the same way nuclei needed a second, lower-threshold round.
#
# --mem/--time match submit_run_loso_final_bins.sbatch (128G/10h, ~453 min
# mean observed bins fold duration from the original bin-level LOSO run).
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
echo "Config: --local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw --local-positive-mode exclude"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso_bins.py --held_out_slide "$HELD_OUT_SLIDE" \
    --local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw --local-positive-mode exclude

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
