#!/bin/bash
#SBATCH --job-name=loso_multipos_semantic
#SBATCH --output=logs/loso_multipos_semantic_%a.out
#SBATCH --error=logs/loso_multipos_semantic_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Tests --local-positive-mode attract on the SEMANTIC criterion (raw source),
# isolated from the spatial one (--local-neighbor-radius none) so the read
# isn't confounded by both mechanisms at once.
#
# Follow-up to two prior results:
#   - submit_run_loso_ablation_anchorsim_raw_lowthresh.sbatch (EXCLUDE mode,
#     same raw criterion): thresholds 0.1-0.3 mask a real 12-44% of pairs
#     (unlike the spatial mask's ~0.02%) and showed a clean, monotonic
#     trade-off -- held_out_cluster_top1 (the PRIMARY retrieval metric, see
#     token_retrieval_accuracy's own "cluster_top1 leads" comment) improved
#     substantially (0.456->0.517 at threshold=0.1), at the cost of held-out
#     exact-cell top1, train fit, and a small leakage increase. See
#     ablation_studies_summary.txt for the full numbers.
#   - submit_run_loso_ablation_multipositive_spatial.sbatch (ATTRACT mode,
#     spatial criterion): inconclusive -- frac_extra_positive was only ~0.02%
#     (random-sampled batches rarely contain spatial neighbors), too sparse
#     for a clean read either way.
# This run combines the two: attract mode (actively pulls flagged pairs
# together, not just declining to push them apart) on the criterion that's
# actually shown a real, substantial effect at a meaningful masked fraction.
# Expectation: likely a STRONGER version of the exclude-mode trade-off above
# (bigger cluster_top1 gain, bigger cost to exact-cell top1/train fit) --
# this is that test.
#
# Threshold fixed at 0.2 (middle of the productive 0.1-0.3 range found
# above) rather than swept again -- if this looks promising, sweep around it
# next.
#
# Check each fold's .out log for local_frac_extra_positive (should land
# in the same ~24% range the exclude-mode 0.2 run showed, since it's the
# same underlying mask) before drawing conclusions.
#
# --time=03:00:00 (up from this campaign's usual 2h): CRC_P2 hit the 2h wall
# twice on the raw-criterion lowthresh sweep (transient slowness, not a bug --
# see that script's own follow-up note), giving this run more headroom.
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
echo "Config: --local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw --local-positive-mode attract --local-neighbor-radius none"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw \
    --local-positive-mode attract --local-neighbor-radius none

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
