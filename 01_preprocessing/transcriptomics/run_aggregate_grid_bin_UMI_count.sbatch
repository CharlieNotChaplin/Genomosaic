#!/bin/bash
#SBATCH --job-name=aggregate_grid_bin
#SBATCH --output=logs/aggregate_grid_bin_%j.out
#SBATCH --error=logs/aggregate_grid_bin_%j.err
#SBATCH -c 8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# CPU-only (no --partition=gpu_node/--gres -- this is sparse-matrix and I/O
# work, no model, unlike this directory's other sbatch scripts which request
# gpu_node out of habit despite being CPU-bound too).
#
# Why --mem=220G: aggregate_grid_bin_UMI_count.py's per-sample aggregation
# step alone is cheap (native square_008um loads are small; only
# CRC_P3_SMAP/CRC_P4_SMAP need the heavier square_002um-based fallback, and
# even those ran fine in ~30GB locally). The actual RAM cost is
# merge_anndatas(..., merge_mode="incremental")'s per-sample step, which
# loads the WHOLE existing combined file into memory before folding in one
# more sample -- so peak usage grows with the combined file itself, not any
# one sample, and is worst on the LAST sample merged. This already OOM'd
# locally (30GB total) partway through CRC_P11mm, the largest slide, being
# folded into the other 9 already-merged samples' combined data.
#
# Default --samples (every slide) is safe to resubmit even after a partial
# run: aggregate_sample() skips any per-sample file that already exists
# (pass --force to redo one), and merge_anndatas' incremental mode skips any
# sample already present in the combined file's Slide_ID column -- so a
# rerun only does real work for whatever didn't finish last time. To
# specifically finish just a known-missing sample instead of touching the
# rest at all, pass it explicitly, e.g.:
#   sbatch run_aggregate_grid_bin_UMI_count.sbatch --samples CRC_P11mm

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Args: $*"
echo "=================================================="

python -u aggregate_grid_bin_UMI_count.py "$@"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
