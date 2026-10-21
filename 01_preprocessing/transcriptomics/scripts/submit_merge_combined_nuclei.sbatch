#!/bin/bash
#SBATCH --job-name=merge_combined_nuclei
#SBATCH --output=logs/merge_combined_nuclei.out
#SBATCH --error=logs/merge_combined_nuclei.err
#SBATCH --partition=gpu_node
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Single job, not an array -- merge_combined_nuclei.py loads every sample's per-sample file into
# RAM at once before concatenating (functions/merge.py: _read_and_prepare_samples), not one at a
# time, so this needs considerably more memory than any individual preprocess.py task. The
# combined outputs are already ~7-8G each on disk (gzip-compressed); uncompressed-in-RAM across
# 10 samples at once is the real driver here. 250G is a starting guess -- lower if the partition
# caps it, raise if it OOMs.
#
# Run only after every sample's per-sample files are current, i.e. after
# submit_preprocess_cell.sbatch's array has finished for all 9 slides (plus CRC_P1, done locally
# earlier). Chain with --dependency, e.g.:
#   ARRAY_JOB=$(sbatch --array=0-8 --parsable submit_preprocess_cell.sbatch)
#   sbatch --dependency=afterok:$ARRAY_JOB submit_merge_combined_nuclei.sbatch
#
# Usage (standalone, once the per-sample files are all ready):
#   sbatch submit_merge_combined_nuclei.sbatch

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# see submit_preprocess_cell.sbatch for why both of these are needed on this cluster
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
python -u merge_combined_nuclei.py
