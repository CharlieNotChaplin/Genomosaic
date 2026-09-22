#!/bin/bash
#SBATCH --job-name=final_loso_7_9
#SBATCH --output=logs/final_loso_7_9_%j.out
#SBATCH --error=logs/final_loso_7_9_%j.err
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Sections 7 (gene-expression probe), 8 (cluster-type annotation) and 9 (spatial
# cluster maps) of evaluate_final_loso.ipynb -- see evaluate_final_loso_7_9.py's
# docstring for what each writes. Needs a GPU only for section 7 (frozen token
# encoder forward pass + the MLP probe); sections 8/9 are CPU work (Leiden).
#
# Default order is 8, 9, 7 (heavy GPU section last), and results are appended to
# the CSVs fold by fold, so if the time limit hits, resubmitting the same command
# picks up where it stopped instead of starting over (--overwrite to discard).
# A time-out loses at most one fold.
#
# --mem=128G: the largest slide (CRC_P11mm, ~786k nuclei) is loaded in full for
# sections 8/9 and once for the section 7 cache; the section 7 cache itself is
# capped at --max-cells-per-slide (default 100000) per slide.
#
# USAGE (run from THIS directory, 04_eval/):
#   sbatch submit_evaluate_final_loso_7_9.sbatch
#   sbatch submit_evaluate_final_loso_7_9.sbatch --sections 8 9
#   sbatch submit_evaluate_final_loso_7_9.sbatch --units nuclei --plot-slides CRC_P2 CRC_P5
# Arguments after the script name are forwarded to evaluate_final_loso_7_9.py
# (see `python evaluate_final_loso_7_9.py --help`).

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Args: $*"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=================================================="

python -u evaluate_final_loso_7_9.py "$@"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
