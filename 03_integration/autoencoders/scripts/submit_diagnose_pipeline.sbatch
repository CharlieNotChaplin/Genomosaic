#!/bin/bash
#SBATCH --job-name=diagnose_pipeline
#SBATCH --output=logs/diagnose_pipeline.out
#SBATCH --error=logs/diagnose_pipeline.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
python -u diagnose_pipeline.py
