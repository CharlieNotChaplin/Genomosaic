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

# numba (a scanpy dependency) decorates a function with @njit(cache=True) at import time and
# tries to write its compile cache next to the source file in site-packages -- which lives on
# beegfs, a network filesystem numba's cache locator can't handle, crashing `import scanpy`
# immediately with "RuntimeError: cannot cache function ...: no locator available". Point the
# cache at node-local storage instead.
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
python -u diagnose_pipeline.py
