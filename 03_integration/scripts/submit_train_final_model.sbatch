#!/bin/bash
#SBATCH --job-name=train_final_model
#SBATCH --output=logs/train_final_model_%j.out
#SBATCH --error=logs/train_final_model_%j.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Trains the deployable final model (all slides, no held-out fold) and writes it to
# data/combined/03_integration/autoencoders/<bin_size>/weights/. See train_final_model.py's
# own docstring for what "the settings we decided on" means and what gets written.
#
# Resources match submit_run_loso_final.sbatch's bins profile (128G/24h): the full-cohort
# training pool here is at least as large as any single LOSO fold's (every slide, not
# all-but-one), so this is the safer default for either unit -- override at submission time
# if nuclei turns out to need less, the same way submit_all_final_loso.sbatch does for LOSO:
#   sbatch --mem=64G --time=02:00:00 --export=ALL,UNIT=nuclei submit_train_final_model.sbatch
#
# UNIT (env, set via --export at submission): nuclei (default) or bins.
#   sbatch --export=ALL,UNIT=nuclei submit_train_final_model.sbatch
#   sbatch --export=ALL,UNIT=bins   submit_train_final_model.sbatch

UNIT="${UNIT:-nuclei}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
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

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Unit: $UNIT"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python -u train_final_model.py --unit "$UNIT"
status=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
exit $status
