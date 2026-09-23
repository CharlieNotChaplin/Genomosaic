#!/bin/bash
#SBATCH --job-name=celltypist
#SBATCH --output=logs/celltypist_%j.out
#SBATCH --error=logs/celltypist_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# CPU-only -- annotate_clusters_celltypist.py only loads already-saved embedding .npz files, reconstructs
# each fold's held-out expression, and runs CellTypist's pretrained classifier on it; no model forward pass
# of our own, so no GPU needed.
#
# Killed by RAM locally (30GB machine) partway through the bins folds -- the *reconstruction* of a fold
# loads the full slide before --max-cells subsamples it, and the largest bins slides (CRC_P11mm ~785K
# units) are exactly what has OOM'd other scripts on that machine all session. 300G here is generous on
# purpose to avoid a repeat.
#
# Prerequisites (do these once, from the HEAD NODE where there's internet -- compute nodes may not have it):
#   conda activate autoencoder
#   pip install celltypist
#   Then either let it download the model on first run (needs internet on whichever node runs the job), or
#   copy the already-downloaded model over so no download is needed at run time:
#     scp ~/.celltypist/data/models/Cells_Intestinal_Tract.pkl \
#         cwolf@<cluster>:~/.celltypist/data/models/Cells_Intestinal_Tract.pkl
#   (same relative path as locally -- celltypist's own default cache dir, so --model-path isn't even needed;
#   it is skipped automatically once the file is already there, same as the local test run did.)
#
# Writes final_loso_cluster_annotation_celltypist.csv, appending as each fold finishes and skipping any
# (unit, variant, slide) already in the file -- safe to resubmit the same command if it times out or fails
# partway through.
#
# USAGE (run from THIS directory, 04_eval/):
#   sbatch submit_annotate_clusters_celltypist.sbatch --units nuclei bins --variants svg colon_xenium_panel svg_semantic_excl_raw_0.2 --max-cells 30000
# Any arguments after the script name are forwarded as-is to annotate_clusters_celltypist.py
# (see `python annotate_clusters_celltypist.py --help`).

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
echo "Args: $*"
echo "=================================================="

python -u annotate_clusters_celltypist.py "$@"
status=$?

echo "=================================================="
echo "Job finished at: $(date)   exit status: $status"
echo "=================================================="
exit $status
