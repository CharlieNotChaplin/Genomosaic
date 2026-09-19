#!/bin/bash
#SBATCH --job-name=congi_grid
#SBATCH --output=logs/congi_grid_%j.out
#SBATCH --error=logs/congi_grid_%j.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Runs run_congi_grid.py, which -- unlike run_spagcn_range.py -- pools nuclei
# onto a coarse grid first (--bin-um, default 100, matching classic Visium's
# spot pitch) rather than running at per-nucleus resolution, since ConGI's
# own dataset/training code assumes Visium-scale spot counts (see
# run_congi_grid.py's own docstring for why). At that pooled scale (typically
# a few thousand bins/sample), --mem=32G is generous headroom, not a tight
# fit -- the memory-budget concerns run_spagcn_range.sbatch documents don't
# apply here. --gres=gpu:1 does matter, though: training runs a real
# DenseNet121 forward+backward pass per batch per epoch (250 epochs by
# default) and is impractically slow on CPU.
#
#   sbatch run_congi_grid.sbatch --samples CRC_P2 --n-clusters-min 7 --n-clusters-max 10
#
# One-time setup this job assumes is already done (see run_congi_grid.py's
# docstring): the `congi` conda env exists with torch 1.7.1/torchvision/
# torchtoolbox/rpy2 + R's mclust package, a standalone `_libvips_only` conda
# env exists alongside it (see _crop_wsi_worker.py -- congi's own env can't
# resolve a working libvips; `mamba install -n congi libvips` against
# congi's own already-large package set pathologically hung its solver for
# 40+ minutes rather than actually being unsatisfiable, so libvips lives in
# its own tiny env instead: `mamba create -p <envs-dir>/_libvips_only
# -c conda-forge libvips -y`), and the DenseNet121 ImageNet weights are
# already cached in ~/.cache/torch/hub/checkpoints/ (copy them over from a
# machine with internet access if this node has none).
#
# run_congi_grid.py finds _libvips_only next to the `congi` env itself
# (sys.prefix's parent) by default -- set CONGI_LIBVIPS_ENV to override if
# it lives elsewhere.

# conda lives in a different place on each cluster -- use whichever exists
for c in /beegfs/home/cwolf/miniconda3 /data/wv377102/miniconda3 "$HOME/miniconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
conda activate congi
export PYTHONUNBUFFERED=1

# Confirmed necessary, not precautionary: with PyTorch, NumPy/OpenBLAS,
# scikit-learn, and numba all loaded in the same process, letting each
# spin up its own multi-threaded BLAS/OpenMP pool deadlocks completely
# (reproduced directly -- sc.pp.pca hung indefinitely past 120s with no
# error, every single-step timing test above it having been sub-minute) and,
# separately, was the likely cause of an earlier flaky "ArpackError: Could
# not build an Arnoldi factorization" (multi-threaded BLAS reduction order
# isn't run-to-run reproducible). Forcing everything single-threaded fixed
# both: the same steps completed in under 20s each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
################################################################

python3 run_congi_grid.py "$@"
