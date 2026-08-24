#!/bin/bash
#SBATCH --job-name=token_sweep
#SBATCH --output=logs/token_sweep.out
#SBATCH --error=logs/token_sweep.err
#SBATCH -c 30
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1

# Single job, not an array -- run_token_sweep.py loops over its GRID
# sequentially within one process/GPU allocation (see that file's docstring
# for why: data prep + token-branch Phase-1 pretraining are shared once
# across every grid point, which is much simpler to guarantee within one
# process than across separate array tasks).
#
# --time budget: train_autoenc_token.py's full 100-epoch run (Phase 1b +
# Phase 3 + eval) took 43.2 min total. Phase 3 does more per-epoch work than
# Phase 1b (full graph pass + pooling + both branches + two InfoNCE losses vs.
# plain MLP forward/backward), so roughly Phase 1b ~10-12 min / Phase 3 ~28-30
# min / eval ~3-5 min of that. This sweep's shared Phase 1 costs about the
# same ~10-12 min ONCE; each grid point is Phase 3 alone at 30 epochs instead
# of 100 (~8-9 min) plus eval (~1-2 min, including the new leakage check) --
# roughly 10 min/grid point. 18 combos * ~10 min + ~12 min shared pretrain =~
# 3-3.5h; 6h leaves real margin without wasting a huge time-limit reservation.
# Check the actual per-run wall-clock printed after the first couple of grid
# points and adjust GRID/REDUCED_EPOCHS/--time together if it's off.
# Resubmitting this exact script after a timeout picks up where it left off
# (see run_token_sweep.py's resumable-via-LOG_PATH logic), so a too-short
# --time just means splitting the sweep across more than one submission, not
# losing progress.

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
python -u run_token_sweep.py
