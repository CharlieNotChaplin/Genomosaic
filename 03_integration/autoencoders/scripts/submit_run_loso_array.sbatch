#!/bin/bash
#SBATCH --job-name=loso_array
#SBATCH --output=logs/loso_%a.out
#SBATCH --error=logs/loso_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# CAPPED-BATCHES experiment: CRC_P11mm is back IN the training pool for every fold
# (run_loso.py's EXCLUDE_SLIDES is now empty) -- back to all 8 slides, --array=0-7.
# Instead of removing it, run_loso.py's CONFIG["max_patches_per_slide"]=11000 caps how
# many of ITS patches (29,959, vs. 8,063-10,983 for every other slide) feed into batches
# per epoch, via PatchGroupedLoader -- a no-op for every other slide, since none of them
# exceed the cap. Motivated by the prior EXCLUDED experiment's result (see conversation
# notes / loso_excl_CRC_P11mm): full exclusion nearly doubled in-sample fit and reduced
# token leakage, but held-out generalization stayed flat while the train/held-out gap
# roughly doubled too -- an overfitting signature from a smaller training pool, not a
# real fix. Capping keeps CRC_P11mm's full patch pool available (a different random
# subset up to the cap each epoch, since PatchGroupedLoader reshuffles every epoch)
# without letting it dominate the gradient budget.
#
# Baseline (all 8 slides, uncapped) results already archived to
# data/combined/03_integration/autoencoders/002um/loso_token_baseline_all8slides/, and
# the CRC_P11mm-excluded experiment's results to .../loso_excl_CRC_P11mm/ -- this run's
# outputs (loso_token_log_{slide}_cap11000.csv etc., automatically tagged by
# run_loso.py's RUN_TAG) can't collide with either.
#
# Requires: run_loso.py, autoencoder_functions_new.py, and this .sbatch file all in
# the SAME flat directory (no scripts/ subdirectory) when the job actually runs --
# that's what makes `from autoencoder_functions_new import *` resolve without any
# PYTHONPATH tricks. Deploy accordingly before submitting, and run `sbatch` from
# that same directory (see the cd below -- $SLURM_SUBMIT_DIR captures it).
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P5 CRC_P6_5mm Normal_P3 Normal_P5)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

# Load Conda environment
source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# SLURM copies this script into a per-job spool directory before running it, so
# $0/dirname "$0" points there, NOT at your actual deployment folder -- that's what
# caused "can't open file '.../job59828/run_loso.py'" (permission-denied mkdir was
# the same root cause: it landed in that read-only spool dir). $SLURM_SUBMIT_DIR is
# the directory `sbatch` was actually invoked FROM -- cd there instead.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
