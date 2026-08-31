#!/bin/bash
#SBATCH --job-name=loso_array
#SBATCH --output=logs/loso_%a.out
#SBATCH --error=logs/loso_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# 10-SLIDE + OVERSAMPLED-NORMAL experiment: CRC_P3_SMAP/CRC_P4_SMAP's preprocessing
# (incl. token_preprocessing.py) is now verified complete, so run_loso.py's
# EXCLUDE_SLIDES is empty and both are deliberately IN the pool for every fold --
# --array=0-9 covers all 10 slides (8 tumor / 2 normal). This is a NEW baseline, not
# comparable fold-by-fold against the earlier 8-slide capped-batches results
# (loso_token_run_cap_per_epoch_patch_size) -- different training pool AND different
# n_slides/decoder width.
#
# Also carries CONFIG["max_patches_per_slide"]=11000 (unchanged, still caps CRC_P11mm's
# 29,959 patches -- a no-op for every other slide) AND the new
# CONFIG["oversample_slides"]: evaluate_disease_signal (conversation notes) showed the
# model didn't reliably generalize "normal tissue" to a second unseen normal patient
# even after rebalancing the diagnostic classifier itself (Normal_P5 stayed near
# chance) -- with only 2 normal vs. 8 tumor slides, their COMBINED per-epoch patch
# exposure is ~16,400 vs. ~96,000 despite each normal slide being individually a
# normal size (~8,000 patches). oversample_slides resamples each normal slide (with
# replacement, real cells only) up to ~24,000 draws/epoch, roughly matching the tumor
# side's total. See PatchGroupedLoader's oversample_target docstring for the mechanism.
#
# Baseline (all 8 slides, uncapped) results already archived to
# data/combined/03_integration/autoencoders/002um/loso_token_baseline_all8slides/, the
# CRC_P11mm-excluded experiment's to .../loso_excl_CRC_P11mm/, and the 8-slide
# capped-batches results to .../loso_token_run_cap_per_epoch_patch_size/ -- this run's
# outputs (loso_token_log_{slide}_cap11000_oversample_normal.csv etc., automatically
# tagged by run_loso.py's RUN_TAG) can't collide with any of those.
#
# Requires: run_loso.py, autoencoder_functions_new.py, and this .sbatch file all in
# the SAME flat directory (no scripts/ subdirectory) when the job actually runs --
# that's what makes `from autoencoder_functions_new import *` resolve without any
# PYTHONPATH tricks. Deploy accordingly before submitting, and run `sbatch` from
# that same directory (see the cd below -- $SLURM_SUBMIT_DIR captures it).
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 CRC_P6_5mm Normal_P3 Normal_P5)
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
