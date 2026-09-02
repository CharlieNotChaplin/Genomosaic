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

# 10-slide + oversampled-normal run: CRC_P3_SMAP/CRC_P4_SMAP preprocessing is now
# complete, run_loso.py's EXCLUDE_SLIDES is empty, --array=0-9 covers all 10 slides
# (8 tumor/2 normal). NOT comparable fold-by-fold against the earlier 8-slide
# capped-batches results (different pool, different n_slides) -- those are archived
# under data/combined/03_integration/autoencoders/002um/loso_token_run_cap_per_epoch_patch_size/
# (and loso_token_baseline_all8slides/, loso_excl_CRC_P11mm/ for the runs before that).
#
# CONFIG still carries max_patches_per_slide=11000 (caps CRC_P11mm) plus the new
# oversample_slides (~24000 draws/epoch for each normal slide, roughly matching the
# tumor side's total) -- see PatchGroupedLoader.oversample_target's docstring.
#
# Requires run_loso.py/autoencoder_functions_new.py/this file in the SAME flat
# directory (no scripts/ subdir) so `from autoencoder_functions_new import *`
# resolves; run `sbatch` from that directory too.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 CRC_P6_5mm Normal_P3 Normal_P5)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# $0/dirname "$0" points at SLURM's ephemeral per-job spool dir, not the real
# deployment folder (caused "can't open file '.../job59828/run_loso.py'" before) --
# $SLURM_SUBMIT_DIR is where `sbatch` was actually invoked from.
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
