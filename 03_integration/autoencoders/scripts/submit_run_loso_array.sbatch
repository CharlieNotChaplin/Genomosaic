#!/bin/bash
#SBATCH --job-name=loso_array
#SBATCH --output=logs/loso_%a.out
#SBATCH --error=logs/loso_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# 10-slide + oversampled-normal run: CRC_P3_SMAP/CRC_P4_SMAP preprocessing is now
# complete, run_loso.py's EXCLUDE_SLIDES is empty, --array=0-9 covers all 10 slides
# (8 tumor/2 normal). NOT comparable fold-by-fold against the earlier 8-slide
# capped-batches results (different pool, different n_slides) -- those are archived
# under data/combined/03_integration/autoencoders/002um/loso_token_run_cap_per_epoch_patch_size/
# (and loso_token_baseline_all8slides/, loso_excl_CRC_P11mm/ for the runs before that).
#
# CRC_P6_5mm dropped from this array (2026-09): ~80-90% of its cells were filtered
# out at QC for low UMI count, so the ~28k survivors are a skewed, non-representative
# slice of that slide's tissue rather than just a smaller sample -- holding it out and
# scoring against it isn't a fair generalization test (its held-out cluster-top1
# retrieval was 0.11 vs. 0.31-0.78 for every other slide). Deliberately NOT added to
# EXCLUDE_SLIDES though: that flag drops a slide from the training pool entirely, and
# the surviving 28k cells are still legitimate post-QC cells worth keeping as training
# signal in every other fold -- only excluded from being the held-out target here.
#
# CONFIG still carries max_patches_per_slide=11000 (caps CRC_P11mm) plus the new
# oversample_slides (~24000 draws/epoch for each normal slide, roughly matching the
# tumor side's total) -- see PatchGroupedLoader.oversample_target's docstring.
#
# run_loso.py resolves both `functions/` (for dual_autoencoder.py) and `data/`
# by walking UP from its own file location until it finds them -- no longer
# tied to a specific flat directory layout. A full repo checkout works as-is;
# a hand-curated cluster deployment just needs a `functions/` folder and a
# `data/` folder (or symlink) somewhere above wherever run_loso.py sits, at
# whatever depths make sense for that deployment.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
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
