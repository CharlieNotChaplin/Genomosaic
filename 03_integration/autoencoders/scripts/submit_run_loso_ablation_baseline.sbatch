#!/bin/bash
#SBATCH --job-name=loso_clean_baseline
#SBATCH --output=logs/loso_clean_baseline_%a.out
#SBATCH --error=logs/loso_clean_baseline_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Clean, CURRENT-CODE baseline on the representative 3-slide subset -- no ablation
# flags at all, just today's defaults (which now include per-unit density
# weighting via compute_local_density_weight).
#
# Why this exists: the "recalibrated_slidestratified_newcap" baseline every
# wave-1/wave-2 delta was computed against predates the density-weighting swap
# (confirmed directly: its checkpoint has no gene_names/scalers, meaning it also
# predates the checkpoint-persistence fix, which itself predates density
# weighting). Every prior wave-1/2 comparison was therefore silently confounded
# by the patch-weight -> density-weight switch on top of whatever factor was
# actually being tested. This run replaces that stale reference point, and
# doubles as the "on" arm of submit_run_loso_ablation_weighting.sbatch's
# density-weighting on/off test.
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "Config: current defaults, no ablation flags (density weighting ON)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
