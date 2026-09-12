#!/bin/bash
#SBATCH --job-name=loso_no_weighting
#SBATCH --output=logs/loso_no_weighting_%a.out
#SBATCH --error=logs/loso_no_weighting_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Density-weighting on/off test: --no-weight-reconstruction disables per-unit
# inverse-density weighting of the transcriptomics reconstruction loss entirely
# (plain unweighted mean instead). The "on" arm is
# submit_run_loso_ablation_baseline.sbatch (today's default, no flags) --
# compare against that, NOT the stale recalibrated_slidestratified_newcap
# baseline (predates this mechanism entirely, see that script's own comment).
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
echo "Config: --no-weight-reconstruction (plain unweighted mean)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --no-weight-reconstruction

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
