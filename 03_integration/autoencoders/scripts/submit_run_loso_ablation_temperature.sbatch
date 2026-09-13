#!/bin/bash
#SBATCH --job-name=loso_temperature
#SBATCH --output=logs/loso_temperature_%a.out
#SBATCH --error=logs/loso_temperature_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Contrastive-loss temperature sweep (see local_infonce): 0.07 has been the value
# in every run to date, inherited from CLIP/SimCLR and never independently
# ablated. Unlike the other ablation factors so far (mostly on/off switches),
# temperature is continuous, so this sweeps 2 values bracketing the default
# (0.05 sharper, 0.2 softer) and compares both against
# submit_run_loso_ablation_baseline.sbatch's own output (temperature=0.07,
# the CLI default) rather than re-running that point here. NOTE: don't add
# 0.07 as an explicit value in this sweep -- run_loso.py's RUN_TAG only tags
# non-default temperatures, so an explicit --temperature 0.07 run would get
# the exact same output filename as the baseline run and silently overwrite it.
#
# 2 temperatures x the usual representative 3-slide subset (CRC_P2, CRC_P4_SMAP,
# Normal_P3) = 6 array tasks. temp index = task_id / 3, slide index = task_id % 3.
TEMPS=(0.05 0.2)
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3)
N_SLIDES=${#SLIDES[@]}
TEMP="${TEMPS[$((SLURM_ARRAY_TASK_ID / N_SLIDES))]}"
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> temperature=$TEMP, held-out slide: $HELD_OUT_SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --temperature "$TEMP"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
