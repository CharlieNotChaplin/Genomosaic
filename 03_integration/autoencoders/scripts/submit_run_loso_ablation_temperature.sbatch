#!/bin/bash
#SBATCH --job-name=loso_temperature
#SBATCH --output=logs/loso_temperature_%a.out
#SBATCH --error=logs/loso_temperature_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Contrastive-loss temperature sweep (see local_infonce): 0.07 has been the value
# in every run to date, inherited from CLIP/SimCLR and never independently
# ablated. Unlike the other ablation factors so far (mostly on/off switches),
# temperature is continuous, so this sweeps values bracketing the default
# (0.03/0.05 sharper, 0.2 softer) and compares against
# submit_run_loso_ablation_baseline.sbatch's own output (temperature=0.07,
# the CLI default) rather than re-running that point here. NOTE: don't add
# 0.07 as an explicit value in this sweep -- run_loso.py's RUN_TAG only tags
# non-default temperatures, so an explicit --temperature 0.07 run would get
# the exact same output filename as the baseline run and silently overwrite it.
#
# 0.05 and 0.2 already have completed results (first round: held_out_top1
# 0.0117 and 0.0053 respectively, vs 0.0107 at the 0.07 baseline -- sharper
# clearly better within that bracket, softer clearly worse). This round adds
# 0.03 to check whether the improving trend below 0.07 continues past 0.05 or
# plateaus/reverses -- the deciding evidence for whether a learnable
# (CLIP-style logit_scale) temperature is worth implementing instead of a
# fixed value. TEMPS keeps 0.05/0.2 for reference/rerun-ability, but the
# default --array=0-2 above only covers the new 0.03 point (index 0) --
# submit with `sbatch --array=3-8 submit_run_loso_ablation_temperature.sbatch`
# if you also want to rerun 0.05/0.2.
#
# temp index = task_id / 3, slide index = task_id % 3.
TEMPS=(0.03 0.05 0.2)
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
