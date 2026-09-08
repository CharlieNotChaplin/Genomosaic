#!/bin/bash
#SBATCH --job-name=loso_ablation_oldcap
#SBATCH --output=logs/loso_ablation_oldcap_%a.out
#SBATCH --error=logs/loso_ablation_oldcap_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Ablation: SlideStratifiedLoader (current loader) + the OLD, pre-recalibration
# max_cells_per_slide=11000/oversample_slides=24000-24000 values -- isolates whether
# the loader swap or the cap/oversample recalibration is responsible for the
# retrieval regression seen when both changes were bundled into one retrain
# (loso_run_with_two_new_changes/: held_out_cell_top1/cluster_top1 both dropped vs.
# the pre-change PatchGroupedLoader run across nearly all 9 folds). See
# run_loso_ablation_oldcap.py's own docstring for the full comparison/rationale.
#
# Same SLIDES/array structure as submit_run_loso_array.sbatch -- diff the two sbatch
# files if you need to confirm nothing else (resources, env) drifted.
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso_ablation_oldcap.py --held_out_slide "$HELD_OUT_SLIDE"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
