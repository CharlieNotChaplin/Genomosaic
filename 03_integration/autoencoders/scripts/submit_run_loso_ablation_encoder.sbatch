#!/bin/bash
#SBATCH --job-name=loso_encoder_sage
#SBATCH --output=logs/loso_encoder_sage_%a.out
#SBATCH --error=logs/loso_encoder_sage_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Encoder architecture test: --encoder sage (GraphSAGE, nn.BatchNorm1d internally,
# hidden_dim=256, no other tunables exposed) against transconv (attention,
# LayerNorm, the universal default in every run to date -- see
# submit_run_loso_ablation_baseline.sbatch's output for the "on" arm). Never
# directly compared before this flag existed; both encoders are otherwise
# unchanged (same latent_dim, same decoder, same everything else in CONFIG).
#
# Same representative 3-slide subset as the rest of this ablation campaign
# (CRC_P2, CRC_P4_SMAP, Normal_P3).
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
echo "Config: --encoder sage"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --encoder sage

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
