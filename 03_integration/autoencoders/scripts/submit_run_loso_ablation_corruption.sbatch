#!/bin/bash
#SBATCH --job-name=loso_corruption
#SBATCH --output=logs/loso_corruption_%a.out
#SBATCH --error=logs/loso_corruption_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Wave-1 sub-ablation: graph/gene denoising corruption, on (edge_dropout_p=0.1,
# gene_dropout_p=0.1) vs off (every existing run's implicit default -- see
# run_loso.py's --edge-dropout-p/--gene-dropout-p, never previously exercised).
# 0.1/0.1 is a first-guess starting value, not evidence-backed -- there's no prior
# run to anchor it against, since this is the first time either flag has been set
# to anything nonzero.
#
# Representative 3-slide subset (not the full 9-fold LOSO -- this and the other
# wave-1/2 sub-ablation scripts are meant to be concise; the full 9-fold run is
# reserved for whichever final config wins): CRC_P2 (tumor, mid-size, otherwise
# unremarkable), CRC_P4_SMAP (tumor, different sample-prep batch), Normal_P3
# (normal). Chosen to span disease status and prep batch while avoiding known
# outliers (CRC_P11mm's size, CRC_P5's disease-signal anomaly).
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
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
