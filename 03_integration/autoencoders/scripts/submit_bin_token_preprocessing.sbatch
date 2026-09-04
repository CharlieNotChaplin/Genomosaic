#!/bin/bash
#SBATCH --job-name=bin_token
#SBATCH --output=logs/bin_token_%a.out
#SBATCH --error=logs/bin_token_%a.err
#SBATCH --partition=gpu_node
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# One task per slide -- bin_token_preprocessing.py is CPU-only (cKDTree radius
# queries against the raw token grid), no GPU needed, unlike
# submit_token_preprocessing.sbatch's leftover --gres=gpu:1 request. Adjust
# SLIDES / the --array range together if you only need a subset.
SLIDES=(CRC_P1 CRC_P2 CRC_P5 CRC_P6_5mm CRC_P11mm CRC_P11mm_2_cut CRC_P3_SMAP CRC_P4_SMAP Normal_P3 Normal_P5)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "Processing slide: $SLIDE"
python -u bin_token_preprocessing.py --slide "$SLIDE"
