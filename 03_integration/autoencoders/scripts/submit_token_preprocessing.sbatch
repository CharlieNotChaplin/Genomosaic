#!/bin/bash
#SBATCH --job-name=autoenc_token
#SBATCH --output=logs/autoenc_token.out
#SBATCH --error=logs/autoenc_token.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 30
#SBATCH --mem=100G
#SBATCH --time=4-24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de


# One task per slide -- token_preprocessing.py is CPU-only (per-cell shapely
# intersections against the token grid), no GPU needed. Adjust SLIDES / the
# --array range together if you only need a subset (e.g. just the two slides
# missing locally: CRC_P6_5mm, CRC_P11mm -> --array=0-1 with SLIDES trimmed).
SLIDES=(CRC_P1 CRC_P2 CRC_P5 CRC_P6_5mm CRC_P11mm Normal_P3 Normal_P5)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
echo "Processing slide: $SLIDE"
python -u token_preprocessing.py --slide "$SLIDE"
