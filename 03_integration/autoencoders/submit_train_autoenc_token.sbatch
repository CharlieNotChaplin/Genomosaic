#!/bin/bash
#SBATCH --job-name=autoenc_token
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=autoenc_token_%j.out
#SBATCH --error=autoenc_token_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$(dirname "$0")"
python -u train_autoenc_token.py
