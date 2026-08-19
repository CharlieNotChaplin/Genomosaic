#!/bin/bash
#SBATCH --job-name=diagnose_pipeline
#SBATCH --output=logs/diagnose_pipeline.out
#SBATCH --error=logs/diagnose_pipeline.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
python -u diagnose_pipeline.py
