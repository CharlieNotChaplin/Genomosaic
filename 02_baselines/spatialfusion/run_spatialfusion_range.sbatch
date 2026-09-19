#!/bin/bash
#SBATCH --job-name=spatialfusion_range
#SBATCH --output=logs/spatialfusion_range_%j.out
#SBATCH --error=logs/spatialfusion_range_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Runs run_spatialfusion_range.py -- the CPU-only half of the SpatialFusion
# pipeline. Unlike run_scgpt_embed.sbatch, this script does NOT request a
# GPU: SpatialFusion's own AE+GCN fusion is pure inference on pretrained,
# bundled checkpoints (no training), explicitly supports --device cpu per
# its own README, and is a small network (GCN hidden_dim=10) -- confirmed
# via its own source (src/spatialfusion/embed/embed.py), not assumed. This
# is the one piece of this baseline safe to schedule on a GPU-free cluster.
#
# Must run AFTER run_scgpt_embed.sbatch for the same --samples (reads its
# output, data/{sample}/02_baselines/spatialfusion/data/scGPT.parquet) and
# after this project's own existing UNI extraction step (reads
# data/{sample}/01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad,
# already present for essentially every sample).
#
#   sbatch run_spatialfusion_range.sbatch --samples CRC_P2 --n-clusters-min 7 --n-clusters-max 10
#
# One-time setup this job assumes is already done:
#   mamba create -p <envs-dir>/spatialfusion python=3.10 -y
#   conda activate spatialfusion
#   pip install "torch==2.4.0+cpu" "torchvision==0.19.0+cpu" --index-url https://download.pytorch.org/whl/cpu
#   pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
#   pip install spatialfusion igraph
# (2.4.0, NOT the README's 2.4.1: dgl 2.4.0 requires torch<=2.4.0, so the
# README's order silently downgrades to the CUDA build of 2.4.0 and drags in
# ~4GB of NVIDIA libraries on a CPU-only install. `igraph` is needed by this
# script's Leiden step and isn't pulled in by spatialfusion itself.)
#
# Tested end-to-end on CPU on CRC_P2 (144k cells): AE+GCN inference took
# seconds; the Leiden resolution search is the slower part. Expect only
# cells with BOTH UNI and scGPT embeddings in the output (see the script).

# conda lives in a different place on each cluster -- use whichever exists
for c in /beegfs/home/cwolf/miniconda3 /data/wv377102/miniconda3 "$HOME/miniconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
conda activate spatialfusion
export PYTHONUNBUFFERED=1
################################################################

python3 run_spatialfusion_range.py "$@"
