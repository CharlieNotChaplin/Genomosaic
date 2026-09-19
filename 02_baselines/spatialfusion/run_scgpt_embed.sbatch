#!/bin/bash
#SBATCH --job-name=scgpt_embed
#SBATCH --output=logs/scgpt_embed_%j.out
#SBATCH --error=logs/scgpt_embed_%j.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Runs run_scgpt_embed.py, the GPU-bound half of the SpatialFusion pipeline:
# extracts per-cell scGPT embeddings (RNA foundation-model side), chunked to
# cap peak memory at this project's nucleus-level cell counts (see the
# script's own docstring). Must run BEFORE run_spatialfusion_range.sbatch for
# a given sample -- that script reads this one's output
# (data/{sample}/02_baselines/spatialfusion/data/scGPT.parquet) and fails
# with a clear FileNotFoundError if it's missing.
#
#   sbatch run_scgpt_embed.sbatch --samples CRC_P2
#
# One-time setup this job assumes is already done:
#   - the `scgpt` conda env exists. A plain `pip install scgpt` does NOT work
#     (each of these was hit and fixed while testing, in this order):
#       mamba create -p <envs-dir>/scgpt python=3.10 pip -y
#       pip install "torch==2.3.0" "numpy<2" --extra-index-url https://download.pytorch.org/whl/cu121
#       pip install scgpt ipython
#     - torch must be exactly 2.3.0: scgpt requires `torchtext`, which was
#       discontinued at 0.18.0 and is ABI-tied to torch 2.3.0 (any other
#       torch fails at import with an undefined-symbol error), and torch 2.3.0
#       needs numpy<2. Left unpinned, pip instead pulls the newest torch plus a
#       multi-GB CUDA 13 stack.
#     - `ipython` is imported by scgpt but not declared as a dependency.
#     - flash-attn is deliberately NOT installed (fragile, CUDA-pinned, only a
#       training-speed optimization; see --use-fast-transformer).
#     Tested on an RTX 4060: ~1s per 64-cell batch, so ~1h for CRC_P2's
#     224k cells; A100 nodes should be faster.
#     If pip runs out of RAM building/unpacking (large CUDA wheels), set
#     TMPDIR to a disk path -- /tmp can be a RAM-backed tmpfs.
#   - the "whole-human" pretrained checkpoint (vocab.json, args.json,
#     best_model.pt) is at weights/scGPT_human/ -- a ~200MB binary, NOT
#     committed to git, fetched via:
#       pip install gdown
#       gdown "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
#     then move the resulting scGPT_human/ folder here manually on any other
#     machine (e.g. this cluster) that doesn't already have it.

# conda lives in a different place on each cluster -- use whichever exists
for c in /beegfs/home/cwolf/miniconda3 /data/wv377102/miniconda3 "$HOME/miniconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
conda activate scgpt
export PYTHONUNBUFFERED=1
################################################################

python3 run_scgpt_embed.py "$@"
