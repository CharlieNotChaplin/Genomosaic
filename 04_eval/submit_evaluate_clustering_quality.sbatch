#!/bin/bash
#SBATCH --job-name=clustering_quality
#SBATCH --output=logs/clustering_quality_%j.out
#SBATCH --error=logs/clustering_quality_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# CPU-only -- evaluate_clustering_quality.py only loads already-saved embedding
# .npz files and clusters/scores them, no model forward pass, so it needs no
# GPU. Deliberately has no --partition=gpu_node / --gres=gpu line (unlike most
# of this project's other sbatch scripts) so the scheduler places it on an
# ordinary compute node instead of tying up a GPU node for CPU-bound work.
#
# The dominant cost is sc.tl.leiden(..., flavor="leidenalg") inside the
# python script's cluster_embedding() -- noticeably slower than the "igraph"
# flavor scanpy itself recommends, and it grows with slide size: CRC_P1
# (~59K nuclei after histology-patch pairing) took a few minutes per
# embedding in local testing; CRC_P11mm (~785K nuclei) was still running
# after 10+ minutes for a SINGLE embedding. A full sweep (both units x all 3
# variants x all 9 slides x 3 embeddings, each re-clustered once "raw" plus
# once per smoothing mode) can run for many hours -- the --time budget above
# is generous on purpose. For a faster/narrower run, pass extra arguments at
# submit time instead of lowering it (see USAGE below).
#
# Silently skips any (unit, variant, slide) with no finished embeddings .npz
# or no label-transfer reference yet (e.g. as of writing, "colon_xenium_panel"
# has no finished runs at all, and bins has none for "svg"/"colon_xenium_panel"
# either) -- safe to resubmit at any point as more LOSO folds finish training.
#
# USAGE (run from THIS directory, 04_eval/, so the plain
# `python -u evaluate_clustering_quality.py` below resolves):
#   sbatch submit_evaluate_clustering_quality.sbatch
#   sbatch submit_evaluate_clustering_quality.sbatch --units nuclei --variants svg colon_xenium_panel
#   sbatch submit_evaluate_clustering_quality.sbatch --slides CRC_P1 CRC_P11mm --smooth-reference
# Any arguments after the script name are forwarded as-is to
# evaluate_clustering_quality.py (see `python evaluate_clustering_quality.py --help`).

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Args: $*"
echo "=================================================="

python -u evaluate_clustering_quality.py "$@"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
