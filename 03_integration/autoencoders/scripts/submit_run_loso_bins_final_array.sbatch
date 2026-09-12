#!/bin/bash
#SBATCH --job-name=loso_bins_final
#SBATCH --output=logs/loso_bins_final_%a.out
#SBATCH --error=logs/loso_bins_final_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full 9-fold LOSO, BIN level, same adopted config as submit_run_loso_final_array.sbatch
# (edge_dropout_p=gene_dropout_p=0.1, local_anchor_sim_threshold=0.9, freeze_token
# left False) -- see that script's own comment for the full rationale, and the
# ablation summary for the numbers.
#
# IMPORTANT CAVEAT carried over from the non-ablated bin LOSO run, still unresolved:
# local_neighbor_radius=100 is the CELL-level value, never re-derived for bin pixel
# pitch (see run_loso_bins.py's own module docstring). This run does not fix that --
# it's still an approximation for bins specifically, on top of the newly-adopted
# corruption/semantic-threshold values (which were only tested on nuclei, not bins --
# graph corruption in particular was flagged as the one factor NOT safe to assume
# transfers, since bins use a regular grid graph vs nuclei's irregular Delaunay).
# This run is the first real bin-level test of that assumption, not a confirmation
# of it.
#
# --mem/--time unchanged from submit_run_loso_bins_array.sbatch (128G/10h) --
# same memory/wall-clock profile expected since nothing about data volume changed.
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
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1 local_anchor_sim_threshold=0.9"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso_bins.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1 --local-anchor-sim-threshold 0.9

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
