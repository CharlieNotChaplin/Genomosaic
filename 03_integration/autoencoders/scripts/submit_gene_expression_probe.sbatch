#!/bin/bash
#SBATCH --job-name=gene_expr_probe
#SBATCH --output=logs/gene_expr_probe.out
#SBATCH --error=logs/gene_expr_probe.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Single job, not an array -- gene_expression_probe.py loops over all 9 evaluated
# LOSO folds itself (skipping CRC_P11mm/CRC_P3_SMAP/CRC_P4_SMAP as held-out targets,
# since their cell_token_embeddings.h5ad was deleted after the LOSO run -- see the
# script's docstring). Only needs the GPU for the small token_enc forward pass;
# everything else (Ridge fit, R^2/Pearson r) is CPU/sklearn.

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python gene_expression_probe.py

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
