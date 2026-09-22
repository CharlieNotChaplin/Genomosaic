#!/bin/bash
#SBATCH --job-name=regen_grouping
#SBATCH --output=logs/regen_grouping_%a.out
#SBATCH --error=logs/regen_grouping_%a.err
#SBATCH --partition=gpu_node
#SBATCH -c 60
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Throwaway one-off (delete after use): regenerates only filtered_custom_bins.h5ad
# and nuclei_grouped_UMI_count.h5ad for every sample (accidentally deleted) --
# stops right after those, no QC/clustering/get_svg.py, since neither file depends
# on anything after them. Submit for every sample either as an array
# (`sbatch --array=0-9 run_regenerate_grouping.sbatch`) or one at a time
# (`sbatch run_regenerate_grouping.sbatch CRC_P1`).
SLIDES=(CRC_P1 CRC_P2 CRC_P5 CRC_P6_5mm CRC_P11mm CRC_P11mm_2_cut CRC_P3_SMAP CRC_P4_SMAP Normal_P3 Normal_P5)
SAMPLE="${1:-${SLIDES[$SLURM_ARRAY_TASK_ID]}}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=== $SAMPLE: regenerating filtered_custom_bins.h5ad + nuclei_grouped_UMI_count.h5ad ==="
python -u regenerate_grouping_files.py --sample "$SAMPLE"
