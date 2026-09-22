#!/bin/bash
#SBATCH --job-name=token_preproc
#SBATCH --output=logs/token_preproc_%a.out
#SBATCH --error=logs/token_preproc_%a.err
#SBATCH --partition=gpu_node
#SBATCH -c 30
#SBATCH --mem=100G
#SBATCH --time=4-24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# One task per slide -- token_preprocessing.py is CPU-only for both units
# (per-cell shapely intersections for --unit nuclei, cKDTree/rectangle-overlap
# queries for --unit bins), no GPU needed despite the gpu_node partition.
#
# Set UNIT via --export when submitting, e.g.:
#   sbatch --array=0-6 --export=UNIT=nuclei submit_token_preprocessing.sbatch
#   sbatch --array=0-9 --export=UNIT=bins submit_token_preprocessing.sbatch
# The resource defaults above are sized for the heavier --unit nuclei job;
# --unit bins only needs ~8 cores/32G, so override at submit time for a
# lighter bins run, e.g.:
#   sbatch --array=0-9 -c 8 --mem=32G --time=4-00:00:00 --export=UNIT=bins submit_token_preprocessing.sbatch
UNIT="${UNIT:?Set UNIT=nuclei or UNIT=bins, e.g. sbatch --export=UNIT=nuclei ...}"

case "$UNIT" in
    nuclei) SLIDES=(CRC_P1 CRC_P2 CRC_P5 CRC_P6_5mm CRC_P11mm Normal_P3 Normal_P5) ;;
    bins)   SLIDES=(CRC_P1 CRC_P2 CRC_P5 CRC_P6_5mm CRC_P11mm CRC_P11mm_2_cut CRC_P3_SMAP CRC_P4_SMAP Normal_P3 Normal_P5) ;;
    *) echo "UNIT must be 'nuclei' or 'bins', got '$UNIT'" >&2; exit 1 ;;
esac
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "Processing slide: $SLIDE (unit=$UNIT)"
python -u token_preprocessing.py --unit "$UNIT" --slide "$SLIDE"
