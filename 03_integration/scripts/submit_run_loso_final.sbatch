#!/bin/bash
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --array=0-26%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
# --job-name/--output/--error/--mem/--time are set at submission time (see submit_all_final_loso.sbatch),
# not hardcoded here, because nuclei and bins need different resources (bins: more RAM, much longer wall
# time -- folds take 8-9h, hence 24h there against 2h for nuclei). One file, two resource profiles.

# Final leave-one-slide-out runs: 9 held-out slides x 3 variants (27 tasks, at most 3 concurrent).
#   tasks 0-8    SVG genes
#   tasks 9-17   10x Xenium colon gene panel
#   tasks 18-26  SVG genes with raw-source semantic exclusion (threshold 0.2, taken over from nuclei
#                and not calibrated for bins)
# UNIT (env, set via --export at submission): nuclei (default) or bins.
# All other settings come from config/training_config.json. CRC_P6_5mm is never held out (see the
# evaluation chapter).
#
# Usage (normally via submit_all_final_loso.sbatch, not called directly):
#   sbatch --job-name=loso_final_nuclei --output=logs/loso_final_nuclei_%a.out --error=logs/loso_final_nuclei_%a.err \
#          --mem=64G --time=02:00:00 --export=ALL,UNIT=nuclei submit_run_loso_final.sbatch
#   sbatch --job-name=loso_final_bins --output=logs/loso_final_bins_%a.out --error=logs/loso_final_bins_%a.err \
#          --mem=128G --time=24:00:00 --export=ALL,UNIT=bins submit_run_loso_final.sbatch

UNIT="${UNIT:-nuclei}"
UNIT_FLAGS=""
[ "$UNIT" = "bins" ] && UNIT_FLAGS="--unit bins"

SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
N_SLIDES=${#SLIDES[@]}
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"
VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / N_SLIDES))

if [ "$VARIANT_IDX" -eq 0 ]; then
    VARIANT_FLAGS=""
    VARIANT_LABEL="svg"
elif [ "$VARIANT_IDX" -eq 1 ]; then
    VARIANT_FLAGS="--gene-panel colon_xenium"
    VARIANT_LABEL="colon_xenium_panel"
else
    VARIANT_FLAGS="--local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw --local-positive-mode exclude"
    VARIANT_LABEL="svg_semantic_excl_raw_0.2"
fi

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

# The env's own libstdc++ (needed by numba/llvmlite) can be shadowed by the cluster's system
# /cm/local/apps/gcc/.../libstdc++.so.6 depending on node/module state, causing an
# "OSError: ... CXXABI_1.3.15 not found" crash at `import scanpy`. Force the env's own copy first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# numba (a scanpy dependency) decorates a function with @njit(cache=True) at import time and
# tries to write its compile cache next to the source file in site-packages -- which lives on
# beegfs, a network filesystem numba's cache locator can't handle, crashing `import scanpy`
# immediately with "RuntimeError: cannot cache function ...: no locator available". Point the
# cache at node-local storage instead.
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> unit: $UNIT, variant: $VARIANT_LABEL, held-out slide: $HELD_OUT_SLIDE"
echo "Variant flags: $VARIANT_FLAGS"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" $UNIT_FLAGS $VARIANT_FLAGS
status=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
exit $status
