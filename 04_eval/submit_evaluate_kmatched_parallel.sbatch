#!/bin/bash
#SBATCH --job-name=eval_kmatched
#SBATCH --output=logs/eval_kmatched_%A_%a.out
#SBATCH --error=logs/eval_kmatched_%A_%a.err
#SBATCH --array=0-4
#SBATCH -c 4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Step 1 of the extended evaluation (evaluate_k_matched_ari.py), split into 5 size-balanced shards that run in
# PARALLEL as an array job: the 785k-nucleus CRC_P11mm gets a shard of its own, the other eight slides are spread over
# the remaining four. Each shard writes its own *.shardIofN.csv files, so no two tasks touch the same file;
# make_thesis_tables.py reads all shards. The work is single-threaded Leiden/kNN, so 4 cores per task is plenty;
# --mem is for the biggest slide (CRC_P11mm) -- the four smaller shards could run with far less (--mem=80G).
#
# What one run now contains (see evaluate_k_matched_ari.py's docstring): matched-k ARI/NMI plus homogeneity/completeness/purity against
# label transfer, SpaceRanger and (CRC_P2) the annotation; cluster partitions of the raw embeddings AND of spatially smoothed embeddings
# (--smooth-hops 1 2 3 for z_trans and z_joint, SVG variant); SpaGCN/STAGATE/SpatialFusion re-scored on the same nuclei. That is roughly
# twice the work of the first version: expect ~4 h for the CRC_P11mm task and <= 2 h for the others.
#
# BEFORE a full rerun (all of it is read from data/ or 02_baselines/ ON THE CLUSTER):
#   - data/<slide>/output/binned_outputs/square_008um/{analysis,spatial}/ for CRC_P5, Normal_P3, Normal_P5 (SpaceRanger clusters; without them
#     those slides get no SpaceRanger columns)
#   - 02_baselines/{spagcn,stagate,spatialfusion}/outputs/*.parquet (otherwise no baseline is re-scored on our nuclei)
# To replace the finished results (new columns) use:   sbatch submit_evaluate_kmatched_parallel.sbatch --overwrite
# Cheaper partial alternatives:   ... --smooth-only   (only the smoothed-embedding sweeps, own files)   or   ... --baselines-only
#
# If a task times out or fails, resubmit only it:   sbatch --array=2 submit_evaluate_kmatched_parallel.sbatch
# (finished folds inside a shard are skipped).
#
# USAGE (from 04_eval/):
#   sbatch submit_evaluate_kmatched_parallel.sbatch
#   sbatch submit_evaluate_kmatched_parallel.sbatch --units bins                            # bin level: own files (..._ari_bins.shard*.csv),
#                                                                                          # no baselines; ~2x the units, allow ~12 h for CRC_P11mm
#   Extra arguments go to the script. Submit nuclei and bins as separate jobs; --overwrite only removes the files of the units it runs.
#   Then run submit_evaluate_extended.sbatch (steps 2-4) separately, or at the same time.

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
echo "Job started at: $(date)   node: $(hostname)   shard: ${SLURM_ARRAY_TASK_ID}/5   args: $*"
echo "=================================================="

python -u evaluate_k_matched_ari.py --embeddings z_trans z_token z_joint --baselines \
    --shard "${SLURM_ARRAY_TASK_ID}/5" "$@"
status=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
exit $status
