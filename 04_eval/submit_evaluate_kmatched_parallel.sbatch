#!/bin/bash
#SBATCH --job-name=eval_kmatched
#SBATCH --output=logs/eval_kmatched_%A_%a.out
#SBATCH --error=logs/eval_kmatched_%A_%a.err
#SBATCH --array=0-4
#SBATCH -c 4
#SBATCH --mem=200G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Step 1 of the extended evaluation (evaluate_k_matched_ari.py), split into 5 size-balanced shards that run in
# PARALLEL as an array job: the 785k-nucleus CRC_P11mm gets a shard of its own, the other eight slides are spread over
# the remaining four. Each shard writes its own *.shardIofN.csv files, so no two tasks touch the same file;
# make_thesis_tables.py reads all shards. The work is single-threaded Leiden/kNN, so 4 cores per task is plenty;
# --mem is for the biggest slide (CRC_P11mm) -- the four smaller shards could run with far less (--mem=80G).
#
# If a task times out or fails, resubmit only it:   sbatch --array=2 submit_evaluate_kmatched_parallel.sbatch
# (finished folds inside a shard are skipped).
#
# USAGE (from 04_eval/):
#   sbatch submit_evaluate_kmatched_parallel.sbatch
#   sbatch submit_evaluate_kmatched_parallel.sbatch --units nuclei bins --variants svg      # extra args go to the script
#   Then run submit_evaluate_extended.sbatch (steps 2-4) separately, or at the same time.

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)   node: $(hostname)   shard: ${SLURM_ARRAY_TASK_ID}/5   args: $*"
echo "=================================================="

python -u evaluate_k_matched_ari.py --embeddings z_trans z_token z_joint --baselines \
    --shard "${SLURM_ARRAY_TASK_ID}/5" "$@"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
