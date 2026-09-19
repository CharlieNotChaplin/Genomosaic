#!/bin/bash
#SBATCH --job-name=eval_extended
#SBATCH --output=logs/eval_extended_%j.out
#SBATCH --error=logs/eval_extended_%j.err
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Extended evaluation for the thesis' domain-identification / disease-signal results. CPU only. Steps run one
# after another; a failing step is reported but does not stop the later ones. Every step appends to its CSV per fold
# and skips folds already done, so if the time limit hits, resubmit the same command and it continues.
#
#   (1. evaluate_k_matched_ari.py -- the cluster-count-matched multi-reference evaluation -- is the slow part and runs
#       separately, in parallel shards: submit_evaluate_kmatched_parallel.sbatch)
#   2. evaluate_disease_signal_weighted.py   tumor/normal probe with and without class re-weighting
#   3. evaluate_umap_overview.py        UMAP figures (slide identity, tumor/normal, partitions before/after refinement)
#   4. annotate_clusters_celltypist.py  only if RUN_CELLTYPIST=1 (needs `pip install celltypist`, model download)
#
# Then, on your machine (after BOTH sbatch scripts finished): python make_thesis_tables.py
#
# USAGE (run from THIS directory, 04_eval/):
#   sbatch submit_evaluate_extended.sbatch
#   RUN_CELLTYPIST=1 sbatch --export=ALL submit_evaluate_extended.sbatch
#
# Memory: the 785k-nucleus slide (CRC_P11mm) is the peak; refinement (refine_domains) is a Python loop and takes
# minutes per call on it. 300G is generous; lower it if the queue is long.

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)   node: $(hostname)   args: $*"
echo "=================================================="

run_step () {
    echo; echo "##### $1 -- $(date)"; shift
    python -u "$@" || echo "!!!!! step failed (exit $?): $*"
}

run_step "1/3 disease signal, weighted vs unweighted" \
    evaluate_disease_signal_weighted.py
run_step "2/3 UMAP overview figures" \
    evaluate_umap_overview.py --slides CRC_P2 CRC_P5 Normal_P3
if [ "${RUN_CELLTYPIST:-0}" = "1" ]; then
    run_step "3/3 CellTypist annotation" annotate_clusters_celltypist.py
else
    echo; echo "##### 3/3 CellTypist skipped (set RUN_CELLTYPIST=1 to run it)"
fi

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
