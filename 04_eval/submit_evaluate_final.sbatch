#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --output=logs/eval_final_%j.out
#SBATCH --error=logs/eval_final_%j.err
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# CPU job that runs the missing parts of the final leave-one-slide-out evaluation (see run_final_evaluation.py).
# Arguments are passed through:
#   sbatch submit_evaluate_final.sbatch
#   sbatch submit_evaluate_final.sbatch --variants svg_semantic_excl_raw_0.2 --steps quality disease sections
#   sbatch submit_evaluate_final.sbatch --units nuclei bins
# Every step skips finished folds, so after a time-out resubmit the same command.

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job started at: $(date)   node: $(hostname)   args: $*"
python -u run_final_evaluation.py "$@"
status=$?
echo "Job finished at: $(date)   exit status: $status"
exit $status
