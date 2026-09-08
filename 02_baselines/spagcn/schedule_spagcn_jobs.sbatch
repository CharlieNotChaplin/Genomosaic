#!/bin/bash
#SBATCH --job-name=spagcn_scheduler
#SBATCH --output=logs/spagcn_scheduler_%j.out
#SBATCH --error=logs/spagcn_scheduler_%j.err
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# This job is just the supervisor loop (schedule_spagcn_jobs.py) -- it does no
# heavy computation itself, only `sbatch`/`squeue` calls every --poll-seconds,
# so -c 1 / --mem=2G is genuinely enough. --time is set long (10 days) since
# it must stay alive for the entire batch; the actual SpaGCN work happens in
# the separate run_spagcn_range.sbatch jobs it submits, each sized per-sample.
#
# Adjust --time if your cluster caps job wall time below 10 days -- if the
# scheduler itself gets killed by the time limit, any jobs it already
# submitted keep running (they're independent SLURM jobs), but no new ones
# will be submitted after that point.

source /data/wv377102/miniconda3/etc/profile.d/conda.sh
conda activate nuclei-seg
export PYTHONUNBUFFERED=1
################################################################

python3 schedule_spagcn_jobs.py "$@"
