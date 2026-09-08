#!/bin/bash
#SBATCH --job-name=spagcn_range
#SBATCH --output=logs/spagcn_range_%j.out
#SBATCH --error=logs/spagcn_range_%j.err
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# This job runs run_spagcn_range.py, which processes every --samples entry
# SEQUENTIALLY within this one process (build adj -> sweep n_clusters ->
# delete everything but the output files -> next sample). Because of that:
#   --mem must cover the LARGEST single sample you list, not the average --
#     see run_all_samples.sh's per-sample nuclei-count/matrix-size table.
#   --time is a SUM across every sample you list (they run one after another,
#     not in parallel), each roughly scaling with its nuclei count squared.
# The #SBATCH defaults above are deliberately small placeholders for a quick
# single small-sample test -- override both at submission time for anything
# larger, e.g.:
#   sbatch --mem=200G --time=30:00:00 run_spagcn_range.sh \
#       --samples CRC_P5 --n-clusters-min 7 --n-clusters-max 10

source /data/wv377102/miniconda3/etc/profile.d/conda.sh
conda activate nuclei-seg
export PYTHONUNBUFFERED=1
################################################################

python3 run_spagcn_range.py "$@"
