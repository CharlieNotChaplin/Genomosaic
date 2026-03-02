#!/bin/bash
#SBATCH -c 60
#SBATCH --mem=490G
#SBATCH --output=../../../../logs/spagcn/find_cluster-%j.out
#SBATCH --error=../../../../logs/spagcn/find_cluster-%j.err
#SBATCH --job-name=spagcn_cluster
#SBATCH --mail-type=END
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
#SBATCH --time=20:00:00
#SBATCH --nodelist=compute-node003
source /data/wv377102/miniconda3/etc/profile.d/conda.sh

conda activate nuclei-seg
################################################################

python3 get_clustering.py

