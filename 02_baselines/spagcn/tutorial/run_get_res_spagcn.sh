#!/bin/bash
#SBATCH -c 60
#SBATCH --mem=490G
#SBATCH --output=../../../../logs/spagcn/findp-%j.out
#SBATCH --error=../../../../logs/spagcn/findp-%j.err
#SBATCH --job-name=spagcn
#SBATCH --mail-type=END
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
#SBATCH --time=20:00:00
#SBATCH --nodelist=compute-node003
source /data/wv377102/miniconda3/etc/profile.d/conda.sh

conda activate nuclei-seg
################################################################

python3 get_res.py

