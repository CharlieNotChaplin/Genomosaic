#!/bin/bash
#SBATCH -c 60
#SBATCH --mem=100G
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --job-name=svg_CRC5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
#SBATCH --time=2-00:00:00

source /data/wv377102/miniconda3/etc/profile.d/conda.sh

conda activate nuclei-seg

#######################################################################

python get_svg.py --sample "$1"

