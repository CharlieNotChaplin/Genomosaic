#!/bin/bash
#SBATCH --job-name=get_svgs
#SBATCH --output=logs/svg_CRC5.out
#SBATCH --error=logs/svg_CRC5.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:2
#SBATCH -c 30
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

source /data/wv377102/miniconda3/etc/profile.d/conda.sh

conda activate nuclei-seg

#######################################################################

python get_svg.py --sample "$1"

