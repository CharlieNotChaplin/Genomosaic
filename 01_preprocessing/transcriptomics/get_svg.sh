#!/bin/bash
#SBATCH -c 60
#SBATCH --mem=128G
#SBATCH --output=CRC11.out
#SBATCH --error=CRC11.err
#SBATCH --job-name=svg_CRC11
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
#SBATCH --time=4-00:00:00

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh

conda activate autoencoder

#######################################################################

python get_svg.py --sample "$1"

