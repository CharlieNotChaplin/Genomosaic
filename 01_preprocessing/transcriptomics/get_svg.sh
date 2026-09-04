#!/bin/bash
#SBATCH -c 60
#SBATCH --mem=128G
#SBATCH --output=CRC3.out
#SBATCH --error=CRC3.err
#SBATCH --job-name=svg_CRC3
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de
#SBATCH --time=4-00:00:00

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh

conda activate autoencoder

#######################################################################

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python get_svg.py --sample "$1" --unit "${2:-cell}"

