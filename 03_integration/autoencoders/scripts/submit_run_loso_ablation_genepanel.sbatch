#!/bin/bash
#SBATCH --job-name=loso_genepanel
#SBATCH --output=logs/loso_genepanel_%a.out
#SBATCH --error=logs/loso_genepanel_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Gene-panel test: --gene-panel colon_xenium trains on the intersection of the
# 10x Xenium Human Colon panel (data/combined/01_preprocessing/gene_panels/
# colon_xenium.json, 325 gene targets) and whatever of those genes actually
# survives this dataset's own var_names -- an externally-defined, disease-
# agnostic gene set, instead of the internally-derived SVG union
# submit_run_loso_ablation_baseline.sbatch uses. See get_gene_panel_mask/
# get_gene_mask in functions/dual_autoencoder.py. run_loso.py prints the
# panel/dataset overlap (n found / n missing) at the start of each fold --
# check the .out log for that before reading too much into the results, since
# an unexpectedly small overlap would mean training on far fewer genes than
# the SVG baseline.
#
# Same representative 3-slide subset as the rest of this ablation campaign
# (CRC_P2, CRC_P4_SMAP, Normal_P3).
SLIDES=(CRC_P2 CRC_P4_SMAP Normal_P3)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "Config: --gene-panel colon_xenium"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --gene-panel colon_xenium

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
