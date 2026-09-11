#!/bin/bash
#SBATCH --job-name=loso_freeze
#SBATCH --output=logs/loso_freeze_%a.out
#SBATCH --error=logs/loso_freeze_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Wave-2 sub-ablation: does freezing token_enc/token_dec during Phase 3 help
# (e.g. gene-expression prediction, via gene_expression_probe.py on the
# resulting checkpoints) or hurt cell<->token retrieval? Pretraining is left ON
# (the default, skip_token_pretrain not passed) -- freezing a never-pretrained,
# randomly-initialized token branch would not be a meaningful test, see
# submit_run_loso_ablation_pretrain.sbatch.
#
# Same representative 3-slide subset as wave 1 (CRC_P2, CRC_P4_SMAP, Normal_P3).
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
echo "Config: freeze_token=True (pretraining kept on)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" --freeze-token

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
