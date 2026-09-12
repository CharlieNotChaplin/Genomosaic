#!/bin/bash
#SBATCH --job-name=loso_final
#SBATCH --output=logs/loso_final_%a.out
#SBATCH --error=logs/loso_final_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full 9-fold LOSO, NUCLEI level, combined config from the wave-1/wave-2 ablation
# studies (see data/combined/03_integration/autoencoders/002um/ablation_studies_summary.txt):
#   - edge_dropout_p=gene_dropout_p=0.1 (adopted -- small held-out gain, less
#     overfitting, no downside in the 3-slide probe)
#   - local_anchor_sim_threshold=0.9 (adopted for completeness -- flat vs. off in
#     the probe, not a validated improvement, but a previously-vetted, non-collapsing
#     setting; see the ablation summary before treating this as a proven win)
#   - freeze_token: NOT adopted (deliberately left at default False) -- froze
#     substantially reduced token-branch Slide_ID leakage (0.86->0.69) but cost
#     ~0.024 mean held-out cluster-top1, judged not worth it here
#   - skip_token_pretrain, local_neighbor_radius: left at defaults (both wave-2/
#     wave-1 tests were flat, no reason to change)
#
# Same 9-slide roster as submit_run_loso_array.sbatch (CRC_P6_5mm excluded, same
# rationale -- skewed post-QC survivor pool, not a fair held-out target).
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
HELD_OUT_SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> held-out slide: $HELD_OUT_SLIDE"
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1 local_anchor_sim_threshold=0.9"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1 --local-anchor-sim-threshold 0.9

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
