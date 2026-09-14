#!/bin/bash
#SBATCH --job-name=loso_final_nuclei
#SBATCH --output=logs/loso_final_nuclei_%a.out
#SBATCH --error=logs/loso_final_nuclei_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-17%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full 9-fold LOSO, NUCLEI level, SVG gene selection (tasks 0-8) vs. the 10x
# Xenium Human Colon gene panel (tasks 9-17) -- see get_gene_panel_mask in
# functions/dual_autoencoder.py and submit_run_loso_ablation_genepanel.sbatch's
# own comment. `%3` caps this at 3 concurrent GPUs regardless of how many array
# tasks are pending.
#
# UPDATED final config, supersedes submit_run_loso_final_array.sbatch's:
#   - edge_dropout_p=gene_dropout_p=0.1 (kept -- unchanged from before, no new
#     evidence against it under full LayerNorm; see ablation_studies_summary.txt)
#   - weight_decay=1e-5 (NEWLY ADOPTED -- small, uniformly favorable movement in
#     the layernorm_full re-run: retrieval up slightly, leakage down slightly;
#     functionally free)
#   - temperature=0.05 (NEWLY ADOPTED, now also the CLI/CONFIG default -- see
#     the temperature sweep in ablation_studies_summary.txt)
#   - local_anchor_sim_threshold: REMOVED (left at the off/None default). The
#     OLD final_array script hardcoded 0.9 as "adopted for completeness" based
#     on an earlier, flat-looking probe -- the layernorm_full re-run of BOTH
#     anchorsim thresholds (0.7 and 0.9) showed both clearly UNDERPERFORM
#     leaving this off (held_out_top1 0.0094/0.0078 vs 0.0107 off). Carrying
#     0.9 forward into this run would have been reusing a since-contradicted
#     decision -- flagging this explicitly rather than silently changing it.
#   - freeze_token, skip_token_pretrain: still NOT adopted (unchanged reasoning)
#   - encoder: still transconv (default) -- sage was tested, no better and a
#     visibly weaker in-sample fit; no reason to switch
#
# Same 9-slide roster as submit_run_loso_final_array.sbatch (CRC_P6_5mm excluded,
# skewed post-QC survivor pool, not a fair held-out target).
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
N_SLIDES=${#SLIDES[@]}
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"

if [ "$((SLURM_ARRAY_TASK_ID / N_SLIDES))" -eq 0 ]; then
    GENE_FLAG=""
    GENE_LABEL="svg"
else
    GENE_FLAG="--gene-panel colon_xenium"
    GENE_LABEL="colon_xenium_panel"
fi

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> gene set: $GENE_LABEL, held-out slide: $HELD_OUT_SLIDE"
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1 weight_decay=1e-5 temperature=0.05 $GENE_FLAG"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1 --weight-decay 1e-5 --temperature 0.05 $GENE_FLAG

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
