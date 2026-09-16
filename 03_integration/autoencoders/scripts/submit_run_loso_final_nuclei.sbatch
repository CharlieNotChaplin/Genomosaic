#!/bin/bash
#SBATCH --job-name=loso_final_nuclei
#SBATCH --output=logs/loso_final_nuclei_%a.out
#SBATCH --error=logs/loso_final_nuclei_%a.err
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-26%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Full 9-fold LOSO, NUCLEI level, 3 variants: SVG gene selection (tasks 0-8),
# the 10x Xenium Human Colon gene panel (tasks 9-17), and SVG + raw-source
# semantic exclusion (tasks 18-26) -- see get_gene_panel_mask in
# functions/dual_autoencoder.py and submit_run_loso_ablation_genepanel.sbatch's
# own comment for the panel, local_infonce's docstring for semantic exclusion.
# `%3` caps this at 3 concurrent GPUs regardless of how many array tasks are
# pending.
#
# UPDATED final config, supersedes submit_run_loso_final_array.sbatch's:
#   - edge_dropout_p=gene_dropout_p=0.1 (kept -- unchanged from before, no new
#     evidence against it under full LayerNorm; see ablation_studies_summary.txt)
#   - weight_decay=1e-5 (ADOPTED -- small, uniformly favorable movement in the
#     layernorm_full re-run: retrieval up slightly, leakage down slightly.
#     Disease-signal later showed a Normal_P3 regression (0.569->0.499) --
#     kept anyway: functionally free, effect small in both directions, not
#     enough to overturn a low-confidence call on one noisy slide.)
#   - temperature=0.05 (ADOPTED -- see the temperature sweep in
#     ablation_studies_summary.txt; corroborated by disease-signal too,
#     0.569->0.632, the cleanest-supported change in this whole campaign)
#   - local_anchor_sim_threshold: left OFF (at the off/None default) for the
#     svg and colon_xenium_panel variants (tasks 0-17). The OLD final_array
#     script hardcoded 0.9 as "adopted for completeness" based on an earlier,
#     flat-looking probe -- the layernorm_full re-run of BOTH anchorsim
#     thresholds (0.7/0.9, PROJECTED source) clearly underperformed leaving
#     this off. The RAW-source version (tasks 18-26 here) did better on
#     retrieval (cluster_top1 0.456->0.502 at threshold 0.2) but consistently
#     WORSE on disease-signal (Normal_P3 0.569->0.460, same direction at
#     0.1/0.2/0.3) -- not adopted as the default, but kept as an explicit
#     side-by-side comparison arm per request, not folded into the main
#     svg/panel variants. threshold=0.2 chosen as the middle of the tested
#     0.1-0.3 range (0.1 had the biggest retrieval gain but the worst train-
#     fit collapse; 0.3 the gentlest of both). Deliberately scoped to SVG only
#     (not also crossed with the gene panel, and not mirrored to bins --
#     semantic exclusion was never calibrated/tested on bins' own token/
#     expression feature space at all, unlike this nuclei-level threshold).
#   - freeze_token, skip_token_pretrain: still NOT adopted (unchanged reasoning)
#   - encoder: still transconv (default) -- sage was tested, no better and a
#     visibly weaker in-sample fit; no reason to switch
#
# Same 9-slide roster as submit_run_loso_final_array.sbatch (CRC_P6_5mm excluded,
# skewed post-QC survivor pool, not a fair held-out target).
SLIDES=(CRC_P1 CRC_P11mm CRC_P11mm_2_cut CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5)
N_SLIDES=${#SLIDES[@]}
HELD_OUT_SLIDE="${SLIDES[$((SLURM_ARRAY_TASK_ID % N_SLIDES))]}"
VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / N_SLIDES))

if [ "$VARIANT_IDX" -eq 0 ]; then
    VARIANT_FLAGS=""
    VARIANT_LABEL="svg"
elif [ "$VARIANT_IDX" -eq 1 ]; then
    VARIANT_FLAGS="--gene-panel colon_xenium"
    VARIANT_LABEL="colon_xenium_panel"
else
    VARIANT_FLAGS="--local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw --local-positive-mode exclude"
    VARIANT_LABEL="svg_semantic_excl_raw_0.2"
fi

source /beegfs/home/cwolf/miniconda3/etc/profile.d/conda.sh
conda activate autoencoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> variant: $VARIANT_LABEL, held-out slide: $HELD_OUT_SLIDE"
echo "Config: edge_dropout_p=0.1 gene_dropout_p=0.1 weight_decay=1e-5 temperature=0.05 $VARIANT_FLAGS"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python run_loso.py --held_out_slide "$HELD_OUT_SLIDE" \
    --edge-dropout-p 0.1 --gene-dropout-p 0.1 --weight-decay 1e-5 --temperature 0.05 $VARIANT_FLAGS

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
