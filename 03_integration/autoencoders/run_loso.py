"""
Leave-one-slide-out generalization test.

does cross-modal (histology <-> transcriptomics) retrieval generalize to a slide
the model has NEVER seen at any stage of training (not Phase 1 pretraining,
not Phase 3 joint training) -- as opposed to every retrieval_accuracy number
so far, which was measured on data the model had already been trained on.

Run this ONCE with a single held-out slide first (default below) and sanity
check the result before committing to the full 5-fold version -- 5-fold means
training the entire Phase 1 + Phase 3 pipeline from scratch 5 times (once per
held-out slide), 5x the cost of a single run.

CONFIG here matches the validated settings from the max_contrast_weight sweep:
hist_condition_on_slide=True (the fix that mattered most), max_contrast_weight=1.0
(the point where the sweep plateaued -- see conversation notes for the full curve).
"""

# ---- imports ----------------------------------------------------------
import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import random
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from plyer import notification
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv

from autoencoder_functions_new import *

def run_it():
    set_seed(42)

    # ---- CONFIG -------------------------------------------------------------
    CONFIG = dict(
        sample="combined",     # kept for consistency with run_sweep.py's CONFIG shape; NOT
                                # used to filter here -- run_leave_one_slide_out does its own
                                # slide filtering (train vs held-out) internally
        bin_size="002um",
        dir_cellseg="cell_seg_stardist",
        loss_fct="ntxent",

        # model
        latent_dim=64,
        proj_dim=32,
        pool_type="attention",
        encoder="transconv",
        hist_condition_on_slide=True,   # validated: leakage 0.90 -> ~0.50 with this on

        # which genes feed trans_enc -- "svg" (default, validated) selects for spatial
        # patterning within a slide, the signal the alignment task actually needs. "hvg"
        # showed lower raw-PCA leakage in a standalone check (conversation notes) but
        # was never tested against retrieval quality -- treat as an experiment, not a
        # replacement, until this run's retrieval numbers are compared against the "svg"
        # baseline for the same held-out slide. "intersection" (SVG & HVG) is a middle
        # ground worth trying too.
        gene_selection="svg",   # "svg" | "hvg" | "intersection"

        # graph / batching
        n_neighs=6,
        num_neighbors=(10, 10),
        patches_per_batch=64,
        max_cells_per_patch=64,

        # loss
        temperature=0.07,
        neighbor_radius=2912,     # 13x patch diameter, currently 224x224 px

        weight_reconstruction_by_patch_freq=True,

        hist_batch_correct=False,
        hist_harmony_npcs=50,

        hist_pretrain_epochs=200,   # generous ceiling -- 100 never early-stopped in testing
        hist_pretrain_lr=1e-3,
        hist_pretrain_weight_decay=1e-5,
        hist_pretrain_val_frac=0.1,
        hist_pretrain_patience=15,

        # training
        n_epochs=100,
        lr=1e-3,
        warmup_epochs=5,
        max_contrast_weight=1.0,

        # Phase 3 hist_enc freeze -- default True matches the validated regime
        # (hist_enc/hist_dec frozen after Phase 1, only trans_enc + pooling +
        # both projection heads train against the contrastive loss). Flip to
        # False to let hist_enc adapt to the contrastive objective too -- see
        # conversation notes: this risks re-opening the Slide_ID-leakage path
        # Phase 1 pretraining + conditioning closed, since the contrastive loss
        # has every incentive to exploit slide-level shortcuts in the histology
        # latent once it's allowed to reshape it. hist_lr_mult only matters when
        # freeze_hist=False -- keeps hist_enc/hist_dec's LR a fraction of trans_enc's
        # so it can't be reshaped as aggressively as a from-scratch component.
        freeze_hist=True,
        hist_lr_mult=0.1,   # only used if freeze_hist=False

        # trans_enc regularization -- first pass at the leakage findings from the
        # frozen CRC_P1 run (train_trans_leakage_mean=0.669 vs hist's 0.554, chance
        # 0.25). trans_enc previously had NO weight_decay at all in Phase 3, unlike
        # hist_enc's Phase 1 regime; trans_dec already defaults to
        # condition_on_slide=True, so the same conditioning+weight_decay mechanism
        # that dropped hist leakage 0.90->0.50 is structurally available here too --
        # it was just missing the weight_decay half. edge_dropout_p/gene_dropout_p
        # add denoising-style corruption on top (see train_step docstring for which
        # leakage channel each targets: edges -> graph-structure/cell-density signal,
        # genes -> over-reliance on a few especially discriminative genes).
        # NOTE: none of this can fix the *biological* component of that leakage --
        # this dataset's 5 slides are 5 different patients, 3 tumor/2 normal, so
        # Slide_ID is confounded 1:1 with patient identity + tumor status. Expect
        # SOME reduction, not a drop to chance the way hist's was.
        trans_weight_decay=1e-5,   # matches hist_pretrain_weight_decay for consistency
        edge_dropout_p=0.0,
        gene_dropout_p=0.0,
    )

    # ---- WHICH SLIDE TO HOLD OUT (single-slide test) ---------------------------
    HELD_OUT_SLIDE = "CRC_P1"   # change this to test a different slide

    # ---- output dirs ----------------------------------------------------------
    basedir = Path("../../data")
    dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
    dir_transcriptomics = basedir / "combined/01_preprocessing" / CONFIG["dir_cellseg"] / CONFIG["bin_size"] / "custom_binning"
    dir_output = basedir / "combined" / "03_integration/autoencoders" / CONFIG["bin_size"]
    dir_output.mkdir(parents=True, exist_ok=True)

    # ---- load data (ALL slides -- filtering into train/held-out happens inside ----
    # ---- run_leave_one_slide_out itself, not here) -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
    f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
    f_trans_coord = gpd.read_parquet(dir_transcriptomics / "nuclei_geometry.parquet")

    print("All slides available:", sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique().tolist()))

    trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
    if not trans_global_id.is_unique:
        raise ValueError("Cannot set Slide_ID+id as index: contains duplicates")
    f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)

    hist_global_id = f_hist_adata.obs["Slide_ID"].astype(str) + "_" + f_hist_adata.obs["Patch_ID"].astype(str)
    if not hist_global_id.is_unique:
        raise ValueError("Cannot set Slide_ID+Patch_ID as index: contains duplicates")
    f_hist_adata.obs = f_hist_adata.obs.set_index(hist_global_id)

    # NOTE: hist_batch_correct (Harmony) is NOT applied here. If you want it, it would need
    # to be fit on training slides only and .transform()-ed onto the held-out slide, the same
    # way build_pipeline_data now does for svg_scaler/hist_scaler -- harmonize_hist_embeddings
    # doesn't currently support that split (it always fits+transforms in one call). Skip it for
    # now; CONFIG["hist_batch_correct"]=False reflects that, not an oversight.

    # ---- single held-out slide, run first --------------------------------------
    log_path = dir_output / "loso_log.csv"
    checkpoint_dir = dir_output / "loso_checkpoints"

    # run_tag disambiguates run_id/checkpoint filename when comparing regimes on the
    # same held_out_slide -- otherwise a second run silently overwrites the first
    # run's checkpoint (same collision class run_sweep.py's BATCH_TAG fixed). Built
    # from whichever non-default regimes are active, so it composes as new toggles
    # get added instead of needing a new if/else per one. "" (all defaults, i.e. the
    # original frozen/no-regularization regime) keeps continuity with the
    # CRC_P2/CRC_P1 rows already in loso_log.csv.
    _tag_parts = []
    if not CONFIG["freeze_hist"]:
        _tag_parts.append("unfrozen")
    # which trans-regularization levers are active -- "transreg" alone (no suffix) is
    # kept reserved for the all-three-active combo to match the run_id already logged
    # for the existing loso_CRC_P1_transreg rows; any OTHER subset gets its own distinct
    # suffix so e.g. weight_decay-only doesn't collide with (and silently overwrite) the
    # full-combo checkpoint/log rows -- same collision class run_sweep.py's BATCH_TAG fixed.
    _reg_active = []
    if CONFIG["trans_weight_decay"] > 0:
        _reg_active.append("wd")
    if CONFIG["edge_dropout_p"] > 0:
        _reg_active.append("edge")
    if CONFIG["gene_dropout_p"] > 0:
        _reg_active.append("gene")
    if len(_reg_active) == 3:
        _tag_parts.append("transreg")
    elif _reg_active:
        _tag_parts.append("transreg_" + "".join(_reg_active))
    if CONFIG["gene_selection"] != "svg":
        _tag_parts.append(CONFIG["gene_selection"])
    RUN_TAG = ("_" + "_".join(_tag_parts)) if _tag_parts else ""

    print(f"\n=== Leave-one-slide-out: holding out {HELD_OUT_SLIDE} (freeze_hist={CONFIG['freeze_hist']}) ===\n")
    metrics, model = run_leave_one_slide_out(
        HELD_OUT_SLIDE, f_hist_adata, f_trans_adata, f_trans_coord, CONFIG, device,
        log_path=log_path, checkpoint_dir=checkpoint_dir, seed=0, verbose=True, run_tag=RUN_TAG,
    )

    print("\n=== Result ===")
    print(f"In-sample (training slides) top-1 retrieval: {metrics['train_top1_retrieval']:.4f} "
          f"(median rank {metrics['train_median_rank']:.0f}, MRR {metrics['train_mrr']:.4f} vs chance {metrics['train_chance_mrr']:.4f})")
    print(f"Held-out ({HELD_OUT_SLIDE}) top-1 retrieval:        {metrics['held_out_top1_retrieval']:.4f} "
          f"(median rank {metrics['held_out_median_rank']:.0f}, MRR {metrics['held_out_mrr']:.4f} vs chance {metrics['held_out_chance_mrr']:.4f})")
    print(f"Gap (train - held-out): {metrics['train_top1_retrieval'] - metrics['held_out_top1_retrieval']:.4f}")
    print(f"Train-slide leakage        -- trans: {metrics['train_trans_leakage_mean']:.4f} | hist: {metrics['train_hist_leakage_mean']:.4f} | "
          f"joint: {metrics['train_joint_leakage_mean']:.4f} (chance = {metrics['train_leakage_chance']:.4f})")
    print(f"Held-out-vs-train leakage  -- trans: {metrics['combined_trans_leakage_mean']:.4f} | hist: {metrics['combined_hist_leakage_mean']:.4f} | "
          f"joint: {metrics['combined_joint_leakage_mean']:.4f} (chance = {metrics['combined_leakage_chance']:.4f})")
    print(f"Held-out spatial autocorr (mean |Moran's I|) -- trans: {metrics['held_out_moran_mean_I_trans']:.4f} | "
          f"hist: {metrics['held_out_moran_mean_I_hist']:.4f} | joint: {metrics['held_out_moran_mean_I_joint']:.4f}")
    print(f"Train-slide-disjoint tumor/normal acc ({metrics['train_loso_disease_n_valid_folds']} valid folds) -- "
          f"trans: {metrics['train_loso_disease_acc_trans']} | hist: {metrics['train_loso_disease_acc_hist']} | "
          f"joint: {metrics['train_loso_disease_acc_joint']}")
    print(f"Held-out ({HELD_OUT_SLIDE}, true label={metrics['held_out_disease_true_label']}) tumor/normal acc -- "
          f"trans: {metrics['held_out_disease_acc_trans']} | hist: {metrics['held_out_disease_acc_hist']} | "
          f"joint: {metrics['held_out_disease_acc_joint']}")
    print(f"\nFull results logged to {log_path}")
    print(f"Patch embeddings (trans/hist/joint, train + held-out) saved to {checkpoint_dir / f'loso_{HELD_OUT_SLIDE}{RUN_TAG}_embeddings.npz'}")

    notification.notify(
        title='Jupyter Notebook Alert',
        message=f'LOSO run for {HELD_OUT_SLIDE} finished!',
        app_name='Python',
        timeout=5,
    )

    # ---- FULL 5-FOLD (uncomment once the single-slide result above looks sane) ----
    # WARNING: trains the entire Phase 1 + Phase 3 pipeline from scratch once per
    # slide -- 5x the cost of the single run above.
    #
    # results_df = run_5fold_loso(
    #     f_hist_adata, f_trans_adata, f_trans_coord, CONFIG, device,
    #     log_path=log_path, checkpoint_dir=checkpoint_dir, seed=0, verbose=True,
    # )
    # print(results_df[["held_out_slide", "train_top1_retrieval", "held_out_top1_retrieval"]])
