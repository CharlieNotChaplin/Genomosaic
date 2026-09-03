"""
Leave-one-slide-out generalization test -- TOKEN BRANCH (cell<->token retrieval).

Tests whether cell-level cross-modal retrieval generalizes to a slide never
seen during training, unlike every other cell_top1/cluster_top1 number
produced so far (measured in-sample).

--held_out_slide picks the fold (default "CRC_P1"). For the full N-fold run,
submit one array task per slide via submit_run_loso_array.sbatch (parallel,
not N x the wall-clock of one run) -- each task's log is named per held-out
slide so concurrent writers don't race on one shared CSV; merge_loso_logs.py
combines them after.

CONFIG mirrors train_autoenc_token.py's validated settings (max_local_contrast_
weight=1.0, local_warmup_epochs=10, local_anchor_sim_threshold=None -- the
18-config sweep winner). Cell-level (trans<->token) only -- the patch-level
(hist<->trans) branch has been removed from this project entirely.
"""

# ---- imports ----------------------------------------------------------
import argparse

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from pathlib import Path
from plyer import notification

from autoencoder_functions_new import *


def run_it(held_out_slide="CRC_P1"):
    set_seed(42)

    # ---- CONFIG -------------------------------------------------------------
    # Mirrors train_autoenc_token.py's validated CONFIG -- see that script for full
    # rationale. hist_pretrain_*/skip_hist_pretrain omitted: the patch-level branch
    # (and its Phase-1 pretraining) has been removed from the project entirely.
    CONFIG = dict(
        sample="combined",     # unused here -- slide filtering happens inside
                                # run_leave_one_slide_out_token itself
        bin_size="002um",
        dir_cellseg="cell_seg_stardist",
        loss_fct="ntxent",

        # model
        latent_dim=64,
        proj_dim=32,
        encoder="transconv",

        gene_selection="svg",   # "svg" | "hvg" | "intersection" -- spatial patterning,
                                 # the signal the alignment task needs

        # graph / batching
        n_neighs=6,
        num_neighbors=(10, 10),
        patches_per_batch=64,
        max_cells_per_patch=64,

        # loss
        temperature=0.07,
        neighbor_radius=2912,       # still required by compute_patch_metadata/
                                     # build_pipeline_data's shared patch-batching path
        local_neighbor_radius=100,  # cell-level analogue, first attempt

        local_anchor_sim_threshold=None,  # sweep showed masking here doesn't help and
                                           # triggers collapse -- see train_autoenc_token.py

        weight_reconstruction_by_patch_freq=True,

        # training (Phase 3)
        n_epochs=100,
        lr=1e-3,
        warmup_epochs=5,

        # token branch (Phase 1b + Phase 3 cell<->token contrastive term)
        token_pretrain_epochs=100,
        token_pretrain_lr=1e-3,
        token_pretrain_weight_decay=1e-5,
        token_pretrain_val_frac=0.1,
        token_pretrain_patience=15,
        token_lr_mult=0.3,
        freeze_token=False,
        max_local_contrast_weight=1.0,   # winner of run_token_sweep.py's 18-config sweep
        local_warmup_epochs=10,

        # Caps CRC_P11mm's per-epoch batch share: PatchGroupedLoader allocates batches
        # proportional to each slide's own patch count, and CRC_P11mm's 29,959 (vs.
        # 8,063-10,983 elsewhere) dominated every epoch. 11000 sits just above the next
        # largest slide -- a no-op for everyone except CRC_P11mm. None = uncapped.
        # (Tried fully EXCLUDING CRC_P11mm instead: reduced leakage and ~doubled in-sample
        # fit, but held-out generalization stayed flat and the train/held-out gap widened
        # -- an overfitting signature, not a fix. Capping is the better tradeoff.)
        max_patches_per_slide=11000,

        # Boosts the 2 normal slides' per-epoch exposure: evaluate_disease_signal showed
        # the model doesn't reliably generalize "normal tissue" to a second unseen normal
        # patient (Normal_P5 stayed near chance even after rebalancing the diagnostic
        # classifier, unlike Normal_P3). Each normal slide is a typical size on its own
        # (~8,000 patches) but only 2 exist vs. 8 tumor slides, so combined exposure is
        # ~16,400 vs. ~96,000. 24000/slide (~3x) roughly balances the two classes per
        # epoch -- a starting point, not tuned. Resampling real cells with replacement
        # (see PatchGroupedLoader.oversample_target), not synthetic augmentation.
        oversample_slides={"Normal_P3": 24000, "Normal_P5": 24000},
    )

    # ---- WHICH SLIDE TO HOLD OUT ------------------------------------------------
    HELD_OUT_SLIDE = held_out_slide

    # ---- slides excluded from the dataset entirely (not just held out) ---------
    # CRC_P3_SMAP/CRC_P4_SMAP's preprocessing is now complete, so both are included --
    # a new 10-slide (8 tumor/2 normal) baseline, not comparable fold-by-fold against
    # the earlier 8-slide capped-batches results (different pool, different n_slides).
    EXCLUDE_SLIDES = []

    # ---- output dirs ----------------------------------------------------------
    basedir = Path("../../data")
    dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
    dir_transcriptomics = basedir / "combined/01_preprocessing" / CONFIG["dir_cellseg"] / CONFIG["bin_size"] / "custom_binning"
    dir_output = basedir / "combined" / "03_integration/autoencoders" / CONFIG["bin_size"]
    dir_output.mkdir(parents=True, exist_ok=True)

    # ---- load data (ALL slides -- filtering into train/held-out happens inside ----
    # ---- run_leave_one_slide_out_token itself, not here) -----------------------
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

    # apply EXCLUDE_SLIDES before token loading, so slides_needed below never looks
    # for their token files
    if EXCLUDE_SLIDES:
        if HELD_OUT_SLIDE in EXCLUDE_SLIDES:
            raise ValueError(f"HELD_OUT_SLIDE={HELD_OUT_SLIDE!r} is also in EXCLUDE_SLIDES "
                              f"{EXCLUDE_SLIDES} -- pick a held-out slide that isn't excluded.")
        print(f"Excluding from the dataset entirely (not just as a held-out choice): {EXCLUDE_SLIDES}")
        f_hist_adata = f_hist_adata[~f_hist_adata.obs["Slide_ID"].isin(EXCLUDE_SLIDES)].copy()
        f_trans_adata = f_trans_adata[~f_trans_adata.obs["Slide_ID"].astype(str).isin(EXCLUDE_SLIDES)].copy()
        f_trans_coord = f_trans_coord[~f_trans_coord["Slide_ID"].isin(EXCLUDE_SLIDES)].copy()

    # ---- load token data --------------------------------------------------------
    # Reads each slide's cell_token_embeddings.h5ad, concatenates, and -- if some
    # slides don't have token data yet -- restricts hist/trans/coord to the slides
    # that do.
    slides_needed = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique())
    token_adatas, slides_with_tokens = [], []
    for s in slides_needed:
        p = basedir / s / "01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad"
        if p.exists():
            token_adatas.append(sc.read_h5ad(p))
            slides_with_tokens.append(s)
        else:
            print(f"WARNING: no token data for slide {s} ({p}) -- run token_preprocessing.py for it first")

    missing = sorted(set(slides_needed) - set(slides_with_tokens))
    if missing:
        print(f"Restricting to slides with token data: {slides_with_tokens} (dropping {missing})")
        f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"].isin(slides_with_tokens)].copy()
        f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"].astype(str).isin(slides_with_tokens)].copy()
        f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"].isin(slides_with_tokens)].copy()
    if not token_adatas:
        raise RuntimeError("no in-scope slide has token data -- run token_preprocessing.py first")
    if HELD_OUT_SLIDE not in slides_with_tokens:
        raise ValueError(f"HELD_OUT_SLIDE={HELD_OUT_SLIDE!r} has no token data -- pick one of {slides_with_tokens}")

    f_token_adata = ad.concat(token_adatas, join="outer")
    token_key = f_token_adata.obs["Slide_ID"].astype(str) + "_" + f_token_adata.obs["id"].astype(str)
    if not token_key.is_unique:
        raise ValueError("Cannot set Slide_ID+id as token index: contains duplicates")
    f_token_adata.obs = f_token_adata.obs.set_index(token_key)
    f_token_adata = f_token_adata[f_token_adata.obs["matched"].astype(bool)].copy()
    print(f"Loaded token data: {f_token_adata.n_obs} cells across {len(slides_with_tokens)} slide(s)")

    # ---- single held-out slide ---------------------------------------------------
    checkpoint_dir = dir_output / "loso_token_checkpoints"

    # run_tag disambiguates checkpoint/log filenames between regimes on the same
    # held_out_slide, so a second run doesn't overwrite the first's outputs
    _tag_parts = []
    if CONFIG["freeze_token"]:
        _tag_parts.append("frozen_token")
    if CONFIG["gene_selection"] != "svg":
        _tag_parts.append(CONFIG["gene_selection"])
    if EXCLUDE_SLIDES:
        _tag_parts.append("excl_" + "_".join(EXCLUDE_SLIDES))
    if CONFIG.get("max_patches_per_slide") is not None:
        _tag_parts.append(f"cap{CONFIG['max_patches_per_slide']}")
    if CONFIG.get("oversample_slides"):
        _tag_parts.append("oversample_normal")
    RUN_TAG = ("_" + "_".join(_tag_parts)) if _tag_parts else ""

    # Named per held-out slide AND run_tag so concurrent array tasks never race on one
    # CSV, and a different regime on the same held-out slide gets its own file instead
    # of overwriting another run's row. merge_loso_logs.py combines them after.
    log_path = dir_output / f"loso_token_log_{HELD_OUT_SLIDE}{RUN_TAG}.csv"

    print(f"\n=== Leave-one-slide-out (token branch): holding out {HELD_OUT_SLIDE} "
          f"(freeze_token={CONFIG['freeze_token']}) ===\n")
    metrics, model, history = run_leave_one_slide_out_token(
        HELD_OUT_SLIDE, f_hist_adata, f_trans_adata, f_trans_coord, f_token_adata, CONFIG, device,
        log_path=log_path, checkpoint_dir=checkpoint_dir, seed=0, verbose=True, run_tag=RUN_TAG,
    )

    print("\n=== Result ===")
    # cluster_top1 (hit = retrieved cell shares the query's independent cluster label) is
    # the headline number, not raw top1: with 100,000s of cells, many are near-duplicates
    # biologically, so exact-cell top1 penalizes a correct-but-not-exact match as a miss.
    # Raw top1 is kept alongside as a secondary diagnostic, not the primary read.
    print(f"In-sample (training slides) cell cluster top-1 retrieval:     {metrics['train_cell_cluster_top1_retrieval']:.4f} "
          f"(exact-cell top-1 {metrics['train_cell_top1_retrieval']:.4f})")
    print(f"Held-out ({HELD_OUT_SLIDE}) cell cluster top-1 retrieval:        {metrics['held_out_cell_cluster_top1_retrieval']:.4f} "
          f"(exact-cell top-1 {metrics['held_out_cell_top1_retrieval']:.4f})")
    print(f"Gap (train - held-out): cluster_top1={metrics['train_cell_cluster_top1_retrieval'] - metrics['held_out_cell_cluster_top1_retrieval']:.4f} "
          f"top1={metrics['train_cell_top1_retrieval'] - metrics['held_out_cell_top1_retrieval']:.4f}")
    print(f"Token pretrain: best val MSE {metrics['token_pretrain_best_val_mse']:.5f} "
          f"(stopped epoch {metrics['token_pretrain_stopped_epoch']})")
    print(f"Train-slide token leakage: {metrics['train_token_leakage_mean']:.4f} "
          f"(chance = {metrics['train_token_leakage_chance']:.4f})")
    print(f"Held-out-vs-train token leakage: {metrics['combined_token_leakage_mean']:.4f} "
          f"(chance = {metrics['combined_token_leakage_chance']:.4f})")
    print(f"Final-epoch local_frac_masked={metrics['final_local_frac_masked']:.4f} "
          f"contrast_local={metrics['final_contrast_local']:.4f} "
          f"(frac_masked > 0.5 would indicate collapse)")
    print(f"\nFull results logged to {log_path}")
    print(f"Cell embeddings (trans/token, train + held-out) saved to "
          f"{checkpoint_dir / f'loso_token_{HELD_OUT_SLIDE}{RUN_TAG}_embeddings.npz'}")

    # Best-effort: plyer's Linux backend needs notify-send/dbus, unavailable on a
    # headless cluster node -- swallow rather than let a non-essential notification
    # fail the job after all real work is already done.
    try:
        notification.notify(
            title='Jupyter Notebook Alert',
            message=f'LOSO-token run for {HELD_OUT_SLIDE} finished!',
            app_name='Python',
            timeout=5,
        )
    except Exception as e:
        print(f"(desktop notification skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held_out_slide", default="CRC_P1",
                         help="Slide to hold out for this fold (default: CRC_P1). "
                              "submit_run_loso_array.sbatch sets this per array task "
                              "for the full N-fold run.")
    args = parser.parse_args()
    run_it(held_out_slide=args.held_out_slide)
