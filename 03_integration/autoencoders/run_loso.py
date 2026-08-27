"""
Leave-one-slide-out generalization test -- TOKEN BRANCH (cell<->token retrieval).

Does cell-level cross-modal retrieval (cell transcriptomics <-> per-cell UNI
tokens) generalize to a slide the model has NEVER seen at any stage of
training (not Phase 1b token pretraining, not Phase 3 joint training) -- as
opposed to every cell_top1/cell_cluster_top1 number produced so far (the
18-config sweep in token_sweep.py included), which was measured on data the
model had already been trained on.

Run this ONCE with a single held-out slide first (default below) and sanity
check the result before committing to the full N-fold version -- N-fold means
training Phase 1b + Phase 3 from scratch once per slide, N times the cost of
a single run.

--held_out_slide selects the fold (default "CRC_P1", matches the original
single-slide sanity check). For the full N-fold run, submit one array task per
slide via submit_run_loso_array.sbatch rather than looping in-process (see
that file) -- each task calls this script with a different --held_out_slide,
running genuinely in parallel instead of N times the wall-clock of one run.
log_path is named per held-out slide specifically so concurrent array tasks
never write to the same CSV (a shared log would race: each task reads-modifies-
writes the whole file, so concurrent writers can silently drop each other's
rows) -- merge_loso_logs.py combines the per-slide logs afterward.

CONFIG here matches the validated settings from train_autoenc_token.py's
CONFIG (see that script's comments for the full rationale/history of each
value): max_local_contrast_weight=1.0, local_warmup_epochs=10,
local_anchor_sim_threshold=None -- the winner of run_token_sweep.py's
18-config sweep. Patch-level (hist<->trans) alignment stays OFF
(max_contrast_weight=0.0, hist_enc/hist_dec left at random init and frozen) --
this project's current focus is entirely the cell/token branch; a
patch-level version of this test already exists as run_leave_one_slide_out
in autoencoder_functions_new.py.
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
    # Mirrors train_autoenc_token.py's validated CONFIG -- see that script for the
    # full rationale behind each value. Only the keys run_leave_one_slide_out_token
    # (and the build_pipeline_data / DualAutoencoder / train() calls it makes) actually
    # reads are included; hist_pretrain_* and skip_hist_pretrain are omitted entirely
    # because that function never calls pretrain_hist_autoencoder at all (hist_enc is
    # always left at random init and frozen immediately, harmless as long as
    # max_contrast_weight stays 0.0 -- same regime skip_hist_pretrain=True encodes).
    CONFIG = dict(
        sample="combined",     # kept for consistency with train_autoenc_token.py's CONFIG shape;
                                # NOT used to filter here -- run_leave_one_slide_out_token does its
                                # own slide filtering (train vs held-out) internally
        bin_size="002um",
        dir_cellseg="cell_seg_stardist",
        loss_fct="ntxent",

        # model
        latent_dim=64,
        proj_dim=32,
        pool_type="attention",
        encoder="transconv",
        # hist_condition_on_slide / token_condition_on_slide left unset -> both default
        # True in DualAutoencoder, matching train_autoenc_token.py (which also never
        # overrides them).

        # which genes feed trans_enc -- "svg" (default, validated) selects for spatial
        # patterning within a slide, the signal the alignment task actually needs.
        # train_autoenc_token.py always uses this implicitly (calls get_svg_union_mask
        # directly); exposed here as a knob since run_leave_one_slide_out_token supports
        # it via get_gene_mask, same as the patch-level LOSO function does.
        gene_selection="svg",   # "svg" | "hvg" | "intersection"

        # graph / batching
        n_neighs=6,
        num_neighbors=(10, 10),
        patches_per_batch=64,
        max_cells_per_patch=64,

        # loss
        temperature=0.07,
        neighbor_radius=2912,      # patch-level neighbor tolerance -- irrelevant here
                                    # (max_contrast_weight=0.0) but still required by
                                    # compute_patch_metadata/build_pipeline_data.
        local_neighbor_radius=100,  # cell-level analogue, see train_autoenc_token.py's
                                     # CONFIG comment -- first attempt, worth iterating on.

        patch_anchor_sim_threshold=0.9,   # irrelevant while max_contrast_weight=0.0,
                                           # kept for fidelity with train_autoenc_token.py.
        local_anchor_sim_threshold=None,  # validated: sweep showed this masking doesn't
                                           # clearly help and was the collapse mechanism's
                                           # entry point -- see train_autoenc_token.py.

        weight_reconstruction_by_patch_freq=True,

        # patch-level (hist<->trans) alignment -- OFF. hist_enc/hist_dec are built but
        # never trained; DualAutoencoder still needs them to exist structurally.
        max_contrast_weight=0.0,

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
        local_warmup_epochs=10,           # (see train_autoenc_token.py for the full history)
    )

    # ---- WHICH SLIDE TO HOLD OUT ------------------------------------------------
    HELD_OUT_SLIDE = held_out_slide

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

    # ---- load token data --------------------------------------------------------
    # Same loading logic as train_autoenc_token.py's step "load token data": reads each
    # slide's cell_token_embeddings.h5ad (from token_preprocessing.py), concatenates,
    # and -- if some slides don't have token data yet -- restricts EVERYTHING (hist,
    # trans, coord) to the slides that do, so run_leave_one_slide_out_token never sees
    # a slide with hist/trans data but no token data.
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

    # NOTE: hist_batch_correct (Harmony) is intentionally not supported here, same as the
    # patch-level run_leave_one_slide_out -- and moot anyway since hist_enc never trains
    # in this regime (max_contrast_weight=0.0).

    # ---- single held-out slide ---------------------------------------------------
    # Named per held-out slide (not a shared "loso_token_log.csv") so that running
    # every slide's fold concurrently as a SLURM array (submit_run_loso_array.sbatch)
    # never has two processes read-modify-write the same CSV at once -- see this
    # script's docstring. merge_loso_logs.py combines them afterward.
    log_path = dir_output / f"loso_token_log_{HELD_OUT_SLIDE}.csv"
    checkpoint_dir = dir_output / "loso_token_checkpoints"

    # run_tag disambiguates run_id/checkpoint filename when comparing regimes on the
    # same held_out_slide -- otherwise a second run silently overwrites the first run's
    # checkpoint (same collision class run_sweep.py's BATCH_TAG fixed). Built from
    # whichever non-default regimes are active, so it composes as new toggles get added.
    _tag_parts = []
    if CONFIG["freeze_token"]:
        _tag_parts.append("frozen_token")
    if CONFIG["gene_selection"] != "svg":
        _tag_parts.append(CONFIG["gene_selection"])
    RUN_TAG = ("_" + "_".join(_tag_parts)) if _tag_parts else ""

    print(f"\n=== Leave-one-slide-out (token branch): holding out {HELD_OUT_SLIDE} "
          f"(freeze_token={CONFIG['freeze_token']}) ===\n")
    metrics, model, history = run_leave_one_slide_out_token(
        HELD_OUT_SLIDE, f_hist_adata, f_trans_adata, f_trans_coord, f_token_adata, CONFIG, device,
        log_path=log_path, checkpoint_dir=checkpoint_dir, seed=0, verbose=True, run_tag=RUN_TAG,
    )

    print("\n=== Result ===")
    print(f"In-sample (training slides) cell top-1 retrieval:     {metrics['train_cell_top1_retrieval']:.4f} "
          f"(cluster top-1 {metrics['train_cell_cluster_top1_retrieval']:.4f})")
    print(f"Held-out ({HELD_OUT_SLIDE}) cell top-1 retrieval:        {metrics['held_out_cell_top1_retrieval']:.4f} "
          f"(cluster top-1 {metrics['held_out_cell_cluster_top1_retrieval']:.4f})")
    print(f"Gap (train - held-out): top1={metrics['train_cell_top1_retrieval'] - metrics['held_out_cell_top1_retrieval']:.4f} "
          f"cluster_top1={metrics['train_cell_cluster_top1_retrieval'] - metrics['held_out_cell_cluster_top1_retrieval']:.4f}")
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

    # Best-effort only: plyer's Linux backend needs notify-send/dbus, neither of which
    # exist on a headless cluster node -- confirmed to raise NotImplementedError there
    # (see loso_err.txt from the CRC_P1 run). That happened AFTER the real work and the
    # CSV/checkpoint writes above, so it never lost anything, but it did leave every job
    # exiting on an uncaught traceback -- confusing across an 8-task array where you're
    # scanning for which folds actually failed. Swallowed here since the notification
    # itself is a non-essential convenience.
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
