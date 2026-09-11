"""
Leave-one-slide-out generalization test -- ABLATION variant of run_loso.py.

Isolates which of the two bundled 2026-09 changes (SlideStratifiedLoader replacing
PatchGroupedLoader, vs. recalibrating max_cells_per_slide/oversample_slides from
patch-count to cell-count values) is responsible for the retrieval regression seen
after retraining with both at once (held_out_cell_top1/cluster_top1 dropped across
nearly all 9 folds vs. the pre-change PatchGroupedLoader run -- see
loso_run_with_two_new_changes/ for both sets of logs).

This script keeps SlideStratifiedLoader (the loader change) but reverts
max_cells_per_slide/oversample_slides to their OLD, pre-recalibration numeric values
(11000 / 24000 each) -- i.e. exactly the values the last KNOWN-GOOD (higher-retrieval)
PatchGroupedLoader run used, now just fed through the new loader instead. If results
come back close to the old PatchGroupedLoader run's numbers, the loader swap is not
the culprit and the recalibration (or the interaction between the two) is. If results
still look like the regressed run, the loader swap itself is implicated.

This is a diagnostic-only variant, not a new default -- the 11000/24000 values are
known to be miscalibrated at cell granularity (see run_loso.py's own CONFIG comments
for why), kept here unchanged ONLY so this run isolates a single variable against the
last good result. Do not promote these numbers back into run_loso.py after this.

Everything else (all other CONFIG values, data loading, eval) is identical to
run_loso.py -- diff the two files if you need to confirm nothing else drifted.

--held_out_slide picks the fold (default "CRC_P1"). For the full N-fold run, submit
via submit_run_loso_ablation_oldcap.sbatch (mirrors submit_run_loso_array.sbatch).
"""

# ---- imports ----------------------------------------------------------
import argparse
import sys

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from pathlib import Path
from plyer import notification

# Finds the ancestor directory that actually contains `functions/` -- the repo
# root when run from a checkout (this file sits two levels under it), or this
# script's own directory when run from a flat cluster deployment that copied
# the `functions/` package alongside it (see submit_run_loso_array.sbatch).
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from functions.dual_autoencoder import *


def run_it(held_out_slide="CRC_P1"):
    set_seed(42)

    # ---- CONFIG -------------------------------------------------------------
    # Identical to run_loso.py's CONFIG except max_cells_per_slide/oversample_slides
    # below, reverted to their pre-recalibration values for this ablation -- see
    # module docstring.
    CONFIG = dict(
        sample="combined",     # unused here -- slide filtering happens inside
                                # run_leave_one_slide_out_token itself
        bin_size="002um",
        dir_cellseg="cell_seg_stardist",
        loss_fct="ntxent",
        token_filename="cell_token_embeddings.h5ad",  # "bin_token_embeddings.h5ad" for bins

        # model
        latent_dim=64,
        proj_dim=32,
        encoder="transconv",

        gene_selection="svg",   # "svg" | "hvg" | "intersection" -- spatial patterning,
                                 # the signal the alignment task needs

        # graph / batching
        # n_neighs: 6 for the (unused-by-default) KNN fallback; 8 for spatial_graph_grid=True
        # (bins -- full Moore/queen neighborhood of a square lattice, see build_spatial_graph).
        # Irrelevant when spatial_graph_delaunay=True (nuclei default), which uses percentile instead.
        n_neighs=6,
        spatial_graph_delaunay=True,  # nuclei/cells: matches get_svg.py's own cell-path choice
        spatial_graph_grid=False,     # bins: set True (with n_neighs=8) for a regular 8um lattice
        num_neighbors=(10, 10),
        # SlideStratifiedLoader kept (this ablation only reverts the caps below, not
        # the loader itself) -- see run_loso.py's CONFIG comment for the loader's own
        # rationale.
        cells_per_batch=1024,

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

        # ABLATION: reverted to the OLD, pre-recalibration values (known miscalibrated
        # at cell granularity -- see run_loso.py) so this run isolates the loader swap
        # as the only variable against the last known-good PatchGroupedLoader result.
        # Do not treat 11000 as a real per-slide cap going forward.
        max_cells_per_slide=11000,

        # ABLATION: same reasoning -- old flat 24000/24000 (a patch-count value, not a
        # calibrated cell-count target), reverted here on purpose. See run_loso.py's
        # CONFIG comment for why this was changed, and why it should stay changed for
        # any run other than this ablation.
        oversample_slides={"Normal_P3": 24000, "Normal_P5": 24000},
    )

    # ---- WHICH SLIDE TO HOLD OUT ------------------------------------------------
    HELD_OUT_SLIDE = held_out_slide

    # ---- slides excluded from the dataset entirely (not just held out) ---------
    EXCLUDE_SLIDES = []

    # ---- output dirs ----------------------------------------------------------
    # Robust to invocation directory: walks up from this file's own location for
    # a `data/` sibling, rather than assuming a fixed relative depth from CWD
    # (broke when this file moved into scripts/, one level deeper than before).
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "data").is_dir():
            basedir = _candidate / "data"
            break
    else:
        raise RuntimeError("Could not find a 'data' directory in any ancestor of this script")
    dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
    dir_transcriptomics = basedir / "combined/01_preprocessing" / CONFIG["dir_cellseg"] / CONFIG["bin_size"] / "custom_binning"
    dir_output = basedir / "combined" / "03_integration/autoencoders" / CONFIG["bin_size"]
    dir_output.mkdir(parents=True, exist_ok=True)

    # ---- load data (ALL slides -- filtering into train/held-out happens inside ----
    # ---- run_leave_one_slide_out_token itself, not here) -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
    f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
    _geom_path = dir_transcriptomics / "nuclei_geometry.parquet"
    f_trans_coord = gpd.read_parquet(_geom_path) if _geom_path.exists() else None

    print("All slides available:", sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique().tolist()))

    trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
    if not trans_global_id.is_unique:
        raise ValueError("Cannot set Slide_ID+id as index: contains duplicates")
    f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)

    hist_global_id = f_hist_adata.obs["Slide_ID"].astype(str) + "_" + f_hist_adata.obs["Patch_ID"].astype(str)
    if not hist_global_id.is_unique:
        raise ValueError("Cannot set Slide_ID+Patch_ID as index: contains duplicates")
    f_hist_adata.obs = f_hist_adata.obs.set_index(hist_global_id)

    if EXCLUDE_SLIDES:
        if HELD_OUT_SLIDE in EXCLUDE_SLIDES:
            raise ValueError(f"HELD_OUT_SLIDE={HELD_OUT_SLIDE!r} is also in EXCLUDE_SLIDES "
                              f"{EXCLUDE_SLIDES} -- pick a held-out slide that isn't excluded.")
        print(f"Excluding from the dataset entirely (not just as a held-out choice): {EXCLUDE_SLIDES}")
        f_hist_adata = f_hist_adata[~f_hist_adata.obs["Slide_ID"].isin(EXCLUDE_SLIDES)].copy()
        f_trans_adata = f_trans_adata[~f_trans_adata.obs["Slide_ID"].astype(str).isin(EXCLUDE_SLIDES)].copy()
        if f_trans_coord is not None:
            f_trans_coord = f_trans_coord[~f_trans_coord["Slide_ID"].isin(EXCLUDE_SLIDES)].copy()

    # ---- load token data --------------------------------------------------------
    slides_needed = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique())
    token_adatas, slides_with_tokens = [], []
    for s in slides_needed:
        p = basedir / s / "01_preprocessing/unimodel/metadata" / CONFIG["token_filename"]
        if p.exists():
            token_adatas.append(sc.read_h5ad(p))
            slides_with_tokens.append(s)
        else:
            print(f"WARNING: no token data for slide {s} ({p}) -- run token_preprocessing.py --unit nuclei for it first")

    missing = sorted(set(slides_needed) - set(slides_with_tokens))
    if missing:
        print(f"Restricting to slides with token data: {slides_with_tokens} (dropping {missing})")
        f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"].isin(slides_with_tokens)].copy()
        f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"].astype(str).isin(slides_with_tokens)].copy()
        if f_trans_coord is not None:
            f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"].isin(slides_with_tokens)].copy()
    if not token_adatas:
        raise RuntimeError("no in-scope slide has token data -- run token_preprocessing.py --unit nuclei first")
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

    _tag_parts = ["slideloaderablation_oldcap"]  # forced first: guarantees this run's
    # log/checkpoint filenames never collide with either the pre-change PatchGroupedLoader
    # run (which used the exact same numeric cap11000/oversample_normal tag) or the
    # post-change cap300000 run -- all three stay independently distinguishable.
    if CONFIG["freeze_token"]:
        _tag_parts.append("frozen_token")
    if CONFIG["gene_selection"] != "svg":
        _tag_parts.append(CONFIG["gene_selection"])
    if EXCLUDE_SLIDES:
        _tag_parts.append("excl_" + "_".join(EXCLUDE_SLIDES))
    if CONFIG.get("max_cells_per_slide") is not None:
        _tag_parts.append(f"cap{CONFIG['max_cells_per_slide']}")
    if CONFIG.get("oversample_slides"):
        _tag_parts.append("oversample_normal")
    RUN_TAG = ("_" + "_".join(_tag_parts)) if _tag_parts else ""

    log_path = dir_output / f"loso_token_log_{HELD_OUT_SLIDE}{RUN_TAG}.csv"

    print(f"\n=== [ABLATION: SlideStratifiedLoader + OLD caps] Leave-one-slide-out "
          f"(token branch): holding out {HELD_OUT_SLIDE} "
          f"(freeze_token={CONFIG['freeze_token']}) ===\n")
    metrics, model, history = run_leave_one_slide_out_token(
        HELD_OUT_SLIDE, f_hist_adata, f_trans_adata, f_trans_coord, f_token_adata, CONFIG, device,
        log_path=log_path, checkpoint_dir=checkpoint_dir, seed=0, verbose=True, run_tag=RUN_TAG,
    )

    print("\n=== Result ===")
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

    try:
        notification.notify(
            title='Jupyter Notebook Alert',
            message=f'LOSO-token ablation (old caps) run for {HELD_OUT_SLIDE} finished!',
            app_name='Python',
            timeout=5,
        )
    except Exception as e:
        print(f"(desktop notification skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held_out_slide", default="CRC_P1",
                         help="Slide to hold out for this fold (default: CRC_P1).")
    args = parser.parse_args()
    run_it(held_out_slide=args.held_out_slide)
