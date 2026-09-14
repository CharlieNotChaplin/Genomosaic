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


def run_it(held_out_slide="CRC_P1", edge_dropout_p=0.0, gene_dropout_p=0.0,
           local_neighbor_radius=100, local_anchor_sim_threshold=None,
           local_anchor_sim_source="projected",
           skip_token_pretrain=False, freeze_token=False, weight_reconstruction=True,
           weight_decay=0.0, encoder_type="transconv", temperature=0.05, gene_panel=None,
           gene_selection="svg"):
    """
    Ablation-relevant knobs are function params (not hardcoded in CONFIG below) so
    __main__'s CLI parsing can override them per sbatch submission without touching
    this file -- defaults here exactly reproduce the previously-hardcoded values,
    so a plain `run_it(held_out_slide=...)` call is unchanged from before.
    """
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
        token_filename="cell_token_embeddings.h5ad",  # "bin_token_embeddings.h5ad" for bins

        # model
        latent_dim=64,
        proj_dim=32,
        # transconv (attention, LayerNorm) is the universal default -- "sage" here
        # tests DualAutoencoder's other encoder (GraphSAGE, still nn.BatchNorm1d
        # internally, never itself ablated against transconv) at its own defaults.
        encoder=encoder_type,

        gene_selection=gene_selection,   # "svg" | "hvg" | "intersection" -- spatial
                                 # patterning, the signal the alignment task needs.
                                 # Ignored entirely if gene_panel is set below (an
                                 # external panel overrides internal selection, see
                                 # get_gene_mask).
        gene_panel_path=None,   # resolved from `gene_panel` just below, once basedir
                                 # is known -- None (default): use gene_selection above.

        # graph / batching
        # n_neighs: 6 for the (unused-by-default) KNN fallback; 8 for spatial_graph_grid=True
        # (bins -- full Moore/queen neighborhood of a square lattice, see build_spatial_graph).
        # Irrelevant when spatial_graph_delaunay=True (nuclei default), which uses percentile instead.
        n_neighs=6,
        spatial_graph_delaunay=True,  # nuclei/cells: matches get_svg.py's own cell-path choice
        spatial_graph_grid=False,     # bins: set True (with n_neighs=8) for a regular 8um lattice
        num_neighbors=(10, 10),
        # Seed cells drawn uniformly at random per slide (SlideStratifiedLoader),
        # not grouped by shared histology-patch membership -- the patch-level
        # pooled contrastive loss that grouping existed for is gone; reconstruction
        # weighting and the cell<->token contrastive term are already per-cell,
        # needing only a per-cell patch-weight lookup. ~1024 was old batches'
        # typical realized size (64 patches x a few cells/patch each, rarely
        # hitting the old 64-cell-per-patch cap) -- a starting point, re-tune
        # against actual throughput/GPU memory now that it's a direct cap.
        cells_per_batch=1024,

        # loss
        # Contrastive logits are divided by this before softmax (see local_infonce) --
        # lower = sharper/more confident distribution, higher = softer. 0.05 (default,
        # changed from the CLIP/SimCLR-inherited 0.07 -- see ablation_studies_summary.txt's
        # "CONTRASTIVE TEMPERATURE SWEEP" section) beat 0.07 on held-out retrieval for
        # every slide in the 3-slide probe; 0.03 didn't improve further (plateau), 0.2
        # was clearly worse. Ablatable via CLI.
        temperature=temperature,
        neighbor_radius=2912,       # still required by compute_patch_metadata/
                                     # build_pipeline_data's shared patch-batching path
        local_neighbor_radius=local_neighbor_radius,  # cell-level analogue, first attempt

        local_anchor_sim_threshold=local_anchor_sim_threshold,  # None (default): the
                                           # 18-config sweep showed masking here doesn't
                                           # help and can trigger collapse -- see
                                           # train_autoenc_token.py. Ablatable via CLI.
        # "projected" (default) reproduces that same tested-and-rejected behavior
        # (checks the model's own live projections). "raw" checks fixed,
        # pre-encoder feature vectors instead -- not yet validated, see
        # local_infonce's own docstring for why this might avoid the collapse
        # feedback loop "projected" is prone to. Only has any effect when
        # local_anchor_sim_threshold is not None.
        local_anchor_sim_source=local_anchor_sim_source,

        # Name predates the compute_local_density_weight swap (see functions/
        # dual_autoencoder.py) -- this now gates the per-unit DENSITY weight, not
        # a patch-frequency one, despite the key name. Never independently
        # ablated against a plain unweighted mean -- see run_it's own param.
        weight_reconstruction_by_patch_freq=weight_reconstruction,

        # Denoising corruption (see train_step's own docstring) -- 0.0/0.0 (default)
        # means this has never actually been exercised in any run to date; both are
        # ablation knobs, not yet-validated defaults.
        edge_dropout_p=edge_dropout_p,
        gene_dropout_p=gene_dropout_p,

        # Ablation hook: True skips Phase-1 token pretraining entirely, so token_enc/
        # token_dec start from the same random init as the rest of the model.
        skip_token_pretrain=skip_token_pretrain,

        # Phase 3's own weight_decay -- 0.0 (default) means never exercised in any
        # run to date, despite MLPDecoder's own docstring describing weight_decay
        # as the actual mechanism that lets slide-conditioning work (the incentive
        # to drop slide identity from the bottleneck once conditioning makes it
        # unnecessary for reconstruction). Phase 1 (token pretraining) already uses
        # AdamW + token_pretrain_weight_decay=1e-5 successfully; train()'s own
        # optimizer was plain Adam until this flag was added (switched to AdamW
        # alongside it, since Adam's L2-style decay doesn't behave like AdamW's
        # decoupled decay once this is actually nonzero).
        weight_decay=weight_decay,

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
        freeze_token=freeze_token,
        max_local_contrast_weight=1.0,   # winner of run_token_sweep.py's 18-config sweep
        local_warmup_epochs=10,

        # Caps CRC_P11mm's per-epoch batch share: SlideStratifiedLoader allocates
        # batches proportional to each slide's own (capped) cell count. Recalibrated
        # (2026-09) against real per-slide cell counts from count_area_filtered_adata.h5ad
        # -- cell:patch ratios turned out NOT uniform across slides, so the old
        # patch-count cap (11000) badly undershot at cell granularity (every slide has
        # 5-70x more cells than that, so it was throttling everyone, not just CRC_P11mm).
        # Real counts (9-slide array, CRC_P6_5mm excluded as usual): CRC_P11mm=785828
        # (outlier), next-largest CRC_P3_SMAP=295977, then CRC_P2=224825 down to
        # Normal_P3=57187. 300000 sits just above CRC_P3_SMAP -- a no-op for every slide
        # except CRC_P11mm, same rationale as the original patch-count value.
        # (Tried fully EXCLUDING CRC_P11mm instead: reduced leakage and ~doubled in-sample
        # fit, but held-out generalization stayed flat and the train/held-out gap widened
        # -- an overfitting signature, not a fix. Capping is the better tradeoff.)
        max_cells_per_slide=300000,

        # Boosts the 2 normal slides' per-epoch exposure: evaluate_disease_signal showed
        # the model doesn't reliably generalize "normal tissue" to a second unseen normal
        # patient (Normal_P5 stayed near chance even after rebalancing the diagnostic
        # classifier, unlike Normal_P3). Recalibrated (2026-09) by applying the ORIGINAL
        # design's own multiplier (~3x a normal slide's natural size) to each slide's real
        # cell count, rather than reusing the old flat 24000/slide (a patch-count value
        # that happened to look uniform only because both normal slides had similar patch
        # counts -- at cell level they don't: Normal_P3=57187 cells, Normal_P5=127584,
        # a 2.2x difference). A flat target would have under-boosted Normal_P5 and
        # over-boosted Normal_P3 relative to their real sizes; scaling each by the same
        # ~3x keeps both slides' relative representation intact while still moving the
        # aggregate tumor:normal per-epoch ratio from ~10:1 (natural) to ~2.4:1 (after
        # capping CRC_P11mm to max_cells_per_slide above), similar magnitude of rebalancing
        # to the original design's own ~2:1 target. Resampling real cells with replacement
        # (see SlideStratifiedLoader.oversample_target), not synthetic augmentation.
        oversample_slides={"Normal_P3": 172000, "Normal_P5": 383000},
    )

    # ---- WHICH SLIDE TO HOLD OUT ------------------------------------------------
    HELD_OUT_SLIDE = held_out_slide

    # ---- slides excluded from the dataset entirely (not just held out) ---------
    # CRC_P3_SMAP/CRC_P4_SMAP's preprocessing is now complete, so both are included --
    # a new 10-slide (8 tumor/2 normal) baseline, not comparable fold-by-fold against
    # the earlier 8-slide capped-batches results (different pool, different n_slides).
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

    # gene_panel is a bare name (e.g. "colon_xenium"), resolved against the fixed
    # panel directory here rather than accepted as a raw path -- keeps the CLI/sbatch
    # invocation short and stops a typo'd path from silently landing outside the
    # repo's own tracked panel set.
    if gene_panel is not None:
        gene_panel_path = basedir / "combined/01_preprocessing/gene_panels" / f"{gene_panel}.json"
        if not gene_panel_path.is_file():
            raise FileNotFoundError(f"No gene panel named {gene_panel!r} at {gene_panel_path}")
        CONFIG["gene_panel_path"] = gene_panel_path

    # ---- load data (ALL slides -- filtering into train/held-out happens inside ----
    # ---- run_leave_one_slide_out_token itself, not here) -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
    f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
    # Bins have no geometry side-car (obsm['spatial'] is already populated directly at
    # load time) -- extract_cell_xy already prioritizes obsm['spatial'] over f_trans_coord,
    # so None here is a safe, fully-supported fallback, not a degraded path.
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

    # apply EXCLUDE_SLIDES before token loading, so slides_needed below never looks
    # for their token files
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
    # Reads each slide's token embeddings (CONFIG["token_filename"] -- cell or bin),
    # concatenates, and -- if some slides don't have token data yet -- restricts
    # hist/trans/coord to the slides that do.
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

    # run_tag disambiguates checkpoint/log filenames between regimes on the same
    # held_out_slide, so a second run doesn't overwrite the first's outputs
    _tag_parts = []
    if CONFIG["freeze_token"]:
        _tag_parts.append("frozen_token")
    if CONFIG.get("skip_token_pretrain"):
        _tag_parts.append("nopretrain")
    if CONFIG.get("edge_dropout_p", 0.0) > 0.0 or CONFIG.get("gene_dropout_p", 0.0) > 0.0:
        _tag_parts.append(f"edrop{CONFIG.get('edge_dropout_p', 0.0)}_gdrop{CONFIG.get('gene_dropout_p', 0.0)}")
    if CONFIG.get("local_neighbor_radius") != 100:
        _tag_parts.append(f"localrad{CONFIG.get('local_neighbor_radius')}")
    if CONFIG.get("local_anchor_sim_threshold") is not None:
        _tag_parts.append(f"anchorsim{CONFIG['local_anchor_sim_threshold']}")
        if CONFIG.get("local_anchor_sim_source", "projected") != "projected":
            _tag_parts.append(f"simsrc_{CONFIG['local_anchor_sim_source']}")
    if not CONFIG.get("weight_reconstruction_by_patch_freq", True):
        _tag_parts.append("noweight")
    if CONFIG.get("weight_decay", 0.0) > 0.0:
        _tag_parts.append(f"wd{CONFIG['weight_decay']}")
    if CONFIG["encoder"] != "transconv":
        _tag_parts.append(f"enc_{CONFIG['encoder']}")
    if CONFIG["temperature"] != 0.05:
        _tag_parts.append(f"temp{CONFIG['temperature']}")
    if CONFIG.get("gene_panel_path") is not None:
        _tag_parts.append(f"panel_{CONFIG['gene_panel_path'].stem}")
    elif CONFIG["gene_selection"] != "svg":
        _tag_parts.append(CONFIG["gene_selection"])
    if EXCLUDE_SLIDES:
        _tag_parts.append("excl_" + "_".join(EXCLUDE_SLIDES))
    if CONFIG.get("max_cells_per_slide") is not None:
        _tag_parts.append(f"cap{CONFIG['max_cells_per_slide']}")
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


def _float_or_none(s):
    """argparse type: accepts a float, or the literal string 'none' (any case)."""
    return None if s.lower() == "none" else float(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held_out_slide", default="CRC_P1",
                         help="Slide to hold out for this fold (default: CRC_P1). "
                              "submit_run_loso_array.sbatch sets this per array task "
                              "for the full N-fold run.")
    # ---- ablation knobs -- all default to exactly the prior hardcoded CONFIG values,
    # so omitting a flag reproduces the pre-ablation-CLI behavior exactly. ----
    parser.add_argument("--edge-dropout-p", type=float, default=0.0,
                         help="Denoising: fraction of graph edges dropped per training "
                              "step, encoder's view only (default: 0.0, i.e. off -- "
                              "never yet exercised in any run before this flag existed).")
    parser.add_argument("--gene-dropout-p", type=float, default=0.0,
                         help="Denoising: fraction of gene entries masked per training "
                              "step (default: 0.0, i.e. off, same caveat as above).")
    parser.add_argument("--local-neighbor-radius", type=_float_or_none, default=100,
                         help="Cell<->token contrastive loss: spatial exclusion radius, "
                              "or 'none' to disable spatial exclusion entirely "
                              "(default: 100, the value every existing run has used).")
    parser.add_argument("--local-anchor-sim-threshold", type=_float_or_none, default=None,
                         help="Cell<->token contrastive loss: semantic (cosine-similarity) "
                              "exclusion threshold, or 'none' to disable (default: none, "
                              "the value every existing run has used).")
    parser.add_argument("--local-anchor-sim-source", choices=["projected", "raw"], default="projected",
                         help="What --local-anchor-sim-threshold checks similarity on "
                              "(no effect if that threshold is unset): 'projected' (default) "
                              "checks the model's own live projections -- tested, "
                              "underperformed leaving masking off entirely (see "
                              "ablation_studies_summary.txt). 'raw' checks fixed, "
                              "pre-encoder feature vectors instead, which can't feed into "
                              "the same collapse feedback loop -- not yet validated, "
                              "thresholds tuned for 'projected' (0.7/0.9) do NOT carry "
                              "over, see local_infonce's docstring before picking a value.")
    parser.add_argument("--skip-token-pretrain", action="store_true",
                         help="Skip Phase-1 token-branch pretraining entirely -- "
                              "token_enc/token_dec start from the same random init as "
                              "the rest of the model (default: pretraining is kept).")
    parser.add_argument("--freeze-token", action="store_true",
                         help="Freeze token_enc/token_dec during Phase 3 joint training "
                              "(default: not frozen). Pair with pretraining left ON "
                              "(the default) -- freezing a randomly-initialized token "
                              "branch is not a meaningful test.")
    parser.add_argument("--no-weight-reconstruction", action="store_true",
                         help="Disable weighting the transcriptomics reconstruction loss "
                              "by per-unit inverse density (default: weighted). Never "
                              "independently ablated against a plain unweighted mean "
                              "before this flag existed.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                         help="Phase 3 (joint training) AdamW weight decay, applied to "
                              "every trainable param group (default: 0.0, every run to "
                              "date). Phase 1 token pretraining already uses AdamW + "
                              "1e-5 successfully -- this tests the same on Phase 3, where "
                              "MLPDecoder's own docstring describes weight_decay as the "
                              "actual mechanism behind slide-conditioning's benefit, "
                              "despite it never having been nonzero here before.")
    parser.add_argument("--encoder", choices=["sage", "transconv"], default="transconv",
                         help="Transcriptomics graph encoder (default: transconv, the "
                              "universal default in every run to date). 'sage' is "
                              "DualAutoencoder's other implementation (GraphSAGE, "
                              "nn.BatchNorm1d internally) -- never itself compared "
                              "against transconv before this flag existed.")
    parser.add_argument("--temperature", type=float, default=0.05,
                         help="Contrastive loss softmax temperature (see local_infonce) "
                              "-- lower is a sharper/more confident distribution, higher "
                              "is softer (default: 0.05, adopted after a fixed-value sweep "
                              "over {0.03, 0.05, 0.07, 0.2} on the 3-slide probe beat the "
                              "CLIP/SimCLR-inherited 0.07 on every held-out slide; see "
                              "ablation_studies_summary.txt).")
    parser.add_argument("--gene-panel", default=None,
                         help="Bare name (no path, no .json) of an external, predesigned "
                              "gene panel under data/combined/01_preprocessing/gene_panels/ "
                              "-- e.g. 'colon_xenium'. OVERRIDES --gene-selection entirely: "
                              "trains on the intersection of this panel's genes and "
                              "whatever's actually in var_names (reported at runtime), "
                              "instead of any internally-derived (svg/hvg/intersection) "
                              "selection (default: none, i.e. use --gene-selection).")
    parser.add_argument("--gene-selection", choices=["svg", "hvg", "intersection"], default="svg",
                         help="Internal gene selection (default: svg, the validated "
                              "regime). Ignored entirely if --gene-panel is given.")
    args = parser.parse_args()
    run_it(
        held_out_slide=args.held_out_slide,
        edge_dropout_p=args.edge_dropout_p,
        gene_dropout_p=args.gene_dropout_p,
        local_neighbor_radius=args.local_neighbor_radius,
        local_anchor_sim_threshold=args.local_anchor_sim_threshold,
        local_anchor_sim_source=args.local_anchor_sim_source,
        skip_token_pretrain=args.skip_token_pretrain,
        freeze_token=args.freeze_token,
        weight_reconstruction=not args.no_weight_reconstruction,
        weight_decay=args.weight_decay,
        encoder_type=args.encoder,
        temperature=args.temperature,
        gene_panel=args.gene_panel,
        gene_selection=args.gene_selection,
    )
