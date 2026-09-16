"""
Leave-one-slide-out generalization test -- BIN-LEVEL variant of run_loso.py.

Mirrors run_loso.py's structure and validated CONFIG exactly (SlideStratifiedLoader,
transconv encoder, cell<->token contrastive alignment, etc.) with only the unit-
specific changes bins actually need:
  - bin_size="008um", dir_cellseg="grid_bin" (not cell_seg_stardist/002um)
  - token_filename="bin_token_embeddings.h5ad" (not cell_token_embeddings.h5ad)
  - spatial_graph_grid=True (regular 8um lattice), not spatial_graph_delaunay --
    n_neighs=8 for the full Moore/queen neighborhood of a square grid (see
    build_spatial_graph's docstring)
  - max_cells_per_slide/oversample_slides recalibrated against REAL bin counts
    (2026-09, from the just-completed full 10-slide bin merge) -- see below for
    the exact numbers and where they came from. NOT the same values as the cell-
    level script: bin:cell ratios are not uniform across slides (confirmed directly
    -- e.g. CRC_P11mm_2_cut's bin count turned out much smaller than a naive
    raw-tissue estimate suggested, once its own Patch_ID-matching was actually run).

CAVEATS not yet resolved (flagging rather than guessing):
  - oversample_slides below is NOT corrected for bin-level token-match rate the way
    the cell-level values were (see diagnose_pipeline.py's per-slide "matched" check)
    -- that correction needs a bin-level equivalent diagnostic, not yet built. Treat
    these as a first pass, same status the cell-level numbers had before that
    correction was applied.
  - local_neighbor_radius=100 is carried over UNCHANGED from the cell-level script,
    where it was calibrated to median nearest-CELL spacing (~27px). Bins sit on a
    regular 8um lattice, a different physical spacing in pixels (see
    token_preprocessing.py's own half_side_px = grid_um / (2 * microns_per_pixel)
    calculation for the actual pitch) -- this value should be re-derived from that
    conversion rather than assumed correct for bins.

--held_out_slide picks the fold (default "CRC_P1"). For the full N-fold run, submit
via submit_run_loso_bins_array.sbatch (mirrors submit_run_loso_array.sbatch).
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

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from functions.dual_autoencoder import *


def run_it(held_out_slide="CRC_P1", edge_dropout_p=0.0, gene_dropout_p=0.0,
           local_neighbor_radius=100, local_anchor_sim_threshold=None,
           local_anchor_sim_source="projected", local_positive_mode="exclude",
           skip_token_pretrain=False, freeze_token=False, weight_reconstruction=True,
           weight_decay=0.0, temperature=0.05, gene_panel=None, gene_selection="svg"):
    """
    Ablation-relevant knobs mirror run_loso.py's own -- see that file's run_it
    docstring. Defaults here exactly reproduce the previously-hardcoded values.
    """
    set_seed(42)

    # ---- CONFIG -------------------------------------------------------------
    # Identical to run_loso.py's CONFIG except the unit-specific block below --
    # diff the two files if you need to confirm nothing else drifted.
    CONFIG = dict(
        sample="combined",
        bin_size="008um",
        dir_cellseg="grid_bin",
        loss_fct="ntxent",
        token_filename="bin_token_embeddings.h5ad",

        # model
        latent_dim=64,
        proj_dim=32,
        encoder="transconv",

        gene_selection=gene_selection,  # ignored entirely if gene_panel is set below
        gene_panel_path=None,   # resolved from `gene_panel` just below, once basedir
                                 # is known -- None (default): use gene_selection above.

        # graph / batching -- bin-specific: regular 8um lattice, not Delaunay
        n_neighs=8,                    # full Moore/queen neighborhood of a square grid
        spatial_graph_delaunay=False,  # mutually exclusive with grid=True below
        spatial_graph_grid=True,
        num_neighbors=(10, 10),
        cells_per_batch=1024,  # same starting point as the cell-level script -- retune
                                # against actual throughput/GPU memory for bins separately

        # loss
        # 0.05 default -- see run_loso.py's own CONFIG comment / ablation_studies_
        # summary.txt's "CONTRASTIVE TEMPERATURE SWEEP" section (nuclei-only sweep,
        # never independently re-tested for bins -- carried over on the same
        # assumption-transfer basis as other nuclei-validated defaults here).
        temperature=temperature,
        neighbor_radius=2912,       # patch-geometry-derived (2x 224px patch diameter),
                                     # not unit-size-derived -- unchanged for bins
        local_neighbor_radius=local_neighbor_radius,  # CARRIED OVER from the cell-level
                                     # value by default, NOT re-derived for bin pitch --
                                     # see module docstring. Ablatable via CLI.

        local_anchor_sim_threshold=local_anchor_sim_threshold,
        # "raw" not yet tested for bins -- nuclei's own productive threshold range
        # (0.1-0.3) was calibrated entirely on nuclei's raw SVG-expression/token
        # feature space; bins' own space may behave differently (different SVG
        # count, different token aggregation). Probe before trusting a value --
        # see submit_run_loso_ablation_bins_semantic_probe.sbatch.
        local_anchor_sim_source=local_anchor_sim_source,
        # "exclude" (default) vs "attract" (multi-positive InfoNCE) -- see
        # local_infonce's own docstring. Not yet tested for bins either.
        local_positive_mode=local_positive_mode,

        # Name predates the compute_local_density_weight swap -- gates the per-unit
        # DENSITY weight now, not a patch-frequency one. See run_loso.py's own note.
        weight_reconstruction_by_patch_freq=weight_reconstruction,

        # Denoising corruption -- see run_loso.py's own CONFIG comment. 0.0/0.0
        # (default) means never yet exercised for bins prior to these flags.
        edge_dropout_p=edge_dropout_p,
        gene_dropout_p=gene_dropout_p,

        skip_token_pretrain=skip_token_pretrain,

        # Phase 3 AdamW weight decay -- see run_loso.py's own CONFIG comment.
        # 1e-5 (matching Phase 1's own, already-validated value) is the nuclei-
        # adopted setting; never independently re-tested for bins.
        weight_decay=weight_decay,

        # training (Phase 3)
        n_epochs=100,
        lr=1e-3,
        warmup_epochs=5,

        # token branch
        token_pretrain_epochs=100,
        token_pretrain_lr=1e-3,
        token_pretrain_weight_decay=1e-5,
        token_pretrain_val_frac=0.1,
        token_pretrain_patience=15,
        token_lr_mult=0.3,
        freeze_token=freeze_token,
        max_local_contrast_weight=1.0,
        local_warmup_epochs=10,

        # Recalibrated (2026-09) against REAL per-slide bin counts from the completed
        # 10-slide bin merge (9-slide array, CRC_P6_5mm excluded as usual):
        #   CRC_P11mm=1,538,238 (outlier), CRC_P3_SMAP=602,274 (next-largest),
        #   CRC_P2=444,991, CRC_P4_SMAP=440,676, CRC_P5=412,811, CRC_P1=332,795,
        #   Normal_P5=285,345, CRC_P11mm_2_cut=283,676, Normal_P3=156,510.
        # 610000 sits just above CRC_P3_SMAP -- a no-op for every slide except
        # CRC_P11mm, same rationale as the cell-level cap.
        max_cells_per_slide=610000,

        # Same ~3x multiplier as the cell-level script's own (original design's)
        # choice, applied to each normal slide's real bin count: Normal_P3
        # 156510*3=469530, Normal_P5 285345*3=856035. NOT corrected for bin-level
        # token-match rate -- see module docstring caveat.
        oversample_slides={"Normal_P3": 470000, "Normal_P5": 856000},
    )

    HELD_OUT_SLIDE = held_out_slide
    EXCLUDE_SLIDES = []

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
    # panel directory -- see run_loso.py's own comment for the full rationale.
    if gene_panel is not None:
        gene_panel_path = basedir / "combined/01_preprocessing/gene_panels" / f"{gene_panel}.json"
        if not gene_panel_path.is_file():
            raise FileNotFoundError(f"No gene panel named {gene_panel!r} at {gene_panel_path}")
        CONFIG["gene_panel_path"] = gene_panel_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
    f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
    # Bins have no geometry side-car (obsm['spatial'] is already populated directly at
    # load time) -- extract_cell_xy already prioritizes obsm['spatial'] over f_trans_coord.
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

    slides_needed = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique())
    token_adatas, slides_with_tokens = [], []
    for s in slides_needed:
        p = basedir / s / "01_preprocessing/unimodel/metadata" / CONFIG["token_filename"]
        if p.exists():
            token_adatas.append(sc.read_h5ad(p))
            slides_with_tokens.append(s)
        else:
            print(f"WARNING: no token data for slide {s} ({p}) -- run token_preprocessing.py --unit bins for it first")

    missing = sorted(set(slides_needed) - set(slides_with_tokens))
    if missing:
        print(f"Restricting to slides with token data: {slides_with_tokens} (dropping {missing})")
        f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"].isin(slides_with_tokens)].copy()
        f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"].astype(str).isin(slides_with_tokens)].copy()
        if f_trans_coord is not None:
            f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"].isin(slides_with_tokens)].copy()
    if not token_adatas:
        raise RuntimeError("no in-scope slide has token data -- run token_preprocessing.py --unit bins first")
    if HELD_OUT_SLIDE not in slides_with_tokens:
        raise ValueError(f"HELD_OUT_SLIDE={HELD_OUT_SLIDE!r} has no token data -- pick one of {slides_with_tokens}")

    f_token_adata = ad.concat(token_adatas, join="outer")
    token_key = f_token_adata.obs["Slide_ID"].astype(str) + "_" + f_token_adata.obs["id"].astype(str)
    if not token_key.is_unique:
        raise ValueError("Cannot set Slide_ID+id as token index: contains duplicates")
    f_token_adata.obs = f_token_adata.obs.set_index(token_key)
    f_token_adata = f_token_adata[f_token_adata.obs["matched"].astype(bool)].copy()
    print(f"Loaded token data: {f_token_adata.n_obs} bins across {len(slides_with_tokens)} slide(s)")

    checkpoint_dir = dir_output / "loso_token_checkpoints"

    _tag_parts = ["bins"]  # forced first: keeps bin-level runs distinguishable from
    # the cell-level script's own log/checkpoint filenames even if cap/oversample
    # values ever coincide numerically.
    if CONFIG["freeze_token"]:
        _tag_parts.append("frozen_token")
    if CONFIG.get("skip_token_pretrain"):
        _tag_parts.append("nopretrain")
    if CONFIG.get("edge_dropout_p", 0.0) > 0.0 or CONFIG.get("gene_dropout_p", 0.0) > 0.0:
        _tag_parts.append(f"edrop{CONFIG.get('edge_dropout_p', 0.0)}_gdrop{CONFIG.get('gene_dropout_p', 0.0)}")
    if CONFIG.get("local_neighbor_radius") != 100:
        _tag_parts.append(f"localrad{CONFIG.get('local_neighbor_radius')}")
    if CONFIG.get("local_positive_mode", "exclude") != "exclude":
        _tag_parts.append(f"posmode_{CONFIG['local_positive_mode']}")
    if CONFIG.get("local_anchor_sim_threshold") is not None:
        _tag_parts.append(f"anchorsim{CONFIG['local_anchor_sim_threshold']}")
        if CONFIG.get("local_anchor_sim_source", "projected") != "projected":
            _tag_parts.append(f"simsrc_{CONFIG['local_anchor_sim_source']}")
    if not CONFIG.get("weight_reconstruction_by_patch_freq", True):
        _tag_parts.append("noweight")
    if CONFIG.get("weight_decay", 0.0) > 0.0:
        _tag_parts.append(f"wd{CONFIG['weight_decay']}")
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

    log_path = dir_output / f"loso_token_log_{HELD_OUT_SLIDE}{RUN_TAG}.csv"

    print(f"\n=== [BIN-LEVEL] Leave-one-slide-out (token branch): holding out {HELD_OUT_SLIDE} "
          f"(freeze_token={CONFIG['freeze_token']}) ===\n")
    # Hand off f_hist_adata/f_trans_adata/f_token_adata via a list + pop() rather
    # than passing the names directly: a plain call would leave THIS frame's own
    # f_hist_adata/f_trans_adata/f_token_adata bindings alive (on the call stack)
    # for the entire duration of run_leave_one_slide_out_token, so its own
    # del+gc.collect() after splitting (see that function's comment) would find a
    # second live reference here and free nothing -- at bin scale (~4.8M rows,
    # tens of GB per array) that's the difference between the split step needing
    # ~1x vs ~2x these objects' combined size. Popping empties this frame's
    # references as each argument is evaluated, so run_leave_one_slide_out_token
    # ends up holding the only reference once the call is entered.
    _handoff = [f_hist_adata, f_trans_adata, f_token_adata]
    del f_hist_adata, f_trans_adata, f_token_adata
    metrics, model, history = run_leave_one_slide_out_token(
        HELD_OUT_SLIDE, _handoff.pop(0), _handoff.pop(0), f_trans_coord, _handoff.pop(0), CONFIG, device,
        log_path=log_path, checkpoint_dir=checkpoint_dir, seed=0, verbose=True, run_tag=RUN_TAG,
    )

    print("\n=== Result ===")
    print(f"In-sample (training slides) bin cluster top-1 retrieval:     {metrics['train_cell_cluster_top1_retrieval']:.4f} "
          f"(exact-bin top-1 {metrics['train_cell_top1_retrieval']:.4f})")
    print(f"Held-out ({HELD_OUT_SLIDE}) bin cluster top-1 retrieval:        {metrics['held_out_cell_cluster_top1_retrieval']:.4f} "
          f"(exact-bin top-1 {metrics['held_out_cell_top1_retrieval']:.4f})")
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
    print(f"Bin embeddings (trans/token, train + held-out) saved to "
          f"{checkpoint_dir / f'loso_token_{HELD_OUT_SLIDE}{RUN_TAG}_embeddings.npz'}")

    try:
        notification.notify(
            title='Jupyter Notebook Alert',
            message=f'LOSO-token (bins) run for {HELD_OUT_SLIDE} finished!',
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
                         help="Slide to hold out for this fold (default: CRC_P1).")
    # ---- ablation knobs -- mirror run_loso.py's own, same defaults/semantics ----
    parser.add_argument("--edge-dropout-p", type=float, default=0.0)
    parser.add_argument("--gene-dropout-p", type=float, default=0.0)
    parser.add_argument("--local-neighbor-radius", type=_float_or_none, default=100,
                         help="'none' disables spatial exclusion. Default 100 is the "
                              "cell-level value carried over UNCHANGED for bins -- see "
                              "this module's docstring caveat.")
    parser.add_argument("--local-anchor-sim-threshold", type=_float_or_none, default=None)
    parser.add_argument("--local-anchor-sim-source", choices=["projected", "raw"], default="projected",
                         help="What --local-anchor-sim-threshold checks similarity on -- "
                              "see run_loso.py's own flag. Never tested for bins; nuclei's "
                              "0.1-0.3 productive range is NOT assumed to transfer.")
    parser.add_argument("--local-positive-mode", choices=["exclude", "attract"], default="exclude",
                         help="'exclude' (default) or 'attract' (multi-positive InfoNCE) -- "
                              "see run_loso.py's own flag. Never tested for bins.")
    parser.add_argument("--skip-token-pretrain", action="store_true")
    parser.add_argument("--freeze-token", action="store_true")
    parser.add_argument("--no-weight-reconstruction", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                         help="Phase 3 AdamW weight decay (default: 0.0). 1e-5 is the "
                              "nuclei-adopted value -- see run_loso.py's own flag.")
    parser.add_argument("--temperature", type=float, default=0.05,
                         help="Contrastive loss softmax temperature (default: 0.05, "
                              "the nuclei-adopted value -- see run_loso.py's own flag "
                              "and ablation_studies_summary.txt; never independently "
                              "re-tested for bins).")
    parser.add_argument("--gene-panel", default=None,
                         help="Bare name of an external gene panel under "
                              "data/combined/01_preprocessing/gene_panels/ -- e.g. "
                              "'colon_xenium'. Overrides --gene-selection entirely. "
                              "See run_loso.py's own flag.")
    parser.add_argument("--gene-selection", choices=["svg", "hvg", "intersection"], default="svg",
                         help="Internal gene selection (default: svg). Ignored entirely "
                              "if --gene-panel is given.")
    args = parser.parse_args()
    run_it(
        held_out_slide=args.held_out_slide,
        edge_dropout_p=args.edge_dropout_p,
        gene_dropout_p=args.gene_dropout_p,
        local_neighbor_radius=args.local_neighbor_radius,
        local_anchor_sim_threshold=args.local_anchor_sim_threshold,
        local_anchor_sim_source=args.local_anchor_sim_source,
        local_positive_mode=args.local_positive_mode,
        skip_token_pretrain=args.skip_token_pretrain,
        freeze_token=args.freeze_token,
        weight_reconstruction=not args.no_weight_reconstruction,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        gene_panel=args.gene_panel,
        gene_selection=args.gene_selection,
    )
