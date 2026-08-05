"""
Runs the max_contrast_weight sweep from autoencoder.ipynb.

Contains only what the sweep cell needs:
  - imports
  - CONFIG
  - output dir setup
  - data loading + preprocessing (SVG mask, spatial graph, pairing,
    standardization, patch metadata, PyG data + loader)
  - the sweep itself (Phase-1 replicated pretrain + Phase-3 sweep over
    max_contrast_weight)

Deliberately OMITTED (not needed to execute the sweep):
  - the standalone full-pipeline single-run training/eval/save block
    (notebook cell 9, steps 7-10: single-config model training, retrieval
    eval, checkpoint save, latent export)
  - all plotting/analysis cells after the sweep

CHANGES vs the previous version of this script:
  - hist_condition_on_slide is no longer left at the library default. It's
    passed explicitly here (True) so this script's behavior doesn't silently
    change if that default is ever revisited again -- this flag was the
    single biggest lever found so far (Slide_ID leakage 0.90 -> ~0.50 with
    it on, same data/seed/regime otherwise; see conversation notes).
  - swept_values now covers 0.0 through 5.0, not just up to 1.0: the earlier
    sweep (against the corrected template) was still climbing at 1.0, not
    plateaued, so the top end wasn't actually found yet.
  - the pre-existing experiment_log.csv on disk (if any) mixes runs from
    BEFORE and AFTER the hist_condition_on_slide fix -- rows before the fix
    have hist_leakage_mean ~0.9, rows after have ~0.5. This script archives
    whatever's there once, then builds a clean log containing only rows
    below a leakage threshold, and appends this run's results to that clean
    log -- so results reviewed afterward can't accidentally mix pre-fix and
    post-fix numbers under similar-looking run_ids.
"""

# ---- imports ----------------------------------------------------------
import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import random
import scanpy as sc
import scipy.sparse as sp
import shutil
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

set_seed(42)

# ---- CONFIG -------------------------------------------------------------
CONFIG = dict(
    sample="combined",     # str -> single slide. list[str] or "combined" -> multi-slide.
    bin_size="002um",
    dir_cellseg="cell_seg_stardist",  # in case of multiple cell segmentation methods
    loss_fct="ntxent",                # optional: ntxent, cosine

    # model
    latent_dim=64,
    proj_dim=32,
    pool_type="attention",            # "attention" or "mean"
    encoder="transconv",              # sage or transconv for transcriptomics encoding
    hist_condition_on_slide=True,     # see module docstring -- the main lever found so far

    # graph / batching
    n_neighs=6,
    num_neighbors=(10, 10),
    patches_per_batch=64,
    max_cells_per_patch=64,

    # loss
    temperature=0.07,
    neighbor_radius=2912,     # 2x patch diameter, currently 224x224 px

    weight_reconstruction_by_patch_freq=True,

    hist_batch_correct=False,
    hist_harmony_npcs=50,

    hist_pretrain_epochs=100,
    hist_pretrain_lr=1e-3,
    hist_pretrain_weight_decay=1e-5,
    hist_pretrain_val_frac=0.1,
    hist_pretrain_patience=15,

    # training
    n_epochs=100,
    lr=1e-3,
    warmup_epochs=5,
    max_contrast_weight=0.0,
)

# ---- output dirs ----------------------------------------------------------
basedir = Path("../../data")
dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
dir_transcriptomics = basedir / "combined/01_preprocessing" / CONFIG["dir_cellseg"] / CONFIG["bin_size"] / "custom_binning"
_run_tag = CONFIG["sample"] if isinstance(CONFIG["sample"], str) else "combined"
dir_output = basedir / _run_tag / "03_integration/autoencoders" / CONFIG["bin_size"]
dir_output.mkdir(parents=True, exist_ok=True)

# ---- load data ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
f_trans_coord = gpd.read_parquet(dir_transcriptomics / "nuclei_geometry.parquet")

if CONFIG["sample"] == "combined":
    # keep every slide already in the loaded (combined) anndata objects -- no filtering
    pass
elif isinstance(CONFIG["sample"], str):
    # single-slide mode (original behavior)
    f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"] == CONFIG["sample"]]
    f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"] == CONFIG["sample"]]
    f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"] == CONFIG["sample"]]
elif isinstance(CONFIG["sample"], (list, tuple)):
    # multi-slide mode: keep only the named slides
    f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"].isin(CONFIG["sample"])]
    f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"].isin(CONFIG["sample"])]
    f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"].isin(CONFIG["sample"])]
print("Training on slides:", sorted(f_trans_adata.obs["Slide_ID"].unique().tolist()))

trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
if not trans_global_id.is_unique:
    raise ValueError("Cannot set Slide_ID+id as index: contains duplicates")
f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)

hist_global_id = f_hist_adata.obs["Slide_ID"].astype(str) + "_" + f_hist_adata.obs["Patch_ID"].astype(str)
if not hist_global_id.is_unique:
    raise ValueError("Cannot set Slide_ID+Patch_ID as index: contains duplicates")
f_hist_adata.obs = f_hist_adata.obs.set_index(hist_global_id)

# ---- optional histology batch correction ----------------------------------
if CONFIG["hist_batch_correct"]:
    f_hist_adata.obsm["X_harmony"] = harmonize_hist_embeddings(
        f_hist_adata, batch_key="Slide_ID", n_pcs=CONFIG["hist_harmony_npcs"]
    )

# ---- preprocessing: SVG mask, spatial graph, pairing, standardize, --------
# ---- patch metadata, PyG data + patch-grouped loader ----------------------
# 1. SVG mask
print("1. SVG mask")
svg_mask = get_svg_union_mask(f_trans_adata)
print(f"{svg_mask.sum()} SVGs selected (union)")

# 2. spatial graph
print("2. building spatial graph")
conn = build_spatial_graph(f_trans_adata, library_key="Slide_ID", n_neighs=CONFIG["n_neighs"], delaunay=False)

# 3. pairing
print("3. paring")
keep_mask, patch_row_trans, hist_emb = build_pairing(f_trans_adata, f_hist_adata)

trans_sub = f_trans_adata[keep_mask].copy()
conn_sub = conn[keep_mask][:, keep_mask]
X_svg = trans_sub.X[:, svg_mask]
X_svg = X_svg.toarray() if sp.issparse(X_svg) else np.asarray(X_svg)

slide_codes, slide_uniques = pd.factorize(trans_sub.obs["Slide_ID"])
n_slides = len(slide_uniques)

# 4. standardize both modalities
print("4. standardising")
X_svg, hist_emb, svg_scaler, hist_scaler = standardize_modalities(X_svg, hist_emb)

# 5. patch metadata: centroid, cell count, inverse-frequency weight
print("5. centroids")
cell_xy = extract_cell_xy(trans_sub, f_trans_coord)
patch_meta, patch_meta_df = compute_patch_metadata(
    patch_row_trans, cell_xy, f_hist_adata,
    neighbor_radius=CONFIG["neighbor_radius"],
)
patch_meta_df.to_csv(dir_output / "patch_metadata.csv", index=False)

# 6. PyG data + patch-grouped loader
print("6. pyg")
data = build_pyg_data(X_svg, conn_sub, patch_row_trans, slide_codes)
patch_to_nodes = build_patch_to_nodes(data.patch_row)
loader = PatchGroupedLoader(
    data, patch_to_nodes, patch_meta["slide_id"],
    num_neighbors=CONFIG["num_neighbors"],
    patches_per_batch=CONFIG["patches_per_batch"],
    max_cells_per_patch=CONFIG["max_cells_per_patch"],
)

hist_emb_t = torch.tensor(hist_emb, dtype=torch.float32)

# ---- log hygiene: archive any pre-existing mixed-history log --------------
# experiment_log.csv on disk (if any) may contain runs from BEFORE the
# hist_condition_on_slide fix (hist_leakage_mean ~0.9) mixed with runs from
# AFTER it (~0.5). Archive once, then build/extend a clean log that only
# ever contains post-fix rows.
raw_log_path = dir_output / "experiment_log.csv"
archived_log_path = dir_output / "experiment_log_pre_conditioning_fix.csv"
if raw_log_path.exists() and not archived_log_path.exists():
    shutil.copy(raw_log_path, archived_log_path)
    print(f"Archived pre-existing log to {archived_log_path}")

LEAKAGE_VALID_THRESHOLD = 0.6  # post-fix runs measured ~0.5; pre-fix ~0.9
log_path = dir_output / "experiment_log_clean.csv"
if not log_path.exists() and raw_log_path.exists():
    prior = pd.read_csv(raw_log_path)
    valid_prior = prior[prior["hist_leakage_mean"] < LEAKAGE_VALID_THRESHOLD].copy()
    valid_prior.to_csv(log_path, index=False)
    print(f"Seeded clean log with {len(valid_prior)} valid rows carried over from {raw_log_path}")

# ---- sweep: max_contrast_weight --------------------------------------------
# Get best contrastive learning

# --- sweep design: documented once, durably, separate from per-run results ---
SWEEP_DESIGN = {
    "swept_param": "max_contrast_weight",
    "swept_values": [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "rationale": (
        "max_contrast_weight was 0.0 in the current CONFIG -- cross-modal "
        "alignment (the actual point of this project) has not been trained yet. "
        "Finer resolution at the low end to catch the point, if any, where "
        "turning contrastive pressure back on starts degrading trans_enc/pooling "
        "the way it once degraded hist_enc (hist_enc itself is frozen and can't "
        "regress further, so this specifically tests the unfrozen half). "
        "Extended to 5.0: the 0.0-1.0 sweep (against the hist_condition_on_slide "
        "-corrected template) was still climbing at 1.0, not plateaued, so the "
        "top end of the curve hadn't been found yet."
    ),
    "held_fixed": {k: v for k, v in CONFIG.items() if k != "max_contrast_weight"},
}
(dir_output / "experiment_sweep_design.json").write_text(json.dumps(SWEEP_DESIGN, indent=2, default=str))
print("Sweep design saved to", dir_output / "experiment_sweep_design.json")

# --- Phase 1: pretrain the histology branch ONCE, shared across the whole sweep ---
model_kwargs = dict(
    n_svg=X_svg.shape[1], hist_dim=hist_emb.shape[1],
    latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
    n_slides=n_slides, pool_type=CONFIG["pool_type"], encoder_type=CONFIG["encoder"],
    hist_condition_on_slide=CONFIG["hist_condition_on_slide"],
)
pretrain_kwargs = dict(
    epochs=CONFIG["hist_pretrain_epochs"], lr=CONFIG["hist_pretrain_lr"],
    weight_decay=CONFIG["hist_pretrain_weight_decay"],
    val_frac=CONFIG["hist_pretrain_val_frac"], patience=CONFIG["hist_pretrain_patience"],
)

# Phase 1, replicated: 5 different random inits, keep the one with lowest Slide_ID
# leakage (not just lowest val MSE -- they're different objectives and don't
# necessarily track together)
template_state, replicate_log = pretrain_hist_replicated(
    model_kwargs, hist_emb_t, patch_meta, pretrain_kwargs,
    n_replicates=5, select_by="leakage", device=device,
)

replicate_df = pd.DataFrame(replicate_log)
replicate_df.to_csv(dir_output / "hist_pretrain_replicate_log.csv", index=False)
print(replicate_df)   # sanity-check variance BEFORE spending time on the sweep

# --- Phase 3 sweep ---
checkpoint_dir = dir_output / "sweep_checkpoints"

# Every run in THIS script execution shares one batch tag, so reruns of the same
# swept_values can never silently collide: checkpoint filenames (which are keyed
# by run_id) won't overwrite an earlier execution's, and duplicate rows in the
# log are unambiguous to resolve by timestamp/batch rather than guesswork. This
# is a real bug that already happened once (three separate executions all using
# the bare "contrastw_{cw}_repl_slide_conditioning_True" run_id silently
# overwrote each other's checkpoints and left stale, differently-seeded rows
# sitting in the log with no way to tell them apart).
import time as _time
BATCH_TAG = _time.strftime("%Y%m%d_%H%M%S")
print(f"Batch tag for this execution: {BATCH_TAG}")

for cw in SWEEP_DESIGN["swept_values"]:
    run_config = dict(CONFIG, max_contrast_weight=cw)
    run_id = f"contrastw_{cw}_repl_slide_conditioning_True_{BATCH_TAG}"
    metrics, _ = run_experiment(
        run_id=run_id, config=run_config, model_template_state=template_state,
        data=data, loader=loader, hist_emb_all=hist_emb_t, patch_meta=patch_meta,
        n_slides=n_slides, n_svg=X_svg.shape[1], hist_dim=hist_emb.shape[1],
        log_path=log_path, checkpoint_dir=checkpoint_dir, device=device,
    )

# --- review ---
results = pd.read_csv(log_path)
print(results[["run_id", "cfg__max_contrast_weight", "top1_retrieval",
                "neighbor_tolerant_retrieval", "hist_leakage_mean"]].sort_values("cfg__max_contrast_weight"))
