"""
Cluster entry point for a Phase-3 hyperparameter sweep over the cell-level
(trans<->token) contrastive branch, headless via `sbatch`/`python
run_token_sweep.py` (same pattern as train_autoenc_token.py -- edit CONFIG/
GRID below and resubmit, rather than CLI flags, so each run's exact settings
stay visible in the script itself alongside its output).

Runs data prep + token-branch Phase-1 pretraining ONCE (shared across every
grid point, via `pretrain_token_once` -- see its docstring for why: Phase 1
doesn't depend on anything being swept, and re-pretraining per grid point
would let a different random Phase-1 outcome confound the comparison), then
calls `run_token_experiment` once per point in GRID below, appending one row
per run to a persistent CSV log at LOG_PATH.

Resumable: if LOG_PATH already has a row for a given run_id (from a prior
partial/interrupted sweep), that grid point is skipped rather than rerun --
matters given SLURM time limits/preemption on a sweep this size.

Motivated by train_autoenc_token.py's most recent run: the corrected
max_local_contrast_weight=0.5 / local_warmup_epochs=20 setting (reverted from
the 1.0 / 5-epoch combination that collapsed -- see that file's CONFIG
comment) trains stably, but was never actually compared against nearby
alternatives. This sweep is that comparison, plus a check on whether
local_anchor_sim_threshold (the false-negative-protection masking directly
implicated in the collapse mechanism -- it checks the model's own live
projections, which is what made the collapse self-reinforcing) is even net
helpful, by including None (disabled) as one of its swept values.

REDUCED_EPOCHS is intentionally lower than train_autoenc_token.py's full
n_epochs=100 -- the collapse in the original bad run was visible within ~2
epochs (contrast_local crashing, local_frac_masked climbing toward 1.0), so
this is enough to catch the same warning sign at every grid point without
paying full-run cost 18+ times over. Re-run the winning config at full
n_epochs=100 (i.e. just edit train_autoenc_token.py's CONFIG) once chosen.
"""
import itertools
import time as _time
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from autoencoder_functions_new import *

_t_start = _time.time()
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# ---- sweep grid -------------------------------------------------------------
# Extend/trim freely -- every combination below is one sequential Phase-3 run
# within this job, so keep len(combos) * REDUCED_EPOCHS within your --time
# budget (check wall-clock on the first few runs and adjust before a huge
# resubmit). local_neighbor_radius is NOT swept here since it feeds
# build_cell_meta (data prep, built once below) rather than train() -- add it
# by moving cell_meta construction inside the loop if you need it swept too.
GRID = {
    "max_local_contrast_weight": [0.25, 0.5, 1.0],
    "local_warmup_epochs": [10, 20, 40],
    "local_anchor_sim_threshold": [0.9, None],
}
REDUCED_EPOCHS = 30  # vs. 100 for the full run -- see module docstring

BIN_SIZE = "002um"
LOG_PATH = Path("../../data/combined/03_integration/autoencoders/002um/token_sweep_log.csv")
CHECKPOINT_DIR = Path("../../data/combined/03_integration/autoencoders/002um/token_sweep_checkpoints")

# ---- base config: everything NOT being swept, held at
# train_autoenc_token.py's currently-validated values -------------------------
BASE_CONFIG = dict(
    latent_dim=64, proj_dim=32, encoder="transconv",
    n_epochs=REDUCED_EPOCHS, lr=1e-3, warmup_epochs=5,
    temperature=0.07,
    weight_reconstruction_by_patch_freq=True,
    freeze_token=False, token_lr_mult=0.3,
    token_condition_on_slide=True,
)

# ---- load data (identical to train_autoenc_token.py's data-prep block) -----
basedir = Path("../../data")
dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
dir_transcriptomics = basedir / "combined/01_preprocessing/cell_seg_stardist" / BIN_SIZE / "custom_binning"

f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
f_trans_coord = gpd.read_parquet(dir_transcriptomics / "nuclei_geometry.parquet")
print("Training on slides:", sorted(f_trans_adata.obs["Slide_ID"].unique().tolist()))

trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
if not trans_global_id.is_unique:
    raise ValueError("Cannot set Slide_ID+id as index: contains duplicates")
f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)

hist_global_id = f_hist_adata.obs["Slide_ID"].astype(str) + "_" + f_hist_adata.obs["Patch_ID"].astype(str)
if not hist_global_id.is_unique:
    raise ValueError("Cannot set Slide_ID+Patch_ID as index: contains duplicates")
f_hist_adata.obs = f_hist_adata.obs.set_index(hist_global_id)

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

f_token_adata = ad.concat(token_adatas, join="outer")
token_key = f_token_adata.obs["Slide_ID"].astype(str) + "_" + f_token_adata.obs["id"].astype(str)
if not token_key.is_unique:
    raise ValueError("Cannot set Slide_ID+id as token index: contains duplicates")
f_token_adata.obs = f_token_adata.obs.set_index(token_key)
f_token_adata = f_token_adata[f_token_adata.obs["matched"].astype(bool)].copy()
print(f"Loaded token data: {f_token_adata.n_obs} cells across {len(slides_with_tokens)} slide(s)")

print("1. SVG mask")
svg_mask = get_svg_union_mask(f_trans_adata)
print(f"{svg_mask.sum()} SVGs selected (union)")

print("2. building spatial graph")
conn = build_spatial_graph(f_trans_adata, library_key="Slide_ID", n_neighs=6, delaunay=False)

print("3. pairing")
keep_mask, patch_row_trans, hist_emb = build_pairing(f_trans_adata, f_hist_adata)
has_token = f_trans_adata.obs_names.isin(f_token_adata.obs_names)
n_dropped = int(keep_mask.sum() - (keep_mask & has_token).sum())
if n_dropped:
    print(f"  dropping {n_dropped} cells with a hist-patch match but no token match")
keep_mask = keep_mask & has_token

trans_sub = f_trans_adata[keep_mask].copy()
conn_sub = conn[keep_mask][:, keep_mask]
X_svg = trans_sub.X[:, svg_mask]
X_svg = X_svg.toarray() if sp.issparse(X_svg) else np.asarray(X_svg)

slide_codes, slide_uniques = pd.factorize(trans_sub.obs["Slide_ID"])
n_slides = len(slide_uniques)

print("4. standardising")
X_svg, hist_emb, svg_scaler, hist_scaler = standardize_modalities(X_svg, hist_emb)

token_sub = f_token_adata[trans_sub.obs_names]
assert (token_sub.obs_names == trans_sub.obs_names).all(), "token/trans row alignment mismatch"
token_emb = np.concatenate(
    [np.asarray(token_sub.obsm["X_token_footprint"]), np.asarray(token_sub.obsm["X_token_halo"])],
    axis=1,
).astype(np.float32)
token_scaler = StandardScaler()
token_emb = token_scaler.fit_transform(token_emb).astype(np.float32)
print(f"  token_emb: {token_emb.shape}")

print("5. centroids")
cell_xy = extract_cell_xy(trans_sub, f_trans_coord)
patch_meta, patch_meta_df = compute_patch_metadata(
    patch_row_trans, cell_xy, f_hist_adata, neighbor_radius=2912,
)
cell_meta = build_cell_meta(cell_xy, slide_codes, neighbor_radius=100)

print("6. pyg")
data = build_pyg_data(X_svg, conn_sub, patch_row_trans, slide_codes)
patch_to_nodes = build_patch_to_nodes(data.patch_row)
loader = PatchGroupedLoader(
    data, patch_to_nodes, patch_meta["slide_id"],
    num_neighbors=(10, 10), patches_per_batch=64, max_cells_per_patch=64,
)

hist_emb_t = torch.tensor(hist_emb, dtype=torch.float32)
token_emb_t = torch.tensor(token_emb, dtype=torch.float32)

# "clusters" is an independent transcriptomics-only clustering already present
# on the raw input adata (not derived from this model's own trans/token
# projections) -- valid independent label for cluster-level retrieval credit,
# see token_retrieval_accuracy's cell_type_labels docstring.
cluster_labels = pd.factorize(trans_sub.obs["clusters"])[0]

# ---- Phase 1: pretrain token_enc/token_dec ONCE, shared across every grid point
print("\n=== pretraining shared token_enc/token_dec (once for the whole sweep) ===")
model_kwargs = dict(
    n_svg=X_svg.shape[1], token_dim=token_emb.shape[1],
    latent_dim=BASE_CONFIG["latent_dim"], proj_dim=BASE_CONFIG["proj_dim"],
    n_slides=n_slides, encoder_type=BASE_CONFIG["encoder"],
)
model_template_state = pretrain_token_once(
    model_kwargs, token_emb_t, cell_meta,
    pretrain_kwargs=dict(epochs=100, lr=1e-3, weight_decay=1e-5, val_frac=0.1, patience=15),
    device=device,
)

# ---- sweep loop --------------------------------------------------------------
keys = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
print(f"\n=== sweeping {len(combos)} configs ===")

existing_run_ids = set()
if LOG_PATH.exists():
    existing_run_ids = set(pd.read_csv(LOG_PATH)["run_id"].astype(str))

for i, combo in enumerate(combos):
    swept = dict(zip(keys, combo))
    config = {**BASE_CONFIG, **swept}
    run_id = "_".join(f"{k}={v}" for k, v in swept.items())

    if run_id in existing_run_ids:
        print(f"\n--- [{i + 1}/{len(combos)}] run_id={run_id} -- already in log, skipping ---")
        continue

    print(f"\n--- [{i + 1}/{len(combos)}] run_id={run_id} ---")
    run_token_experiment(
        run_id, config, model_template_state, data, loader, patch_meta,
        token_emb_t, cell_meta, n_slides, X_svg.shape[1], token_emb.shape[1],
        cluster_labels, log_path=LOG_PATH, checkpoint_dir=CHECKPOINT_DIR, device=device,
    )

print(f"\nDone. Sweep log: {LOG_PATH}")
print(f"Total wall-clock time: {(_time.time() - _t_start) / 60:.1f} min")
