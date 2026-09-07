"""
Cluster entry point for the token-augmented dual autoencoder (autoenc_token.ipynb),
adapted to run headless via `sbatch`/`python train_autoenc_token.py`. Cell-level
only (trans<->token): pretrain token branch -> freeze -> joint contrastive
training -> eval -> save. The patch-level (hist<->trans) branch has been
removed from this project entirely.

Before running: per-slide token data must exist at
data/{slide}/01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad
(from `token_preprocessing.py --slide <slide>`, run separately).

To change slides/config, edit CONFIG below before submitting -- duplicate this
file or override per sbatch job rather than adding CLI flags, so each run's
settings stay visible alongside its output.
"""

# ---- imports ----------------------------------------------------------
import sys
import time as _time

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Finds the ancestor directory that actually contains `functions/` -- the repo
# root when run from a checkout (this file sits two levels under it), or this
# script's own directory when run from a flat cluster deployment that copied
# the `functions/` package alongside it.
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from functions.dual_autoencoder import *

_t_start = _time.time()
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# ---- CONFIG -------------------------------------------------------------
CONFIG = dict(
    sample="combined",       # str -> single slide. list[str] or "combined" -> multi-slide.
    bin_size="002um",
    dir_cellseg="cell_seg_stardist",  # in case of multiple cell segmentation methods
    loss_fct="ntxent",                # optional: ntxent, cosine
    token_filename="cell_token_embeddings.h5ad",  # "bin_token_embeddings.h5ad" for bins

    # model
    latent_dim=64,                      # latent dim for encoders
    proj_dim=32,                        # dimension of projection head for contrastive learning
    encoder="transconv",                # sage or transconv for transcriptomics encoding

    # graph / batching
    # n_neighs: 6 for the (unused-by-default) KNN fallback; 8 for spatial_graph_grid=True
    # (bins -- full Moore/queen neighborhood of a square lattice, see build_spatial_graph).
    # Irrelevant when spatial_graph_delaunay=True (nuclei default), which uses percentile instead.
    n_neighs=6,
    spatial_graph_delaunay=True,  # nuclei/cells: matches get_svg.py's own cell-path choice
    spatial_graph_grid=False,     # bins: set True (with n_neighs=8) for a regular 8um lattice
    num_neighbors=(10, 10),
    # Seed cells drawn uniformly at random per slide (SlideStratifiedLoader), not
    # grouped by shared histology-patch membership -- see run_loso.py's CONFIG
    # comment for why. ~1024 approximates the old loader's typical realized batch
    # size; re-tune against actual throughput/GPU memory now that it's a direct cap.
    cells_per_batch=1024,

    # loss
    temperature=0.07,
    neighbor_radius=2912,     # still required by compute_patch_metadata/build_pipeline_data's
                                # shared patch-batching path (2x patch diameter, 224x224px)

    # false-negative protection: excludes transcriptomically near-duplicate,
    # non-matching cells from hard negatives (see patch_level_infonce). None disables.
    # None (was 0.9): the 18-config sweep showed this masking doesn't clearly help
    # (None beat 0.9 in most pairs, e.g. weight=1.0/warmup=10: cell_top1 0.0663 vs
    # 0.0579) and it was the collapse mechanism (checks the model's own live
    # projections, so drifting projections get masked out of the negative set).
    local_anchor_sim_threshold=None,

    weight_reconstruction_by_patch_freq=True,

    # training
    n_epochs=100,
    lr=1e-3,
    warmup_epochs=5,

    # --- token branch ---
    token_pretrain_epochs=100,
    token_pretrain_lr=1e-3,
    token_pretrain_weight_decay=1e-5,
    token_pretrain_val_frac=0.1,
    token_pretrain_patience=15,
    # token_enc/token_dec are already well-pretrained (Phase 1) -- reduced LR so
    # the joint objective fine-tunes gently rather than overwriting that structure
    # at the same rate as trans_enc (training from scratch this phase).
    token_lr_mult=0.3,
    # False (was True): goal is predicting trans embeddings from histology alone,
    # which needs token_enc's actual output to move toward trans_enc's embedding
    # space, not just the narrow contrastive projection heads. freeze_token=True
    # previously blocked all Phase-2 gradient to token_enc regardless of this flag
    # (a bug, now fixed) -- token_dec stays attached so loss_recon_token keeps
    # regularizing against collapse while token_enc moves.
    freeze_token=False,

    # median nearest-cell spacing ~27px. Was 50 (~1.85x spacing) -- too small,
    # produced a checkerboard-like cluster artifact from near cells still being
    # treated as hard negatives. 100 (~3.7x) is a first attempt at more room.
    local_neighbor_radius=100,

    # History: 1.0/5-epoch-warmup collapsed (contrast_local -> 0 for ~90 epochs,
    # top-1=0 on all 7 slides) -- unfreezing token_enc while ramping to full
    # strength in 5 epochs let its projections drift together fast, and
    # local_anchor_sim_threshold's masking is circular (checks the model's own
    # live projections), so drifting pairs got masked out of the negative set,
    # a self-reinforcing loop. Reverted to 0.5/20, stable (cell_top1=0.088). The
    # 18-config sweep then found the weight itself wasn't the danger -- every
    # warmup>=10 config stayed stable, and higher weight + shorter warmup both
    # improved retrieval. Best of 18: weight=1.0/warmup=10 -> cell_top1=0.0663,
    # cluster_top1=0.5015 (vs. 0.0532/0.4923 at 0.5/20).
    max_local_contrast_weight=1.0,
    local_warmup_epochs=10,
)

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
_run_tag = CONFIG["sample"] if isinstance(CONFIG["sample"], str) else "combined"
dir_output = basedir / _run_tag / "03_integration/autoencoders" / CONFIG["bin_size"] / "with_tokens"
dir_output.mkdir(parents=True, exist_ok=True)
print("Writing outputs to:", dir_output)

# ---- load transcriptomics + patch histology (histology only for cell<->patch pairing) ----
f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
# Bins have no geometry side-car (obsm['spatial'] is already populated directly at
# load time) -- extract_cell_xy already prioritizes obsm['spatial'] over f_trans_coord,
# so None here is a safe, fully-supported fallback, not a degraded path.
_geom_path = dir_transcriptomics / "nuclei_geometry.parquet"
f_trans_coord = gpd.read_parquet(_geom_path) if _geom_path.exists() else None

if CONFIG["sample"] == "combined":
    pass
elif isinstance(CONFIG["sample"], str):
    f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"] == CONFIG["sample"]]
    f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"] == CONFIG["sample"]]
    if f_trans_coord is not None:
        f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"] == CONFIG["sample"]]
elif isinstance(CONFIG["sample"], (list, tuple)):
    f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"].isin(CONFIG["sample"])]
    f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"].isin(CONFIG["sample"])]
    if f_trans_coord is not None:
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

# ---- load token data --------------------------------------------------
# Loads each in-scope slide's token embeddings (CONFIG["token_filename"] -- cell or
# bin, from token_preprocessing.py / bin_token_preprocessing.py), concatenates them,
# and -- if some in-scope slides don't have token data yet -- restricts training to
# the slides that do (rather than silently zero-filling, which would let the model
# learn "zero vector = slide X" as a shortcut).
slides_needed = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique())
token_adatas, slides_with_tokens = [], []
for s in slides_needed:
    p = basedir / s / "01_preprocessing/unimodel/metadata" / CONFIG["token_filename"]
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
    if f_trans_coord is not None:
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

# ---- preprocessing: SVG mask, spatial graph, pairing, standardize, --------
# ---- patch/cell metadata, PyG data + slide-stratified random loader --------
print("1. SVG mask")
svg_mask = get_svg_union_mask(f_trans_adata)
print(f"{svg_mask.sum()} SVGs selected (union)")

print("2. building spatial graph")
conn = build_spatial_graph(f_trans_adata, library_key="Slide_ID", n_neighs=CONFIG["n_neighs"],
                            delaunay=CONFIG.get("spatial_graph_delaunay", True),
                            grid=CONFIG.get("spatial_graph_grid", False))

print("3. pairing")
keep_mask, patch_row_trans, hist_emb = build_pairing(f_trans_adata, f_hist_adata)

# 3b. token pairing: further restrict keep_mask to cells that also have token data
# Index.isin() already returns a plain ndarray, not a Series -- no .values needed
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
# footprint + halo, concatenated -- kept as two distinct signals rather than
# blended into one average (see token_preprocessing.py docstring).
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
    patch_row_trans, cell_xy, f_hist_adata,
    neighbor_radius=CONFIG["neighbor_radius"],
    slide_categories=slide_uniques,
)
patch_meta_df.to_csv(dir_output / "patch_metadata.csv", index=False)

cell_meta = build_cell_meta(cell_xy, slide_codes, neighbor_radius=CONFIG["local_neighbor_radius"])

print("6. pyg")
data = build_pyg_data(X_svg, conn_sub, patch_row_trans, slide_codes)
loader = SlideStratifiedLoader(
    data, data.slide_code,
    num_neighbors=CONFIG["num_neighbors"], cells_per_batch=CONFIG["cells_per_batch"],
)

token_emb_t = torch.tensor(token_emb, dtype=torch.float32)

# ---- model ------------------------------------------------------------
model = DualAutoencoder(
    n_svg=X_svg.shape[1],
    latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
    n_slides=n_slides,
    encoder_type=CONFIG["encoder"],
    trans_condition_on_slide=CONFIG.get("trans_condition_on_slide", True),
    token_dim=token_emb.shape[1],
)
model.to(device)

# ---- Phase 1: pretrain token branch -----------------------------------
print("\n=== Phase 1: pretrain token branch ===")
pretrain_token_autoencoder(
    model, token_emb_t, cell_meta,
    epochs=CONFIG["token_pretrain_epochs"], lr=CONFIG["token_pretrain_lr"],
    weight_decay=CONFIG["token_pretrain_weight_decay"],
    val_frac=CONFIG["token_pretrain_val_frac"], patience=CONFIG["token_pretrain_patience"],
    device=device,
)

# ---- Phase 2: freeze token branch (gated on CONFIG["freeze_token"]) ----
if CONFIG["freeze_token"]:
    freeze_token_encoder(model, freeze_decoder=True)

# ---- Phase 3: joint training (cell-level contrastive) ------------------
print("\n=== Phase 3: joint contrastive training ===")
train(
    model, loader, patch_meta,
    n_epochs=CONFIG["n_epochs"], lr=CONFIG["lr"], warmup_epochs=CONFIG["warmup_epochs"],
    weight_reconstruction=CONFIG["weight_reconstruction_by_patch_freq"],
    device=device,
    token_emb_all=token_emb_t, cell_meta=cell_meta,
    max_local_contrast_weight=CONFIG["max_local_contrast_weight"],
    local_warmup_epochs=CONFIG["local_warmup_epochs"],
    freeze_token=CONFIG["freeze_token"],
    token_lr_mult=CONFIG["token_lr_mult"],
    local_anchor_sim_threshold=CONFIG["local_anchor_sim_threshold"],
)

# ---- save checkpoint immediately -----------------------------------------
# Saved right after training, before eval, so a crash in token_retrieval_accuracy
# (e.g. an OOM building the full-cohort similarity matrix) can never lose a
# completed training run -- the eval below is diagnostic, not load-bearing.
checkpoint = {
    "model_state_dict": model.state_dict(),
    "epoch": CONFIG["n_epochs"] - 1,
    "n_svg": X_svg.shape[1],
    "latent_dim": CONFIG["latent_dim"],
    "proj_dim": CONFIG["proj_dim"],
    "n_slides": n_slides,
    "token_dim": token_emb.shape[1],
    "svg_scaler": svg_scaler,
    "token_scaler": token_scaler,
    "config": CONFIG,
}
torch.save(checkpoint, dir_output / "dual_ae_checkpoint.pt")

# ---- eval ---------------------------------------------------------------
print("\n=== Eval ===")
cell_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cell_meta.items()}
# "clusters" is an independent transcriptomics-only clustering already present on
# the raw input adata (not derived from this model's own trans/token projections),
# so it's a valid independent label for cluster-level retrieval credit -- see
# token_retrieval_accuracy's cell_type_labels docstring for why that matters.
cluster_labels = pd.factorize(trans_sub.obs["clusters"])[0]
token_eval_results = token_retrieval_accuracy(
    model, data, token_emb_t, cell_meta_device, device=device,
    cell_type_labels=cluster_labels,
)

# ---- save latents ----------------------------------------------------
# Reuses token_retrieval_accuracy's own full-population encode instead of a
# second encode_full_graph_batched call.
z_trans_all = token_eval_results["z_trans_cell_full"]
z_token_all = token_eval_results["z_token_cell_full"]

np.savez(
    dir_output / "dual_ae_latents.npz",
    z_trans_cell=z_trans_all,
    z_token_cell=z_token_all,
    patch_row=data.patch_row.numpy(),
    slide_code=data.slide_code.numpy(),
)

trans_sub.obsm["X_dae_trans_cell"] = z_trans_all
trans_sub.obsm["X_dae_token_cell"] = z_token_all
trans_sub.obsm["X_dae_trans_token_concat"] = np.concatenate([z_trans_all, z_token_all], axis=1)

trans_sub.write_h5ad(dir_output / "trans_with_dae_latents.h5ad")
print("Done. Outputs written to:", dir_output)
print(f"Total wall-clock time: {(_time.time() - _t_start) / 60:.1f} min")
