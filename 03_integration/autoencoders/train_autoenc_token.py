"""
Cluster entry point for the token-augmented dual autoencoder (autoenc_token.ipynb),
adapted to run headless via `sbatch`/`python train_autoenc_token.py` instead of a
Jupyter kernel. Same three-branch model (cell transcriptomics + patch histology +
cell-level token histology) and same phases (pretrain hist -> pretrain token ->
freeze both -> joint contrastive training -> eval -> save), just without notebook
magics or inline plotting.

Before running: per-slide token data must exist at
data/{slide}/01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad,
produced by `token_preprocessing.py --slide <slide>` (run separately, once per
slide -- not part of this script). That script itself reads the smaller,
already-QC'd raw token file, WSI_patch_token_embeddings_filtered.h5ad, not the
unfiltered WSI_patch_token_embeddings.h5ad -- confirm both exist under
data/{slide}/01_preprocessing/unimodel/metadata/ on the cluster before submitting.

To change which slides/config this run uses, edit CONFIG below before submitting
(same pattern as run_token_sweep.py in this directory) -- e.g. duplicate
this file or override CONFIG["sample"]/CONFIG["use_tokens"] per sbatch job rather
than adding CLI flags, so each run's exact settings stay visible in the script
itself alongside its output.
"""

# ---- imports ----------------------------------------------------------
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

from autoencoder_functions_new import *

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

    # model
    latent_dim=64,                      # latent dim for encoders
    proj_dim=32,                        # dimension of projection head for contrastive learning
    pool_type="attention",              # "attention" or "mean" -> pooling type for pooling cells to patch level
    encoder="transconv",                # sage or transconv for transcriptomics encoding

    # graph / batching
    n_neighs=6,                         # how many neighbors in knn spatial graph of cells
    num_neighbors=(10, 10),
    patches_per_batch=64,
    max_cells_per_patch=64,

    # loss
    temperature=0.07,
    neighbor_radius=2912,     # 2x patch diameter, currently 224x224 px

    # false-negative protection: exclude transcriptomically near-duplicate,
    # non-matching patches/cells from the hard-negative set (cosine-similarity
    # cutoff on the trans-side projection), same rationale as neighbor_radius
    # but gated on expression similarity instead of spatial distance -- see
    # `patch_level_infonce`'s docstring. Set to None to disable.
    patch_anchor_sim_threshold=0.9,
    local_anchor_sim_threshold=0.9,

    weight_reconstruction_by_patch_freq=True,

    hist_batch_correct=False,
    hist_harmony_npcs=50,

    hist_pretrain_epochs=200,
    hist_pretrain_lr=1e-3,
    hist_pretrain_weight_decay=1e-5,
    hist_pretrain_val_frac=0.1,
    hist_pretrain_patience=15,

    # training
    n_epochs=100,
    lr=1e-3,
    warmup_epochs=5,
    max_contrast_weight=0.0,

    # --- token branch (this script only) ---
    use_tokens=True,                    # master toggle -- False reproduces autoencoder.ipynb's
                                         # behavior exactly (DualAutoencoder gets token_dim=None).
    token_pretrain_epochs=100,
    token_pretrain_lr=1e-3,
    token_pretrain_weight_decay=1e-5,
    token_pretrain_val_frac=0.1,
    token_pretrain_patience=15,
    # token_lr_mult<1.0: token_enc/token_dec are already well-pretrained (Phase 1b,
    # pure reconstruction) -- once unfrozen below, they get a reduced LR relative
    # to trans_enc/trans_dec so the joint contrastive+reconstruction objective
    # fine-tunes that pretrained structure gently rather than overwriting it at
    # the same rate as trans_enc, which is being trained from scratch this phase.
    token_lr_mult=0.3,
    # False (was True): downstream goal is predicting trans embeddings from
    # histology alone (see conversation notes) -- that needs token_enc's actual
    # OUTPUT (X_dae_token_cell, saved post-training) to move toward trans_enc's
    # embedding space, not just the narrow proj_token/proj_trans_local heads used
    # inside the contrastive loss. With freeze_token=True, token_enc never
    # received gradient in Phase 3 at all regardless of this flag or
    # max_local_contrast_weight (train_autoenc_token.py's Phase 2 called
    # freeze_token_encoder() unconditionally) -- fixed below so this flag now
    # actually controls it. token_dec stays attached (not separately frozen) so
    # loss_recon_token keeps regularizing token_enc against collapsing to a
    # degenerate, purely-retrieval-optimized solution while it moves toward
    # trans_enc's space.
    freeze_token=False,

    local_neighbor_radius=100,          # px -- cell-level analogue of neighbor_radius (2912 for
                                         # patches). Median nearest-cell spacing measured ~27px in
                                         # this dataset. Was 50 (~1.85x spacing) -- too small: cells
                                         # a few widths apart (still very likely the same local
                                         # tissue/cell-type) were still treated as hard negatives,
                                         # producing a fine checkerboard-like alternation in the
                                         # resulting clusters (see conversation notes). 100 (~3.7x
                                         # spacing) is a first attempt at giving more room before
                                         # anything gets pushed apart -- worth iterating on.
    # REVERTED 1.0 -> 0.5, warmup 5 -> 20: the 1.0/5-epoch combination collapsed.
    # Actual run log: contrast_local crashed 7.29 -> 0.03 -> 0.0001 within the
    # first 2 epochs and sat at exactly 0.0000 for ~90 straight epochs; eval
    # showed top-1 retrieval = 0.000 on all 7 slides. Root cause: unfreezing
    # token_enc AND raising this weight to full strength within
    # local_warmup_epochs=5 both at once pushed token_enc's projections to drift
    # fast -- and local_anchor_sim_threshold's masking is circular (it checks
    # the model's OWN live projections), so as they drifted together even
    # slightly, more pairs crossed the 0.9 similarity threshold and got masked
    # OUT of the negative set, removing the very gradient that would have
    # corrected the drift -- a self-reinforcing collapse with no way back out
    # (confirmed directly: a synthetic test with deliberately collapsed
    # projections reproduces frac_masked=1.0, loss=0.0, exactly matching the
    # real log -- see patch_level_infonce's `return_stats`/`frac_masked`,
    # logged per-epoch now as `local_frac_masked` specifically to catch this
    # within a couple epochs instead of after a full wasted run). This run:
    # back to the old 0.5 ceiling (isolates whether unfreezing token_enc ALONE
    # is stable before also pushing the weight higher) and a much slower
    # 20-epoch ramp (gives token_enc time to adapt gradually rather than being
    # hit at full strength almost immediately).
    max_local_contrast_weight=0.5,
    local_warmup_epochs=20,

    # skip Phase 1a (histology-patch pretraining) entirely -- with
    # max_contrast_weight=0.0 above, the patch-level trans<->hist alignment
    # never trains anyway, so hist_enc/hist_dec/pool/proj_trans/proj_hist never
    # receive any gradient in Phase 3 either (their only path is
    # contrast_weight * loss_contrast, which is always zero) -- X_dae_hist_patch
    # would be a real pretrained-but-unaligned embedding either way, but
    # X_dae_trans_patch (pool output) is already untrained noise regardless of
    # whether Phase 1a runs. Skipping it saves a real, dedicated
    # hist_pretrain_epochs-epoch training loop for output nothing here uses.
    # Flip back to False once patch-level alignment is actually being trained
    # (max_contrast_weight > 0).
    skip_hist_pretrain=True,
)

# ---- output dirs ----------------------------------------------------------
basedir = Path("../../data")
dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
dir_transcriptomics = basedir / "combined/01_preprocessing" / CONFIG["dir_cellseg"] / CONFIG["bin_size"] / "custom_binning"
_run_tag = CONFIG["sample"] if isinstance(CONFIG["sample"], str) else "combined"
_variant_tag = "with_tokens" if CONFIG["use_tokens"] else "no_tokens"
dir_output = basedir / _run_tag / "03_integration/autoencoders" / CONFIG["bin_size"] / _variant_tag
dir_output.mkdir(parents=True, exist_ok=True)
print("Writing outputs to:", dir_output)

# ---- load transcriptomics + patch histology (identical to autoencoder.ipynb) ----
f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
f_trans_coord = gpd.read_parquet(dir_transcriptomics / "nuclei_geometry.parquet")

if CONFIG["sample"] == "combined":
    pass
elif isinstance(CONFIG["sample"], str):
    f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"] == CONFIG["sample"]]
    f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"] == CONFIG["sample"]]
    f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"] == CONFIG["sample"]]
elif isinstance(CONFIG["sample"], (list, tuple)):
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

# ---- load token data (skipped entirely when use_tokens=False) -------------
# Loads each in-scope slide's cell_token_embeddings.h5ad (from token_preprocessing.py),
# concatenates them, and -- if some in-scope slides don't have token data yet --
# restricts training to the slides that do (rather than silently zero-filling,
# which would let the model learn "zero vector = slide X" as a shortcut).
if CONFIG["use_tokens"]:
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
        raise RuntimeError("use_tokens=True but no in-scope slide has token data -- run token_preprocessing.py first")

    f_token_adata = ad.concat(token_adatas, join="outer")
    token_key = f_token_adata.obs["Slide_ID"].astype(str) + "_" + f_token_adata.obs["id"].astype(str)
    if not token_key.is_unique:
        raise ValueError("Cannot set Slide_ID+id as token index: contains duplicates")
    f_token_adata.obs = f_token_adata.obs.set_index(token_key)
    f_token_adata = f_token_adata[f_token_adata.obs["matched"].astype(bool)].copy()
    print(f"Loaded token data: {f_token_adata.n_obs} cells across {len(slides_with_tokens)} slide(s)")
else:
    f_token_adata = None

# ---- optional histology batch correction ----------------------------------
if CONFIG["hist_batch_correct"]:
    f_hist_adata.obsm["X_harmony"] = harmonize_hist_embeddings(
        f_hist_adata, batch_key="Slide_ID", n_pcs=CONFIG["hist_harmony_npcs"]
    )

# ---- preprocessing: SVG mask, spatial graph, pairing, standardize, --------
# ---- patch/cell metadata, PyG data + patch-grouped loader ------------------
print("1. SVG mask")
svg_mask = get_svg_union_mask(f_trans_adata)
print(f"{svg_mask.sum()} SVGs selected (union)")

print("2. building spatial graph")
conn = build_spatial_graph(f_trans_adata, library_key="Slide_ID", n_neighs=CONFIG["n_neighs"], delaunay=False)

print("3. pairing")
keep_mask, patch_row_trans, hist_emb = build_pairing(f_trans_adata, f_hist_adata)

# 3b. token pairing: further restrict keep_mask to cells that also have token data
if CONFIG["use_tokens"]:
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

if CONFIG["use_tokens"]:
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
)
patch_meta_df.to_csv(dir_output / "patch_metadata.csv", index=False)

if CONFIG["use_tokens"]:
    cell_meta = build_cell_meta(cell_xy, slide_codes, neighbor_radius=CONFIG["local_neighbor_radius"])
else:
    cell_meta = None

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
token_emb_t = torch.tensor(token_emb, dtype=torch.float32) if CONFIG["use_tokens"] else None

# ---- model ------------------------------------------------------------
model = DualAutoencoder(
    n_svg=X_svg.shape[1], hist_dim=hist_emb.shape[1],
    latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
    n_slides=n_slides,
    pool_type=CONFIG["pool_type"], encoder_type=CONFIG["encoder"],
    token_dim=(token_emb.shape[1] if CONFIG["use_tokens"] else None),
)
model.to(device)

# ---- Phase 1a: pretrain histology (patch) branch -----------------------
if CONFIG["skip_hist_pretrain"]:
    print("\n=== Phase 1a: SKIPPED (skip_hist_pretrain=True) -- "
          "hist_enc/hist_dec left at random init, harmless since max_contrast_weight=0.0 ===")
else:
    print("\n=== Phase 1a: pretrain histology branch ===")
    pretrain_hist_autoencoder(
        model, hist_emb_t, patch_meta,
        epochs=CONFIG["hist_pretrain_epochs"], lr=CONFIG["hist_pretrain_lr"],
        weight_decay=CONFIG["hist_pretrain_weight_decay"],
        val_frac=CONFIG["hist_pretrain_val_frac"], patience=CONFIG["hist_pretrain_patience"],
        device=device,
    )

# ---- Phase 1b: pretrain token (local) branch (skipped when use_tokens=False) ----
if CONFIG["use_tokens"]:
    print("\n=== Phase 1b: pretrain token branch ===")
    pretrain_token_autoencoder(
        model, token_emb_t, cell_meta,
        epochs=CONFIG["token_pretrain_epochs"], lr=CONFIG["token_pretrain_lr"],
        weight_decay=CONFIG["token_pretrain_weight_decay"],
        val_frac=CONFIG["token_pretrain_val_frac"], patience=CONFIG["token_pretrain_patience"],
        device=device,
    )

# ---- Phase 2: freeze histology branches -------------------------------
# hist_enc always freezes here regardless of CONFIG (max_contrast_weight=0.0
# means it never trains in Phase 3 anyway -- see Phase 1a's skip message).
# token_enc freezing is now gated on CONFIG["freeze_token"] -- previously
# called unconditionally, which meant freeze_token/max_local_contrast_weight/
# token_lr_mult had no way to actually let token_enc adapt during Phase 3, no
# matter how they were set (see conversation notes: downstream goal is a
# trans-predictive token embedding, which needs token_enc's actual output to
# move, not just the projection heads used inside the contrastive loss).
freeze_hist_encoder(model, freeze_decoder=True)
if CONFIG["use_tokens"] and CONFIG["freeze_token"]:
    freeze_token_encoder(model, freeze_decoder=True)

# ---- Phase 3: joint training (patch-level + cell-level contrastive) ----
print("\n=== Phase 3: joint contrastive training ===")
train(
    model, loader, hist_emb_t, patch_meta,
    n_epochs=CONFIG["n_epochs"], lr=CONFIG["lr"], warmup_epochs=CONFIG["warmup_epochs"],
    max_contrast_weight=CONFIG["max_contrast_weight"],
    weight_reconstruction=CONFIG["weight_reconstruction_by_patch_freq"],
    freeze_hist=True,
    device=device,
    token_emb_all=token_emb_t, cell_meta=cell_meta,
    max_local_contrast_weight=(CONFIG["max_local_contrast_weight"] if CONFIG["use_tokens"] else 0.0),
    local_warmup_epochs=CONFIG["local_warmup_epochs"],
    freeze_token=CONFIG["freeze_token"],
    token_lr_mult=CONFIG["token_lr_mult"],
    patch_anchor_sim_threshold=CONFIG["patch_anchor_sim_threshold"],
    local_anchor_sim_threshold=(CONFIG["local_anchor_sim_threshold"] if CONFIG["use_tokens"] else None),
)

# ---- save checkpoint immediately -----------------------------------------
# Saved right after training, before eval, so a crash in retrieval_accuracy
# (e.g. an OOM building the full-cohort similarity matrix) can never lose a
# completed training run -- the eval below is diagnostic, not load-bearing.
checkpoint = {
    "model_state_dict": model.state_dict(),
    "epoch": CONFIG["n_epochs"] - 1,
    "n_svg": X_svg.shape[1],
    "hist_dim": hist_emb.shape[1],
    "latent_dim": CONFIG["latent_dim"],
    "proj_dim": CONFIG["proj_dim"],
    "n_slides": n_slides,
    "pool_type": CONFIG["pool_type"],
    "svg_scaler": svg_scaler,
    "hist_scaler": hist_scaler,
    "config": CONFIG,
}
if CONFIG["use_tokens"]:
    checkpoint["token_dim"] = token_emb.shape[1]
    checkpoint["token_scaler"] = token_scaler
torch.save(checkpoint, dir_output / "dual_ae_checkpoint.pt")

# ---- eval ---------------------------------------------------------------
print("\n=== Eval ===")
patch_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in patch_meta.items()}
eval_results = retrieval_accuracy(model, data, hist_emb_t, patch_meta_device, device=device)

if CONFIG["use_tokens"]:
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
with torch.no_grad():
    z_trans_all = encode_full_graph_batched(
        model.trans_enc, data, device,
        num_neighbors=CONFIG["num_neighbors"], batch_size=4096,
    ).cpu().numpy()
    z_hist_all = model.hist_enc(hist_emb_t.to(device)).cpu().numpy()
    if CONFIG["use_tokens"]:
        z_token_all = model.token_enc(token_emb_t.to(device)).cpu().numpy()
z_patch_trans_all = eval_results["z_patch_trans_full"]  # one row per unique patch, pooled

save_kwargs = dict(
    z_trans_cell=z_trans_all,
    z_hist_per_patch=z_hist_all,
    z_patch_trans=z_patch_trans_all,
    patch_row=data.patch_row.numpy(),
    slide_code=data.slide_code.numpy(),
)
if CONFIG["use_tokens"]:
    save_kwargs["z_token_cell"] = z_token_all
np.savez(dir_output / "dual_ae_latents.npz", **save_kwargs)

trans_sub.obsm["X_dae_trans_cell"] = z_trans_all
trans_sub.obsm["X_dae_trans_patch"] = z_patch_trans_all[data.patch_row.numpy()]
trans_sub.obsm["X_dae_hist_patch"] = z_hist_all[data.patch_row.numpy()]
trans_sub.obsm["X_dae_joint_patch"] = np.concatenate(
    [trans_sub.obsm["X_dae_trans_patch"], trans_sub.obsm["X_dae_hist_patch"]], axis=1
)
if CONFIG["use_tokens"]:
    trans_sub.obsm["X_dae_token_cell"] = z_token_all
    # local + regional histology, fused with transcriptomic identity, per cell
    trans_sub.obsm["X_dae_trans_token_hist_concat"] = np.concatenate(
        [trans_sub.obsm["X_dae_trans_cell"], trans_sub.obsm["X_dae_token_cell"], trans_sub.obsm["X_dae_hist_patch"]],
        axis=1,
    )

trans_sub.write_h5ad(dir_output / "trans_with_dae_latents.h5ad")
print("Done. Outputs written to:", dir_output)
print(f"Total wall-clock time: {(_time.time() - _t_start) / 60:.1f} min")
