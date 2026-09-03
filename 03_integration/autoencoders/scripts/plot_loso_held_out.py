"""
Spatial plot of a leave-one-slide-out run's held-out-slide embeddings.

run_leave_one_slide_out_token's saved *_embeddings.npz has held_out_z_trans/
held_out_z_token but NOT spatial coordinates or cell IDs, so it can't be plotted
on the tissue as-is. Rather than rerunning the full LOSO pipeline (expensive --
retrains Phase 1b + Phase 3 from scratch), this script cheaply reconstructs just
the held-out slide's cell ordering (data loading + build_pairing + token
restriction -- no spatial graph, no scaling, no model forward pass) so the
already-saved z_token/z_trans rows can be mapped back to (x, y) positions.

This ONLY works because build_pairing/extract_cell_xy are deterministic
(pure key-matching / lookups, no randomness) -- restricting to one slide before
or after these steps gives identical row ordering, verified by the shape assert
below. If that assert ever fires, the reconstruction has drifted from
run_leave_one_slide_out_token's actual filtering (e.g. a fixed bug re-run with a
different keep_mask) and the npz should be regenerated instead of trusted here.
"""

import sys

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

# Finds the ancestor directory that actually contains `functions/` -- the repo
# root when run from a checkout (this file sits three levels under it), or this
# script's own directory when run from a flat cluster deployment that copied
# the `functions/` package alongside it.
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from functions.dual_autoencoder import build_pairing, extract_cell_xy

HELD_OUT_SLIDE = "CRC_P1"
RUN_TAG = ""   # must match whatever run_loso.py used (see loso_token_log.csv's run_id)

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
dir_transcriptomics = basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning"
checkpoint_dir = basedir / "combined/03_integration/autoencoders/002um/loso_token_checkpoints"

# ---- reload just the held-out slide's cells, same order of operations as ----
# ---- build_pipeline_data (pairing -> token restriction), no other slides involved ----
print(f"Loading {HELD_OUT_SLIDE} only...")
f_hist_adata = sc.read_h5ad(dir_histology / "filtered_WSI_patch_embeddings.h5ad")
f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad")
f_trans_coord = gpd.read_parquet(dir_transcriptomics / "nuclei_geometry.parquet")

f_hist_adata = f_hist_adata[f_hist_adata.obs["Slide_ID"] == HELD_OUT_SLIDE].copy()
f_trans_adata = f_trans_adata[f_trans_adata.obs["Slide_ID"] == HELD_OUT_SLIDE].copy()
f_trans_coord = f_trans_coord[f_trans_coord["Slide_ID"] == HELD_OUT_SLIDE].copy()

trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)
hist_global_id = f_hist_adata.obs["Slide_ID"].astype(str) + "_" + f_hist_adata.obs["Patch_ID"].astype(str)
f_hist_adata.obs = f_hist_adata.obs.set_index(hist_global_id)

f_token_adata = sc.read_h5ad(basedir / HELD_OUT_SLIDE / "01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad")
token_key = f_token_adata.obs["Slide_ID"].astype(str) + "_" + f_token_adata.obs["id"].astype(str)
f_token_adata.obs = f_token_adata.obs.set_index(token_key)
f_token_adata = f_token_adata[f_token_adata.obs["matched"].astype(bool)].copy()

keep_mask, _, _ = build_pairing(f_trans_adata, f_hist_adata)
has_token = f_trans_adata.obs_names.isin(f_token_adata.obs_names)
keep_mask = keep_mask & has_token
trans_sub = f_trans_adata[keep_mask].copy()

cell_xy = extract_cell_xy(trans_sub, f_trans_coord)
cluster_labels = trans_sub.obs["clusters"].values
print(f"Reconstructed {trans_sub.n_obs} cells for {HELD_OUT_SLIDE}")

# ---- load the already-computed embeddings (no retraining needed) ----
npz_path = checkpoint_dir / f"loso_token_{HELD_OUT_SLIDE}{RUN_TAG}_embeddings.npz"
npz = np.load(npz_path)
z_token = npz["held_out_z_token"]
z_trans = npz["held_out_z_trans"]

assert z_token.shape[0] == trans_sub.n_obs, (
    f"row count mismatch: {z_token.shape[0]} embeddings vs {trans_sub.n_obs} reconstructed cells -- "
    "see this script's docstring before trusting the plot below"
)

# ---- cluster the held-out token embedding, compare spatially against the ----
# ---- independent trans-only clustering (ground truth reference, not derived ----
# ---- from this model at all) ----
n_clusters = len(np.unique(cluster_labels))
token_clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(z_token)
trans_clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(z_trans)
# plain concatenation, same convention as retrieval_accuracy's z_joint_patch = cat([z_trans, z_hist])
# -- no re-normalization, so whichever branch has larger-magnitude features can dominate the
# resulting joint clusters more than a naive "50/50 combined" reading would suggest.
z_joint = np.concatenate([z_trans, z_token], axis=1)
joint_clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(z_joint)

fig, axes = plt.subplots(1, 4, figsize=(26, 6))
for ax, labels, title in [
    (axes[0], pd.factorize(cluster_labels)[0], "independent trans-only clusters\n(ground truth reference)"),
    (axes[1], trans_clusters, f"KMeans(k={n_clusters}) on held-out z_trans"),
    (axes[2], token_clusters, f"KMeans(k={n_clusters}) on held-out z_token"),
    (axes[3], joint_clusters, f"KMeans(k={n_clusters}) on held-out z_joint\n(z_trans concat z_token)"),
]:
    sc_plot = ax.scatter(cell_xy[:, 0], cell_xy[:, 1], c=labels, cmap="tab10", s=1)
    ax.set_title(f"{HELD_OUT_SLIDE}: {title}")
    ax.invert_yaxis()
    ax.set_aspect("equal")

plt.tight_layout()
out_path = checkpoint_dir / f"loso_token_{HELD_OUT_SLIDE}{RUN_TAG}_spatial_clusters.png"
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
