# import
import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from plyer import notification
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv


from autoencoder_functions import *


CONFIG = dict(
    sample="combined",     # str -> single slide. list[str] or "combined" -> multi-slide.
    bin_size="002um",
    dir_cellseg="cell_seg_stardist",  # in case of multiple cell segmentation methods
    loss_fct="ntxent",                # optional: ntxent, cosine
 
    # model
    latent_dim=64,
    proj_dim=32,
    pool_type="attention",            # "attention" or "mean"
    encoder="transconv",                   # sage or transconv for transcriptomics encoding
 
    # graph / batching
    n_neighs=6,                       
    num_neighbors=(10, 10),           
    patches_per_batch=16,            
    max_cells_per_patch=64,           
 
    # loss
    temperature=0.07,
    neighbor_radius=None,            
                                       
    weight_reconstruction_by_patch_freq=True,   
 
    # training
    n_epochs=30,
    lr=1e-3,
    warmup_epochs=5,
    max_contrast_weight=1.0,
)
 
xpca_hist = False
xpca_trans = False
xpca_col_hist = "X_pca"
xpca_col_trans = "X_pca"


basedir = Path("../../data")
dir_histology = basedir / "combined/01_preprocessing/unimodel/metadata"
dir_transcriptomics = basedir / "combined/01_preprocessing" / CONFIG["dir_cellseg"] / CONFIG["bin_size"] / "custom_binning"
_run_tag = CONFIG["sample"] if isinstance(CONFIG["sample"], str) else "combined"
dir_output = basedir / _run_tag / "03_integration/autoencoders" / CONFIG["bin_size"]
dir_output.mkdir(parents=True, exist_ok=True)

import anndata as ad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- load --
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

# 4. standardize both modalities (NEW)
print("4. standardising")
X_svg, hist_emb, svg_scaler, hist_scaler = standardize_modalities(X_svg, hist_emb)

# 5. patch metadata: centroid, cell count, inverse-frequency weight (NEW)
print("5. centroids")
cell_xy = extract_cell_xy(trans_sub, f_trans_coord)
patch_meta, patch_meta_df = compute_patch_metadata(
    patch_row_trans, cell_xy, f_hist_adata,
    neighbor_radius=CONFIG["neighbor_radius"],
)
patch_meta_df.to_csv(dir_output / "patch_metadata.csv", index=False)

# 6. PyG data + patch-grouped loader (NEW loader; replaces plain NeighborLoader)
print("6. pyg")
data = build_pyg_data(X_svg, conn_sub, patch_row_trans, slide_codes)
patch_to_nodes = build_patch_to_nodes(data.patch_row)
loader = PatchGroupedLoader(
    data, patch_to_nodes,
    num_neighbors=CONFIG["num_neighbors"],
    patches_per_batch=CONFIG["patches_per_batch"],
    max_cells_per_patch=CONFIG["max_cells_per_patch"],
)

# 7. model + training
print("7. model training")
model = DualAutoencoder(
    n_svg=X_svg.shape[1], hist_dim=hist_emb.shape[1],
    latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
    n_slides=n_slides, pool_type=CONFIG["pool_type"],
    encoder_type=CONFIG["encoder"],
)
hist_emb_t = torch.tensor(hist_emb, dtype=torch.float32)

train(
    model, loader, hist_emb_t, patch_meta,
    n_epochs=CONFIG["n_epochs"], lr=CONFIG["lr"], warmup_epochs=CONFIG["warmup_epochs"],
    max_contrast_weight=CONFIG["max_contrast_weight"], temperature=CONFIG["temperature"],
    weight_reconstruction=CONFIG["weight_reconstruction_by_patch_freq"],
    device=device,
)

# 8. eval
print("8. eval")
patch_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in patch_meta.items()}
eval_results = retrieval_accuracy(model, data, hist_emb_t, patch_meta_device, device=device)

# 9. save model + scalers + config
print("9. saving...")
torch.save({
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
}, dir_output / "dual_ae_checkpoint.pt")

# 10. save latents (cell-level AND patch-level) + metadata
with torch.no_grad():
    z_trans_all = encode_full_graph_batched(
        model.trans_enc, data, device,
        num_neighbors=CONFIG["num_neighbors"], batch_size=4096,
    ).cpu().numpy()
    z_hist_all = model.hist_enc(hist_emb_t.to(device)).cpu().numpy()
z_patch_trans_all = eval_results["z_patch_trans_full"]  # one row per unique patch, pooled

np.savez(
    dir_output / "dual_ae_latents.npz",
    z_trans_cell=z_trans_all,                              # per-cell trans latent
    z_hist_per_patch=z_hist_all,                            # per-patch hist latent
    z_patch_trans=z_patch_trans_all,                        # per-patch pooled trans latent (NEW)
    patch_row=data.patch_row.numpy(),                       # cell -> patch index mapping
    slide_code=data.slide_code.numpy(),
)

# save anndata: per-cell trans latent, per-cell (via lookup) pooled patch latents of both modalities
trans_sub.obsm["X_dae_trans_cell"] = z_trans_all
trans_sub.obsm["X_dae_trans_patch"] = z_patch_trans_all[data.patch_row.numpy()]
trans_sub.obsm["X_dae_hist_patch"] = z_hist_all[data.patch_row.numpy()]
trans_sub.obsm["X_dae_joint_patch"] = np.concatenate(
    [trans_sub.obsm["X_dae_trans_patch"], trans_sub.obsm["X_dae_hist_patch"]], axis=1
)

trans_sub.write_h5ad(dir_output / "trans_with_dae_latents.h5ad")

print("Done. Outputs written to:", dir_output)

# Trigger a system notification
notification.notify(
    title='Jupyter Notebook Alert',
    message='Your script has finished running!',
    app_name='Jupyter',
    timeout=5 # Notification disappears after 5 seconds
)