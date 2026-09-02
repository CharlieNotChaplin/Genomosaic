# import
import anndata as ad
import geopandas as gpd
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
from torch_geometric.utils import dropout_edge


######### Model definitions
class SVGGraphEncoder_SAGE(nn.Module):
    """GraphSAGE encoder for SVG features + spatial neighborhood context. Produces
    one latent vector per CELL -- this is the object the reconstruction loss
    supervises, and the input to patch-level pooling below."""
 
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64, n_slides=5, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.out = nn.Linear(hidden_dim, latent_dim)
 
    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        z = self.out(h)
        return z


class SVGGraphEncoder_TransConv(nn.Module):
    """
    Uses multi-head attention (PyG `TransformerConv`) over spatial neighborhoods
    to learn dynamic, data-dependent cell-cell interaction weights, instead of
    SAGEConv's fixed mean/sum neighbor aggregation. Each head can specialize in a
    different kind of neighbor relationship (e.g. one head attending to cells of
    a similar type, another to boundary/transition cells).

    Note on dimensions: `TransformerConv(in, out, heads=H, concat=True)` returns
    H*out features (each head's output is concatenated), so `out_per_head` is set
    to hidden_dim // heads to keep the layer's total output width == hidden_dim.
    """

    def __init__(self, in_dim, hidden_dim=256, latent_dim=64, n_slides=5,
                 dropout=0.1, heads=4, edge_dim=None):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        out_per_head = hidden_dim // heads

        self.conv1 = TransformerConv(
            in_dim, out_per_head, heads=heads, concat=True,
            dropout=dropout, edge_dim=edge_dim,
        )
        self.conv2 = TransformerConv(
            hidden_dim, out_per_head, heads=heads, concat=True,
            dropout=dropout, edge_dim=edge_dim,
        )
        # LayerNorm (per-node, over the feature dim) rather than BatchNorm (per-feature,
        # over the batch/node-count dim) -- see conversation notes on why this matters
        # for this attention-based, graph-sampled setup.
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.norm2(self.conv2(h, edge_index, edge_attr)))
        return self.out(h)
 
 
class AttentionPool(nn.Module):
    """
    Gated attention pooling (Ilse et al., 2018 style) over cells within a patch.
    Learns which cells drive the patch's histology-aligned identity -- attention
    weights are interpretable per cell (e.g. can be cross-referenced against cell
    type annotations to see which populations anchor a patch's morphology).
    """
 
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.attn_V = nn.Linear(in_dim, hidden_dim)
        self.attn_U = nn.Linear(in_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
 
    def forward(self, z_cells, patch_index, num_patches):
        a = torch.tanh(self.attn_V(z_cells)) * torch.sigmoid(self.attn_U(z_cells))
        scores = self.attn_w(a).squeeze(-1)
        attn_weights = segment_softmax(scores, patch_index, num_patches)
        pooled = segment_sum(z_cells * attn_weights.unsqueeze(-1), patch_index, num_patches)
        return pooled, attn_weights
 
 
class MLPDecoder(nn.Module):
    """
    Optionally conditions on slide identity, one-hot-concatenated onto the latent
    before decoding. Conditioning alone only gives the decoder a "free" channel to
    explain away which slide a patch came from -- it removes the *requirement* for
    the encoder to keep that in its bottleneck, without directly penalizing it for
    keeping it anyway. The actual incentive to drop it comes from weight_decay on
    the encoder's optimizer (see `DualAutoencoder`'s `hist_condition_on_slide`
    docstring for the full mechanism and the measured effect): once conditioning
    makes slide identity unnecessary for reconstruction, L2 regularization pushes
    the optimizer toward not encoding it anyway. `condition_on_slide=True` (this
    class's default) is the validated setting for `hist_dec`, paired with
    `pretrain_hist_autoencoder`'s weight_decay -- Slide_ID leakage 0.495 vs 0.900
    unconditioned, same data/seed/regime.
    """
    def __init__(self, latent_dim, hidden_dims, out_dim, n_slides=5, condition_on_slide=True):
        super().__init__()
        self.condition_on_slide = condition_on_slide
        in_dim = latent_dim + (n_slides if condition_on_slide else 0)
        layers, d = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.n_slides = n_slides

    def forward(self, z, slide_code=None):
        if self.condition_on_slide:
            onehot = F.one_hot(slide_code, num_classes=self.n_slides).float()
            z = torch.cat([z, onehot], dim=-1)
        return self.net(z)
 
 
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
class ProjectionHead(nn.Module):
    """Separate small head for contrastive space, keeps z_recon space free."""
 
    def __init__(self, latent_dim, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )
 
    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)
 
 
class DualAutoencoder(nn.Module):
    def __init__(self, n_svg, hist_dim, latent_dim=64, proj_dim=32, n_slides=5,
                 pool_type="attention", encoder_type="sage", transconv_heads=4,
                 hist_condition_on_slide=True, token_dim=None, token_hidden=(512, 128),
                 token_condition_on_slide=True):
        """
        token_dim: dimensionality of the per-cell local histology token feature
            (footprint+halo ViT token embedding, see `token_preprocessing.py`).
            None (default) builds the model with no token branch, unaffecting
            existing checkpoints/pipelines. Passing a value adds a third branch
            (`token_enc`/`token_dec`/`proj_token`, mirroring `hist_enc`/`hist_dec`/
            `proj_hist`), plus `proj_trans_local` kept separate from `proj_trans`
            so the cell-level and patch-level contrastive objectives don't share
            one projection subspace.

            Token embeddings are per-CELL, not per-patch -- no pooling/broadcast,
            since footprint/halo aggregation already makes the correspondence
            close to one-to-one (median nucleus ~21px vs. 14px tokens).

        hist_condition_on_slide: whether `hist_dec` also gets the slide one-hot
            concatenated onto z. Default True (previously False, on the theory
            that conditioning only removes the *requirement* to encode slide
            identity without penalizing it -- incomplete reasoning, since
            `pretrain_hist_autoencoder`'s weight_decay actively pushes the
            encoder to drop now-unnecessary information, not just permits it).
            Measured, same data/seed: condition_on_slide=True -> Slide_ID
            leakage 0.495; False -> 0.900 (chance=0.20). Keep True.
        """
        super().__init__()
        if encoder_type == "sage":
            self.trans_enc = SVGGraphEncoder_SAGE(n_svg, 256, latent_dim, n_slides)
        elif encoder_type == "transconv":
            self.trans_enc = SVGGraphEncoder_TransConv(n_svg, 256, latent_dim, n_slides, heads=transconv_heads)
        else:
            raise ValueError(f"Unsupported ecnoder_type: '{encoder_type}'. Choose from 'sage', 'transconv'.")
        self.trans_dec = MLPDecoder(latent_dim, [256, 512], n_svg, n_slides)
        self.hist_enc = MLPEncoder(hist_dim, [512, 128], latent_dim)
        self.hist_dec = MLPDecoder(latent_dim, [128, 512], hist_dim, n_slides,
                                    condition_on_slide=hist_condition_on_slide)
        self.proj_trans = ProjectionHead(latent_dim, proj_dim)
        self.proj_hist = ProjectionHead(latent_dim, proj_dim)

        self.pool_type = pool_type
        self.pool = AttentionPool(latent_dim) if pool_type == "attention" else None

        self.token_dim = token_dim
        if token_dim is not None:
            self.token_enc = MLPEncoder(token_dim, list(token_hidden), latent_dim)
            self.token_dec = MLPDecoder(latent_dim, list(token_hidden)[::-1], token_dim, n_slides,
                                         condition_on_slide=token_condition_on_slide)
            self.proj_token = ProjectionHead(latent_dim, proj_dim)
            self.proj_trans_local = ProjectionHead(latent_dim, proj_dim)
        else:
            self.token_enc = None
            self.token_dec = None
            self.proj_token = None
            self.proj_trans_local = None


################### functions

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Enforce deterministic algorithm behavior in PyTorch/cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finish_note():
    from plyer import notification
    notification.notify(
        title='Jupyter Notebook Alert',
        message='Your script has finished running!',
        app_name='Jupyter',
        timeout=5 # Notification disappears after 5 seconds
    )

def get_svg_union_mask(trans_adata, per_sample_key="spatially_variable_per_sample"):
    """Union across samples: gene is an SVG if flagged in >=1 sample."""
    if per_sample_key in trans_adata.varm:
        sv = np.asarray(trans_adata.varm[per_sample_key]).astype(bool)
        return sv.any(axis=1)
    return (trans_adata.var["spatially_variable_n_samples"] >= 1).values


def get_hvg_union_mask(trans_adata, per_sample_key="highly_variable_per_sample"):
    """Union across samples: gene is an HVG if flagged in >=1 sample. Mirrors
    get_svg_union_mask exactly, reading the HVG-equivalent per-sample columns."""
    if per_sample_key in trans_adata.varm:
        hv = np.asarray(trans_adata.varm[per_sample_key]).astype(bool)
        return hv.any(axis=1)
    return (trans_adata.var["highly_variable_n_samples"] >= 1).values


def get_gene_mask(trans_adata, gene_selection="svg"):
    """
    Dispatches to the gene-selection mask named by `gene_selection`:
      "svg": spatially-variable-genes union (default, validated regime) -- genes
        selected for spatial patterning WITHIN a slide, which is the signal the
        cross-modal patch-level alignment task actually needs.
      "hvg": highly-variable-genes union -- selected on overall variance alone,
        no preference for spatial relevance. A raw-PCA leakage check found this
        set leaks Slide_ID somewhat less than SVG (see conversation notes), but
        that was never checked against retrieval quality -- treat as an
        experiment to run via `run_leave_one_slide_out`, not a validated default.
      "intersection": SVG & HVG -- a middle ground, keeping only genes that are
        both spatially patterned and highly variable.
    """
    if gene_selection == "svg":
        return get_svg_union_mask(trans_adata)
    elif gene_selection == "hvg":
        return get_hvg_union_mask(trans_adata)
    elif gene_selection == "intersection":
        return get_svg_union_mask(trans_adata) & get_hvg_union_mask(trans_adata)
    else:
        raise ValueError(f"Unsupported gene_selection: {gene_selection!r}. Choose from 'svg', 'hvg', 'intersection'.")


def build_spatial_graph(trans_adata, library_key="Slide_ID", n_neighs=6, delaunay=False, percentile=95.0):
    """Uses squidpy so that neighbors are only ever formed within a sample."""
    
    if delaunay:
        # Delaunay triangulation within each library slide
        sq.gr.spatial_neighbors_delaunay(
            trans_adata,
            spatial_key="spatial",
            library_key=library_key,
            percentile=percentile,  # Prunes long Delaunay edges beyond 95th percentile distance
        )
    else:
        # KNN graph within each library slide
        sq.gr.spatial_neighbors_knn(
            trans_adata,
            spatial_key="spatial",
            library_key=library_key,
            n_neighs=n_neighs,
        )

    # Extract connectivity matrix
    conn = trans_adata.obsp["spatial_connectivities"]
    
    # Verify no cross-sample edges were formed
    slide_ids = trans_adata.obs[library_key].values
    coo = conn.tocoo()
    cross = (slide_ids[coo.row] != slide_ids[coo.col]).sum()
    assert cross == 0, f"Found {cross} cross-sample edges - check library_key"
    
    return conn
 
 
def harmonize_hist_embeddings(hist_adata, batch_key="Slide_ID", n_pcs=50, random_state=42):
    """
    Runs Harmony (harmonypy) batch correction on the raw histology foundation-model
    embeddings, keyed on `batch_key` (typically Slide_ID) -- attacks the confound
    at its source (the frozen UNI embeddings already carry strong acquisition/
    center signal, see conversation notes / the Robustness Index literature)
    rather than only fighting it from the autoencoder/training side.

    PCA-reduces to `n_pcs` first: standard Harmony usage (it's a clustering-based
    correction, more effective and much cheaper in a moderate-dim space than
    running it directly on the full 1536-dim embedding), and downstream `hist_enc`
    was always going to compress further anyway.

    Meant to be run ONCE on the full, unique-patch `f_hist_adata` right after
    loading it:
        f_hist_adata.obsm["X_harmony"] = harmonize_hist_embeddings(f_hist_adata)
    then pass `hist_use_rep="X_harmony"` to `build_pairing` to use this corrected
    representation as the autoencoder's histology input instead of the raw
    embedding.

    Returns:
      X_harmony: (n_obs, n_pcs) float32 array, aligned to hist_adata.obs row order
    """
    import harmonypy
    from sklearn.decomposition import PCA

    X = hist_adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    pca = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = pca.fit_transform(X)

    meta = hist_adata.obs[[batch_key]].copy()
    ho = harmonypy.run_harmony(X_pca, meta, [batch_key], random_state=random_state)
    # harmonypy 2.0.0's Z_corr comes back in the same (N, n_pcs) orientation the
    # input was given in (verified empirically -- older harmonypy versions
    # returned (n_pcs, N) and needed a transpose here; this one doesn't).
    X_harmony = np.asarray(ho.Z_corr).astype(np.float32)
    assert X_harmony.shape == X_pca.shape, (
        f"unexpected harmonypy output orientation: got {X_harmony.shape}, expected {X_pca.shape}"
    )
    return X_harmony


def build_pairing(trans_adata, hist_adata, trans_patch_col="Patch_ID", hist_patch_col="Patch_ID",
                   slide_col="Slide_ID", hist_use_rep=None):
    """
    Returns:
      keep_mask: (n_trans,) bool array, nuclei with a matching histology patch
      patch_row_trans: (n_kept,) int array, which unique histology-patch row each
                        kept nucleus belongs to
      hist_emb: (n_unique_patches, hist_dim) float32 array, one row per unique patch

    hist_use_rep: if given (e.g. "X_harmony"), pulls hist_emb from
      `hist_adata.obsm[hist_use_rep]` instead of `hist_adata.X` -- use this to feed
      a batch-corrected representation (see `harmonize_hist_embeddings`) to the
      autoencoder instead of the raw foundation-model embedding. None preserves
      the original behavior.
    """
    trans_key = (
        trans_adata.obs[slide_col].astype(str) + "_" + trans_adata.obs[trans_patch_col].astype(str)
    )
    hist_key = (
        hist_adata.obs[slide_col].astype(str) + "_" + hist_adata.obs[hist_patch_col].astype(str)
    )
    assert hist_key.is_unique, "hist.obs has duplicate Slide_ID+Patch_ID rows"

    key_to_hist_row = {k: i for i, k in enumerate(hist_key)}

    missing = ~trans_key.isin(key_to_hist_row)
    if missing.any():
        print(f"WARNING: {missing.sum()} / {len(trans_key)} nuclei have no matching "
              f"hist patch - dropping them.")
    keep_mask = ~missing.values

    patch_key_trans = trans_key.values[keep_mask]
    patch_row_trans = np.array([key_to_hist_row[k] for k in patch_key_trans])

    if hist_use_rep is not None:
        hist_emb = np.asarray(hist_adata.obsm[hist_use_rep], dtype=np.float32)
    else:
        hist_X = hist_adata.X
        hist_emb = hist_X.toarray() if sp.issparse(hist_X) else np.asarray(hist_X)
        hist_emb = hist_emb.astype(np.float32)

    return keep_mask, patch_row_trans, hist_emb
 
 
def standardize_modalities(X_svg, hist_emb):
    """
    Z-score both modalities before training.
 
    As written, this fits ONE global scaler across whatever cells/patches you pass
    in. With a single slide that's unambiguous. With multiple slides passed in at
    once, this pools all slides into one mean/std -- fine if you want batch effects
    between slides to remain part of the signal, but if slides differ a lot in
    intensity/exposure/staining, a global scaler can let one slide's scale dominate.
    If that's a concern, standardize per-slide instead (compute mean/std within each
    Slide_ID group, subtract/divide per-group) before concatenating, or pass a
    `groups` array here and extend this function accordingly. If you later add a
    held-out validation slide, fit the scaler on training slides only and
    `.transform()` (never `.fit_transform()`) the held-out one.
    """
    svg_scaler = StandardScaler()
    X_svg_std = svg_scaler.fit_transform(X_svg).astype(np.float32)
 
    hist_scaler = StandardScaler()
    hist_emb_std = hist_scaler.fit_transform(hist_emb).astype(np.float32)
 
    return X_svg_std, hist_emb_std, svg_scaler, hist_scaler
 

def extract_cell_xy(trans_sub, f_trans_coord=None):
    """
    Extracts (N, 2) array of (x, y) cell centroids aligned with `trans_sub.obs_names`.
    
    Priority:
    1. trans_sub.obsm['spatial']
    2. f_trans_coord['centroid'] column or geometry
    """
    # Try extracting directly from AnnData .obsm['spatial']
    if "spatial" in trans_sub.obsm:
        xy = np.asarray(trans_sub.obsm["spatial"], dtype=np.float32)
        if xy.shape[0] == trans_sub.n_obs and xy.shape[1] == 2:
            return xy

    # Fall back to f_trans_coord if provided
    if f_trans_coord is not None:
        coords = f_trans_coord.copy()
        
        # Ensure proper index alignment
        if "id" in coords.columns:
            coords = coords.set_index(coords["id"].astype(str))
        
        # Subset & align order to match trans_sub
        coords = coords.loc[trans_sub.obs_names]
        
        # Check if 'centroid' column exists
        if "centroid" in coords.columns:
            # If centroid column contains Shapely Points
            if hasattr(coords["centroid"], "x"):
                x_vals = coords["centroid"].x.values
                y_vals = coords["centroid"].y.values
            else:
                # If centroid is a column of tuple/list/array objects or Shapely objects
                centroids = coords["centroid"].values
                x_vals = [p.x if hasattr(p, "x") else p[0] for p in centroids]
                y_vals = [p.y if hasattr(p, "y") else p[1] for p in centroids]
            return np.column_stack([x_vals, y_vals]).astype(np.float32)
            
        # Fall back to computing centroid from polygon geometry column
        elif hasattr(coords, "geometry") and coords.geometry is not None:
            centroids = coords.geometry.centroid
            return np.column_stack([centroids.x.values, centroids.y.values]).astype(np.float32)

    # Raise error if no spatial information could be resolved
    raise ValueError(
        "Could not extract spatial coordinates! "
        "Neither 'spatial' key found in `trans_sub.obsm` nor valid centroids in `f_trans_coord`."
    )
 
 
def compute_patch_metadata(patch_row_trans, cell_xy, hist_adata, key_to_hist_row=None,
                            trans_sub=None, slide_col="Slide_ID", patch_col="Patch_ID",
                            neighbor_radius=None, slide_categories=None):
    """
    Builds one row of metadata per unique histology patch, aligned by integer
    `patch_row` id (the same id used everywhere else in this script, i.e. row index
    into `hist_emb`). Includes:
      - centroid: mean (x, y) position of nuclei assigned to the patch
      - n_cells: number of paired nuclei
      - weight: inverse-frequency weight (mean-normalized to ~1), for use in both
        the patch-level contrastive loss and (optionally) the cell reconstruction loss
      - slide_id: integer code per patch. IMPORTANT for multi-slide runs: raw
        spatial coordinates are per-slide, and different slides commonly have
        overlapping coordinate ranges (e.g. both indexed from ~(0,0)). A patch on
        slide A and an unrelated patch on slide B can end up with a small
        `cdist` distance purely by coincidence. `slide_id` lets the neighbor-radius
        masking in `patch_level_infonce` require "same slide AND close distance"
        instead of just "close distance", so cross-slide pairs are never
        incorrectly excluded from the negative set.
    Also returns a plain pandas DataFrame version for saving/inspection.

    slide_categories: canonical, ordered list of slide names to code against (pass the
      same `slide_uniques` used for the cell/graph side). Previously this ran its own
      independent `pd.factorize` here, which could assign different codes than the
      cell-side factorize (different objects, different first-appearance order),
      occasionally producing a patch-level slide_id >= n_slides and crashing hist_dec's
      one-hot with a device-side CUDA assert. Coding against the shared list fixes this.
    """
    n_hist_patches = hist_adata.shape[0]
 
    df = pd.DataFrame({
        "patch_row": patch_row_trans,
        "x": cell_xy[:, 0],
        "y": cell_xy[:, 1],
    })
    stats = df.groupby("patch_row").agg(x=("x", "mean"), y=("y", "mean"), n_cells=("x", "size"))
 
    centroid = np.zeros((n_hist_patches, 2), dtype=np.float32)
    n_cells = np.zeros(n_hist_patches, dtype=np.float32)
    centroid[stats.index.values] = stats[["x", "y"]].values
    n_cells[stats.index.values] = stats["n_cells"].values
 
    freq = n_cells / max(n_cells.sum(), 1.0)
    weight = np.zeros(n_hist_patches, dtype=np.float32)
    nonzero = n_cells > 0
    weight[nonzero] = 1.0 / freq[nonzero]
    weight[nonzero] = weight[nonzero] / weight[nonzero].mean()  # normalize so mean weight ~= 1
 
    if slide_col in hist_adata.obs:
        if slide_categories is not None:
            slide_id = pd.Categorical(
                hist_adata.obs[slide_col].astype(str), categories=list(slide_categories)
            ).codes.astype(np.int64)
            assert (slide_id >= 0).all(), (
                "hist_adata has a Slide_ID not present in slide_categories -- "
                "the two objects' slide sets have diverged."
            )
        else:
            slide_id, _slide_uniques = pd.factorize(hist_adata.obs[slide_col].values)
    else:
        slide_id = np.zeros(n_hist_patches, dtype=np.int64)  # single-slide fallback

    meta_tensors = {
        "centroid": torch.tensor(centroid, dtype=torch.float32),
        "n_cells": torch.tensor(n_cells, dtype=torch.float32),
        "weight": torch.tensor(weight, dtype=torch.float32),
        "slide_id": torch.tensor(slide_id, dtype=torch.long),
        "neighbor_radius": neighbor_radius,
    }

    meta_df = pd.DataFrame({
        "patch_row": np.arange(n_hist_patches),
        "Patch_ID": hist_adata.obs[patch_col].values if patch_col in hist_adata.obs else None,
        "Slide_ID": hist_adata.obs[slide_col].values if slide_col in hist_adata.obs else None,
        "centroid_x": centroid[:, 0],
        "centroid_y": centroid[:, 1],
        "n_cells": n_cells,
        "weight": weight,
    })

    return meta_tensors, meta_df


def build_cell_meta(cell_xy, slide_codes, neighbor_radius=None):
    """
    CELL-level analogue of `compute_patch_metadata`, for the token branch's
    cell<->token contrastive term. Deliberately minimal compared to the patch
    version -- there's no pooling/weighting/dedup step to do at the cell level,
    since (unlike patches) every row here already IS the unit the loss operates
    on, one-to-one, with no broadcast.

    `cell_xy`/`slide_codes` must be row-aligned with `data.x` (i.e. the same
    order as `X_svg`/`trans_sub`), since `train_step` indexes into these tensors
    using `batch.n_id` (global node ids) directly.

    neighbor_radius: same role as `patch_meta["neighbor_radius"]`, but should be
    MUCH smaller here -- patches are 224px, cells are tens of px apart (median
    nearest-cell spacing ~27px in this dataset). A radius on that order (e.g.
    40-50px) excludes only genuinely adjacent cells from the negative set,
    rather than the whole local neighborhood.
    """
    return {
        "xy": torch.tensor(np.asarray(cell_xy), dtype=torch.float32),
        "slide_id": torch.tensor(np.asarray(slide_codes), dtype=torch.long),
        "neighbor_radius": neighbor_radius,
    }


def build_pyg_data(X_svg, conn, patch_row_trans, slide_codes):
    edge_index = torch.tensor(np.vstack(conn.tocoo().nonzero()), dtype=torch.long)
    data = Data(
        x=torch.tensor(X_svg, dtype=torch.float32),
        edge_index=edge_index,
    )
    data.patch_row = torch.tensor(patch_row_trans, dtype=torch.long)
    data.slide_code = torch.tensor(slide_codes, dtype=torch.long)
    return data
 
 
def build_patch_to_nodes(patch_row_tensor):
    """dict: global patch_row id -> LongTensor of node indices belonging to that patch."""
    patch_row_np = patch_row_tensor.numpy()
    order = np.argsort(patch_row_np, kind="stable")
    sorted_patches = patch_row_np[order]
    unique_patches, start_idx = np.unique(sorted_patches, return_index=True)
    groups = np.split(order, start_idx[1:])
    return {int(p): torch.tensor(g, dtype=torch.long) for p, g in zip(unique_patches, groups)}
 
 
class PatchGroupedLoader:
    """
    Groups cells by patch so each mini-batch contains complete (or capped) patches,
    instead of the arbitrary partial patches a plain node-level NeighborLoader would
    produce -- needed for patch-level pooling/contrastive alignment to be meaningful.

    Batches are SLIDE-STRATIFIED (never mixed): otherwise the patch-level contrastive
    loss can shortcut via slide/scan identity instead of genuine morphology-
    transcriptomics correspondence. See `patch_level_infonce`'s `slide_ids` masking
    for a second, independent safeguard.

    Reuses NeighborLoader's subgraph sampling via `collate_fn`, driving the seed-node
    groups ourselves. Relies on `collate_fn(index)` being directly callable with an
    arbitrary index tensor (standard in recent PyG); if that breaks, fall back to a
    fresh `NeighborLoader(data, input_nodes=seed_nodes, batch_size=len(seed_nodes))`
    per group.
    """

    def __init__(self, data, patch_to_nodes, patch_slide_id, num_neighbors=(10, 10),
                 patches_per_batch=16, max_cells_per_patch=64, shuffle=True,
                 max_patches_per_slide=None, oversample_target=None):
        """
        max_patches_per_slide: caps how many of a slide's OWN patches feed into batches
          per epoch (None = no cap). __len__/__iter__ give each slide a batch count
          proportional to its own patch count, so an outsized slide (e.g. CRC_P11mm at
          29,959 vs. 8,063-10,983 elsewhere) would dominate every epoch's gradient
          updates; capping truncates its (reshuffled) list before chunking, so a
          different random subset is used each epoch and the full pool still gets
          exposure over time, just not all at once.

        oversample_target: dict[int, int], slide code -> minimum patch draws per epoch
          (WITH replacement) -- the mirror-image problem to max_patches_per_slide: too
          FEW slides of a class rather than one slide too big (e.g. 2 normal vs. 8
          tumor slides here, each individually a normal size). A slide below its target
          gets padded via random.choices from its own real patches (not synthetic data;
          still passes through this loader's per-batch subsampling and the model's
          dropout, so repeats aren't identical training examples). Applied after
          max_patches_per_slide. None (default) leaves a slide at its natural count.
        """
        self.patch_to_nodes = patch_to_nodes
        self.patches_per_batch = patches_per_batch
        self.max_cells_per_patch = max_cells_per_patch
        self.shuffle = shuffle
        self.max_patches_per_slide = max_patches_per_slide
        self.oversample_target = oversample_target or {}

        # group patches by slide up front; `patch_slide_id` is indexed the same way
        # as `patch_meta["slide_id"]` (one entry per patch_row id)
        self.patches_by_slide = {}
        for p in patch_to_nodes.keys():
            sid = int(patch_slide_id[p])
            self.patches_by_slide.setdefault(sid, []).append(p)

        # --- FIX: Pass torch.arange(data.num_nodes) so all node indices are valid ---
        self._sampler_loader = NeighborLoader(
            data, 
            num_neighbors=list(num_neighbors),
            input_nodes=torch.arange(data.num_nodes), 
            batch_size=1,
        )

    def __len__(self):
        total = 0
        for sid, plist in self.patches_by_slide.items():
            n = len(plist) if self.max_patches_per_slide is None else min(len(plist), self.max_patches_per_slide)
            n = max(n, self.oversample_target.get(sid, 0))
            total += (n + self.patches_per_batch - 1) // self.patches_per_batch
        return total

    def __iter__(self):
        # build all batches slide-by-slide first (each batch confined to one slide),
        # then shuffle the batch ORDER so training still sees slides interleaved
        # across steps -- only within-batch mixing is what causes the shortcut.
        all_batches = []
        for sid, plist in self.patches_by_slide.items():
            plist = plist.copy()
            if self.shuffle:
                random.shuffle(plist)
            if self.max_patches_per_slide is not None:
                plist = plist[:self.max_patches_per_slide]
            target = self.oversample_target.get(sid)
            if target is not None and len(plist) < target:
                # pad up to target by resampling this slide's own patches WITH
                # replacement -- plist is already shuffled, so every real patch is
                # used at least once before any repeats occur
                plist = plist + random.choices(plist, k=target - len(plist))
                if self.shuffle:
                    random.shuffle(plist)
            for start in range(0, len(plist), self.patches_per_batch):
                all_batches.append(plist[start:start + self.patches_per_batch])

        if self.shuffle:
            random.shuffle(all_batches)

        for batch_patch_ids in all_batches:
            seed_chunks, local_patch_index = [], []
            for local_idx, p in enumerate(batch_patch_ids):
                nodes = self.patch_to_nodes[p]
                if len(nodes) > self.max_cells_per_patch:
                    perm = torch.randperm(len(nodes))[:self.max_cells_per_patch]
                    nodes = nodes[perm]
                seed_chunks.append(nodes)
                local_patch_index.append(torch.full((len(nodes),), local_idx, dtype=torch.long))
            seed_nodes = torch.cat(seed_chunks)
            batch = self._sampler_loader.collate_fn(seed_nodes)
            # NeighborLoader guarantees seed nodes are the first len(seed_nodes) rows.
            batch.local_patch_index = torch.cat(local_patch_index)
            batch.batch_patch_ids = torch.tensor(batch_patch_ids, dtype=torch.long)
            yield batch
 
 
@torch.no_grad()
def encode_full_graph_batched(encoder, data, device, num_neighbors=(10, 10), batch_size=4096):
    """
    Encode every node in `data` through a graph encoder (e.g. `model.trans_enc`)
    without ever materializing the whole graph's edges/attention at once.

    Needed for TransformerConv in particular: unlike SAGEConv's cheap neighbor-mean,
    TransformerConv stores per-edge query/key/value/attention tensors of shape
    (num_edges, heads, out_per_head). A single full-graph forward pass allocates
    all of that simultaneously, which is what causes OOM at eval/latent-export time
    even when mini-batch training (which only ever sees small sampled subgraphs)
    fits comfortably.

    This reuses the same sampling machinery as `PatchGroupedLoader`/training
    (`NeighborLoader`, `num_neighbors`), just iterating over every node as a seed
    in chunks of `batch_size`, so each forward pass only ever touches one small
    sampled subgraph at a time. Results are written back into a full-size output
    tensor keyed by each seed node's global id (`batch.n_id`), so the final
    ordering exactly matches `data`'s original node order regardless of batching/
    shuffling.
    """
    encoder.eval()
    out_dim = encoder.out.out_features
    out = torch.zeros(data.num_nodes, out_dim, device=device)

    loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        input_nodes=torch.arange(data.num_nodes),
        batch_size=batch_size,
        shuffle=False,
    )
    for batch in loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        z = encoder(x, edge_index)
        n_seed = batch.batch_size  # NeighborLoader: seed nodes are always the first n_seed rows
        seed_global_ids = batch.n_id[:n_seed].to(device)
        out[seed_global_ids] = z[:n_seed]
    return out


# ---- segment (group-wise) ops, self-contained (no torch_scatter dependency) ----
 
def segment_sum(src, index, num_segments):
    out_shape = (num_segments,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out
 
 
def segment_mean_pool(z, index, num_segments):
    s = segment_sum(z, index, num_segments)
    counts = segment_sum(torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype), index, num_segments)
    return s / counts.clamp(min=1.0)
 
 
def segment_softmax(scores, index, num_segments):
    seg_max = torch.full((num_segments,), float("-inf"), device=scores.device, dtype=scores.dtype)
    seg_max.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    scores = scores - seg_max[index]
    exp_scores = scores.exp()
    seg_sum = segment_sum(exp_scores, index, num_segments)
    return exp_scores / (seg_sum[index] + 1e-12)
 
 
# ---- loss ----
 
def patch_level_infonce(p_trans, p_hist, centroids, slide_ids=None, temperature=0.07,
                         neighbor_radius=None, weight=None, anchor_sim_threshold=None,
                         return_stats=False):
    """
    One-to-one patch-level InfoNCE (post-pooling: each row is a genuine patch, not
    a cell). Optionally:
      - `neighbor_radius`: excludes spatially close, non-matching patches from the
        negative set instead of forcing them apart -- nearby tissue can be
        genuinely similar in both modalities under spatial autocorrelation.
      - `anchor_sim_threshold`: same false-negative problem, but for patches that
        are transcriptomically near-duplicate regardless of distance (e.g. two
        separate tumor nests of the same subtype) -- without it, InfoNCE pushes
        them apart via spurious non-biological signal (staining, slide identity)
        as a tie-breaker. Checked on both `p_trans`-vs-`p_trans` and
        `p_hist`-vs-`p_hist` similarity, since a pair can be a near-duplicate in
        one modality but fail a strict cutoff in the other from noise/dropout
        alone (checking only one side missed real false negatives -- traced a
        checkerboard-like leiden-cluster artifact back to exactly this gap).
        Masking-only, never an added attractive term. Assumes L2-normalized
        inputs, so this is a cosine-similarity cutoff in [-1, 1] (e.g. 0.9 =
        near-duplicates only).
      - `weight`: inverse-frequency weighting per patch.

    `return_stats=True` also returns `frac_masked` (fraction of candidate
    negatives excluded by any mask) -- a collapse early-warning signal, since
    `anchor_sim_threshold` checks the model's own live projections: if they start
    collapsing together, more pairs get masked, removing the gradient that would
    correct it (self-reinforcing). Watch for `frac_masked` climbing toward 1.0
    early rather than only discovering `contrast_local` stuck at ~0 after a full
    run.

    MULTI-SLIDE NOTE: if a batch mixes slides, pass `slide_ids` -- raw coordinates
    aren't comparable across slides (overlapping ranges can make an unrelated
    cross-slide pair look spatially "close"), so masking additionally requires
    same-slide. In practice a no-op with slide-stratified batches; kept as a
    backstop.
    """
    logits = p_trans @ p_hist.T / temperature  # (P, P)
    P = logits.shape[0]
    device = logits.device

    # Hard-exclude cross-slide pairs from the negative set entirely (independent of
    # neighbor_radius below). With slide-stratified batches (see PatchGroupedLoader)
    # this is normally a no-op, since every patch in a batch already shares a slide --
    # it's kept as a defensive backstop so the loss can never exploit slide identity
    # as a shortcut even if batch composition changes elsewhere.
    if slide_ids is not None:
        same_slide = slide_ids.unsqueeze(0) == slide_ids.unsqueeze(1)
        cross_slide = ~same_slide
        cross_slide.fill_diagonal_(False)  # never mask out the true positive
        logits = logits.masked_fill(cross_slide, float("-inf"))

    if neighbor_radius is not None:
        dist = torch.cdist(centroids, centroids)
        neighbor_mask = dist < neighbor_radius
        if slide_ids is not None:
            same_slide = slide_ids.unsqueeze(0) == slide_ids.unsqueeze(1)
            neighbor_mask = neighbor_mask & same_slide
        neighbor_mask.fill_diagonal_(False)  # never mask out the true positive
        logits = logits.masked_fill(neighbor_mask, float("-inf"))

    if anchor_sim_threshold is not None:
        trans_sim = p_trans @ p_trans.T  # cosine sim, p_trans is L2-normalized
        hist_sim = p_hist @ p_hist.T     # p_token for the local/cell-level call
        false_neg_mask = (trans_sim > anchor_sim_threshold) | (hist_sim > anchor_sim_threshold)
        if slide_ids is not None:
            same_slide = slide_ids.unsqueeze(0) == slide_ids.unsqueeze(1)
            false_neg_mask = false_neg_mask & same_slide
        false_neg_mask.fill_diagonal_(False)  # never mask out the true positive
        logits = logits.masked_fill(false_neg_mask, float("-inf"))

    if return_stats:
        off_diag = ~torch.eye(P, dtype=torch.bool, device=device)
        frac_masked = (logits[off_diag] == float("-inf")).float().mean().item()
        stats = {"frac_masked": frac_masked, "n_candidates": P}

    idx = torch.arange(P, device=device)

    log_prob_t2h = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_t2h = -log_prob_t2h[idx, idx]

    logits_h2t = logits.T
    log_prob_h2t = logits_h2t - torch.logsumexp(logits_h2t, dim=1, keepdim=True)
    loss_h2t = -log_prob_h2t[idx, idx]

    per_patch_loss = 0.5 * (loss_t2h + loss_h2t)

    if weight is not None:
        loss = (per_patch_loss * weight).sum() / weight.sum().clamp(min=1e-8)
    else:
        loss = per_patch_loss.mean()

    if return_stats:
        return loss, stats
    return loss
 
 
def contrast_weight_schedule(epoch, warmup_epochs=5, max_weight=1.0):
    """Ramp contrastive loss weight up linearly; reconstruction stabilizes first."""
    return max_weight * min(1.0, epoch / warmup_epochs)


def vicreg_regularizer(z, var_target=1.0, eps=1e-4):
    """
    VICReg-style variance + covariance regularization (Bardes, Ponce & LeCun,
    2021) on a batch of latent vectors `z` (shape (B, D)). Penalizes collapse:
      - variance term: pushes per-dimension std up toward `var_target` via
        relu(target - std), stopping one dimension (or a correlated cluster)
        from soaking up most of the variance.
      - covariance term: pushes off-diagonal covariances toward zero, so
        dimensions can't be redundant proxies for the same direction.

    Unlike decoder conditioning (slide_code/tile_code), which only removes the
    *requirement* to encode a nuisance direction without penalizing it (why
    tile-conditioning alone changed nothing), this is an active penalty on z --
    doesn't need to know what the nuisance variable is, unlike a GRL
    classifier/regressor (unstable in this codebase historically).

    Requires B > 1 (returns 0 for B=1, where std/covariance are undefined).

    var_target: the paper's default (1.0) assumes roughly unit-scale
    representations; check `z.std(dim=0)` early in a run if the loss saturates
    at ~0 immediately (target too low) or stays large throughout (fighting the
    other losses too hard -- lower vicreg_weight or var_target instead).
    """
    B, D = z.shape
    if B < 2:
        return z.new_tensor(0.0)

    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
    loss_var = F.relu(var_target - std).mean()

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (B - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    loss_cov = off_diag.pow(2).sum() / D

    return loss_var + loss_cov


def pretrain_hist_autoencoder(model, hist_emb_all, patch_meta, epochs=100, batch_size=256,
                               lr=1e-3, weight_decay=1e-5, val_frac=0.1, patience=15,
                               device="cuda", verbose=True, seed=0):
    """
    Phase-1 pretraining: hist_enc/hist_dec only -- no graph machinery, no
    contrastive loss, no trans branch (plain full-dataset MLP training loop).

    Rebuilt to match a known-good external baseline that preserved anatomical
    structure well, where the original version (fixed LR, no validation, no
    early stopping) didn't:
      - pure MSE only -- avoids the unprincipled-combined-loss gradient-conflict
        problem. Add `vicreg_regularizer` deliberately if wanted; not wired in
        by default since pure MSE alone matched the working baseline.
      - AdamW with real weight decay, instead of unregularized Adam.
      - ReduceLROnPlateau instead of one fixed LR throughout.
      - a genuine validation split (`val_frac`), early stopping (`patience`),
        restoring the best-val-loss checkpoint at the end.

    Leave `hist_condition_on_slide=True` (default) -- see DualAutoencoder's
    docstring for the measured effect.

    Meant to run before `train()`. Follow with `freeze_hist_encoder(model)` and
    `train(..., freeze_hist=True)` for the joint phase.
    """
    import copy

    model.to(device)
    opt = torch.optim.AdamW(
        list(model.hist_enc.parameters()) + list(model.hist_dec.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    hist_emb_all = hist_emb_all.to(device)
    slide_id = patch_meta["slide_id"].to(device)
    N = hist_emb_all.shape[0]

    n_val = max(1, int(N * val_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.hist_enc.train()
        model.hist_dec.train()
        train_perm = train_idx[torch.randperm(len(train_idx))]
        tot_recon, nb = 0.0, 0
        for start in range(0, len(train_perm), batch_size):
            idx = train_perm[start:start + batch_size]
            x = hist_emb_all[idx]
            sid = slide_id[idx]
            z = model.hist_enc(x)
            recon = model.hist_dec(z, slide_code=sid)
            loss = F.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_recon += loss.item() * len(idx)
            nb += len(idx)
        train_loss = tot_recon / nb

        model.hist_enc.eval()
        model.hist_dec.eval()
        with torch.no_grad():
            x = hist_emb_all[val_idx]
            sid = slide_id[val_idx]
            z = model.hist_enc(x)
            recon = model.hist_dec(z, slide_code=sid)
            val_loss = F.mse_loss(recon, x).item()
        scheduler.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"[hist pretrain] epoch {epoch:4d} | train MSE {train_loss:.5f} | "
                  f"val MSE {val_loss:.5f} | lr {opt.param_groups[0]['lr']:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "hist_enc": copy.deepcopy(model.hist_enc.state_dict()),
                "hist_dec": copy.deepcopy(model.hist_dec.state_dict()),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"[hist pretrain] early stopping at epoch {epoch} (best val MSE {best_val:.5f})")
                break

    if best_state is not None:
        model.hist_enc.load_state_dict(best_state["hist_enc"])
        model.hist_dec.load_state_dict(best_state["hist_dec"])
        if verbose:
            print(f"[hist pretrain] restored best checkpoint (val MSE {best_val:.5f})")

    return {"best_val_mse": best_val, "stopped_epoch": epoch}


def freeze_hist_encoder(model, freeze_decoder=True):
    """
    Freezes hist_enc (and, by default, hist_dec) in place: sets requires_grad=False
    on their parameters so `train()`'s optimizer (which filters on requires_grad,
    see below) never updates them, and marks them .eval() so BatchNorm running
    stats stop updating too (requires_grad=False alone does NOT freeze BatchNorm's
    running mean/var -- those update from forward-pass statistics regardless of
    gradient flags, so both need to be handled).

    Call after `pretrain_hist_autoencoder`, before the joint/contrastive `train()`
    phase. `train_step`/`train`'s `freeze_hist=True` re-applies .eval() to these
    submodules each step (since `model.train()` at the top of `train_step`
    otherwise flips every submodule, including frozen ones, back to train mode).
    """
    for p in model.hist_enc.parameters():
        p.requires_grad_(False)
    model.hist_enc.eval()
    if freeze_decoder:
        for p in model.hist_dec.parameters():
            p.requires_grad_(False)
        model.hist_dec.eval()


def pretrain_token_autoencoder(model, token_emb_all, cell_meta, epochs=100, batch_size=1024,
                                lr=1e-3, weight_decay=1e-5, val_frac=0.1, patience=15,
                                device="cuda", verbose=True, seed=0):
    """
    Phase-1 pretraining for the token branch (`token_enc`/`token_dec`) -- CELL-level
    analogue of `pretrain_hist_autoencoder`, same rationale and same recipe (pure
    MSE, AdamW+weight_decay, ReduceLROnPlateau, held-out val split, early stopping,
    restore best checkpoint). Batched over CELLS via `cell_meta["slide_id"]`
    instead of over patches via `patch_meta["slide_id"]` -- everything else is
    identical in structure.

    Requires `model.token_enc`/`model.token_dec` to exist, i.e. `DualAutoencoder`
    was constructed with `token_dim` set.

    Meant to run BEFORE `train()`, alongside (order doesn't matter relative to)
    `pretrain_hist_autoencoder`. Follow with `freeze_token_encoder(model)` and
    `train(..., freeze_token=True)` for the joint phase.
    """
    import copy

    if model.token_enc is None:
        raise ValueError("model has no token branch (token_dim was None) -- nothing to pretrain")

    model.to(device)
    opt = torch.optim.AdamW(
        list(model.token_enc.parameters()) + list(model.token_dec.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    token_emb_all = token_emb_all.to(device)
    slide_id = cell_meta["slide_id"].to(device)
    N = token_emb_all.shape[0]

    n_val = max(1, int(N * val_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.token_enc.train()
        model.token_dec.train()
        train_perm = train_idx[torch.randperm(len(train_idx))]
        tot_recon, nb = 0.0, 0
        for start in range(0, len(train_perm), batch_size):
            idx = train_perm[start:start + batch_size]
            x = token_emb_all[idx]
            sid = slide_id[idx]
            z = model.token_enc(x)
            recon = model.token_dec(z, slide_code=sid)
            loss = F.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_recon += loss.item() * len(idx)
            nb += len(idx)
        train_loss = tot_recon / nb

        model.token_enc.eval()
        model.token_dec.eval()
        with torch.no_grad():
            x = token_emb_all[val_idx]
            sid = slide_id[val_idx]
            z = model.token_enc(x)
            recon = model.token_dec(z, slide_code=sid)
            val_loss = F.mse_loss(recon, x).item()
        scheduler.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"[token pretrain] epoch {epoch:4d} | train MSE {train_loss:.5f} | "
                  f"val MSE {val_loss:.5f} | lr {opt.param_groups[0]['lr']:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "token_enc": copy.deepcopy(model.token_enc.state_dict()),
                "token_dec": copy.deepcopy(model.token_dec.state_dict()),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"[token pretrain] early stopping at epoch {epoch} (best val MSE {best_val:.5f})")
                break

    if best_state is not None:
        model.token_enc.load_state_dict(best_state["token_enc"])
        model.token_dec.load_state_dict(best_state["token_dec"])
        if verbose:
            print(f"[token pretrain] restored best checkpoint (val MSE {best_val:.5f})")

    return {"best_val_mse": best_val, "stopped_epoch": epoch}


def freeze_token_encoder(model, freeze_decoder=True):
    """
    Freezes `token_enc` (and, by default, `token_dec`) in place -- CELL-level
    analogue of `freeze_hist_encoder`, same mechanism (requires_grad=False +
    .eval() for BatchNorm). Call after `pretrain_token_autoencoder`, before the
    joint `train(..., freeze_token=True)` phase.
    """
    if model.token_enc is None:
        return
    for p in model.token_enc.parameters():
        p.requires_grad_(False)
    model.token_enc.eval()
    if freeze_decoder:
        for p in model.token_dec.parameters():
            p.requires_grad_(False)
        model.token_dec.eval()


def train_step(model, batch, hist_emb_all, patch_meta, opt, contrast_weight,
               temperature=0.07, weight_reconstruction=True,
               freeze_hist=False, edge_dropout_p=0.0, gene_dropout_p=0.0, device="cuda",
               token_emb_all=None, cell_meta=None, local_contrast_weight=0.0, freeze_token=False,
               patch_anchor_sim_threshold=None, local_anchor_sim_threshold=None):
    """
    token_emb_all / cell_meta / local_contrast_weight / freeze_token: cell-level
    analogue of hist_emb_all/patch_meta/contrast_weight/freeze_hist. token_emb_all
    and cell_meta["xy"]/["slide_id"] must be row-aligned with data.x -- seed cells
    are looked up via batch.n_id[:n_seed] (NeighborLoader puts seed nodes first).
    Defaults reproduce pre-token behavior for models with no token branch.

    patch_anchor_sim_threshold / local_anchor_sim_threshold: forwarded to
    patch_level_infonce for the patch- and cell-level calls respectively -- two
    separate thresholds since pooled per-patch expression is smoother/more
    separable than raw per-cell expression. Both default to None (off).

    edge_dropout_p / gene_dropout_p: denoising-style corruption applied only to
    what trans_enc sees (reconstruction target stays clean). Targets two Slide_ID
    leakage channels: edge_dropout_p attacks local graph structure (slide-specific
    cell density), gene_dropout_p attacks over-reliance on slide-discriminative
    genes. Neither touches leakage that reflects genuine biology (Slide_ID is
    confounded 1:1 with patient/tumor status here -- no input corruption can
    separate real biological signal from technical at this sample size).
    """
    model.train()
    if freeze_hist:
        # model.train() re-enabled BatchNorm on every submodule -- pin back to eval
        # only the ones actually frozen (checked directly, since freeze_decoder=False
        # can leave hist_dec trainable while hist_enc is frozen).
        if not any(p.requires_grad for p in model.hist_enc.parameters()):
            model.hist_enc.eval()
        if not any(p.requires_grad for p in model.hist_dec.parameters()):
            model.hist_dec.eval()
    if freeze_token and model.token_enc is not None:
        # same reasoning as the freeze_hist block above, mirrored for the token branch
        if not any(p.requires_grad for p in model.token_enc.parameters()):
            model.token_enc.eval()
        if not any(p.requires_grad for p in model.token_dec.parameters()):
            model.token_dec.eval()
    opt.zero_grad()
 
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    n_seed = batch.local_patch_index.shape[0]

    x_in = F.dropout(x, p=gene_dropout_p, training=True) if gene_dropout_p > 0 else x
    edge_index_in = dropout_edge(edge_index, p=edge_dropout_p, training=True)[0] if edge_dropout_p > 0 else edge_index

    z_full = model.trans_enc(x_in, edge_index_in)
    z_cells = z_full[:n_seed]  # cell-level latents for seed cells only

    # --- cell-level reconstruction (unchanged in spirit: still per cell) ---
    # target is the CLEAN x, not x_in -- see docstring above
    slide_code = batch.slide_code[:n_seed].to(device)
    x_recon = model.trans_dec(z_cells, slide_code)
    recon_per_cell = F.mse_loss(x_recon, x[:n_seed], reduction="none").mean(dim=1)
 
    if weight_reconstruction:
        batch_patch_ids = batch.batch_patch_ids.to(device)
        local_patch_index = batch.local_patch_index.to(device)
        per_patch_weight = patch_meta["weight"][batch_patch_ids]
        cell_weight = per_patch_weight[local_patch_index]
        loss_recon_trans = (recon_per_cell * cell_weight).sum() / cell_weight.sum().clamp(min=1e-8)
    else:
        loss_recon_trans = recon_per_cell.mean()
 
    # --- patch-level branch (histology pooling/encoding + patch-level contrastive) ---
    # Skipped when frozen AND zero-weighted: neither loss term can produce a nonzero
    # gradient then, so the pooling/hist_enc/hist_dec/proj forward passes were pure
    # wasted compute (this project's regime always runs freeze_hist=True,
    # max_contrast_weight=0.0 -- kept in the architecture only for old checkpoints /
    # patch-level eval code). Mirrors the token branch's own "else: zero tensor" case.
    skip_patch_branch = freeze_hist and contrast_weight == 0
    if not skip_patch_branch:
        # pool cell latents into one patch latent per patch present in this batch
        local_patch_index = batch.local_patch_index.to(device)
        num_local_patches = batch.batch_patch_ids.shape[0]
        if model.pool_type == "attention":
            z_patch_trans, _attn_w = model.pool(z_cells, local_patch_index, num_local_patches)
        else:
            z_patch_trans = segment_mean_pool(z_cells, local_patch_index, num_local_patches)

        # histology side: one embedding per patch already, no duplication needed
        batch_patch_ids = batch.batch_patch_ids.to(device)
        batch_slide_ids = patch_meta["slide_id"][batch_patch_ids]
        hist_batch = hist_emb_all[batch_patch_ids]
        z_patch_hist = model.hist_enc(hist_batch)
        hist_recon = model.hist_dec(z_patch_hist, slide_code=batch_slide_ids)
        loss_recon_hist = F.mse_loss(hist_recon, hist_batch)

        # patch-level contrastive alignment
        p_trans_patch = model.proj_trans(z_patch_trans)
        p_hist_patch = model.proj_hist(z_patch_hist)

        centroids = patch_meta["centroid"][batch_patch_ids]
        weights = patch_meta["weight"][batch_patch_ids]
        loss_contrast = patch_level_infonce(
            p_trans_patch, p_hist_patch, centroids, slide_ids=batch_slide_ids,
            temperature=temperature, neighbor_radius=patch_meta["neighbor_radius"], weight=weights,
            anchor_sim_threshold=patch_anchor_sim_threshold,
        )
    else:
        loss_recon_hist = z_cells.new_tensor(0.0)
        loss_contrast = z_cells.new_tensor(0.0)

    # --- token side: CELL-level local branch, genuinely one-to-one, no pooling ---
    # Always computed (mirrors loss_recon_hist's unconditional-even-when-frozen
    # pattern above, kept for logging/monitoring parity) whenever the model has a
    # token branch and token data was supplied; contributes zero gradient to
    # token_enc/token_dec once frozen, same as loss_recon_hist does for hist_enc/dec.
    if model.token_enc is not None and token_emb_all is not None:
        seed_global_ids = batch.n_id[:n_seed].to(device)
        token_x = token_emb_all[seed_global_ids]
        cell_slide_ids = cell_meta["slide_id"][seed_global_ids]

        z_token_cells = model.token_enc(token_x)
        token_recon = model.token_dec(z_token_cells, slide_code=cell_slide_ids)
        loss_recon_token = F.mse_loss(token_recon, token_x)

        p_trans_local = model.proj_trans_local(z_cells)
        p_token = model.proj_token(z_token_cells)
        cell_xy = cell_meta["xy"][seed_global_ids]
        loss_contrast_local, local_mask_stats = patch_level_infonce(
            p_trans_local, p_token, cell_xy, slide_ids=cell_slide_ids,
            temperature=temperature, neighbor_radius=cell_meta["neighbor_radius"],
            anchor_sim_threshold=local_anchor_sim_threshold,
            return_stats=True,
        )
        local_frac_masked = local_mask_stats["frac_masked"]
    else:
        loss_recon_token = z_cells.new_tensor(0.0)
        loss_contrast_local = z_cells.new_tensor(0.0)
        local_frac_masked = 0.0

    loss = (loss_recon_trans + loss_recon_hist + contrast_weight * loss_contrast
            + loss_recon_token + local_contrast_weight * loss_contrast_local)
    loss.backward()
    opt.step()

    return {
        "loss": loss.item(),
        "recon_trans": loss_recon_trans.item(),
        "recon_hist": loss_recon_hist.item(),
        "contrast": loss_contrast.item(),
        "recon_token": loss_recon_token.item(),
        "contrast_local": loss_contrast_local.item(),
        "local_frac_masked": local_frac_masked,
    }


def train(model, loader, hist_emb_all, patch_meta, n_epochs=30, lr=1e-3, warmup_epochs=5,
          max_contrast_weight=1.0, temperature=0.07,
          weight_reconstruction=True, freeze_hist=False, hist_lr_mult=1.0,
          weight_decay=0.0, edge_dropout_p=0.0, gene_dropout_p=0.0, device="cuda",
          token_emb_all=None, cell_meta=None, max_local_contrast_weight=0.0,
          local_warmup_epochs=None, freeze_token=False, token_lr_mult=1.0,
          patch_anchor_sim_threshold=None, local_anchor_sim_threshold=None):
    """
    token_emb_all / cell_meta: token branch's cell-level analogue of
      hist_emb_all/patch_meta (see `build_cell_meta`). None (default) trains as
      before, including for models with no token branch.

    patch_anchor_sim_threshold / local_anchor_sim_threshold: forwarded to
      `patch_level_infonce` -- cosine-similarity cutoff excluding
      transcriptomically near-duplicate, non-matching patches/cells from hard
      negatives (same false-negative protection as `neighbor_radius`, gated on
      expression similarity instead of distance). Default None (off).

    max_local_contrast_weight / local_warmup_epochs: separate ramp schedule for
      the cell<->token term vs. max_contrast_weight/warmup_epochs (patch-level).
      local_warmup_epochs defaults to warmup_epochs if not given.

    freeze_token / token_lr_mult: token-branch analogues of freeze_hist/
      hist_lr_mult (gentler LR on an already-pretrained branch).

    freeze_hist: assumes `freeze_hist_encoder(model)` already ran -- frozen
      params are excluded from the optimizer automatically via requires_grad.

    hist_lr_mult: LR multiplier for hist_enc/hist_dec, only relevant when
      freeze_hist=False. Keep <1.0 so contrastive/recon_hist losses reshape the
      histology tower more gently than trans_enc (training from scratch) --
      full `lr` risks undoing the Slide_ID-leakage fix Phase-1 pretraining
      bought, since loss_contrast can exploit slide shortcuts once hist_enc is
      allowed to move.

    weight_decay: applied to every trainable param group -- trans_enc's
      equivalent of what pretrain_hist_autoencoder already gives hist_enc/dec.

    edge_dropout_p / gene_dropout_p: forwarded to `train_step` (denoising
      mechanism, see its docstring).
    """
    model.to(device)
    hist_param_ids = {id(p) for p in list(model.hist_enc.parameters()) + list(model.hist_dec.parameters())}
    token_param_ids = set()
    if model.token_enc is not None:
        token_param_ids = {id(p) for p in list(model.token_enc.parameters()) + list(model.token_dec.parameters())}
    special_param_ids = hist_param_ids | token_param_ids
    hist_params = [p for p in model.parameters() if p.requires_grad and id(p) in hist_param_ids]
    token_params = [p for p in model.parameters() if p.requires_grad and id(p) in token_param_ids]
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in special_param_ids]
    param_groups = [{"params": other_params, "lr": lr, "weight_decay": weight_decay}]
    if hist_params:
        param_groups.append({"params": hist_params, "lr": lr * hist_lr_mult, "weight_decay": weight_decay})
    if token_params:
        param_groups.append({"params": token_params, "lr": lr * token_lr_mult, "weight_decay": weight_decay})
    opt = torch.optim.Adam(param_groups)
    hist_emb_all = hist_emb_all.to(device)
    patch_meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in patch_meta.items()}

    use_tokens = model.token_enc is not None and token_emb_all is not None
    if use_tokens:
        token_emb_all = token_emb_all.to(device)
        cell_meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cell_meta.items()}
    local_warmup_epochs = local_warmup_epochs if local_warmup_epochs is not None else warmup_epochs

    # Per-epoch history, returned (not just printed) so callers running many
    # configs unattended (e.g. run_token_experiment/a sweep) can check for a
    # repeat of the local_frac_masked collapse (see CONFIG's comment on
    # max_local_contrast_weight in train_autoenc_token.py) without a human
    # having to scroll each run's stdout -- existing callers that ignore the
    # return value are unaffected.
    history = []
    for epoch in range(n_epochs):
        cw = contrast_weight_schedule(epoch, warmup_epochs, max_contrast_weight)
        lcw = contrast_weight_schedule(epoch, local_warmup_epochs, max_local_contrast_weight) if use_tokens else 0.0
        epoch_stats = {"loss": 0.0, "recon_trans": 0.0, "recon_hist": 0.0, "contrast": 0.0,
                        "recon_token": 0.0, "contrast_local": 0.0, "local_frac_masked": 0.0}
        n_batches = 0
        for batch in loader:
            stats = train_step(model, batch, hist_emb_all, patch_meta, opt, cw,
                                temperature=temperature,
                                weight_reconstruction=weight_reconstruction,
                                freeze_hist=freeze_hist,
                                edge_dropout_p=edge_dropout_p, gene_dropout_p=gene_dropout_p,
                                device=device,
                                token_emb_all=token_emb_all if use_tokens else None,
                                cell_meta=cell_meta if use_tokens else None,
                                local_contrast_weight=lcw, freeze_token=freeze_token,
                                patch_anchor_sim_threshold=patch_anchor_sim_threshold,
                                local_anchor_sim_threshold=local_anchor_sim_threshold)
            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            n_batches += 1
        epoch_stats = {k: v / n_batches for k, v in epoch_stats.items()}
        print(f"epoch {epoch:03d} | contrast_w={cw:.2f}" + (f" | local_w={lcw:.2f}" if use_tokens else "") + " | " +
              " | ".join(f"{k}={v:.4f}" for k, v in epoch_stats.items()))
        history.append({"epoch": epoch, "contrast_w": cw, "local_w": lcw, **epoch_stats})

    return history


@torch.no_grad()
def retrieval_accuracy(model, data, hist_emb_all, patch_meta, device="cuda",
                        eval_num_neighbors=(10, 10), eval_batch_size=4096):
    """
    Full-graph patch-level retrieval eval. Reports strict top-1 accuracy plus a
    neighborhood-tolerant top-1 (counts a hit if the predicted patch's centroid
    lies within `patch_meta['neighbor_radius']` of the true one) -- two adjacent,
    genuinely similar patches shouldn't be penalized like a distant confusion.

    `trans_enc` runs via `encode_full_graph_batched`, not a single full-graph
    call: with TransformerConv, materializing per-edge attention for every edge
    at once can OOM even though mini-batch training fits fine.

    Also reports median_rank/mrr -- top-1 over thousands of candidates is harsh
    and can read as ~0 even when the true match is consistently near the top.

    Returns patch-level embeddings for reuse downstream: z_trans_patch,
    z_hist_patch, z_joint_patch (concat of the two), row-aligned to
    `unique_patches`. These are PRE-projection encoder outputs, not
    proj_trans/proj_hist's -- standard practice (SimCLR etc.) is that the
    pre-projection representation transfers better to downstream probes.
    """
    model.eval()
    z_cells_full = encode_full_graph_batched(
        model.trans_enc, data, device,
        num_neighbors=eval_num_neighbors, batch_size=eval_batch_size,
    )

    patch_row = data.patch_row.to(device)
    num_patches = int(patch_row.max().item()) + 1
    if model.pool_type == "attention":
        z_patch_trans_full, _ = model.pool(z_cells_full, patch_row, num_patches)
    else:
        z_patch_trans_full = segment_mean_pool(z_cells_full, patch_row, num_patches)

    unique_patches = torch.unique(patch_row)
    hist_batch = hist_emb_all.to(device)[unique_patches]
    z_patch_hist = model.hist_enc(hist_batch)
    z_patch_trans = z_patch_trans_full[unique_patches]

    p_trans = model.proj_trans(z_patch_trans)
    p_hist = model.proj_hist(z_patch_hist)

    sims = p_trans @ p_hist.T
    pred = sims.argmax(dim=1)
    true_col = torch.arange(len(unique_patches), device=device)  # one-to-one, in `unique_patches` order
    top1_acc = (pred == true_col).float().mean().item()

    n_candidates = len(unique_patches)
    diag = sims.diagonal()
    # Chunked to avoid materializing a second full n_candidates x n_candidates
    # tensor (`sims` itself is already n_candidates^2 and, at full-cohort scale
    # across all slides, that's the largest tensor this function can afford --
    # a second one of the same size is what OOMs).
    rank = torch.empty(n_candidates, device=device)
    for start in range(0, n_candidates, eval_batch_size):
        end = min(start + eval_batch_size, n_candidates)
        rank[start:end] = (sims[start:end] >= diag[start:end].unsqueeze(1)).sum(dim=1).float()
    mrr = (1.0 / rank).mean().item()
    median_rank = rank.median().item()
    chance_mrr = float(np.sum(1.0 / np.arange(1, n_candidates + 1)) / n_candidates)

    neighbor_radius = patch_meta.get("neighbor_radius")
    neighbor_acc = None
    if neighbor_radius is not None:
        centroids = patch_meta["centroid"][unique_patches].to(device)
        slide_ids = patch_meta["slide_id"][unique_patches].to(device)
        dist = (centroids[pred] - centroids).norm(dim=1)
        same_slide = slide_ids[pred] == slide_ids  # raw coords aren't comparable across slides
        neighbor_acc = ((dist < neighbor_radius) & same_slide).float().mean().item()

    print(f"Top-1 patch retrieval accuracy: {top1_acc:.3f} (chance = {1/n_candidates:.5f})")
    if neighbor_acc is not None:
        print(f"Neighborhood-tolerant top-1 accuracy (radius={neighbor_radius}): {neighbor_acc:.3f}")
    print(f"Median rank: {median_rank:.1f} / {n_candidates} | MRR: {mrr:.4f} (chance MRR = {chance_mrr:.5f})")

    z_joint_patch = torch.cat([z_patch_trans, z_patch_hist], dim=1)

    return {"top1": top1_acc, "neighbor_tolerant": neighbor_acc,
            "median_rank": median_rank, "mrr": mrr, "chance_mrr": chance_mrr, "n_candidates": n_candidates,
            "z_patch_trans_full": z_patch_trans_full.cpu().numpy(),
            "z_trans_patch": z_patch_trans.cpu().numpy(),
            "z_hist_patch": z_patch_hist.cpu().numpy(),
            "z_joint_patch": z_joint_patch.cpu().numpy(),
            "unique_patches": unique_patches.cpu().numpy()}


@torch.no_grad()
def token_retrieval_accuracy(model, data, token_emb_all, cell_meta, device="cuda",
                              eval_num_neighbors=(10, 10), eval_batch_size=4096,
                              max_cells_per_slide=5000, seed=0, cell_type_labels=None):
    """
    CELL-level analogue of `retrieval_accuracy`, for the token branch.

    Full-population retrieval isn't feasible (an N x N matrix over hundreds of
    thousands of cells is 10^11+ entries). Instead subsamples up to
    `max_cells_per_slide` cells PER SLIDE, computing top-1 within each slide's
    subsample separately -- pooling across slides would let the metric be
    satisfied by recognizing slide identity instead of genuine correspondence.

    `cell_type_labels`: optional (n_cells,) array, row-aligned to `data` -- an
    INDEPENDENT cluster assignment (e.g. leiden on trans-only embedding, not
    this model's own projections, which would be circular). If given, also
    reports cluster-level top-1 (a hit if the retrieved cell shares the query's
    cluster label) -- strict exact-ID top-1 is harsh when many cells are
    genuine biological near-duplicates, same relief `neighbor_tolerant` gives
    for spatial proximity.

    Returns full-dataset z_trans_cell_full/z_token_cell_full for reuse
    downstream (e.g. evaluate_latent_leakage).
    """
    if model.token_enc is None:
        raise ValueError("model has no token branch (token_dim was None)")

    model.eval()
    z_cells_full = encode_full_graph_batched(
        model.trans_enc, data, device,
        num_neighbors=eval_num_neighbors, batch_size=eval_batch_size,
    )
    token_emb_all = token_emb_all.to(device)
    slide_id_all = cell_meta["slide_id"].to(device)
    xy_all = cell_meta["xy"].to(device)
    neighbor_radius = cell_meta.get("neighbor_radius")
    if cell_type_labels is not None:
        cell_type_labels = torch.as_tensor(cell_type_labels, device=device)

    z_token_full = model.token_enc(token_emb_all)

    rng = np.random.default_rng(seed)
    results_per_slide = {}
    all_top1, all_n, all_cluster_top1 = [], [], []
    for sid in torch.unique(slide_id_all).cpu().numpy():
        idx = (slide_id_all == int(sid)).nonzero(as_tuple=True)[0]
        if len(idx) > max_cells_per_slide:
            perm = torch.tensor(rng.choice(len(idx), size=max_cells_per_slide, replace=False), device=idx.device)
            idx = idx[perm]

        p_trans_local = model.proj_trans_local(z_cells_full[idx])
        p_token = model.proj_token(z_token_full[idx])
        sims = p_trans_local @ p_token.T
        pred = sims.argmax(dim=1)
        true_col = torch.arange(len(idx), device=device)
        top1 = (pred == true_col).float().mean().item()

        neighbor_acc = None
        if neighbor_radius is not None:
            xy = xy_all[idx]
            dist = (xy[pred] - xy).norm(dim=1)
            neighbor_acc = (dist < neighbor_radius).float().mean().item()

        cluster_top1 = None
        if cell_type_labels is not None:
            labels = cell_type_labels[idx]
            cluster_top1 = (labels[pred] == labels).float().mean().item()
            all_cluster_top1.append(cluster_top1)

        results_per_slide[int(sid)] = {"top1": top1, "neighbor_tolerant": neighbor_acc,
                                        "cluster_top1": cluster_top1, "n": len(idx)}
        all_top1.append(top1)
        all_n.append(len(idx))

    # cluster_top1 leads: exact-cell top1 is overly harsh once a slide has many
    # biological near-duplicates (see this function's docstring), so it's reported
    # second, as a diagnostic rather than the primary read.
    mean_top1 = float(np.average(all_top1, weights=all_n))
    mean_cluster_top1 = None
    if all_cluster_top1:
        mean_cluster_top1 = float(np.average(all_cluster_top1, weights=all_n))
        print(f"Cell<->token retrieval, n-weighted mean cluster-level top-1: {mean_cluster_top1:.3f}")
    print(f"Cell<->token retrieval, n-weighted mean exact-cell top-1 across slides: {mean_top1:.3f}")
    for sid, r in results_per_slide.items():
        msg = f"  slide {sid}:"
        if r["cluster_top1"] is not None:
            msg += f" cluster_top1={r['cluster_top1']:.3f} |"
        msg += f" top1={r['top1']:.3f}"
        if r["neighbor_tolerant"] is not None:
            msg += f" | neighbor_tolerant={r['neighbor_tolerant']:.3f}"
        msg += f" (n={r['n']}, chance={1 / r['n']:.5f})"
        print(msg)

    return {"mean_top1": mean_top1, "mean_cluster_top1": mean_cluster_top1, "per_slide": results_per_slide,
            "z_trans_cell_full": z_cells_full.cpu().numpy(),
            "z_token_cell_full": z_token_full.cpu().numpy()}


# ---- experiment harness: reproducible sweeps with a persistent, inspectable log ----

def evaluate_latent_leakage(z, slide_id, cv=5, max_iter=5000):
    """
    Slide_ID leakage diagnostic on an ALREADY-COMPUTED embedding `z` (n_patches x dim,
    numpy array or tensor): checks how well a plain logistic regression can recover
    Slide_ID from it via cross-validated accuracy (chance = 1/n_slides).

    Use this directly on whatever embedding you actually intend to use downstream --
    trans-only, hist-only, or the joint concat -- not just hist_enc's output.
    `evaluate_hist_leakage` is a thin wrapper around this (runs hist_enc first, kept
    for the existing call sites); reach for this one directly when you already have
    the embedding computed (e.g. from `retrieval_accuracy`'s `z_trans_patch` /
    `z_hist_patch` / `z_joint_patch`).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()
    y = slide_id.cpu().numpy() if torch.is_tensor(slide_id) else np.asarray(slide_id)

    clf = LogisticRegression(max_iter=max_iter)
    scores = cross_val_score(clf, z, y, cv=cv)
    return {
        "leakage_mean": float(scores.mean()),
        "leakage_scores": scores.tolist(),
        "leakage_chance": 1.0 / len(np.unique(y)),
    }


def evaluate_hist_leakage(model, hist_emb_all, slide_id, device="cuda", cv=5, max_iter=5000):
    """
    Slide_ID leakage diagnostic: encodes every patch with `model.hist_enc` and
    delegates to `evaluate_latent_leakage` for the actual cross-validated check.

    Row-aligned directly against `hist_emb_all`/`slide_id` (the same tensors used
    throughout training -- pass `patch_meta["slide_id"]`) rather than going
    through any AnnData obs lookup or deduplication step. This deliberately
    sidesteps the Patch_ID-collision bug found in the notebook's own
    `patch_adata_global` construction: Patch_ID repeats across slides (only
    11,974 unique values across 44,592 patches in this dataset), so
    deduplicating on it alone silently drops ~73% of patches and mixes up which
    slide's copy survives. Use this function for any leakage check going
    forward instead of rebuilding that pipeline.
    """
    model.eval()
    with torch.no_grad():
        z = model.hist_enc(hist_emb_all.to(device)).cpu().numpy()
    return evaluate_latent_leakage(z, slide_id, cv=cv, max_iter=max_iter)


def evaluate_latent_spatial_autocorr(z, centroids, n_neighs=6):
    """
    Unsupervised sanity check: computes Moran's I per latent dimension via squidpy,
    using a KNN graph built directly on patch centroids. Call this on ONE slide's
    patches at a time (pass that slide's z/centroids only) -- mixing slides would
    build nonsensical cross-slide neighbor edges since raw coordinates aren't
    comparable across slides (same caveat as retrieval_accuracy's neighbor_tolerant
    check).

    A real (non-noise) spatial embedding should be spatially smooth -- nearby patches
    more similar than distant ones -- so this is a necessary-but-not-sufficient check:
    near-zero mean |I| across dims suggests the embedding is ~noise for this slide;
    high |I| confirms spatial structure exists but doesn't by itself prove that
    structure is shared cross-modal biology rather than e.g. a smooth staining or
    scanning artifact -- same ambiguity as any single unsupervised diagnostic here.
    """
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()
    if torch.is_tensor(centroids):
        centroids = centroids.detach().cpu().numpy()

    dims = [f"dim_{i}" for i in range(z.shape[1])]
    a = ad.AnnData(X=z.astype(np.float32))
    a.var_names = dims
    a.obsm["spatial"] = centroids

    sq.gr.spatial_neighbors_knn(a, spatial_key="spatial", n_neighs=n_neighs)
    sq.gr.spatial_autocorr(a, mode="moran", genes=dims)

    moran_df = a.uns["moranI"]
    return {
        "moran_mean_I": float(moran_df["I"].mean()),
        "moran_median_I": float(moran_df["I"].median()),
        "moran_frac_sig": float((moran_df["pval_norm_fdr_bh"] < 0.05).mean()),
        "moran_per_dim": moran_df,
    }


def evaluate_disease_signal(z_train, train_slide_name, z_held, held_slide_name,
                             tumor_prefix="CRC", max_iter=5000, class_weight=None):
    """
    Tests whether an embedding captures real tumor-vs-normal biological signal --
    since tumor status is confounded 1:1 with Slide_ID/patient in this cohort, some
    of the Slide_ID leakage measured elsewhere (evaluate_latent_leakage) is plausibly
    real biology, not a technical artifact to suppress.

    Two checks, both genuinely slide-disjoint (unlike a naive pooled cross_val_score):
      - train_loso_acc: leave-one-TRAINING-slide-out (sklearn LeaveOneGroupOut) --
        does the classifier separate tumor/normal on a training slide it wasn't fit
        on? Degenerate folds (down to one class) are skipped; n_valid_folds reports
        how many contributed.
      - held_out_acc: fits on all training-slide patches, predicts on the TRUE
        held-out slide (never seen). Its true label is constant, so this is "fraction
        classified as its own known label," not two-sided accuracy -- read alongside
        train_loso_acc, not alone.

    class_weight: forwarded to LogisticRegression (e.g. "balanced" for training-pool
      class imbalance, e.g. only 1 normal slide left among 7 when a normal slide is
      held out). This classifier is fit post-hoc on frozen embeddings, so rebalancing
      only moves ITS decision boundary -- it separates a genuine embedding-quality
      gap from an artifact of this diagnostic's own imbalance: held_out_acc
      recovering once balanced means the embeddings were fine; staying low means they
      genuinely don't represent the minority class well.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    if torch.is_tensor(z_train):
        z_train = z_train.detach().cpu().numpy()
    if torch.is_tensor(z_held):
        z_held = z_held.detach().cpu().numpy()

    groups = np.asarray(train_slide_name)
    y_train = np.array([1 if str(s).startswith(tumor_prefix) else 0 for s in train_slide_name])
    y_held = np.array([1 if str(s).startswith(tumor_prefix) else 0 for s in held_slide_name])

    fold_scores = []
    if len(np.unique(groups)) >= 2:
        for tr_idx, te_idx in LeaveOneGroupOut().split(z_train, y_train, groups=groups):
            if len(np.unique(y_train[tr_idx])) < 2:
                continue  # degenerate fold: training portion left with only one class
            clf_fold = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
            clf_fold.fit(z_train[tr_idx], y_train[tr_idx])
            fold_scores.append(clf_fold.score(z_train[te_idx], y_train[te_idx]))
    train_loso_acc = float(np.mean(fold_scores)) if fold_scores else None

    held_out_acc = None
    if len(y_held) and len(np.unique(y_train)) >= 2:
        clf = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
        clf.fit(z_train, y_train)
        held_out_acc = float((clf.predict(z_held) == y_held).mean())

    return {
        "train_loso_acc": train_loso_acc,
        "n_valid_folds": len(fold_scores),
        "held_out_acc": held_out_acc,
        "held_out_true_label": int(y_held[0]) if len(y_held) else None,
    }


def pretrain_hist_once(model_kwargs, hist_emb_all, patch_meta, pretrain_kwargs, device="cuda"):
    """
    Runs `pretrain_hist_autoencoder` exactly ONCE on a throwaway model built from
    `model_kwargs`, and returns just the resulting hist_enc/hist_dec state dicts
    for reuse across many `run_experiment` calls via its `model_template_state`
    argument.

    Phase 1 doesn't depend on anything a Phase-3 sweep (contrast weight, freeze
    strategy, patches_per_batch, ...) would vary, so pretraining it once and
    reusing the result keeps every sweep run's comparison clean -- only Phase 3
    differs between runs, not also a different random Phase-1 outcome -- and
    avoids redundantly re-pretraining an identical histology encoder (up to 100
    epochs over the full dataset) for every point in the sweep.
    """
    import copy
    template_model = DualAutoencoder(**model_kwargs)
    pretrain_hist_autoencoder(template_model, hist_emb_all, patch_meta, **pretrain_kwargs, device=device)
    return {
        "hist_enc": copy.deepcopy(template_model.hist_enc.state_dict()),
        "hist_dec": copy.deepcopy(template_model.hist_dec.state_dict()),
    }


def pretrain_hist_replicated(model_kwargs, hist_emb_all, patch_meta, pretrain_kwargs,
                              n_replicates=5, seeds=None, select_by="leakage", device="cuda"):
    """
    Runs `pretrain_hist_autoencoder` `n_replicates` times from different random
    inits, returns the BEST replicate's hist_enc/hist_dec state dicts plus a
    per-replicate log (seed, stopped epoch, val MSE, leakage) so run-to-run
    variance is visible before trusting any single number.

    Motivated by an observed case: two Phase-1 runs, identical setup, differing
    only in random init, gave Slide_ID leakage of 0.336 vs 0.906.

    select_by: 'leakage' (default) picks lowest Slide_ID leakage -- leakage and
      reconstruction quality are different objectives and don't necessarily
      track each other. 'val_loss' selects by reconstruction quality instead.

    Returns best_template_state (hist_enc/hist_dec state dicts, for
    run_experiment's model_template_state) and replicate_log (per-replicate
    seed/stopped_epoch/best_val_mse/leakage_mean/leakage_chance).
    """
    import copy

    if seeds is None:
        seeds = list(range(n_replicates))

    replicate_log = []
    best_score = None
    best_state = None

    for seed in seeds:
        set_seed(seed)
        model = DualAutoencoder(**model_kwargs)
        model.to(device)
        pretrain_info = pretrain_hist_autoencoder(
            model, hist_emb_all, patch_meta, **pretrain_kwargs, device=device,
        )
        leak = evaluate_hist_leakage(model, hist_emb_all, patch_meta["slide_id"], device=device)

        entry = {
            "seed": seed,
            "stopped_epoch": pretrain_info["stopped_epoch"],
            "best_val_mse": pretrain_info["best_val_mse"],
            "leakage_mean": leak["leakage_mean"],
            "leakage_chance": leak["leakage_chance"],
        }
        replicate_log.append(entry)
        print(f"[replicate seed={seed}] stopped_epoch={entry['stopped_epoch']} "
              f"val_mse={entry['best_val_mse']:.5f} leakage={entry['leakage_mean']:.4f} "
              f"(chance={entry['leakage_chance']:.4f})")

        score = entry["leakage_mean"] if select_by == "leakage" else entry["best_val_mse"]
        if best_score is None or score < best_score:
            best_score = score
            best_state = {
                "hist_enc": copy.deepcopy(model.hist_enc.state_dict()),
                "hist_dec": copy.deepcopy(model.hist_dec.state_dict()),
            }

    print(f"Selected seed with best {select_by} = {best_score:.4f}")
    return best_state, replicate_log


def pretrain_token_once(model_kwargs, token_emb_all, cell_meta, pretrain_kwargs, device="cuda"):
    """
    Token-branch analogue of `pretrain_hist_once`: runs `pretrain_token_autoencoder`
    exactly ONCE on a throwaway model built from `model_kwargs` (which must include
    `token_dim`), and returns just the resulting token_enc/token_dec state dicts
    for reuse across many `run_token_experiment` calls via its
    `model_template_state` argument -- same rationale as the hist version: Phase 1
    doesn't depend on anything a Phase-3 sweep varies, so pretraining once and
    reusing the result keeps every sweep run's comparison clean (only Phase 3
    differs between runs, not also a different random Phase-1 outcome), and avoids
    redundantly re-pretraining an identical token encoder for every sweep point.
    """
    import copy
    template_model = DualAutoencoder(**model_kwargs)
    pretrain_token_autoencoder(template_model, token_emb_all, cell_meta, **pretrain_kwargs, device=device)
    return {
        "token_enc": copy.deepcopy(template_model.token_enc.state_dict()),
        "token_dec": copy.deepcopy(template_model.token_dec.state_dict()),
    }


def run_experiment(run_id, config, model_template_state, data, loader, hist_emb_all, patch_meta,
                    n_slides, n_svg, hist_dim, log_path, checkpoint_dir=None, seed=0, device="cuda"):
    """
    Runs ONE Phase-3 (joint/contrastive) training run starting from a shared,
    already phase-1-pretrained-and-frozen histology encoder
    (`model_template_state`, from `pretrain_hist_once`), evaluates it, and
    appends one row to a persistent CSV log at `log_path` (created on first
    call) so config + metrics survive kernel restarts.

    `config`: full parameter dict for this run, logged with a `cfg__` prefix so
      the log is self-documenting.

    `seed`: fixed (default 0) -- trans_enc/pooling/projection heads are freshly
      initialized on every call; an uncontrolled random init would confound a
      sweep the same way it did on the Phase-1 side before that was fixed. Keep
      the same across a sweep; only change deliberately (e.g. replicate-checking
      the winning config).

    Metrics logged: top1_retrieval/neighbor_tolerant_retrieval (primary
    cross-modal patch retrieval quality) and hist_leakage_mean/chance.

    Returns (metrics_dict, trained_model) -- the model isn't discarded, so a
    specific run's result can be inspected without reloading its checkpoint.
    """
    import copy, csv, time
    from pathlib import Path

    set_seed(seed)
    model = DualAutoencoder(
        n_svg=n_svg, hist_dim=hist_dim,
        latent_dim=config["latent_dim"], proj_dim=config["proj_dim"],
        n_slides=n_slides, pool_type=config["pool_type"], encoder_type=config["encoder"],
        hist_condition_on_slide=config.get("hist_condition_on_slide", True),
    )
    model.hist_enc.load_state_dict(model_template_state["hist_enc"])
    model.hist_dec.load_state_dict(model_template_state["hist_dec"])
    model.to(device)

    freeze_hist_encoder(model, freeze_decoder=config.get("freeze_decoder", True))

    train(
        model, loader, hist_emb_all, patch_meta,
        n_epochs=config["n_epochs"], lr=config["lr"], warmup_epochs=config["warmup_epochs"],
        max_contrast_weight=config["max_contrast_weight"], temperature=config["temperature"],
        weight_reconstruction=config["weight_reconstruction_by_patch_freq"],
        freeze_hist=True, device=device,
    )

    patch_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in patch_meta.items()}
    eval_results = retrieval_accuracy(model, data, hist_emb_all, patch_meta_device, device=device)
    leak = evaluate_hist_leakage(model, hist_emb_all, patch_meta["slide_id"], device=device)

    metrics = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top1_retrieval": eval_results["top1"],
        "neighbor_tolerant_retrieval": eval_results["neighbor_tolerant"],
        "hist_leakage_mean": leak["leakage_mean"],
        "hist_leakage_chance": leak["leakage_chance"],
    }
    row = {**{f"cfg__{k}": v for k, v in config.items()}, **metrics}

    # Reconcile against whatever's already on disk rather than blindly appending:
    # CONFIG gains new keys over time, so a plain DictWriter.writerow() can append
    # a row with a different field count than what's already there, corrupting the
    # file (hit this: "Expected 32 fields... saw 33"). Union-column concat instead
    # keeps the file consistent -- older rows get NaN in newer columns.
    log_path = Path(log_path)
    row_df = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, row_df], ignore_index=True, sort=False)
    else:
        combined = row_df
    combined.to_csv(log_path, index=False)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config, "metrics": metrics},
            checkpoint_dir / f"run_{run_id}.pt",
        )

    print(f"[run {run_id}] top1={metrics['top1_retrieval']:.4f} "
          f"neighbor_tol={metrics['neighbor_tolerant_retrieval']:.4f} "
          f"leakage={metrics['hist_leakage_mean']:.4f} (chance={metrics['hist_leakage_chance']:.4f})")

    return metrics, model


def run_token_experiment(run_id, config, model_template_state, data, loader, hist_emb_all, patch_meta,
                          token_emb_all, cell_meta, n_slides, n_svg, hist_dim, token_dim,
                          cell_type_labels, log_path, checkpoint_dir=None, seed=0, device="cuda",
                          leakage_subsample=20_000):
    """
    Token-branch analogue of `run_experiment`: runs ONE Phase-3 joint training run
    for the cell-level (trans<->token) contrastive branch and logs the result.

    Starts from a shared, already-pretrained token_enc/token_dec
    (`model_template_state`, from `pretrain_token_once` -- reused across sweep
    points so Phase-1 variance never confounds the comparison) plus hist_enc/dec
    at random init, immediately frozen (harmless while max_contrast_weight=0.0).

    `config`: full parameter dict for this run, logged with a `cfg__` prefix.

    `cell_type_labels`: independent cluster/cell-type labels, row-aligned to
      `data`/`cell_meta`, forwarded to `token_retrieval_accuracy` for the
      cluster-tolerant top-1 metric (strict exact-cell top-1 is harsh when many
      cells are genuine biological near-duplicates).

    `seed`: fixed (default 0) so the swept config is the only thing that differs
      between runs.

    `leakage_subsample`: subsamples `evaluate_latent_leakage`'s cross-validated
      logistic regression, too slow at full dataset size (1M+ cells).

    Metrics logged: cell_top1/cell_cluster_top1, token_leakage_mean/chance, plus
    the last epoch's local_frac_masked/contrast_local -- a collapsed run shows up
    directly in the sweep log rather than only in that run's stdout.

    Returns (metrics_dict, trained_model, history) for tracing a suspicious
    final-epoch number back to when it happened.
    """
    import copy, csv, time
    from pathlib import Path

    set_seed(seed)
    model = DualAutoencoder(
        n_svg=n_svg, hist_dim=hist_dim, token_dim=token_dim,
        latent_dim=config["latent_dim"], proj_dim=config["proj_dim"],
        n_slides=n_slides, pool_type=config["pool_type"], encoder_type=config["encoder"],
        hist_condition_on_slide=config.get("hist_condition_on_slide", True),
        token_condition_on_slide=config.get("token_condition_on_slide", True),
    )
    model.token_enc.load_state_dict(model_template_state["token_enc"])
    model.token_dec.load_state_dict(model_template_state["token_dec"])
    model.to(device)

    freeze_hist_encoder(model, freeze_decoder=True)
    if config.get("freeze_token", False):
        freeze_token_encoder(model, freeze_decoder=True)

    history = train(
        model, loader, hist_emb_all, patch_meta,
        n_epochs=config["n_epochs"], lr=config["lr"], warmup_epochs=config["warmup_epochs"],
        max_contrast_weight=config.get("max_contrast_weight", 0.0), temperature=config["temperature"],
        weight_reconstruction=config["weight_reconstruction_by_patch_freq"],
        freeze_hist=True, device=device,
        token_emb_all=token_emb_all, cell_meta=cell_meta,
        max_local_contrast_weight=config["max_local_contrast_weight"],
        local_warmup_epochs=config.get("local_warmup_epochs"),
        freeze_token=config.get("freeze_token", False),
        token_lr_mult=config.get("token_lr_mult", 1.0),
        patch_anchor_sim_threshold=config.get("patch_anchor_sim_threshold"),
        local_anchor_sim_threshold=config.get("local_anchor_sim_threshold"),
    )

    cell_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cell_meta.items()}
    eval_results = token_retrieval_accuracy(
        model, data, token_emb_all, cell_meta_device, device=device,
        cell_type_labels=cell_type_labels,
    )

    rng = np.random.default_rng(0)
    n_cells = eval_results["z_token_cell_full"].shape[0]
    sub_idx = rng.choice(n_cells, size=min(leakage_subsample, n_cells), replace=False)
    token_leak = evaluate_latent_leakage(
        eval_results["z_token_cell_full"][sub_idx], cell_meta["slide_id"].cpu().numpy()[sub_idx],
    )

    final_epoch = history[-1]
    metrics = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cell_top1_retrieval": eval_results["mean_top1"],
        "cell_cluster_top1_retrieval": eval_results["mean_cluster_top1"],
        "token_leakage_mean": token_leak["leakage_mean"],
        "token_leakage_chance": token_leak["leakage_chance"],
        "final_local_frac_masked": final_epoch["local_frac_masked"],
        "final_contrast_local": final_epoch["contrast_local"],
        "final_recon_token": final_epoch["recon_token"],
    }
    row = {**{f"cfg__{k}": v for k, v in config.items()}, **metrics}

    # Same union-column reconciliation as run_experiment -- see its docstring
    # for why a plain DictWriter.writerow() silently corrupts the file once the
    # config surface gains a new key between runs.
    log_path = Path(log_path)
    row_df = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, row_df], ignore_index=True, sort=False)
    else:
        combined = row_df
    combined.to_csv(log_path, index=False)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config, "metrics": metrics, "history": history},
            checkpoint_dir / f"run_{run_id}.pt",
        )

    collapse_flag = " [POSSIBLE COLLAPSE]" if final_epoch["local_frac_masked"] > 0.5 else ""
    print(f"[run {run_id}] cell_top1={metrics['cell_top1_retrieval']:.4f} "
          f"cluster_top1={metrics['cell_cluster_top1_retrieval']:.4f} "
          f"token_leakage={metrics['token_leakage_mean']:.4f} (chance={metrics['token_leakage_chance']:.4f}) "
          f"frac_masked={metrics['final_local_frac_masked']:.4f}{collapse_flag}")

    return metrics, model, history


# ---- leave-one-slide-out evaluation ----------------------------------------

def build_pipeline_data(f_hist_adata, f_trans_adata, f_trans_coord, CONFIG,
                         svg_mask=None, svg_scaler=None, hist_scaler=None, verbose=True,
                         f_token_adata=None, token_scaler=None, local_neighbor_radius=None,
                         max_patches_per_slide=None, oversample_slides=None):
    """
    Runs the SVG-mask -> spatial-graph -> pairing -> standardize -> patch-metadata
    -> PyG-data-and-loader pipeline, factored out so it can run on an arbitrary
    SLICE of slides -- needed for leave-one-slide-out, where this construction
    happens twice (training slides, then held-out), with the held-out call
    reusing the training call's fitted feature selection and scalers.

    Expects f_hist_adata/f_trans_adata/f_trans_coord already filtered to the
    slides this call should use, with index-setting already applied.

    svg_mask: if given, use this instead of recomputing -- REQUIRED for the
      held-out call, since recomputing on it alone would leak its own
      spatial-variability pattern into gene selection.
    svg_scaler / hist_scaler: if given, .transform() (not .fit()) with these --
      same leakage rationale.
    f_token_adata / token_scaler / local_neighbor_radius: optional token-branch
      analogue of the hist args above; None reproduces the original hist/trans-
      only behavior. Pass the training call's fitted token_scaler for the
      held-out slice, same leakage rationale.

    Returns a dict: svg_mask, svg_scaler, hist_scaler, X_svg, hist_emb,
    hist_emb_t, trans_sub, patch_meta, patch_meta_df, data, loader, n_slides,
    n_svg, hist_dim, and (only when f_token_adata is given) token_scaler,
    token_emb, token_emb_t, cell_meta, token_dim.
    """
    if svg_mask is None:
        svg_mask = get_svg_union_mask(f_trans_adata)
    if verbose:
        print(f"  SVG mask: {svg_mask.sum()} genes")

    conn = build_spatial_graph(f_trans_adata, library_key="Slide_ID",
                                n_neighs=CONFIG["n_neighs"], delaunay=False)

    keep_mask, patch_row_trans, hist_emb = build_pairing(f_trans_adata, f_hist_adata)

    # token pairing: further restrict keep_mask to cells that also have token data --
    # same rationale/order as train_autoenc_token.py's step 3b (restrict BEFORE
    # trans_sub/conn_sub/X_svg are sliced, so every downstream array stays row-aligned).
    if f_token_adata is not None:
        has_token = f_trans_adata.obs_names.isin(f_token_adata.obs_names)
        n_dropped = int(keep_mask.sum() - (keep_mask & has_token).sum())
        if verbose and n_dropped:
            print(f"  dropping {n_dropped} cells with a hist-patch match but no token match")
        # patch_row_trans is row-aligned to keep_mask's True entries -- must be narrowed
        # by the same token-based restriction or it goes stale vs. trans_sub/cell_xy
        # below (previously missed this, which crashed compute_patch_metadata's
        # pd.DataFrame(...) once a slide had a large token/trans id mismatch).
        patch_row_trans = patch_row_trans[has_token[keep_mask]]
        keep_mask = keep_mask & has_token

    trans_sub = f_trans_adata[keep_mask].copy()
    conn_sub = conn[keep_mask][:, keep_mask]
    X_svg = trans_sub.X[:, svg_mask]
    X_svg = X_svg.toarray() if sp.issparse(X_svg) else np.asarray(X_svg)

    slide_codes, slide_uniques = pd.factorize(trans_sub.obs["Slide_ID"])
    n_slides = len(slide_uniques)

    # {slide_name: target} -> {slide_code: target} against this call's own
    # slide_uniques; a name absent from this slice is just ignored, not an error.
    oversample_target = None
    if oversample_slides:
        name_to_code = {name: code for code, name in enumerate(slide_uniques)}
        oversample_target = {
            name_to_code[name]: target for name, target in oversample_slides.items() if name in name_to_code
        }

    if svg_scaler is None or hist_scaler is None:
        X_svg, hist_emb, svg_scaler, hist_scaler = standardize_modalities(X_svg, hist_emb)
    else:
        X_svg = svg_scaler.transform(X_svg).astype(np.float32)
        hist_emb = hist_scaler.transform(hist_emb).astype(np.float32)

    if f_token_adata is not None:
        token_sub = f_token_adata[trans_sub.obs_names]
        assert (token_sub.obs_names == trans_sub.obs_names).all(), "token/trans row alignment mismatch"
        token_emb = np.concatenate(
            [np.asarray(token_sub.obsm["X_token_footprint"]), np.asarray(token_sub.obsm["X_token_halo"])],
            axis=1,
        ).astype(np.float32)
        if token_scaler is None:
            token_scaler = StandardScaler()
            token_emb = token_scaler.fit_transform(token_emb).astype(np.float32)
        else:
            token_emb = token_scaler.transform(token_emb).astype(np.float32)
        if verbose:
            print(f"  token_emb: {token_emb.shape}")

    cell_xy = extract_cell_xy(trans_sub, f_trans_coord)
    patch_meta, patch_meta_df = compute_patch_metadata(
        patch_row_trans, cell_xy, f_hist_adata, neighbor_radius=CONFIG["neighbor_radius"],
        slide_categories=slide_uniques,
    )

    if f_token_adata is not None:
        cell_meta = build_cell_meta(cell_xy, slide_codes, neighbor_radius=local_neighbor_radius)

    data = build_pyg_data(X_svg, conn_sub, patch_row_trans, slide_codes)
    patch_to_nodes = build_patch_to_nodes(data.patch_row)
    loader = PatchGroupedLoader(
        data, patch_to_nodes, patch_meta["slide_id"],
        num_neighbors=CONFIG["num_neighbors"], patches_per_batch=CONFIG["patches_per_batch"],
        max_cells_per_patch=CONFIG["max_cells_per_patch"], max_patches_per_slide=max_patches_per_slide,
        oversample_target=oversample_target,
    )
    hist_emb_t = torch.tensor(hist_emb, dtype=torch.float32)

    result = {
        "svg_mask": svg_mask, "svg_scaler": svg_scaler, "hist_scaler": hist_scaler,
        "X_svg": X_svg, "hist_emb": hist_emb, "hist_emb_t": hist_emb_t,
        "trans_sub": trans_sub, "patch_meta": patch_meta, "patch_meta_df": patch_meta_df,
        "data": data, "loader": loader, "n_slides": n_slides,
        "n_svg": X_svg.shape[1], "hist_dim": hist_emb.shape[1],
    }
    if f_token_adata is not None:
        result.update({
            "token_scaler": token_scaler, "token_emb": token_emb,
            "token_emb_t": torch.tensor(token_emb, dtype=torch.float32),
            "cell_meta": cell_meta, "token_dim": token_emb.shape[1],
        })
    return result


def run_leave_one_slide_out(held_out_slide, f_hist_adata, f_trans_adata, f_trans_coord,
                             CONFIG, device, log_path=None, checkpoint_dir=None,
                             seed=0, verbose=True, run_tag=""):
    """
    run_tag: appended to run_id/checkpoint filename ("loso_{slide}{run_tag}") so
      comparison runs on the same held_out_slide don't overwrite each other.
      Empty by default.

    Trains the full pipeline on every slide EXCEPT held_out_slide, then
    evaluates cross-modal retrieval on held_out_slide alone -- tissue never
    seen at any training stage. This is the actual generalization test: every
    prior evaluation ran retrieval_accuracy on data the model had trained on.

    Reports retrieval_accuracy on both training slides (in-sample) and the
    held-out slide -- the GAP between them is what tells you whether this
    generalizes.

    Deliberately NOT reported for the held-out slide:
      - hist_leakage: no other slide within a single held-out slide to
        classify against -- meaningless here, not just hard to compute.
      - hist_dec/trans_dec reconstruction: both decoders condition on slide
        identity via a one-hot sized for the TRAINING slides only -- the
        held-out slide has no valid slot at all. retrieval_accuracy never
        touches either decoder, so it's the right metric here.

    seed: fixed (default 0) so an uncontrolled random init doesn't confound
      fold-to-fold comparison.
    """
    import time
    from pathlib import Path

    all_slides = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique().tolist())
    if held_out_slide not in all_slides:
        raise ValueError(f"held_out_slide {held_out_slide!r} not found among {all_slides}")
    train_slides = [s for s in all_slides if s != held_out_slide]

    if verbose:
        print(f"[LOSO] held out: {held_out_slide} | train: {train_slides}")

    hist_slide = f_hist_adata.obs["Slide_ID"].astype(str)
    trans_slide = f_trans_adata.obs["Slide_ID"].astype(str)
    coord_slide = f_trans_coord["Slide_ID"].astype(str)

    f_hist_train = f_hist_adata[hist_slide != held_out_slide].copy()
    f_trans_train = f_trans_adata[trans_slide != held_out_slide].copy()
    f_coord_train = f_trans_coord[coord_slide != held_out_slide]

    f_hist_held = f_hist_adata[hist_slide == held_out_slide].copy()
    f_trans_held = f_trans_adata[trans_slide == held_out_slide].copy()
    f_coord_held = f_trans_coord[coord_slide == held_out_slide]

    # gene selection computed ONCE from TRAIN-only data -- same rationale as svg_scaler/
    # hist_scaler being fit-on-train-only: computing it fresh on the held-out slide alone
    # would leak its own variability pattern into what "counts" as a selected gene.
    gene_selection = CONFIG.get("gene_selection", "svg")
    gene_mask = get_gene_mask(f_trans_train, gene_selection=gene_selection)
    if verbose:
        print(f"  gene selection ({gene_selection}): {gene_mask.sum()} genes")

    if verbose:
        print("  building training-slide pipeline data...")
    train_pipe = build_pipeline_data(f_hist_train, f_trans_train, f_coord_train, CONFIG,
                                      svg_mask=gene_mask, verbose=verbose)

    if verbose:
        print("  building held-out-slide pipeline data (reusing training svg_mask/scalers)...")
    held_pipe = build_pipeline_data(
        f_hist_held, f_trans_held, f_coord_held, CONFIG,
        svg_mask=train_pipe["svg_mask"], svg_scaler=train_pipe["svg_scaler"],
        hist_scaler=train_pipe["hist_scaler"], verbose=verbose,
    )

    model_kwargs = dict(
        n_svg=train_pipe["n_svg"], hist_dim=train_pipe["hist_dim"],
        latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
        n_slides=train_pipe["n_slides"], pool_type=CONFIG["pool_type"], encoder_type=CONFIG["encoder"],
        hist_condition_on_slide=CONFIG.get("hist_condition_on_slide", True),
    )
    pretrain_kwargs = dict(
        epochs=CONFIG["hist_pretrain_epochs"], lr=CONFIG["hist_pretrain_lr"],
        weight_decay=CONFIG["hist_pretrain_weight_decay"],
        val_frac=CONFIG["hist_pretrain_val_frac"], patience=CONFIG["hist_pretrain_patience"],
    )

    set_seed(seed)
    model = DualAutoencoder(**model_kwargs)
    pretrain_info = pretrain_hist_autoencoder(
        model, train_pipe["hist_emb_t"], train_pipe["patch_meta"],
        device=device, verbose=verbose, **pretrain_kwargs,
    )
    freeze_hist = CONFIG.get("freeze_hist", True)
    if freeze_hist:
        freeze_hist_encoder(model, freeze_decoder=CONFIG.get("freeze_decoder", True))

    train(
        model, train_pipe["loader"], train_pipe["hist_emb_t"], train_pipe["patch_meta"],
        n_epochs=CONFIG["n_epochs"], lr=CONFIG["lr"], warmup_epochs=CONFIG["warmup_epochs"],
        max_contrast_weight=CONFIG["max_contrast_weight"], temperature=CONFIG["temperature"],
        weight_reconstruction=CONFIG["weight_reconstruction_by_patch_freq"],
        freeze_hist=freeze_hist, hist_lr_mult=CONFIG.get("hist_lr_mult", 1.0),
        weight_decay=CONFIG.get("trans_weight_decay", 0.0),
        edge_dropout_p=CONFIG.get("edge_dropout_p", 0.0), gene_dropout_p=CONFIG.get("gene_dropout_p", 0.0),
        device=device,
    )

    train_patch_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in train_pipe["patch_meta"].items()}
    if verbose:
        print("  in-sample (training slides) retrieval:")
    train_eval = retrieval_accuracy(model, train_pipe["data"], train_pipe["hist_emb_t"],
                                     train_patch_meta_device, device=device)

    held_patch_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v)
                               for k, v in held_pipe["patch_meta"].items()}
    if verbose:
        print(f"  held-out ({held_out_slide}) retrieval -- the actual generalization test:")
    held_eval = retrieval_accuracy(model, held_pipe["data"], held_pipe["hist_emb_t"],
                                    held_patch_meta_device, device=device)

    # Slide_ID leakage on TRAINING slides only (meaningless for the held-out slide,
    # see docstring). Checked on trans/hist/joint so a frozen-vs-unfrozen comparison
    # shows which representation absorbs leakage. Safe to use numeric
    # patch_meta["slide_id"] here since both it and z_*_patch come from the same
    # build_pipeline_data call.
    train_slide_id_np = train_pipe["patch_meta"]["slide_id"].cpu().numpy()[train_eval["unique_patches"]]
    train_leak_trans = evaluate_latent_leakage(train_eval["z_trans_patch"], train_slide_id_np, cv=min(5, len(train_slides)))
    train_leak_hist = evaluate_latent_leakage(train_eval["z_hist_patch"], train_slide_id_np, cv=min(5, len(train_slides)))
    train_leak_joint = evaluate_latent_leakage(train_eval["z_joint_patch"], train_slide_id_np, cv=min(5, len(train_slides)))
    if verbose:
        print(f"  train-slide leakage -- trans: {train_leak_trans['leakage_mean']:.3f} | "
              f"hist: {train_leak_hist['leakage_mean']:.3f} | joint: {train_leak_joint['leakage_mean']:.3f} "
              f"(chance = {train_leak_hist['leakage_chance']:.3f})")

    # Held-out-vs-train leakage: does the held-out slide's patches separate out as
    # their own cluster, or blend in like another training slide? Needs string
    # Slide_ID labels here, not numeric slide_id codes -- those are factorized
    # separately per build_pipeline_data call, so code 0 differs between train_pipe
    # and held_pipe.
    train_slide_name = train_pipe["patch_meta_df"]["Slide_ID"].values[train_eval["unique_patches"]]
    held_slide_name = held_pipe["patch_meta_df"]["Slide_ID"].values[held_eval["unique_patches"]]
    combined_slide_name = np.concatenate([train_slide_name, held_slide_name])
    combined_leak_trans = evaluate_latent_leakage(
        np.concatenate([train_eval["z_trans_patch"], held_eval["z_trans_patch"]]), combined_slide_name)
    combined_leak_hist = evaluate_latent_leakage(
        np.concatenate([train_eval["z_hist_patch"], held_eval["z_hist_patch"]]), combined_slide_name)
    combined_leak_joint = evaluate_latent_leakage(
        np.concatenate([train_eval["z_joint_patch"], held_eval["z_joint_patch"]]), combined_slide_name)
    if verbose:
        print(f"  held-out-vs-train leakage (n_slides={len(all_slides)}) -- trans: {combined_leak_trans['leakage_mean']:.3f} | "
              f"hist: {combined_leak_hist['leakage_mean']:.3f} | joint: {combined_leak_joint['leakage_mean']:.3f} "
              f"(chance = {combined_leak_hist['leakage_chance']:.3f})")

    # Unsupervised spatial-smoothness sanity check on the held-out slide's embeddings --
    # doesn't need cross-modal matching or labels at all, just asks "does this look like
    # tissue architecture or noise" (see evaluate_latent_spatial_autocorr docstring for
    # the necessary-but-not-sufficient caveat).
    held_centroids_np = held_pipe["patch_meta"]["centroid"].cpu().numpy()[held_eval["unique_patches"]]
    moran_trans = evaluate_latent_spatial_autocorr(held_eval["z_trans_patch"], held_centroids_np)
    moran_hist = evaluate_latent_spatial_autocorr(held_eval["z_hist_patch"], held_centroids_np)
    moran_joint = evaluate_latent_spatial_autocorr(held_eval["z_joint_patch"], held_centroids_np)
    if verbose:
        print(f"  held-out spatial autocorr (mean |Moran's I|) -- trans: {moran_trans['moran_mean_I']:.4f} | "
              f"hist: {moran_hist['moran_mean_I']:.4f} | joint: {moran_joint['moran_mean_I']:.4f}")

    # Tumor-vs-normal signal -- the flip side of the leakage checks above: this dataset's
    # 5 slides are 3 tumor (CRC_*) / 2 normal (Normal_*), 1 patient each, so tumor status
    # is confounded 1:1 with Slide_ID. Some of the Slide_ID leakage measured above is
    # plausibly this real, desired biological signal rather than a technical artifact --
    # this checks that directly, on trans/hist/joint, both within the training pool
    # (slide-disjoint) and as a genuine zero-shot check on the held-out slide.
    disease_trans = evaluate_disease_signal(train_eval["z_trans_patch"], train_slide_name,
                                             held_eval["z_trans_patch"], held_slide_name)
    disease_hist = evaluate_disease_signal(train_eval["z_hist_patch"], train_slide_name,
                                            held_eval["z_hist_patch"], held_slide_name)
    disease_joint = evaluate_disease_signal(train_eval["z_joint_patch"], train_slide_name,
                                             held_eval["z_joint_patch"], held_slide_name)
    if verbose:
        print(f"  train-slide-disjoint tumor/normal accuracy ({disease_trans['n_valid_folds']} valid folds) -- "
              f"trans: {disease_trans['train_loso_acc']} | hist: {disease_hist['train_loso_acc']} | "
              f"joint: {disease_joint['train_loso_acc']}")
        print(f"  held-out ({held_out_slide}, true label={disease_trans['held_out_true_label']}) tumor/normal "
              f"accuracy -- trans: {disease_trans['held_out_acc']} | hist: {disease_hist['held_out_acc']} | "
              f"joint: {disease_joint['held_out_acc']}")

    metrics = {
        "run_id": f"loso_{held_out_slide}{run_tag}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "held_out_slide": held_out_slide,
        "train_slides": ",".join(train_slides),
        "train_top1_retrieval": train_eval["top1"],
        "train_neighbor_tolerant_retrieval": train_eval["neighbor_tolerant"],
        "train_median_rank": train_eval["median_rank"],
        "train_mrr": train_eval["mrr"],
        "train_chance_mrr": train_eval["chance_mrr"],
        "held_out_top1_retrieval": held_eval["top1"],
        "held_out_neighbor_tolerant_retrieval": held_eval["neighbor_tolerant"],
        "held_out_median_rank": held_eval["median_rank"],
        "held_out_mrr": held_eval["mrr"],
        "held_out_chance_mrr": held_eval["chance_mrr"],
        "hist_pretrain_best_val_mse": pretrain_info["best_val_mse"],
        "hist_pretrain_stopped_epoch": pretrain_info["stopped_epoch"],
        "train_trans_leakage_mean": train_leak_trans["leakage_mean"],
        "train_hist_leakage_mean": train_leak_hist["leakage_mean"],
        "train_joint_leakage_mean": train_leak_joint["leakage_mean"],
        "train_leakage_chance": train_leak_hist["leakage_chance"],
        "combined_trans_leakage_mean": combined_leak_trans["leakage_mean"],
        "combined_hist_leakage_mean": combined_leak_hist["leakage_mean"],
        "combined_joint_leakage_mean": combined_leak_joint["leakage_mean"],
        "combined_leakage_chance": combined_leak_hist["leakage_chance"],
        "held_out_moran_mean_I_trans": moran_trans["moran_mean_I"],
        "held_out_moran_mean_I_hist": moran_hist["moran_mean_I"],
        "held_out_moran_mean_I_joint": moran_joint["moran_mean_I"],
        "train_loso_disease_acc_trans": disease_trans["train_loso_acc"],
        "train_loso_disease_acc_hist": disease_hist["train_loso_acc"],
        "train_loso_disease_acc_joint": disease_joint["train_loso_acc"],
        "train_loso_disease_n_valid_folds": disease_trans["n_valid_folds"],
        "held_out_disease_acc_trans": disease_trans["held_out_acc"],
        "held_out_disease_acc_hist": disease_hist["held_out_acc"],
        "held_out_disease_acc_joint": disease_joint["held_out_acc"],
        "held_out_disease_true_label": disease_trans["held_out_true_label"],
    }

    if log_path is not None:
        row = {**{f"cfg__{k}": v for k, v in CONFIG.items()}, **metrics}
        row_df = pd.DataFrame([row])
        log_path = Path(log_path)
        if log_path.exists():
            existing = pd.read_csv(log_path)
            combined_log = pd.concat([existing, row_df], ignore_index=True, sort=False)
        else:
            combined_log = row_df
        combined_log.to_csv(log_path, index=False)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "config": CONFIG, "metrics": metrics},
            checkpoint_dir / f"loso_{held_out_slide}{run_tag}.pt",
        )
        # Joint/trans-only/hist-only patch embeddings, saved alongside the checkpoint --
        # these ARE the actual deliverable (a combined latent embedding to use downstream),
        # not just an eval byproduct, so they're persisted rather than only returned in
        # memory for the duration of the script.
        np.savez(
            checkpoint_dir / f"loso_{held_out_slide}{run_tag}_embeddings.npz",
            train_z_trans=train_eval["z_trans_patch"], train_z_hist=train_eval["z_hist_patch"],
            train_z_joint=train_eval["z_joint_patch"], train_patch_row=train_eval["unique_patches"],
            train_slide_name=train_slide_name,
            held_out_z_trans=held_eval["z_trans_patch"], held_out_z_hist=held_eval["z_hist_patch"],
            held_out_z_joint=held_eval["z_joint_patch"], held_out_patch_row=held_eval["unique_patches"],
        )

    if verbose:
        gap = metrics["train_top1_retrieval"] - metrics["held_out_top1_retrieval"]
        print(f"[LOSO {held_out_slide}] train_top1={metrics['train_top1_retrieval']:.4f} "
              f"held_out_top1={metrics['held_out_top1_retrieval']:.4f} (gap={gap:.4f})")

    return metrics, model


def run_5fold_loso(f_hist_adata, f_trans_adata, f_trans_coord, CONFIG, device,
                    log_path=None, checkpoint_dir=None, seed=0, verbose=True):
    """
    Runs run_leave_one_slide_out once per slide (full leave-one-slide-out CV).
    Expensive: trains the entire Phase 1 + Phase 3 pipeline from scratch once
    per slide (5x the cost of a single run_leave_one_slide_out call) -- run a
    single held-out slide first via run_leave_one_slide_out directly and sanity
    check it before committing to this.
    """
    all_slides = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique().tolist())
    results = []
    for held_out in all_slides:
        metrics, _ = run_leave_one_slide_out(
            held_out, f_hist_adata, f_trans_adata, f_trans_coord, CONFIG, device,
            log_path=log_path, checkpoint_dir=checkpoint_dir, seed=seed, verbose=verbose,
        )
        results.append(metrics)
    return pd.DataFrame(results)


def run_leave_one_slide_out_token(held_out_slide, f_hist_adata, f_trans_adata, f_trans_coord,
                                   f_token_adata, CONFIG, device, log_path=None, checkpoint_dir=None,
                                   seed=0, verbose=True, run_tag="", leakage_subsample=20_000):
    """
    Token-branch (cell<->token) analogue of `run_leave_one_slide_out`: does
    cell-level cross-modal retrieval generalize to a slide never seen at any
    training stage (Phase 1b token pretrain or Phase 3 joint)? Every other
    cell-level retrieval number produced so far was measured in-sample.

    Mirrors train_autoenc_token.py's regime: hist_enc/hist_dec frozen at random
    init (max_contrast_weight=0.0), token_enc/token_dec pretrained once on
    training slides only, then Phase 3 trains the cell<->token contrastive term.

    CONFIG["max_patches_per_slide"]: forwarded to the training pipeline's
      PatchGroupedLoader to cap a disproportionately large slide's per-epoch
      batch share. (Tried fully excluding such a slide instead: reduced leakage
      and roughly doubled in-sample fit, but held-out generalization stayed
      flat and the gap widened -- overfitting, not a fix. Capping keeps its
      data available, reshuffled each epoch, without letting it dominate.)

    f_hist_adata/f_trans_adata/f_trans_coord/f_token_adata: already loaded,
      concatenated, and index-set (this function only does the train/held-out
      split and pipeline construction).

    Reports cell-level top1/cluster_top1 on both training slides (in-sample)
    and the held-out slide -- the gap tells you whether this generalizes.
    token_leakage isn't reported for the held-out slide (no other slide within
    it to classify against).

    leakage_subsample: caps evaluate_latent_leakage's cost at full population
      size (1M+ cells), same as run_token_experiment.
    seed: fixed (default 0) since trans_enc/pooling/projection heads are freshly
      initialized here -- an uncontrolled random init would confound fold-to-fold
      comparison.
    """
    import time
    from pathlib import Path

    all_slides = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique().tolist())
    if held_out_slide not in all_slides:
        raise ValueError(f"held_out_slide {held_out_slide!r} not found among {all_slides}")
    train_slides = [s for s in all_slides if s != held_out_slide]

    if verbose:
        print(f"[LOSO-token] held out: {held_out_slide} | train: {train_slides}")

    hist_slide = f_hist_adata.obs["Slide_ID"].astype(str)
    trans_slide = f_trans_adata.obs["Slide_ID"].astype(str)
    coord_slide = f_trans_coord["Slide_ID"].astype(str)
    token_slide = f_token_adata.obs["Slide_ID"].astype(str)

    f_hist_train = f_hist_adata[hist_slide != held_out_slide].copy()
    f_trans_train = f_trans_adata[trans_slide != held_out_slide].copy()
    f_coord_train = f_trans_coord[coord_slide != held_out_slide]
    f_token_train = f_token_adata[token_slide != held_out_slide].copy()

    f_hist_held = f_hist_adata[hist_slide == held_out_slide].copy()
    f_trans_held = f_trans_adata[trans_slide == held_out_slide].copy()
    f_coord_held = f_trans_coord[coord_slide == held_out_slide]
    f_token_held = f_token_adata[token_slide == held_out_slide].copy()

    # gene selection computed ONCE from TRAIN-only data -- same rationale as the
    # patch-level function: computing it fresh on the held-out slide alone would leak
    # its own variability pattern into what "counts" as a selected gene.
    gene_selection = CONFIG.get("gene_selection", "svg")
    gene_mask = get_gene_mask(f_trans_train, gene_selection=gene_selection)
    if verbose:
        print(f"  gene selection ({gene_selection}): {gene_mask.sum()} genes")

    if verbose:
        print("  building training-slide pipeline data (hist + token)...")
    # max_patches_per_slide/oversample_slides only matter for the TRAINING pipeline's
    # loader (they reshape per-epoch batch share -- see PatchGroupedLoader's
    # docstring); the held-out pipe's own loader (built below) is never iterated over
    # during training or eval, only its "data" is used, so both are no-ops left at
    # CONFIG's default (None) for that call.
    train_pipe = build_pipeline_data(
        f_hist_train, f_trans_train, f_coord_train, CONFIG, svg_mask=gene_mask,
        f_token_adata=f_token_train, local_neighbor_radius=CONFIG.get("local_neighbor_radius"),
        max_patches_per_slide=CONFIG.get("max_patches_per_slide"),
        oversample_slides=CONFIG.get("oversample_slides"), verbose=verbose,
    )

    if verbose:
        print("  building held-out-slide pipeline data (reusing training svg_mask/scalers)...")
    held_pipe = build_pipeline_data(
        f_hist_held, f_trans_held, f_coord_held, CONFIG,
        svg_mask=train_pipe["svg_mask"], svg_scaler=train_pipe["svg_scaler"],
        hist_scaler=train_pipe["hist_scaler"], f_token_adata=f_token_held,
        token_scaler=train_pipe["token_scaler"], local_neighbor_radius=CONFIG.get("local_neighbor_radius"),
        verbose=verbose,
    )

    model_kwargs = dict(
        n_svg=train_pipe["n_svg"], hist_dim=train_pipe["hist_dim"], token_dim=train_pipe["token_dim"],
        latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
        n_slides=train_pipe["n_slides"], pool_type=CONFIG["pool_type"], encoder_type=CONFIG["encoder"],
        hist_condition_on_slide=CONFIG.get("hist_condition_on_slide", True),
        token_condition_on_slide=CONFIG.get("token_condition_on_slide", True),
    )
    pretrain_kwargs = dict(
        epochs=CONFIG["token_pretrain_epochs"], lr=CONFIG["token_pretrain_lr"],
        weight_decay=CONFIG["token_pretrain_weight_decay"],
        val_frac=CONFIG["token_pretrain_val_frac"], patience=CONFIG["token_pretrain_patience"],
    )

    set_seed(seed)
    model = DualAutoencoder(**model_kwargs)
    pretrain_info = pretrain_token_autoencoder(
        model, train_pipe["token_emb_t"], train_pipe["cell_meta"],
        device=device, verbose=verbose, **pretrain_kwargs,
    )

    # hist_enc/hist_dec always frozen at random init here -- harmless as long as
    # max_contrast_weight=0.0 (see train_autoenc_token.py's CONFIG comment: patch-level
    # alignment isn't being trained while this project's focus is the cell/token branch).
    freeze_hist_encoder(model, freeze_decoder=True)
    freeze_token = CONFIG.get("freeze_token", False)
    if freeze_token:
        freeze_token_encoder(model, freeze_decoder=True)

    # Fail fast with a clear error instead of an opaque async CUDA "index out of
    # bounds" assert if train_step's seed_global_ids gathers would go out of range --
    # both are indexed by batch.n_id, which ranges over [0, data.num_nodes).
    n_nodes = train_pipe["data"].num_nodes
    assert train_pipe["token_emb_t"].shape[0] == n_nodes, (
        f"token_emb_t has {train_pipe['token_emb_t'].shape[0]} rows but the training "
        f"graph has {n_nodes} nodes -- train_step's token_emb_all[seed_global_ids] "
        f"gather would go out of bounds."
    )
    assert len(train_pipe["cell_meta"]["slide_id"]) == n_nodes, (
        f"cell_meta['slide_id'] has {len(train_pipe['cell_meta']['slide_id'])} entries "
        f"but the training graph has {n_nodes} nodes -- train_step's "
        f"cell_meta['slide_id'][seed_global_ids] gather would go out of bounds."
    )
    # Permanent guard against the compute_patch_metadata slide-code bug class (fixed via
    # slide_categories=) reappearing anywhere upstream.
    n_slides = train_pipe["n_slides"]
    assert train_pipe["patch_meta"]["slide_id"].max().item() < n_slides, (
        f"patch_meta['slide_id'] has a code >= n_slides ({n_slides}) -- "
        f"hist_dec's F.one_hot(slide_code, num_classes=n_slides) would go out of bounds."
    )

    history = train(
        model, train_pipe["loader"], train_pipe["hist_emb_t"], train_pipe["patch_meta"],
        n_epochs=CONFIG["n_epochs"], lr=CONFIG["lr"], warmup_epochs=CONFIG["warmup_epochs"],
        max_contrast_weight=CONFIG.get("max_contrast_weight", 0.0), temperature=CONFIG["temperature"],
        weight_reconstruction=CONFIG["weight_reconstruction_by_patch_freq"],
        freeze_hist=True, device=device,
        token_emb_all=train_pipe["token_emb_t"], cell_meta=train_pipe["cell_meta"],
        max_local_contrast_weight=CONFIG["max_local_contrast_weight"],
        local_warmup_epochs=CONFIG.get("local_warmup_epochs"),
        freeze_token=freeze_token, token_lr_mult=CONFIG.get("token_lr_mult", 1.0),
        patch_anchor_sim_threshold=CONFIG.get("patch_anchor_sim_threshold"),
        local_anchor_sim_threshold=CONFIG.get("local_anchor_sim_threshold"),
    )

    # cluster labels (independent, pre-existing trans-only clustering) computed
    # separately per side -- codes don't need to match between train/held-out since
    # each is evaluated as its own closed candidate pool (see token_retrieval_accuracy).
    train_cluster_labels = pd.factorize(train_pipe["trans_sub"].obs["clusters"])[0]
    held_cluster_labels = pd.factorize(held_pipe["trans_sub"].obs["clusters"])[0]

    train_cell_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v)
                               for k, v in train_pipe["cell_meta"].items()}
    if verbose:
        print("  in-sample (training slides) cell<->token retrieval:")
    train_eval = token_retrieval_accuracy(
        model, train_pipe["data"], train_pipe["token_emb_t"], train_cell_meta_device,
        device=device, cell_type_labels=train_cluster_labels,
    )

    held_cell_meta_device = {k: (v.to(device) if torch.is_tensor(v) else v)
                              for k, v in held_pipe["cell_meta"].items()}
    if verbose:
        print(f"  held-out ({held_out_slide}) cell<->token retrieval -- the actual generalization test:")
    held_eval = token_retrieval_accuracy(
        model, held_pipe["data"], held_pipe["token_emb_t"], held_cell_meta_device,
        device=device, cell_type_labels=held_cluster_labels,
    )

    # token leakage on the TRAINING slides only (meaningful: n_slides-1 classes to
    # distinguish), subsampled the same way run_token_experiment does -- cross-validated
    # logistic regression is too slow at full cell-population size.
    rng = np.random.default_rng(0)
    n_train_cells = train_eval["z_token_cell_full"].shape[0]
    train_sub_idx = rng.choice(n_train_cells, size=min(leakage_subsample, n_train_cells), replace=False)
    train_leak_token = evaluate_latent_leakage(
        train_eval["z_token_cell_full"][train_sub_idx],
        train_pipe["cell_meta"]["slide_id"].cpu().numpy()[train_sub_idx],
    )
    if verbose:
        print(f"  train-slide token leakage: {train_leak_token['leakage_mean']:.3f} "
              f"(chance = {train_leak_token['leakage_chance']:.3f})")

    # held-out-vs-train leakage: does a classifier separate the held-out slide's cells
    # out as their own obviously-distinct cluster, or do they blend in like just another
    # training slide would? Needs string Slide_ID labels (not the numeric slide_id codes,
    # which are factorized separately within each build_pipeline_data call) -- trans_sub
    # carries the actual Slide_ID strings, row-aligned to z_token_cell_full the same way
    # patch_meta_df is for the patch-level version (no unique_patches indexing needed here
    # since every kept cell becomes exactly one graph node, unlike patches which can be
    # dropped/deduplicated).
    n_held_cells = held_eval["z_token_cell_full"].shape[0]
    held_sub_idx = rng.choice(n_held_cells, size=min(leakage_subsample, n_held_cells), replace=False)
    combined_z = np.concatenate([
        train_eval["z_token_cell_full"][train_sub_idx], held_eval["z_token_cell_full"][held_sub_idx],
    ])
    combined_slide_name = np.concatenate([
        train_pipe["trans_sub"].obs["Slide_ID"].values[train_sub_idx],
        held_pipe["trans_sub"].obs["Slide_ID"].values[held_sub_idx],
    ])
    combined_leak_token = evaluate_latent_leakage(combined_z, combined_slide_name)
    if verbose:
        print(f"  held-out-vs-train token leakage (n_slides={len(all_slides)}): "
              f"{combined_leak_token['leakage_mean']:.3f} (chance = {combined_leak_token['leakage_chance']:.3f})")

    metrics = {
        "run_id": f"loso_token_{held_out_slide}{run_tag}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "held_out_slide": held_out_slide,
        "train_slides": ",".join(train_slides),
        "train_cell_top1_retrieval": train_eval["mean_top1"],
        "train_cell_cluster_top1_retrieval": train_eval["mean_cluster_top1"],
        "held_out_cell_top1_retrieval": held_eval["mean_top1"],
        "held_out_cell_cluster_top1_retrieval": held_eval["mean_cluster_top1"],
        "token_pretrain_best_val_mse": pretrain_info["best_val_mse"],
        "token_pretrain_stopped_epoch": pretrain_info["stopped_epoch"],
        "train_token_leakage_mean": train_leak_token["leakage_mean"],
        "train_token_leakage_chance": train_leak_token["leakage_chance"],
        "combined_token_leakage_mean": combined_leak_token["leakage_mean"],
        "combined_token_leakage_chance": combined_leak_token["leakage_chance"],
        "final_local_frac_masked": history[-1]["local_frac_masked"],
        "final_contrast_local": history[-1]["contrast_local"],
    }

    if log_path is not None:
        row = {**{f"cfg__{k}": v for k, v in CONFIG.items()}, **metrics}
        row_df = pd.DataFrame([row])
        log_path = Path(log_path)
        if log_path.exists():
            existing = pd.read_csv(log_path)
            combined_log = pd.concat([existing, row_df], ignore_index=True, sort=False)
        else:
            combined_log = row_df
        combined_log.to_csv(log_path, index=False)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "config": CONFIG, "metrics": metrics, "history": history},
            checkpoint_dir / f"loso_token_{held_out_slide}{run_tag}.pt",
        )
        # Full (uncapped) cell embeddings, saved alongside the checkpoint as the actual
        # deliverable (same rationale as the patch-level function) -- can be large (a
        # multi-hundred-MB .npz) when the training pool is 1M+ cells.
        np.savez(
            checkpoint_dir / f"loso_token_{held_out_slide}{run_tag}_embeddings.npz",
            train_z_trans=train_eval["z_trans_cell_full"], train_z_token=train_eval["z_token_cell_full"],
            train_slide_name=train_pipe["trans_sub"].obs["Slide_ID"].values,
            held_out_z_trans=held_eval["z_trans_cell_full"], held_out_z_token=held_eval["z_token_cell_full"],
        )

    if verbose:
        gap = metrics["train_cell_top1_retrieval"] - metrics["held_out_cell_top1_retrieval"]
        cluster_gap = metrics["train_cell_cluster_top1_retrieval"] - metrics["held_out_cell_cluster_top1_retrieval"]
        print(f"[LOSO-token {held_out_slide}] train_top1={metrics['train_cell_top1_retrieval']:.4f} "
              f"held_out_top1={metrics['held_out_cell_top1_retrieval']:.4f} (gap={gap:.4f}) | "
              f"train_cluster={metrics['train_cell_cluster_top1_retrieval']:.4f} "
              f"held_out_cluster={metrics['held_out_cell_cluster_top1_retrieval']:.4f} (gap={cluster_gap:.4f})")

    return metrics, model, history


def run_5fold_loso_token(f_hist_adata, f_trans_adata, f_trans_coord, f_token_adata, CONFIG, device,
                          log_path=None, checkpoint_dir=None, seed=0, verbose=True):
    """
    Runs `run_leave_one_slide_out_token` once per slide (full leave-one-slide-out CV
    for the cell/token branch). Expensive: trains Phase 1b (token pretrain) + Phase 3
    from scratch once per slide -- run a single held-out slide first via
    `run_leave_one_slide_out_token` directly and sanity check it before committing
    to this.
    """
    all_slides = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique().tolist())
    results = []
    for held_out in all_slides:
        metrics, _, _ = run_leave_one_slide_out_token(
            held_out, f_hist_adata, f_trans_adata, f_trans_coord, f_token_adata, CONFIG, device,
            log_path=log_path, checkpoint_dir=checkpoint_dir, seed=seed, verbose=verbose,
        )
        results.append(metrics)
    return pd.DataFrame(results)