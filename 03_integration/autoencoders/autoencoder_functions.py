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
 
 
class GradientReversalFunction(torch.autograd.Function):
    """Identity on the forward pass; negates (and scales) the gradient on the
    backward pass. This is the standard trick behind domain-adversarial training
    (Ganin & Lempitsky, DANN): whatever loss is computed *after* this layer still
    gets minimized normally by the parameters after it, but the parameters
    *before* it are pushed in the OPPOSITE direction -- i.e. explicitly penalized
    for making that loss easy to minimize."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


class SlideClassifier(nn.Module):
    """
    Domain-adversarial de-confounding head for a latent (e.g. z_patch_hist).

    Deliberately a PLAIN LINEAR layer (hidden_dim=None) by default, matching the
    sklearn LogisticRegression used as the leakage diagnostic -- this way,
    adversarial training directly opposes exactly the family of function used to
    measure the leak, so a drop in this loss during training should correspond
    directly to a drop in that diagnostic's accuracy.

    Unlike reconstruction-based conditioning (which only removes the *requirement*
    to keep slide info in the latent), this actively penalizes the encoder for
    letting slide identity remain linearly predictable from its output, via the
    gradient-reversal layer above.
    """
    def __init__(self, latent_dim, n_slides, hidden_dim=None, grl_lambda=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambd=grl_lambda)
        if hidden_dim is None:
            self.net = nn.Linear(latent_dim, n_slides)
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_slides),
            )

    def set_lambda(self, lambd):
        self.grl.lambd = lambd

    def forward(self, z):
        return self.net(self.grl(z))


def domain_adv_lambda_schedule(epoch, n_epochs, gamma=5.0):
    """
    Standard DANN ramp (Ganin & Lempitsky): lambda goes from 0 -> 1 smoothly over
    training rather than jumping straight to full adversarial strength, so the
    encoder isn't fighting the domain classifier before either has learned
    anything useful yet.
    """
    p = epoch / max(n_epochs, 1)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


class DualAutoencoder(nn.Module):
    def __init__(self, n_svg, hist_dim, latent_dim=64, proj_dim=32, n_slides=5, pool_type="attention",
                 encoder_type="sage", transconv_heads=4):
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
                                    condition_on_slide=True)
        self.proj_trans = ProjectionHead(latent_dim, proj_dim)
        self.proj_hist = ProjectionHead(latent_dim, proj_dim)
 
        self.pool_type = pool_type
        self.pool = AttentionPool(latent_dim) if pool_type == "attention" else None
        self.slide_clf_hist = SlideClassifier(latent_dim, n_slides)  # domain-adversarial de-confounder


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
 
 
def build_pairing(trans_adata, hist_adata, trans_patch_col="Patch_ID", hist_patch_col="Patch_ID",
                   slide_col="Slide_ID"):
    """
    Returns:
      keep_mask: (n_trans,) bool array, nuclei with a matching histology patch
      patch_row_trans: (n_kept,) int array, which unique histology-patch row each
                        kept nucleus belongs to
      hist_emb: (n_unique_patches, hist_dim) float32 array, one row per unique patch
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
                            neighbor_radius=None):
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
    instead of the arbitrary partial patches a plain node-level NeighborLoader
    would produce. This makes patch-level pooling and patch-level contrastive
    alignment meaningful.

    SLIDE-STRATIFIED BATCHES: each batch's patches are drawn from a single slide
    only (never mixed). This is deliberate: if batches mix slides, the patch-level
    contrastive loss can satisfy InfoNCE by learning to recognize slide/scan
    identity -- a much easier, more linearly-separable shortcut than genuine
    morphology-transcriptomics correspondence -- rather than the intended
    alignment. Restricting negatives to same-slide patches removes that shortcut
    entirely, since slide identity carries no information within a single-slide
    batch. See `patch_level_infonce`'s `slide_ids` masking for a second,
    independent safeguard against this.

    Internally reuses a NeighborLoader's own subgraph-sampling machinery via its
    `collate_fn`, but we drive the seed-node groups ourselves (one group of patches
    per batch) rather than letting NeighborLoader chunk seed nodes arbitrarily.

    NOTE: this relies on PyG's `NeighborLoader`/`NodeLoader.collate_fn(index)` being
    directly callable with an arbitrary index tensor. This is standard in recent PyG
    versions; if your PyG version's internals differ, the fallback is to construct a
    fresh `NeighborLoader(data, input_nodes=seed_nodes, batch_size=len(seed_nodes))`
    per group and take its single batch instead.
    """

    def __init__(self, data, patch_to_nodes, patch_slide_id, num_neighbors=(10, 10),
                 patches_per_batch=16, max_cells_per_patch=64, shuffle=True):
        self.patch_to_nodes = patch_to_nodes
        self.patches_per_batch = patches_per_batch
        self.max_cells_per_patch = max_cells_per_patch
        self.shuffle = shuffle

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
        for plist in self.patches_by_slide.values():
            total += (len(plist) + self.patches_per_batch - 1) // self.patches_per_batch
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
                         neighbor_radius=None, weight=None):
    """
    One-to-one patch-level InfoNCE (post-pooling: each row is a genuine patch, not
    a cell). Optionally:
      - excludes spatially close, non-matching patches from the negative set
        (`neighbor_radius`) instead of forcing them apart -- nearby tissue can be
        genuinely similar in both modalities under spatial autocorrelation.
      - applies inverse-frequency weighting per patch (`weight`).
 
    MULTI-SLIDE NOTE: if a batch mixes patches from more than one slide, pass
    `slide_ids` (int tensor, one per patch, same order as `centroids`). Raw
    spatial coordinates aren't comparable across slides -- two different slides
    often share overlapping coordinate ranges, so a patch on slide A can be
    coincidentally "close" in raw distance to an unrelated patch on slide B.
    Without `slide_ids`, the neighbor mask would incorrectly treat that
    coincidence as a reason to exclude a perfectly good negative. When
    `slide_ids` is given, masking additionally requires both patches to be on
    the same slide.
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
 
    idx = torch.arange(P, device=device)
 
    log_prob_t2h = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_t2h = -log_prob_t2h[idx, idx]
 
    logits_h2t = logits.T
    log_prob_h2t = logits_h2t - torch.logsumexp(logits_h2t, dim=1, keepdim=True)
    loss_h2t = -log_prob_h2t[idx, idx]
 
    per_patch_loss = 0.5 * (loss_t2h + loss_h2t)
 
    if weight is not None:
        return (per_patch_loss * weight).sum() / weight.sum().clamp(min=1e-8)
    return per_patch_loss.mean()
 
 
def contrast_weight_schedule(epoch, warmup_epochs=5, max_weight=1.0):
    """Ramp contrastive loss weight up linearly; reconstruction stabilizes first."""
    return max_weight * min(1.0, epoch / warmup_epochs)
 
 
def train_step(model, batch, hist_emb_all, patch_meta, opt, contrast_weight,
               domain_adv_weight=0.0, temperature=0.07, weight_reconstruction=True, device="cuda"):
    model.train()
    opt.zero_grad()
 
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    n_seed = batch.local_patch_index.shape[0]
 
    z_full = model.trans_enc(x, edge_index)
    z_cells = z_full[:n_seed]  # cell-level latents for seed cells only
 
    # --- cell-level reconstruction (unchanged in spirit: still per cell) ---
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
 
    # --- pool cell latents into one patch latent per patch present in this batch ---
    local_patch_index = batch.local_patch_index.to(device)
    num_local_patches = batch.batch_patch_ids.shape[0]
    if model.pool_type == "attention":
        z_patch_trans, _attn_w = model.pool(z_cells, local_patch_index, num_local_patches)
    else:
        z_patch_trans = segment_mean_pool(z_cells, local_patch_index, num_local_patches)
 
    # --- histology side: one embedding per patch already, no duplication needed ---
    batch_patch_ids = batch.batch_patch_ids.to(device)
    batch_slide_ids = patch_meta["slide_id"][batch_patch_ids]
    hist_batch = hist_emb_all[batch_patch_ids]
    z_patch_hist = model.hist_enc(hist_batch)
    hist_recon = model.hist_dec(z_patch_hist, slide_code=batch_slide_ids)
    loss_recon_hist = F.mse_loss(hist_recon, hist_batch)
 
    # --- patch-level contrastive alignment ---
    p_trans_patch = model.proj_trans(z_patch_trans)
    p_hist_patch = model.proj_hist(z_patch_hist)
 
    centroids = patch_meta["centroid"][batch_patch_ids]
    weights = patch_meta["weight"][batch_patch_ids]
    loss_contrast = patch_level_infonce(
        p_trans_patch, p_hist_patch, centroids, slide_ids=batch_slide_ids,
        temperature=temperature, neighbor_radius=patch_meta["neighbor_radius"], weight=weights,
    )

    # --- domain-adversarial de-confounding: actively penalize z_patch_hist for
    # letting slide identity remain (linearly) predictable, via gradient reversal.
    # This is what actually removes the leak that reconstruction/conditioning alone
    # only permitted-but-didn't-require removing. ---
    if domain_adv_weight > 0:
        slide_logits = model.slide_clf_hist(z_patch_hist)
        loss_domain = F.cross_entropy(slide_logits, batch_slide_ids)
    else:
        loss_domain = torch.tensor(0.0, device=device)
 
    loss = loss_recon_trans + loss_recon_hist + contrast_weight * loss_contrast + domain_adv_weight * loss_domain
    loss.backward()
    if domain_adv_weight > 0:
        # GRL-based adversarial terms are prone to runaway logit growth (encoder and
        # classifier escalating against each other) rather than settling at the
        # intended chance-level equilibrium. Clipping caps how much any single step
        # can push things, which is what keeps this a minimax game instead of a
        # divergence.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt.step()
 
    return {
        "loss": loss.item(),
        "recon_trans": loss_recon_trans.item(),
        "recon_hist": loss_recon_hist.item(),
        "contrast": loss_contrast.item(),
        "domain": loss_domain.item() if domain_adv_weight > 0 else 0.0,
    }
 
 
def train(model, loader, hist_emb_all, patch_meta, n_epochs=30, lr=1e-3, warmup_epochs=5,
          max_contrast_weight=1.0, max_domain_adv_weight=0.0, temperature=0.07,
          weight_reconstruction=True, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist_emb_all = hist_emb_all.to(device)
    patch_meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in patch_meta.items()}
 
    for epoch in range(n_epochs):
        cw = contrast_weight_schedule(epoch, warmup_epochs, max_contrast_weight)
        dw = max_domain_adv_weight * domain_adv_lambda_schedule(epoch, n_epochs)
        model.slide_clf_hist.set_lambda(domain_adv_lambda_schedule(epoch, n_epochs))
        epoch_stats = {"loss": 0.0, "recon_trans": 0.0, "recon_hist": 0.0, "contrast": 0.0, "domain": 0.0}
        n_batches = 0
        for batch in loader:
            stats = train_step(model, batch, hist_emb_all, patch_meta, opt, cw,
                                domain_adv_weight=dw, temperature=temperature,
                                weight_reconstruction=weight_reconstruction,
                                device=device)
            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            n_batches += 1
        epoch_stats = {k: v / n_batches for k, v in epoch_stats.items()}
        print(f"epoch {epoch:03d} | contrast_w={cw:.2f} | domain_w={dw:.2f} | " +
              " | ".join(f"{k}={v:.4f}" for k, v in epoch_stats.items()))
 
 
@torch.no_grad()
def retrieval_accuracy(model, data, hist_emb_all, patch_meta, device="cuda",
                        eval_num_neighbors=(10, 10), eval_batch_size=4096):
    """
    Full-graph patch-level retrieval eval. Reports strict top-1 accuracy AND a
    neighborhood-tolerant top-1 accuracy (counts a "miss" as a hit if the predicted
    patch's centroid lies within `patch_meta['neighbor_radius']` of the true
    patch's centroid) -- since two adjacent, genuinely similar patches shouldn't be
    penalized the way a confusion with a distant, unrelated patch should be.

    NOTE: `trans_enc` is run via `encode_full_graph_batched` rather than a single
    `model.trans_enc(data.x, data.edge_index)` call. With TransformerConv this
    matters: a single full-graph pass materializes per-edge attention tensors for
    every edge in the dataset simultaneously and can OOM, even though mini-batch
    training (which only ever sees small sampled subgraphs) fits fine. Chunked
    inference reuses the same `NeighborLoader` sampling strategy as training,
    just applied to every node in manageable pieces.
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
 
    p_trans = model.proj_trans(z_patch_trans_full[unique_patches])
    p_hist = model.proj_hist(z_patch_hist)
 
    sims = p_trans @ p_hist.T
    pred = sims.argmax(dim=1)
    true_col = torch.arange(len(unique_patches), device=device)  # one-to-one, in `unique_patches` order
    top1_acc = (pred == true_col).float().mean().item()
 
    neighbor_radius = patch_meta.get("neighbor_radius")
    neighbor_acc = None
    if neighbor_radius is not None:
        centroids = patch_meta["centroid"][unique_patches].to(device)
        slide_ids = patch_meta["slide_id"][unique_patches].to(device)
        dist = (centroids[pred] - centroids).norm(dim=1)
        same_slide = slide_ids[pred] == slide_ids  # raw coords aren't comparable across slides
        neighbor_acc = ((dist < neighbor_radius) & same_slide).float().mean().item()
 
    print(f"Top-1 patch retrieval accuracy: {top1_acc:.3f} (chance = {1/len(unique_patches):.5f})")
    if neighbor_acc is not None:
        print(f"Neighborhood-tolerant top-1 accuracy (radius={neighbor_radius}): {neighbor_acc:.3f}")
 
    return {"top1": top1_acc, "neighbor_tolerant": neighbor_acc,
            "z_patch_trans_full": z_patch_trans_full.cpu().numpy()}