# import
import anndata as ad
import gc
import geopandas as gpd
import json
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
from scipy.spatial import cKDTree
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
 
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
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

    def __init__(self, in_dim, hidden_dim=256, latent_dim=64,
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

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.norm2(self.conv2(h, edge_index, edge_attr)))
        return self.out(h)
 
 
class MLPDecoder(nn.Module):
    """
    Optionally conditions on slide identity, one-hot-concatenated onto the latent
    before decoding. Conditioning alone only gives the decoder a "free" channel to
    explain away which cell/patch's slide it came from -- it removes the
    *requirement* for the encoder to keep that in its bottleneck, without directly
    penalizing it for keeping it anyway. The actual incentive to drop it comes
    from weight_decay on the encoder's optimizer: once conditioning makes slide
    identity unnecessary for reconstruction, L2 regularization pushes the
    optimizer toward not encoding it anyway. `condition_on_slide=True` (this
    class's default) is the validated setting, paired with weight_decay on
    whichever encoder feeds this decoder (trans_enc/token_enc).

    Uses LayerNorm (adopted 2026-09, previously BatchNorm1d, ablated against it
    -- see git history / ablation_studies_summary.txt) for the same reason as
    MLPEncoder: this decoder also sits on single-slide batches
    (SlideStratifiedLoader). No leakage angle here specifically (slide identity is
    already given directly via condition_on_slide), and no held-out reconstruction
    metric exists to independently check generalization the way held-out retrieval
    did for token_enc's own switch -- adopted on the strength of the encoder result
    plus the shared batch-composition mechanism, not an independently re-derived
    held-out check.
    """
    def __init__(self, latent_dim, hidden_dims, out_dim, n_slides=10, condition_on_slide=True):
        super().__init__()
        self.condition_on_slide = condition_on_slide
        in_dim = latent_dim + (n_slides if condition_on_slide else 0)
        layers, d = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
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
    """
    Uses LayerNorm (adopted 2026-09, previously BatchNorm1d, ablated directly
    against it -- see git history / ablation_studies_summary.txt). LayerNorm more
    than doubled in-sample fit (train_top1 0.090->0.196 mean across a 3-slide
    probe), cut token-branch leakage substantially (train_leak 0.809->0.658), and
    left held-out retrieval flat (0.455->0.459) -- a clean overfitting-vs-
    generalization dissociation, with the leakage reduction essentially free
    (unlike freeze_token, which cut leakage further but cost real held-out
    retrieval). Unlike trans_enc's own BatchNorm->LayerNorm switch
    (SVGGraphEncoder_TransConv), this one isn't motivated by attention -- this is
    a plain MLP; the motivation is that this encoder's input batches are always
    single-slide (SlideStratifiedLoader), and its raw output (z_token) is the
    thing directly measured for Slide_ID leakage, unlike the decoders (already
    explicitly slide-conditioned, nothing left to leak via batch statistics).
    """
    def __init__(self, in_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
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
    def __init__(self, n_svg, latent_dim=64, proj_dim=32, n_slides=5,
                 encoder_type="sage", transconv_heads=4,
                 token_dim=None, token_hidden=(512, 128),
                 token_condition_on_slide=True, trans_condition_on_slide=True):
        """
        token_dim: dimensionality of the per-cell local histology token feature
            (footprint+halo ViT token embedding, see `token_preprocessing.py`).
            None (default) builds the model with no token branch. Passing a
            value adds `token_enc`/`token_dec`/`proj_token`, plus
            `proj_trans_local` kept separate from the cell reconstruction path
            so the cell-level contrastive objective doesn't share `trans_dec`'s
            projection subspace.

            Token embeddings are per-CELL, not per-patch -- no pooling/broadcast,
            since footprint/halo aggregation already makes the correspondence
            close to one-to-one (median nucleus ~21px vs. 14px tokens).

        trans_condition_on_slide: whether `trans_dec` receives a one-hot slide
            code alongside z_trans (see `MLPDecoder`'s own docstring for the
            rationale). True (this class's prior, hardcoded behavior) is still
            the default -- pass False to ablate it, e.g. to test whether the
            latent absorbs slide-identifying technical variance without this
            escape hatch.
        """
        super().__init__()
        if encoder_type == "sage":
            self.trans_enc = SVGGraphEncoder_SAGE(n_svg, 256, latent_dim)
        elif encoder_type == "transconv":
            self.trans_enc = SVGGraphEncoder_TransConv(n_svg, 256, latent_dim, heads=transconv_heads)
        else:
            raise ValueError(f"Unsupported ecnoder_type: '{encoder_type}'. Choose from 'sage', 'transconv'.")
        self.trans_dec = MLPDecoder(latent_dim, [256, 512], n_svg, n_slides,
                                     condition_on_slide=trans_condition_on_slide)

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


def load_gene_panel(panel_path):
    """
    Loads a predesigned external gene panel (e.g. a 10x Xenium panel JSON, see
    data/combined/01_preprocessing/gene_panels/) and returns the set of gene
    SYMBOLS it targets (payload.targets[].type.data.name). Only descriptor=="gene"
    entries are kept -- these panel JSONs also list non-gene probes (Xenium's
    negative controls, descriptor=="negative_control", carry a probe name but no
    Ensembl id and can never match anything in var_names).
    """
    with open(panel_path) as f:
        panel = json.load(f)
    targets = panel["payload"]["targets"]
    return {t["type"]["data"]["name"] for t in targets if t["type"]["descriptor"] == "gene"}


def get_gene_panel_mask(trans_adata, panel_path):
    """
    Boolean mask over trans_adata.var_names, True for genes present in BOTH the
    named external panel and this dataset's own gene expression matrix. Unlike
    get_svg_union_mask/get_hvg_union_mask (which select a subset of an
    already-fixed gene universe), an external panel is defined independently of
    this data and routinely lists genes this dataset never measured or kept
    (different platform, different QC/filtering) -- the match against var_names
    can't be assumed total, so this reports the overlap rather than asserting it.
    """
    panel_genes = load_gene_panel(panel_path)
    var_names = pd.Index(trans_adata.var_names)
    mask = var_names.isin(panel_genes)
    n_found, n_panel = int(mask.sum()), len(panel_genes)
    print(f"  gene panel ({Path(panel_path).stem}): {n_found}/{n_panel} panel genes "
          f"found in this dataset's {len(var_names)} genes "
          f"({n_panel - n_found} panel genes not present in var_names).")
    return mask


def get_gene_mask(trans_adata, gene_selection="svg", gene_panel_path=None):
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

    gene_panel_path: if given, OVERRIDES gene_selection entirely -- loads an
      external, predesigned gene panel (e.g. a Xenium panel JSON) and selects
      genes present in both that panel and this dataset's var_names, instead of
      any internally-derived (svg/hvg/intersection) selection. Internal
      selection needs nothing beyond `gene_selection`; an external panel needs
      this path AND is checked against var_names, since (unlike svg/hvg) it
      isn't guaranteed to be a subset of this dataset's genes to begin with.
    """
    if gene_panel_path is not None:
        return get_gene_panel_mask(trans_adata, gene_panel_path)
    if gene_selection == "svg":
        return get_svg_union_mask(trans_adata)
    elif gene_selection == "hvg":
        return get_hvg_union_mask(trans_adata)
    elif gene_selection == "intersection":
        return get_svg_union_mask(trans_adata) & get_hvg_union_mask(trans_adata)
    else:
        raise ValueError(f"Unsupported gene_selection: {gene_selection!r}. Choose from 'svg', 'hvg', 'intersection'.")


def build_spatial_graph(trans_adata, library_key="Slide_ID", n_neighs=6, delaunay=False, percentile=95.0, grid=False):
    """Uses squidpy so that neighbors are only ever formed within a sample.

    grid: True for bin units (a regular 8um lattice) -- uses squidpy's dedicated
      grid builder (same one get_svg.py's --unit bin path uses for its own Moran's I
      graph) instead of a generic Euclidean KNN/Delaunay search, since it reasons
      about the lattice directly rather than approximating it. n_neighs=8 (not the
      default 6, tuned for hexagonal Visium spots) gives the full Moore/queen
      neighborhood (4 orthogonal + 4 diagonal) for a square lattice -- there's a
      clean distance gap before the 9th-nearest point, so 8 is unambiguous. Takes
      priority over delaunay (mutually exclusive: a unit is either irregular
      nuclei or a regular bin grid, never both).
    delaunay: True for cell/nuclei units -- matches get_svg.py's own cell-path
      choice (coord_type="generic", delaunay=True) for the same irregularly-
      spaced points, rather than an arbitrary fixed-k cutoff. Long boundary edges
      are pruned beyond the `percentile`-th distance, same as get_svg.py relies on
      implicitly via squidpy's default pruning.
    """
    if grid:
        # Grid graph within each library slide (regular lattice, e.g. 8um bins)
        sq.gr.spatial_neighbors_grid(
            trans_adata,
            spatial_key="spatial",
            library_key=library_key,
            n_neighs=n_neighs,
        )
    elif delaunay:
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
    Matches each nucleus to the histology patch it falls in. The reconstruction
    loss itself no longer weights by this (see `compute_local_density_weight`
    instead); the remaining reason this is still called is that `keep_mask`
    restricts training to cells with a histology-patch match, and
    `patch_row_trans` feeds `compute_patch_metadata`'s `slide_id`, used by
    `run_leave_one_slide_out_token`'s bounds assertion against `trans_dec`'s
    one-hot slide-coding.

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
                            trans_sub=None, slide_col="Slide_ID",
                            neighbor_radius=None, slide_categories=None):
    """
    Builds one row of metadata per unique histology patch, aligned by integer
    `patch_row` id (the same id used everywhere else in this script, i.e. row index
    into `hist_emb`). Includes:
      - centroid: mean (x, y) position of nuclei assigned to the patch
      - n_cells: number of paired nuclei
      - weight: inverse-frequency weight (mean-normalized to ~1). No longer used
        to weight the reconstruction loss (see `compute_local_density_weight`
        instead) -- kept for whoever still wants a patch-level density readout.
      - slide_id: integer code per patch, used by `run_leave_one_slide_out_token`'s
        bounds assertion against `trans_dec`'s one-hot slide-coding. IMPORTANT for
        multi-slide runs: raw spatial coordinates are per-slide, and different
        slides commonly have overlapping coordinate ranges (e.g. both indexed from
        ~(0,0)) -- `slide_id` also lets other code require "same slide AND close
        distance" instead of just "close distance", so cross-slide pairs are never
        incorrectly conflated.

    slide_categories: canonical, ordered list of slide names to code against (pass the
      same `slide_uniques` used for the cell/graph side). Previously this ran its own
      independent `pd.factorize` here, which could assign different codes than the
      cell-side factorize (different objects, different first-appearance order),
      occasionally producing a patch-level slide_id >= n_slides and crashing a decoder's
      one-hot (F.one_hot(slide_code, n_slides)) with a device-side CUDA assert. Coding
      against the shared list fixes this.
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

    return meta_tensors


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


def compute_local_density_weight(cell_xy, slide_codes, k=15):
    """
    Per-unit inverse-density weight for the reconstruction loss's
    `weight_reconstruction` branch (see `train_step`), replacing
    `compute_patch_metadata`'s patch-bucketed `weight`. Density is estimated
    directly per unit from its own local spatial neighborhood -- the distance
    to its k-th nearest neighbor WITHIN THE SAME SLIDE (raw coordinates aren't
    comparable across slides, same convention as `build_spatial_graph`/
    `local_infonce`'s `neighbor_radius` masking) -- via the standard 2D
    kNN density estimator density ~ k / (pi * r_k^2), rather than "how many
    units share this unit's 224px histology patch". Two units immediately
    adjacent to each other now get almost the same weight, even where a patch
    boundary would previously have fallen between them.

    Returns 1/density as a (n_units,) float32 tensor, normalized so the mean
    weight across all units is ~1 (same normalization convention
    `compute_patch_metadata`'s weight used).

    `cell_xy`/`slide_codes` must be row-aligned with `data.x` (same contract as
    `build_cell_meta`), since `train_step` indexes into the result using
    `batch.n_id` (global node ids) directly.
    """
    cell_xy = np.asarray(cell_xy, dtype=np.float64)
    slide_codes = np.asarray(slide_codes)
    density = np.zeros(len(cell_xy), dtype=np.float64)

    for code in np.unique(slide_codes):
        mask = slide_codes == code
        xy = cell_xy[mask]
        n = len(xy)
        k_eff = min(k, n - 1)
        if k_eff < 1:
            density[mask] = 1.0  # degenerate (single-unit slide) -- weight stays neutral
            continue
        tree = cKDTree(xy)
        dist, _ = tree.query(xy, k=k_eff + 1)  # column 0 is self, at distance 0
        r_k = np.maximum(dist[:, -1], 1e-6)  # guard against coincident points
        density[mask] = k_eff / (np.pi * r_k ** 2)

    weight = 1.0 / density
    weight = weight / weight.mean()
    return torch.tensor(weight, dtype=torch.float32)


def build_pyg_data(X_svg, conn, slide_codes):
    edge_index = torch.tensor(np.vstack(conn.tocoo().nonzero()), dtype=torch.long)
    data = Data(
        x=torch.tensor(X_svg, dtype=torch.float32),
        edge_index=edge_index,
    )
    data.slide_code = torch.tensor(slide_codes, dtype=torch.long)
    return data
 
 
class SlideStratifiedLoader:
    """
    Slide-stratified random mini-batch loader: seed cells are drawn uniformly at
    random per slide (batches are never mixed across slides -- see
    `local_infonce`'s `slide_ids` masking for why that matters), then
    expanded via `NeighborLoader`'s standard k-hop neighbor sampling for message
    passing.

    Supersedes the earlier `PatchGroupedLoader`, which grouped seed cells by
    shared histology-patch membership so a (since-removed) patch-level pooled
    contrastive loss saw complete patches per batch. That constraint no longer
    applies: the token/cell contrastive term (`local_infonce` called on
    `p_trans_local`/`p_token` in `train_step`) and the cell reconstruction loss
    are already both genuinely per-cell, needing only a per-cell density-weight
    LOOKUP (via `batch.n_id`, see `train_step`) rather than patch-complete
    batches. Plain per-slide random seed sampling is simpler, and gives more
    spatially diverse (less clustered, so less frequently excluded by
    `neighbor_radius` masking) in-batch negatives for the contrastive term
    besides.

    Reuses NeighborLoader's subgraph sampling via `collate_fn`, driving the
    seed-node groups ourselves. Relies on `collate_fn(index)` being directly
    callable with an arbitrary index tensor (standard in recent PyG); if that
    breaks, fall back to a fresh
    `NeighborLoader(data, input_nodes=seed_nodes, batch_size=len(seed_nodes))`
    per batch.
    """

    def __init__(self, data, slide_codes, num_neighbors=(10, 10), cells_per_batch=512,
                 shuffle=True, max_cells_per_slide=None, oversample_target=None):
        """
        slide_codes: LongTensor, one entry per node in `data`, aligned with
          `data.x` (i.e. `data.slide_code`) -- used to group node indices by
          slide up front.

        max_cells_per_slide / oversample_target: direct cell-count analogues of
        `PatchGroupedLoader`'s same-named patch-count parameters. max_cells_per_slide
        caps how many of a slide's own cells feed into batches per epoch (None =
        no cap); __len__/__iter__ give each slide a batch count proportional to
        its own (capped) cell count, so an outsized slide doesn't dominate every
        epoch's gradient updates -- capping truncates its (reshuffled) pool before
        chunking, so a different random subset is used each epoch and the full
        pool still gets exposure over time, just not all at once.

        oversample_target: dict[int, int], slide code -> minimum cell draws per
          epoch (WITH replacement) -- the mirror-image problem to
          max_cells_per_slide: too FEW slides of a class rather than one slide
          too big. A slide below its target gets padded via `np.random.choice`
          from its own real cells (not synthetic data). Applied after
          max_cells_per_slide. None (default) leaves a slide at its natural count.

        Both caps carry over the same rationale as before, but now operate on
        cell counts directly instead of patch counts as a proxy -- re-derive the
        numeric targets against actual per-slide cell counts rather than
        assuming the old patch-based values transfer unchanged.
        """
        self.cells_per_batch = cells_per_batch
        self.shuffle = shuffle
        self.max_cells_per_slide = max_cells_per_slide
        self.oversample_target = oversample_target or {}

        slide_codes_np = slide_codes.numpy() if torch.is_tensor(slide_codes) else np.asarray(slide_codes)
        self.nodes_by_slide = {
            int(sid): np.where(slide_codes_np == sid)[0] for sid in np.unique(slide_codes_np)
        }

        self._sampler_loader = NeighborLoader(
            data,
            num_neighbors=list(num_neighbors),
            input_nodes=torch.arange(data.num_nodes),
            batch_size=1,
        )

    def __len__(self):
        total = 0
        for sid, nodes in self.nodes_by_slide.items():
            n = len(nodes) if self.max_cells_per_slide is None else min(len(nodes), self.max_cells_per_slide)
            n = max(n, self.oversample_target.get(sid, 0))
            total += (n + self.cells_per_batch - 1) // self.cells_per_batch
        return total

    def __iter__(self):
        # build all batches slide-by-slide first (each batch confined to one slide),
        # then shuffle the batch ORDER so training still sees slides interleaved
        # across steps -- only within-batch mixing is what causes the shortcut.
        all_batches = []
        for sid, nodes in self.nodes_by_slide.items():
            nodes = nodes.copy()
            if self.shuffle:
                np.random.shuffle(nodes)
            if self.max_cells_per_slide is not None:
                nodes = nodes[:self.max_cells_per_slide]
            target = self.oversample_target.get(sid)
            if target is not None and len(nodes) < target:
                # pad up to target by resampling this slide's own cells WITH
                # replacement -- nodes is already shuffled, so every real cell is
                # used at least once before any repeats occur
                pad = np.random.choice(nodes, size=target - len(nodes), replace=True)
                nodes = np.concatenate([nodes, pad])
                if self.shuffle:
                    np.random.shuffle(nodes)
            for start in range(0, len(nodes), self.cells_per_batch):
                all_batches.append(nodes[start:start + self.cells_per_batch])

        if self.shuffle:
            random.shuffle(all_batches)

        for seed_nodes_np in all_batches:
            seed_nodes = torch.from_numpy(seed_nodes_np).long()
            batch = self._sampler_loader.collate_fn(seed_nodes)
            # NeighborLoader guarantees seed nodes are the first len(seed_nodes) rows.
            batch.n_seed = len(seed_nodes)
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

    This reuses the same sampling machinery as `SlideStratifiedLoader`/training
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
 
# ---- loss ----

def local_infonce(p_trans, p_hist, centroids, slide_ids=None, temperature=0.07,
                   neighbor_radius=None, weight=None, anchor_sim_threshold=None,
                   anchor_sim_source="projected", raw_trans=None, raw_hist=None,
                   positive_mode="exclude", return_stats=False):
    """
    One-to-one InfoNCE between a unit's (cell or bin) transcriptomics and local
    histology projections -- each row is a genuine unit, not a pooled patch (the
    old patch-level branch this once served is gone; see this project's history
    for that design). Optionally:
      - `neighbor_radius`: excludes spatially close, non-matching units from the
        negative set instead of forcing them apart -- nearby tissue can be
        genuinely similar in both modalities under spatial autocorrelation.
      - `anchor_sim_threshold`: same false-negative problem, but for units that
        are transcriptomically near-duplicate regardless of distance (e.g. two
        separate tumor nests of the same subtype) -- without it, InfoNCE pushes
        them apart via spurious non-biological signal (staining, slide identity)
        as a tie-breaker. Checked on both a transcriptomics-side and a
        histology-side similarity (source controlled by `anchor_sim_source`,
        see below), since a pair can be a near-duplicate in one modality but
        fail a strict cutoff in the other from noise/dropout alone (checking
        only one side missed real false negatives -- traced a checkerboard-like
        leiden-cluster artifact back to exactly this gap). Masking-only, never
        an added attractive term. Assumes L2-normalized inputs, so this is a
        cosine-similarity cutoff in [-1, 1] (e.g. 0.9 = near-duplicates only).
      - `anchor_sim_source`: "projected" (default) checks `p_trans`-vs-`p_trans`
        and `p_hist`-vs-`p_hist` -- the model's own LIVE, currently-training
        projections. Ablated and found to hurt more than help (see
        ablation_studies_summary.txt): the projections are exactly what the
        model is actively learning, so if it starts collapsing (embeddings
        drifting together for non-biological reasons, e.g. slide leakage), more
        pairs cross the threshold and get masked, removing the gradient that
        would correct the collapse -- self-reinforcing.
        "raw" checks `raw_trans`-vs-`raw_trans` and `raw_hist`-vs-`raw_hist`
        instead (L2-normalized here) -- fixed, pre-encoder feature vectors
        (e.g. `x[:n_seed]`/`token_x` in `train_step`) that never change during
        training, so this can't feed back into the same collapse loop. Not yet
        validated (see run_it's own gene panel-style ablation script) -- the
        raw-feature cosine-similarity distribution is unrelated to the
        projected-space one, so a threshold value tuned for "projected"
        (e.g. 0.7/0.9) does NOT carry over; re-tune against this mode's own
        `frac_masked`/`contrast_local` (printed every epoch by `train`) before
        trusting a specific value. Requires `raw_trans`/`raw_hist`.
      - `weight`: inverse-frequency weighting per unit. Not passed by
        `train_step`'s own call today (defaults to a plain per-unit mean) --
        left available for a caller that wants it.
      - `positive_mode`: what to do with pairs flagged by `neighbor_radius`/
        `anchor_sim_threshold`. "exclude" (default, every run to date):
        standard InfoNCE with false negatives masked to -inf, removed from
        the softmax denominator entirely -- the behavior described above.
        "attract": multi-positive InfoNCE (SupCon-style) instead -- flagged
        pairs are promoted to ADDITIONAL positives (alongside the true diagonal
        match) rather than merely excluded, and stay IN the softmax
        denominator (still valid candidates, just also rewarded when matched).
        Mechanically stronger and riskier than "exclude": a wrongly-flagged
        pair now actively corrupts the training signal (pulls a genuinely
        different pair together) instead of just costing one negative. Not
        yet validated -- untested combination with anchor_sim_source="raw"
        vs "projected" carries the same live-projection collapse risk noted
        above, likely more acutely (actively attracting toward a drifting
        target is a more direct collapse route than merely not repelling it).
        Changes what one-to-one retrieval metrics (token_retrieval_accuracy)
        mean -- a spatial/semantic neighbor becomes a second "correct" answer
        by design, so held-out exact-cell top-1 may drop even if the
        underlying representations are more useful; not accounted for in
        today's eval.

    `return_stats=True` also returns `frac_masked` (fraction of off-diagonal
    candidates excluded from the softmax denominator -- only nonzero in
    positive_mode="exclude", plus whatever cross-slide exclusion applies in
    either mode) and `frac_extra_positive` (fraction of off-diagonal
    candidates treated as additional positives -- only nonzero in
    positive_mode="attract"). In "exclude" mode, `frac_masked` is a collapse
    early-warning signal when `anchor_sim_source="projected"`: if the live
    projections start collapsing together, more pairs get masked, removing
    the gradient that would correct it (self-reinforcing). Watch for
    `frac_masked` climbing toward 1.0 early rather than only discovering
    `contrast_local` stuck at ~0 after a full run.

    MULTI-SLIDE NOTE: if a batch mixes slides, pass `slide_ids` -- raw coordinates
    aren't comparable across slides (overlapping ranges can make an unrelated
    cross-slide pair look spatially "close"), so masking additionally requires
    same-slide. In practice a no-op with slide-stratified batches; kept as a
    backstop.
    """
    logits = p_trans @ p_hist.T / temperature  # (P, P)
    P = logits.shape[0]
    device = logits.device
    same_slide = slide_ids.unsqueeze(0) == slide_ids.unsqueeze(1) if slide_ids is not None else None

    # Hard-exclude cross-slide pairs from the negative set entirely, in EITHER
    # positive_mode -- not a "maybe false negative" judgment call like the two
    # masks below, just a batch-composition fact (independent of neighbor_radius/
    # anchor_sim_threshold). With slide-stratified batches (see SlideStratifiedLoader)
    # this is normally a no-op, since every batch already shares a slide --
    # it's kept as a defensive backstop so the loss can never exploit slide identity
    # as a shortcut even if batch composition changes elsewhere.
    if same_slide is not None:
        cross_slide = ~same_slide
        cross_slide.fill_diagonal_(False)  # never mask out the true positive
        logits = logits.masked_fill(cross_slide, float("-inf"))

    # Candidate false negatives -- computed identically regardless of positive_mode,
    # only what happens to them (excluded below vs. promoted to positives in the
    # loss) differs.
    flagged_mask = torch.zeros(P, P, dtype=torch.bool, device=device)

    if neighbor_radius is not None:
        dist = torch.cdist(centroids, centroids)
        neighbor_mask = dist < neighbor_radius
        if same_slide is not None:
            neighbor_mask = neighbor_mask & same_slide
        neighbor_mask.fill_diagonal_(False)  # never flag the true positive
        flagged_mask = flagged_mask | neighbor_mask

    if anchor_sim_threshold is not None:
        if anchor_sim_source == "projected":
            trans_sim = p_trans @ p_trans.T  # cosine sim, p_trans is L2-normalized
            hist_sim = p_hist @ p_hist.T     # p_token for the local/cell-level call
        elif anchor_sim_source == "raw":
            if raw_trans is None or raw_hist is None:
                raise ValueError("anchor_sim_source='raw' requires raw_trans and raw_hist "
                                  "(fixed, pre-encoder feature vectors) to be passed in.")
            raw_trans_n = F.normalize(raw_trans, dim=-1)
            raw_hist_n = F.normalize(raw_hist, dim=-1)
            trans_sim = raw_trans_n @ raw_trans_n.T
            hist_sim = raw_hist_n @ raw_hist_n.T
        else:
            raise ValueError(f"Unsupported anchor_sim_source: {anchor_sim_source!r}. "
                              f"Choose from 'projected', 'raw'.")
        false_neg_mask = (trans_sim > anchor_sim_threshold) | (hist_sim > anchor_sim_threshold)
        if same_slide is not None:
            false_neg_mask = false_neg_mask & same_slide
        false_neg_mask.fill_diagonal_(False)  # never flag the true positive
        flagged_mask = flagged_mask | false_neg_mask

    if positive_mode == "exclude":
        logits = logits.masked_fill(flagged_mask, float("-inf"))
    elif positive_mode != "attract":
        raise ValueError(f"Unsupported positive_mode: {positive_mode!r}. "
                          f"Choose from 'exclude', 'attract'.")
    # "attract": flagged_mask deliberately NOT applied to logits here -- flagged
    # pairs stay in the softmax denominator as valid candidates, and are instead
    # added to the positive set in the loss below.

    if return_stats:
        off_diag = ~torch.eye(P, dtype=torch.bool, device=device)
        frac_masked = (logits[off_diag] == float("-inf")).float().mean().item()
        frac_extra_positive = flagged_mask[off_diag].float().mean().item() if positive_mode == "attract" else 0.0
        stats = {"frac_masked": frac_masked, "frac_extra_positive": frac_extra_positive, "n_candidates": P}

    idx = torch.arange(P, device=device)

    if positive_mode == "attract":
        # Multi-positive InfoNCE (SupCon-style): every flagged pair PLUS the true
        # diagonal match counts as a positive, averaged inside the log rather than
        # picking one -- denominator (logsumexp) still runs over all candidates.
        pos_mask = torch.eye(P, dtype=torch.bool, device=device) | flagged_mask
        pos_count = pos_mask.sum(dim=1).clamp(min=1)

        log_prob_t2h = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss_t2h = -(log_prob_t2h * pos_mask).sum(dim=1) / pos_count

        logits_h2t = logits.T
        pos_mask_h2t = pos_mask.T  # symmetric by construction (both source masks are symmetric)
        pos_count_h2t = pos_mask_h2t.sum(dim=1).clamp(min=1)
        log_prob_h2t = logits_h2t - torch.logsumexp(logits_h2t, dim=1, keepdim=True)
        loss_h2t = -(log_prob_h2t * pos_mask_h2t).sum(dim=1) / pos_count_h2t
    else:
        log_prob_t2h = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss_t2h = -log_prob_t2h[idx, idx]

        logits_h2t = logits.T
        log_prob_h2t = logits_h2t - torch.logsumexp(logits_h2t, dim=1, keepdim=True)
        loss_h2t = -log_prob_h2t[idx, idx]

    per_unit_loss = 0.5 * (loss_t2h + loss_h2t)

    if weight is not None:
        loss = (per_unit_loss * weight).sum() / weight.sum().clamp(min=1e-8)
    else:
        loss = per_unit_loss.mean()

    if return_stats:
        return loss, stats
    return loss
 
 
def contrast_weight_schedule(epoch, warmup_epochs=5, max_weight=1.0):
    """Ramp contrastive loss weight up linearly; reconstruction stabilizes first."""
    return max_weight * min(1.0, epoch / warmup_epochs)


def pretrain_token_autoencoder(model, token_emb_all, cell_meta, epochs=100, batch_size=1024,
                                lr=1e-3, weight_decay=1e-5, val_frac=0.1, patience=15,
                                device="cuda", verbose=True, seed=0):
    """
    Phase-1 pretraining for the token branch (`token_enc`/`token_dec`): no graph
    machinery, no contrastive loss, no trans branch -- pure MSE, AdamW+weight_decay,
    ReduceLROnPlateau, a genuine held-out validation split, early stopping,
    restoring the best-val-loss checkpoint at the end. Batched over cells via
    `cell_meta["slide_id"]`.

    Requires `model.token_enc`/`model.token_dec` to exist, i.e. `DualAutoencoder`
    was constructed with `token_dim` set.

    Meant to run before `train()`. Follow with `freeze_token_encoder(model)` and
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

    # token_emb_all stays on CPU -- moving the WHOLE training-pool matrix to GPU
    # upfront fit at cell scale (~2.1M rows) but OOMs at bin scale (up to ~4.79M
    # rows -- a single allocation exceeding a 40GB GPU's entire capacity, confirmed
    # directly from a bin-level LOSO run's CUDA OOM traceback). Only each batch is
    # moved to GPU below, inside the loop, matching how the main Phase-3 loop
    # (SlideStratifiedLoader) already handles data too large to keep resident on GPU.
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
            x = token_emb_all[idx].to(device)
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
            x = token_emb_all[val_idx].to(device)
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
    Freezes `token_enc` (and, by default, `token_dec`) in place: sets
    requires_grad=False so `train()`'s optimizer (which filters on
    requires_grad) never updates them, and marks them .eval() so BatchNorm
    running stats stop updating too (requires_grad=False alone does NOT freeze
    those). Call after `pretrain_token_autoencoder`, before the joint
    `train(..., freeze_token=True)` phase.
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


def train_step(model, batch, density_weight_all, opt, temperature=0.07, weight_reconstruction=True,
               edge_dropout_p=0.0, gene_dropout_p=0.0, device="cuda",
               token_emb_all=None, cell_meta=None, local_contrast_weight=0.0, freeze_token=False,
               local_anchor_sim_threshold=None, local_anchor_sim_source="projected",
               local_positive_mode="exclude"):
    """
    density_weight_all: per-unit inverse-density weight (see
    `compute_local_density_weight`), used by `weight_reconstruction` to weight
    the CELL reconstruction loss -- looked up per seed cell via `batch.n_id`
    (global node ids, same lookup token_emb_all/cell_meta already use below),
    not tied to any histology-patch grouping.

    token_emb_all / cell_meta: token_emb_all and cell_meta["xy"]/["slide_id"]
    must be row-aligned with data.x -- seed cells are looked up via
    batch.n_id[:n_seed] (NeighborLoader puts seed nodes first).

    local_anchor_sim_threshold / local_anchor_sim_source: forwarded to
    local_infonce -- cosine-similarity cutoff excluding transcriptomically
    near-duplicate, non-matching cells from hard negatives (see local_infonce's
    docstring). When source=="raw", the clean x[:n_seed]/token_x computed below
    are passed as the fixed comparison vectors -- no extra work needed here,
    they already exist for reconstruction/the token branch.

    local_positive_mode: forwarded to local_infonce -- "exclude" (default)
    masks flagged (neighbor/semantic) pairs out of the negative set; "attract"
    promotes them to additional positives instead (multi-positive InfoNCE).
    See local_infonce's own docstring for the tradeoffs.

    edge_dropout_p / gene_dropout_p: denoising-style corruption applied only to
    what trans_enc sees (reconstruction target stays clean). Targets two Slide_ID
    leakage channels: edge_dropout_p attacks local graph structure (slide-specific
    cell density), gene_dropout_p attacks over-reliance on slide-discriminative
    genes. Neither touches leakage that reflects genuine biology (Slide_ID is
    confounded 1:1 with patient/tumor status here -- no input corruption can
    separate real biological signal from technical at this sample size).
    """
    model.train()
    if freeze_token and model.token_enc is not None:
        # model.train() re-enabled BatchNorm on every submodule -- pin back to eval
        # only the ones actually frozen (checked directly, since freeze_decoder=False
        # can leave token_dec trainable while token_enc is frozen).
        if not any(p.requires_grad for p in model.token_enc.parameters()):
            model.token_enc.eval()
        if not any(p.requires_grad for p in model.token_dec.parameters()):
            model.token_dec.eval()
    opt.zero_grad()

    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    n_seed = batch.n_seed

    x_in = F.dropout(x, p=gene_dropout_p, training=True) if gene_dropout_p > 0 else x
    edge_index_in = dropout_edge(edge_index, p=edge_dropout_p, training=True)[0] if edge_dropout_p > 0 else edge_index

    z_full = model.trans_enc(x_in, edge_index_in)
    z_cells = z_full[:n_seed]  # cell-level latents for seed cells only

    # --- cell-level reconstruction ---
    # target is the CLEAN x, not x_in -- see docstring above
    slide_code = batch.slide_code[:n_seed].to(device)
    x_recon = model.trans_dec(z_cells, slide_code)
    recon_per_cell = F.mse_loss(x_recon, x[:n_seed], reduction="none").mean(dim=1)

    if weight_reconstruction:
        cell_weight = density_weight_all[batch.n_id[:n_seed].to(device)]
        loss_recon_trans = (recon_per_cell * cell_weight).sum() / cell_weight.sum().clamp(min=1e-8)
    else:
        loss_recon_trans = recon_per_cell.mean()

    # --- token side: CELL-level local branch, genuinely one-to-one, no pooling ---
    # Always computed (contributes zero gradient to token_enc/token_dec once frozen)
    # whenever the model has a token branch and token data was supplied.
    if model.token_enc is not None and token_emb_all is not None:
        seed_global_ids = batch.n_id[:n_seed].to(device)
        # token_emb_all stays on CPU (see train()'s comment) -- index with the
        # still-CPU id tensor and move only this batch's slice to device.
        token_x = token_emb_all[batch.n_id[:n_seed]].to(device)
        cell_slide_ids = cell_meta["slide_id"][seed_global_ids]

        z_token_cells = model.token_enc(token_x)
        token_recon = model.token_dec(z_token_cells, slide_code=cell_slide_ids)
        loss_recon_token = F.mse_loss(token_recon, token_x)

        p_trans_local = model.proj_trans_local(z_cells)
        p_token = model.proj_token(z_token_cells)
        cell_xy = cell_meta["xy"][seed_global_ids]
        loss_contrast_local, local_mask_stats = local_infonce(
            p_trans_local, p_token, cell_xy, slide_ids=cell_slide_ids,
            temperature=temperature, neighbor_radius=cell_meta["neighbor_radius"],
            anchor_sim_threshold=local_anchor_sim_threshold,
            anchor_sim_source=local_anchor_sim_source,
            raw_trans=x[:n_seed], raw_hist=token_x,
            positive_mode=local_positive_mode,
            return_stats=True,
        )
        local_frac_masked = local_mask_stats["frac_masked"]
        local_frac_extra_positive = local_mask_stats["frac_extra_positive"]
    else:
        loss_recon_token = z_cells.new_tensor(0.0)
        loss_contrast_local = z_cells.new_tensor(0.0)
        local_frac_masked = 0.0
        local_frac_extra_positive = 0.0

    loss = loss_recon_trans + loss_recon_token + local_contrast_weight * loss_contrast_local
    loss.backward()
    opt.step()

    return {
        "loss": loss.item(),
        "recon_trans": loss_recon_trans.item(),
        "recon_token": loss_recon_token.item(),
        "contrast_local": loss_contrast_local.item(),
        "local_frac_masked": local_frac_masked,
        "local_frac_extra_positive": local_frac_extra_positive,
    }


def train(model, loader, density_weight_all, n_epochs=30, lr=1e-3, warmup_epochs=5, temperature=0.07,
          weight_reconstruction=True,
          weight_decay=0.0, edge_dropout_p=0.0, gene_dropout_p=0.0, device="cuda",
          token_emb_all=None, cell_meta=None, max_local_contrast_weight=0.0,
          local_warmup_epochs=None, freeze_token=False, token_lr_mult=1.0,
          local_anchor_sim_threshold=None, local_anchor_sim_source="projected",
          local_positive_mode="exclude"):
    """
    density_weight_all: per-unit inverse-density weight from
      `compute_local_density_weight`, required (not None) when
      weight_reconstruction=True -- see `train_step`'s docstring.

    token_emb_all / cell_meta: see `build_cell_meta`. None (default) trains as
      before, including for models with no token branch.

    local_anchor_sim_threshold / local_anchor_sim_source: forwarded to
      `local_infonce` -- cosine-similarity cutoff excluding transcriptomically
      near-duplicate, non-matching cells from hard negatives (same
      false-negative protection as `neighbor_radius`, gated on expression
      similarity instead of distance). threshold default None (off); source
      default "projected" (see local_infonce's own docstring for "raw").

    local_positive_mode: forwarded to `local_infonce` -- "exclude" (default,
      every run to date) or "attract" (multi-positive InfoNCE, see
      local_infonce's own docstring for the tradeoffs).

    max_local_contrast_weight / local_warmup_epochs: ramp schedule for the
      cell<->token contrastive term. local_warmup_epochs defaults to
      warmup_epochs if not given.

    freeze_token / token_lr_mult: gentler LR on an already-pretrained token
      branch (pretrain_token_autoencoder + freeze_token_encoder beforehand).

    weight_decay: applied to every trainable param group.

    edge_dropout_p / gene_dropout_p: forwarded to `train_step` (denoising
      mechanism, see its docstring).
    """
    model.to(device)
    token_param_ids = set()
    if model.token_enc is not None:
        token_param_ids = {id(p) for p in list(model.token_enc.parameters()) + list(model.token_dec.parameters())}
    token_params = [p for p in model.parameters() if p.requires_grad and id(p) in token_param_ids]
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in token_param_ids]
    param_groups = [{"params": other_params, "lr": lr, "weight_decay": weight_decay}]
    if token_params:
        param_groups.append({"params": token_params, "lr": lr * token_lr_mult, "weight_decay": weight_decay})
    # AdamW, not Adam: decoupled weight decay, matching pretrain_token_autoencoder's
    # own optimizer -- plain Adam's weight_decay is an L2 penalty added to the
    # gradient, which interacts inconsistently with Adam's per-parameter adaptive
    # learning rates (the reason AdamW exists at all). Identical to the prior Adam
    # at weight_decay=0.0 (every run to date), since the two only differ once decay
    # is actually nonzero.
    opt = torch.optim.AdamW(param_groups)
    if weight_reconstruction:
        if density_weight_all is None:
            raise ValueError("weight_reconstruction=True requires density_weight_all (see compute_local_density_weight)")
        # Small (n_units,) tensor -- unlike token_emb_all, no per-batch slicing needed.
        density_weight_all = density_weight_all.to(device)

    use_tokens = model.token_enc is not None and token_emb_all is not None
    if use_tokens:
        # token_emb_all deliberately kept on CPU -- moving the whole training-pool
        # token matrix to GPU upfront fits at cell scale but OOMs at bin scale (up
        # to ~4.79M rows, a single allocation exceeding a 40GB GPU's entire capacity,
        # confirmed directly from a bin-level LOSO run's CUDA OOM traceback).
        # train_step indexes each batch's slice directly, same per-batch pattern
        # already used in pretrain_token_autoencoder.
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
        lcw = contrast_weight_schedule(epoch, local_warmup_epochs, max_local_contrast_weight) if use_tokens else 0.0
        epoch_stats = {"loss": 0.0, "recon_trans": 0.0, "recon_token": 0.0,
                        "contrast_local": 0.0, "local_frac_masked": 0.0,
                        "local_frac_extra_positive": 0.0}
        n_batches = 0
        for batch in loader:
            stats = train_step(model, batch, density_weight_all, opt,
                                temperature=temperature,
                                weight_reconstruction=weight_reconstruction,
                                edge_dropout_p=edge_dropout_p, gene_dropout_p=gene_dropout_p,
                                device=device,
                                token_emb_all=token_emb_all if use_tokens else None,
                                cell_meta=cell_meta if use_tokens else None,
                                local_contrast_weight=lcw, freeze_token=freeze_token,
                                local_anchor_sim_threshold=local_anchor_sim_threshold,
                                local_anchor_sim_source=local_anchor_sim_source,
                                local_positive_mode=local_positive_mode)
            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            n_batches += 1
        epoch_stats = {k: v / n_batches for k, v in epoch_stats.items()}
        print(f"epoch {epoch:03d}" + (f" | local_w={lcw:.2f}" if use_tokens else "") + " | " +
              " | ".join(f"{k}={v:.4f}" for k, v in epoch_stats.items()))
        history.append({"epoch": epoch, "local_w": lcw, **epoch_stats})

    return history


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
    slide_id_all = cell_meta["slide_id"].to(device)
    xy_all = cell_meta["xy"].to(device)
    neighbor_radius = cell_meta.get("neighbor_radius")
    if cell_type_labels is not None:
        cell_type_labels = torch.as_tensor(cell_type_labels, device=device)

    # token_emb_all stays on CPU and is encoded in chunks -- same rationale as
    # train()/train_step: a wholesale .to(device) on the full training-pool matrix
    # OOMs at bin scale. z_token_full itself (latent-dim wide, not token-dim wide)
    # is small enough to keep resident on device for the rest of this function.
    # (this whole function is already @torch.no_grad()-decorated)
    z_token_chunks = []
    for start in range(0, token_emb_all.shape[0], eval_batch_size):
        chunk = token_emb_all[start:start + eval_batch_size].to(device)
        z_token_chunks.append(model.token_enc(chunk))
    z_token_full = torch.cat(z_token_chunks, dim=0)

    rng = np.random.default_rng(seed)
    results_per_slide = {}
    all_top1, all_n, all_cluster_top1, all_neighbor_acc = [], [], [], []
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
            all_neighbor_acc.append(neighbor_acc)

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
    mean_neighbor_tolerant = None
    if all_neighbor_acc:
        mean_neighbor_tolerant = float(np.average(all_neighbor_acc, weights=all_n))
        print(f"Cell<->token retrieval, n-weighted mean neighbor-tolerant top-1: {mean_neighbor_tolerant:.3f}")
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

    return {"mean_top1": mean_top1, "mean_cluster_top1": mean_cluster_top1,
            "mean_neighbor_tolerant": mean_neighbor_tolerant, "per_slide": results_per_slide,
            "z_trans_cell_full": z_cells_full.cpu().numpy(),
            "z_token_cell_full": z_token_full.cpu().numpy()}


# ---- experiment harness: reproducible sweeps with a persistent, inspectable log ----

def evaluate_latent_leakage(z, slide_id, cv=5, max_iter=5000):
    """
    Slide_ID leakage diagnostic on an ALREADY-COMPUTED embedding `z` (n_cells x dim,
    numpy array or tensor): checks how well a plain logistic regression can recover
    Slide_ID from it via cross-validated accuracy (chance = 1/n_slides).

    Use this directly on whatever embedding you actually intend to use downstream --
    e.g. `token_retrieval_accuracy`'s `z_trans_cell_full` / `z_token_cell_full`.
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


def refine_domains(cluster_labels, conn, mode="hard"):
    """
    Spatial smoothing of cluster labels into contiguous domains, applied as a
    post-processing step over an EXISTING sparse spatial connectivity matrix --
    reuses whatever `build_spatial_graph` already built for a slide (Delaunay for
    nuclei, grid for bins), rather than constructing a second graph purely for
    this. Plain clustering on z (e.g. Leiden) groups units by embedding similarity
    alone, so two feature-identical but spatially disjoint regions can end up in
    the same cluster; this reassigns labels using only spatial neighbors, which a
    plain clustering step never looks at.

    Call this on ONE slide's units at a time (same cross-slide-coordinate caveat
    as evaluate_latent_spatial_autocorr/retrieval_accuracy's neighbor_tolerant
    check) -- `conn` must already be restricted to that slide's own edges.

    mode="hard": unconditional mode over each unit's spatial neighbors --
      every unit is reassigned to whatever label is most common among its
      neighbors, every call.
    mode="spagcn": SpaGCN's own conditional rule (Hu et al., "SpaGCN", Nature
      Methods 2021; vendored in this repo under 02_baselines/spagcn/) -- only
      reassigns a unit when its current label is a clear minority among
      neighbors (held by fewer than half) AND some other label is a clear
      majority (held by more than half); otherwise the label is left alone.
      SpaGCN's original implementation assumes a fixed Visium hex (6) or old-ST
      square (4) neighborhood rebuilt from a dense distance matrix -- this
      version instead uses however many neighbors `conn` actually gives each
      unit (variable for Delaunay, fixed at 8 for a bin grid), so it isn't a
      literal drop-in of their function, just the same rule.

    Ties (no strict majority) leave the unit's own label unchanged in both modes.
    """
    if torch.is_tensor(cluster_labels):
        cluster_labels = cluster_labels.detach().cpu().numpy()
    labels = np.asarray(cluster_labels)
    conn = conn.tocsr()
    refined = labels.copy()

    for i in range(len(labels)):
        nbrs = conn.indices[conn.indptr[i]:conn.indptr[i + 1]]
        if len(nbrs) == 0:
            continue
        vals, counts = np.unique(labels[nbrs], return_counts=True)
        maj_count = counts.max()
        is_tie = (counts == maj_count).sum() > 1
        maj_label = vals[counts.argmax()]

        if mode == "hard":
            if not is_tie:
                refined[i] = maj_label
        elif mode == "spagcn":
            self_count = counts[vals == labels[i]].sum() if labels[i] in vals else 0
            if self_count < len(nbrs) / 2 and maj_count > len(nbrs) / 2:
                refined[i] = maj_label
        else:
            raise ValueError(f"Unsupported mode: {mode!r}. Choose from 'hard', 'spagcn'.")

    return refined


def evaluate_disease_signal(z_train, train_slide_name, z_held, held_slide_name,
                             tumor_prefix="CRC", max_iter=5000, class_weight=None):
    """
    Tests whether an embedding captures real tumor-vs-normal biological signal --
    since tumor status is confounded 1:1 with Slide_ID/patient in this cohort, some
    of the Slide_ID leakage measured elsewhere (evaluate_latent_leakage) is plausibly
    real biology, not a technical artifact to suppress.

    Two checks, both genuinely slide-disjoint (unlike a naive pooled cross_val_score):
      - train_loso_acc/train_loso_f1: leave-one-TRAINING-slide-out (sklearn
        LeaveOneGroupOut) -- does the classifier separate tumor/normal on a
        training slide it wasn't fit on? Degenerate folds (down to one class)
        are skipped; n_valid_folds reports how many contributed. f1 (macro) is
        reported alongside accuracy because these folds are exactly as class-imbalanced
        as the slide split itself -- e.g. 1 normal vs. 7 tumor training slides
        once a normal slide is held out -- so accuracy alone can look fine
        while the minority class's own performance doesn't.
      - held_out_acc: fits on all training-slide patches, predicts on the TRUE
        held-out slide (never seen). Its true label is constant, so this is "fraction
        classified as its own known label," not two-sided accuracy -- read alongside
        train_loso_acc, not alone. No held_out_f1: F1 needs both classes present in
        y_true to mean anything, and a single held-out slide is only ever one class.

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
    from sklearn.metrics import f1_score

    if torch.is_tensor(z_train):
        z_train = z_train.detach().cpu().numpy()
    if torch.is_tensor(z_held):
        z_held = z_held.detach().cpu().numpy()

    groups = np.asarray(train_slide_name)
    y_train = np.array([1 if str(s).startswith(tumor_prefix) else 0 for s in train_slide_name])
    y_held = np.array([1 if str(s).startswith(tumor_prefix) else 0 for s in held_slide_name])

    # F1 alongside accuracy: train_loso's classes are as imbalanced as the slide split
    # itself (e.g. 1 normal vs 7 tumor training slides once one normal slide is held
    # out) -- accuracy alone can look fine while the minority class's own F1 doesn't.
    fold_scores, fold_f1s = [], []
    if len(np.unique(groups)) >= 2:
        for tr_idx, te_idx in LeaveOneGroupOut().split(z_train, y_train, groups=groups):
            if len(np.unique(y_train[tr_idx])) < 2:
                continue  # degenerate fold: training portion left with only one class
            clf_fold = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
            clf_fold.fit(z_train[tr_idx], y_train[tr_idx])
            fold_scores.append(clf_fold.score(z_train[te_idx], y_train[te_idx]))
            fold_f1s.append(f1_score(y_train[te_idx], clf_fold.predict(z_train[te_idx]),
                                      labels=[0, 1], average="macro", zero_division=0))
    train_loso_acc = float(np.mean(fold_scores)) if fold_scores else None
    train_loso_f1 = float(np.mean(fold_f1s)) if fold_f1s else None

    held_out_acc = None
    if len(y_held) and len(np.unique(y_train)) >= 2:
        clf = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
        clf.fit(z_train, y_train)
        held_out_acc = float((clf.predict(z_held) == y_held).mean())

    return {
        "train_loso_acc": train_loso_acc,
        "train_loso_f1": train_loso_f1,
        "n_valid_folds": len(fold_scores),
        "held_out_acc": held_out_acc,
        "held_out_true_label": int(y_held[0]) if len(y_held) else None,
    }



def pretrain_token_once(model_kwargs, token_emb_all, cell_meta, pretrain_kwargs, device="cuda"):
    """
    Runs `pretrain_token_autoencoder` exactly ONCE on a throwaway model built
    from `model_kwargs` (which must include `token_dim`), and returns just the
    resulting token_enc/token_dec state dicts for reuse across many
    `run_token_experiment` calls via its `model_template_state` argument: Phase 1
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


def run_token_experiment(run_id, config, model_template_state, data, loader, density_weight_all,
                          token_emb_all, cell_meta, n_slides, n_svg, token_dim,
                          cell_type_labels, log_path, checkpoint_dir=None, seed=0, device="cuda",
                          leakage_subsample=20_000):
    """
    Runs ONE Phase-3 joint training run for the cell-level (trans<->token)
    contrastive branch and logs the result.

    Starts from a shared, already-pretrained token_enc/token_dec
    (`model_template_state`, from `pretrain_token_once` -- reused across sweep
    points so Phase-1 variance never confounds the comparison).

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
        n_svg=n_svg, token_dim=token_dim,
        latent_dim=config["latent_dim"], proj_dim=config["proj_dim"],
        n_slides=n_slides, encoder_type=config["encoder"],
        token_condition_on_slide=config.get("token_condition_on_slide", True),
        trans_condition_on_slide=config.get("trans_condition_on_slide", True),
    )
    model.token_enc.load_state_dict(model_template_state["token_enc"])
    model.token_dec.load_state_dict(model_template_state["token_dec"])
    model.to(device)

    if config.get("freeze_token", False):
        freeze_token_encoder(model, freeze_decoder=True)

    history = train(
        model, loader, density_weight_all,
        n_epochs=config["n_epochs"], lr=config["lr"], warmup_epochs=config["warmup_epochs"],
        temperature=config["temperature"],
        weight_reconstruction=config["weight_reconstruction_by_patch_freq"],
        device=device,
        token_emb_all=token_emb_all, cell_meta=cell_meta,
        max_local_contrast_weight=config["max_local_contrast_weight"],
        local_warmup_epochs=config.get("local_warmup_epochs"),
        freeze_token=config.get("freeze_token", False),
        token_lr_mult=config.get("token_lr_mult", 1.0),
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
                         max_cells_per_slide=None, oversample_slides=None):
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
    hist_emb_t, trans_sub, patch_meta, density_weight, data, loader, n_slides,
    n_svg, hist_dim, and (only when f_token_adata is given) token_scaler,
    token_emb, token_emb_t, cell_meta, token_dim.
    """
    if svg_mask is None:
        svg_mask = get_svg_union_mask(f_trans_adata)
    if verbose:
        print(f"  SVG mask: {svg_mask.sum()} genes")

    conn = build_spatial_graph(f_trans_adata, library_key="Slide_ID",
                                n_neighs=CONFIG["n_neighs"],
                                delaunay=CONFIG.get("spatial_graph_delaunay", True),
                                grid=CONFIG.get("spatial_graph_grid", False))

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
    # patch_meta is kept only for run_leave_one_slide_out_token's slide_id bounds
    # assertion -- not the reconstruction loss's weight source anymore (see below).
    patch_meta = compute_patch_metadata(
        patch_row_trans, cell_xy, f_hist_adata, neighbor_radius=CONFIG["neighbor_radius"],
        slide_categories=slide_uniques,
    )
    # Per-unit inverse-density weight for weight_reconstruction (see train_step) --
    # supersedes patch_meta["weight"] as the reconstruction loss's actual weight
    # source (see compute_local_density_weight's docstring for why).
    density_weight = compute_local_density_weight(cell_xy, slide_codes)

    if f_token_adata is not None:
        cell_meta = build_cell_meta(cell_xy, slide_codes, neighbor_radius=local_neighbor_radius)

    data = build_pyg_data(X_svg, conn_sub, slide_codes)
    loader = SlideStratifiedLoader(
        data, data.slide_code,
        num_neighbors=CONFIG["num_neighbors"], cells_per_batch=CONFIG["cells_per_batch"],
        max_cells_per_slide=max_cells_per_slide, oversample_target=oversample_target,
    )
    hist_emb_t = torch.tensor(hist_emb, dtype=torch.float32)

    result = {
        "svg_mask": svg_mask, "svg_scaler": svg_scaler, "hist_scaler": hist_scaler,
        "X_svg": X_svg, "hist_emb": hist_emb, "hist_emb_t": hist_emb_t,
        "trans_sub": trans_sub, "patch_meta": patch_meta,
        "density_weight": density_weight,
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


def run_leave_one_slide_out_token(held_out_slide, f_hist_adata, f_trans_adata, f_trans_coord,
                                   f_token_adata, CONFIG, device, log_path=None, checkpoint_dir=None,
                                   seed=0, verbose=True, run_tag="", leakage_subsample=20_000):
    """
    Does cell-level cross-modal (trans<->token) retrieval generalize to a slide
    never seen at any training stage (Phase 1b token pretrain or Phase 3
    joint)? Every other cell-level retrieval number produced so far was
    measured in-sample.

    Mirrors train_autoenc_token.py's regime: token_enc/token_dec pretrained
    once on training slides only, then Phase 3 trains the cell<->token
    contrastive term.

    CONFIG["max_cells_per_slide"]: forwarded to the training pipeline's
      SlideStratifiedLoader to cap a disproportionately large slide's per-epoch
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
    token_slide = f_token_adata.obs["Slide_ID"].astype(str)

    f_hist_train = f_hist_adata[hist_slide != held_out_slide].copy()
    f_trans_train = f_trans_adata[trans_slide != held_out_slide].copy()
    f_token_train = f_token_adata[token_slide != held_out_slide].copy()

    f_hist_held = f_hist_adata[hist_slide == held_out_slide].copy()
    f_trans_held = f_trans_adata[trans_slide == held_out_slide].copy()
    f_token_held = f_token_adata[token_slide == held_out_slide].copy()

    # The full, un-split f_hist_adata/f_trans_adata/f_token_adata are never read
    # again below -- only the six train/held splits above are. At bin scale
    # (~4.8M rows, 1536-dim token embeddings) each one is tens of GB, and keeping
    # the originals resident alongside their own train+held copies nearly triples
    # peak memory for no reason (confirmed directly from a bin-level LOSO run's
    # CPU-RAM ArrayMemoryError on the f_token_train copy). Free them immediately.
    del f_hist_adata, f_trans_adata, f_token_adata, hist_slide, trans_slide, token_slide
    gc.collect()

    # f_trans_coord is None for bins (no nuclei_geometry.parquet side-car -- bins
    # carry obsm['spatial'] directly, see extract_cell_xy's own None-safe priority
    # order), same None convention run_loso.py's own EXCLUDE_SLIDES/token-restriction
    # filtering already guards for -- this split needs the identical guard.
    if f_trans_coord is not None:
        coord_slide = f_trans_coord["Slide_ID"].astype(str)
        f_coord_train = f_trans_coord[coord_slide != held_out_slide]
        f_coord_held = f_trans_coord[coord_slide == held_out_slide]
    else:
        f_coord_train = None
        f_coord_held = None

    # gene selection computed ONCE from TRAIN-only data: computing it fresh on the
    # held-out slide alone would leak its own variability pattern into what "counts"
    # as a selected gene.
    gene_selection = CONFIG.get("gene_selection", "svg")
    gene_panel_path = CONFIG.get("gene_panel_path")
    gene_mask = get_gene_mask(f_trans_train, gene_selection=gene_selection, gene_panel_path=gene_panel_path)
    if verbose:
        label = f"panel:{Path(gene_panel_path).stem}" if gene_panel_path is not None else gene_selection
        print(f"  gene selection ({label}): {gene_mask.sum()} genes")

    if verbose:
        print("  building training-slide pipeline data (hist + token)...")
    # max_cells_per_slide/oversample_slides only matter for the TRAINING pipeline's
    # loader (they reshape per-epoch batch share -- see SlideStratifiedLoader's
    # docstring); the held-out pipe's own loader (built below) is never iterated over
    # during training or eval, only its "data" is used, so both are no-ops left at
    # CONFIG's default (None) for that call.
    train_pipe = build_pipeline_data(
        f_hist_train, f_trans_train, f_coord_train, CONFIG, svg_mask=gene_mask,
        f_token_adata=f_token_train, local_neighbor_radius=CONFIG.get("local_neighbor_radius"),
        max_cells_per_slide=CONFIG.get("max_cells_per_slide"),
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
        n_svg=train_pipe["n_svg"], token_dim=train_pipe["token_dim"],
        latent_dim=CONFIG["latent_dim"], proj_dim=CONFIG["proj_dim"],
        n_slides=train_pipe["n_slides"], encoder_type=CONFIG["encoder"],
        token_condition_on_slide=CONFIG.get("token_condition_on_slide", True),
        trans_condition_on_slide=CONFIG.get("trans_condition_on_slide", True),
    )
    pretrain_kwargs = dict(
        epochs=CONFIG["token_pretrain_epochs"], lr=CONFIG["token_pretrain_lr"],
        weight_decay=CONFIG["token_pretrain_weight_decay"],
        val_frac=CONFIG["token_pretrain_val_frac"], patience=CONFIG["token_pretrain_patience"],
    )

    set_seed(seed)
    model = DualAutoencoder(**model_kwargs)
    # skip_token_pretrain: ablation hook for "is Phase-1 pretraining useful at all" --
    # False (default) reproduces existing behavior exactly. When True, token_enc/
    # token_dec start from the same random init as the rest of the model and go
    # straight into Phase 3 -- best_val_mse/stopped_epoch are reported as NaN/-1
    # rather than omitted, so the log CSV keeps a fixed column set across both arms.
    if CONFIG.get("skip_token_pretrain", False):
        pretrain_info = {"best_val_mse": float("nan"), "stopped_epoch": -1}
    else:
        pretrain_info = pretrain_token_autoencoder(
            model, train_pipe["token_emb_t"], train_pipe["cell_meta"],
            device=device, verbose=verbose, **pretrain_kwargs,
        )

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
        f"trans_dec's F.one_hot(slide_code, num_classes=n_slides) would go out of bounds."
    )

    history = train(
        model, train_pipe["loader"], train_pipe["density_weight"],
        n_epochs=CONFIG["n_epochs"], lr=CONFIG["lr"], warmup_epochs=CONFIG["warmup_epochs"],
        temperature=CONFIG["temperature"],
        weight_reconstruction=CONFIG["weight_reconstruction_by_patch_freq"],
        device=device,
        token_emb_all=train_pipe["token_emb_t"], cell_meta=train_pipe["cell_meta"],
        max_local_contrast_weight=CONFIG["max_local_contrast_weight"],
        local_warmup_epochs=CONFIG.get("local_warmup_epochs"),
        freeze_token=freeze_token, token_lr_mult=CONFIG.get("token_lr_mult", 1.0),
        local_anchor_sim_threshold=CONFIG.get("local_anchor_sim_threshold"),
        local_anchor_sim_source=CONFIG.get("local_anchor_sim_source", "projected"),
        local_positive_mode=CONFIG.get("local_positive_mode", "exclude"),
        edge_dropout_p=CONFIG.get("edge_dropout_p", 0.0),
        gene_dropout_p=CONFIG.get("gene_dropout_p", 0.0),
        weight_decay=CONFIG.get("weight_decay", 0.0),
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
        "train_cell_neighbor_tolerant_retrieval": train_eval["mean_neighbor_tolerant"],
        "held_out_cell_top1_retrieval": held_eval["mean_top1"],
        "held_out_cell_cluster_top1_retrieval": held_eval["mean_cluster_top1"],
        "held_out_cell_neighbor_tolerant_retrieval": held_eval["mean_neighbor_tolerant"],
        "token_pretrain_best_val_mse": pretrain_info["best_val_mse"],
        "token_pretrain_stopped_epoch": pretrain_info["stopped_epoch"],
        "train_token_leakage_mean": train_leak_token["leakage_mean"],
        "train_token_leakage_chance": train_leak_token["leakage_chance"],
        "combined_token_leakage_mean": combined_leak_token["leakage_mean"],
        "combined_token_leakage_chance": combined_leak_token["leakage_chance"],
        "final_local_frac_masked": history[-1]["local_frac_masked"],
        "final_local_frac_extra_positive": history[-1]["local_frac_extra_positive"],
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
        # gene_names / *_scaler: fit ONLY on the training slides (see build_pipeline_data's
        # svg_mask/svg_scaler/hist_scaler/token_scaler docstring -- the held-out call reuses
        # these exact fitted objects rather than refitting, to avoid leakage). Saved here so
        # a genuinely new sample can later be matched to and preprocessed for this specific
        # trained model (same gene subset in the same order, same normalize_total+log1p ->
        # StandardScaler.transform() the model actually saw) without needing to re-run this
        # LOSO split to regenerate them -- previously these existed only in train_pipe, in
        # memory, for the duration of this one call. token_scaler is only present when the
        # token branch was used (mirrors train_pipe's own optional key).
        gene_names = train_pipe["trans_sub"].var_names[train_pipe["svg_mask"]].tolist()
        torch.save(
            {
                "model_state_dict": model.state_dict(), "config": CONFIG, "metrics": metrics, "history": history,
                "gene_names": gene_names,
                "svg_scaler": train_pipe["svg_scaler"], "hist_scaler": train_pipe["hist_scaler"],
                "token_scaler": train_pipe.get("token_scaler"),
            },
            checkpoint_dir / f"loso_token_{held_out_slide}{run_tag}.pt",
        )
        # Plain-text gene panel, one gene per line, in the exact column order the model's
        # trans_enc expects -- lets a new sample's panel be checked/matched against this
        # model without loading torch at all.
        (checkpoint_dir / f"loso_token_{held_out_slide}{run_tag}_gene_panel.txt").write_text(
            "\n".join(gene_names) + "\n"
        )
        # Full (uncapped) cell embeddings, saved alongside the checkpoint as the actual
        # deliverable -- can be large (a multi-hundred-MB .npz) when the training pool
        # is 1M+ cells.
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