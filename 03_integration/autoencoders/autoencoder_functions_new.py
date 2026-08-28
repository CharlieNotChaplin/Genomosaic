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
            (footprint+halo area-weighted ViT token embedding, see
            `token_preprocessing.py`). None (default) builds the model exactly as
            before -- no token branch at all -- so existing checkpoints/pipelines
            (`autoencoder.ipynb`) are unaffected. Passing a value adds a THIRD,
            structurally independent branch (`token_enc`/`token_dec`/`proj_token`),
            mirroring `hist_enc`/`hist_dec`/`proj_hist` exactly, plus a dedicated
            `proj_trans_local` -- kept separate from `proj_trans` so the cell-level
            (trans vs. token) and patch-level (pooled-trans vs. hist) contrastive
            objectives don't have to share one projection subspace for two
            different alignment targets (same reasoning as `proj_trans`/`proj_hist`
            being separate from each other, see `ProjectionHead` docstring).

            Token embeddings are per-CELL, not per-patch -- there is no pooling
            step and no broadcast, since footprint/halo aggregation already makes
            the token<->cell correspondence close to genuinely one-to-one (see
            conversation notes: median nucleus diameter ~21px vs. 14px tokens,
            ~27px median cell-cell spacing).

        hist_condition_on_slide: whether `hist_dec` also gets the slide one-hot
            concatenated onto z before decoding. DEFAULT TRUE, reversed from an
            earlier default of False.

            This was previously set to False on the theory that conditioning only
            removes the *requirement* for z to encode slide identity, without
            actively penalizing it for doing so anyway -- so it "shouldn't" help
            much. That reasoning was incomplete: it ignored `weight_decay` in
            `pretrain_hist_autoencoder`'s AdamW optimizer. With L2 regularization
            active, the optimizer isn't just permitted to drop unneeded
            information, it's actively pushed toward the lowest-norm solution
            that still minimizes reconstruction loss -- and once conditioning
            makes slide identity unnecessary for that, weight decay gives the
            encoder a real incentive to drop it, not merely permission.

            Directly measured, same data/seed/training regime, nothing else
            changed: condition_on_slide=True -> Slide_ID leakage 0.495;
            condition_on_slide=False -> 0.900 (chance = 0.20). Keep this True
            unless deliberately re-testing the unconditioned case.
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

    slide_categories: canonical, ordered list of slide names to code against (pass
      the SAME `slide_uniques` that `pd.factorize(trans_sub.obs["Slide_ID"])`
      produced for the cell/graph side). Without this, this function used to run
      its own independent `pd.factorize` here -- harmless-looking, but two
      factorizations of "the same" slide set computed over different objects
      (this function's per-PATCH `hist_adata` vs. the per-CELL `trans_sub` used
      elsewhere) assign codes in order of first appearance in EACH object, which
      can differ. In a LOSO run this occasionally produced a patch-level
      `slide_id` whose max code equaled or exceeded the model's `n_slides` (sized
      from the cell-side factorize alone), crashing `hist_dec`'s
      `F.one_hot(slide_code, num_classes=n_slides)` with a device-side CUDA
      assert (confirmed via CUDA_LAUNCH_BLOCKING=1 on the CRC_P1-held-out fold).
      Coding against the shared `slide_categories` list instead guarantees this
      function's codes reference the exact same slide->code mapping as the rest
      of the pipeline, with every code guaranteed < len(slide_categories) == n_slides.
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
                 patches_per_batch=16, max_cells_per_patch=64, shuffle=True,
                 max_patches_per_slide=None):
        """
        max_patches_per_slide: caps how many of a slide's OWN patches feed into batches
          per epoch (None = no cap, every patch used every epoch, original behavior).
          Motivated by a slide that's disproportionately large relative to the rest of
          the training pool (e.g. CRC_P11mm at 29,959 patches vs. 8,063-10,983 for every
          other slide in this project's dataset -- see conversation notes): since
          __len__/__iter__ give each slide a batch count proportional to its OWN patch
          count, an outsized slide gets a proportionally outsized share of every epoch's
          gradient updates. Capping truncates that slide's (reshuffled) patch list before
          chunking into batches, so it still contributes a full batch quota like everyone
          else, not zero -- and because __iter__ reshuffles every epoch, a DIFFERENT
          random subset of its patches is used each time, so the full patch pool still
          gets exposure across the course of training, just not all at once. Applies
          per-slide, so any slide below the cap is completely unaffected.
        """
        self.patch_to_nodes = patch_to_nodes
        self.patches_per_batch = patches_per_batch
        self.max_cells_per_patch = max_cells_per_patch
        self.shuffle = shuffle
        self.max_patches_per_slide = max_patches_per_slide

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
            n = len(plist) if self.max_patches_per_slide is None else min(len(plist), self.max_patches_per_slide)
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
      - excludes spatially close, non-matching patches from the negative set
        (`neighbor_radius`) instead of forcing them apart -- nearby tissue can be
        genuinely similar in both modalities under spatial autocorrelation.
      - excludes TRANSCRIPTOMICALLY near-duplicate, non-matching patches from the
        negative set (`anchor_sim_threshold`) -- the same false-negative problem
        as `neighbor_radius`, but for patches that are biologically alike (e.g.
        two separate tumor nests of the same subtype) rather than spatially close.
        Without this, InfoNCE still tries to push `p_trans[i]` and `p_hist[j]`
        apart whenever `p_trans[i]` and `p_trans[j]` are already near-identical,
        even though there's no way to satisfy that without the model inventing
        some spurious, non-biological distinguishing signal (staining/scan
        artifacts, slide identity) purely as a tie-breaker -- exactly the kind of
        shortcut the rest of this pipeline's leakage-fighting machinery exists to
        avoid. Checked on BOTH `p_trans`-vs-`p_trans` AND `p_hist`-vs-`p_hist`
        similarity (either clearing the threshold is enough to exclude a pair) --
        originally trans-only, extended to also check the second modality since
        two genuinely-similar nearby cells/patches can easily fail a strict
        transcriptomic-similarity cutoff from expression noise/dropout alone
        (sparse Visium HD data especially) while still being obviously
        near-duplicate in histology/token space, or vice versa; checking only one
        side left real false negatives uncaught (see conversation notes: a fine
        checkerboard-like alternation between two leiden clusters traced back to
        exactly this gap). Note this is still a moving target early in training:
        both projections are themselves being trained by this same loss (or, for
        the local/token call, by a freshly-initialized `proj_token` sitting on
        top of an already-frozen-and-pretrained `token_enc`), so the similarity
        judgment can drift as training progresses rather than being a fixed
        reference. `p_trans`/`p_hist` are assumed L2-normalized (true for
        `ProjectionHead`'s output), so this is a plain cosine-similarity cutoff
        in [-1, 1]; e.g. 0.9 excludes only near-duplicates, not merely similar
        patches. Masking-only (never an added attractive term) -- excluded pairs
        become "don't care" for this loss, not a forced match, so genuinely
        divergent cases (same expression, different morphology) remain free to
        stay divergent rather than being pulled together.
      - applies inverse-frequency weighting per patch (`weight`).

    `return_stats=True`: also returns a dict with `frac_masked` -- the fraction
    of the P*(P-1) off-diagonal (candidate-negative) entries excluded by ANY of
    the three masks above. This is a collapse early-warning signal: if
    `anchor_sim_threshold` masking is circular (it checks the model's own live,
    still-training projections) and those projections start collapsing toward
    each other, more pairs cross the threshold and get masked, which removes
    the very gradient that would otherwise correct the collapse -- a
    self-reinforcing loop. `frac_masked` climbing toward 1.0 within the first
    few epochs is the visible symptom, catchable in minutes instead of only
    discovering a fully collapsed `contrast_local` (stuck at ~0.0 for the rest
    of training, chance-level retrieval at eval) after a full run finishes --
    see conversation notes for a run where exactly this happened.

    MULTI-SLIDE NOTE: if a batch mixes patches from more than one slide, pass
    `slide_ids` (int tensor, one per patch, same order as `centroids`). Raw
    spatial coordinates aren't comparable across slides -- two different slides
    often share overlapping coordinate ranges, so a patch on slide A can be
    coincidentally "close" in raw distance to an unrelated patch on slide B.
    Without `slide_ids`, the neighbor mask would incorrectly treat that
    coincidence as a reason to exclude a perfectly good negative. When
    `slide_ids` is given, masking additionally requires both patches to be on
    the same slide -- this applies to `anchor_sim_threshold` too, for the same
    reason (in practice a no-op with slide-stratified batches, kept as the same
    defensive backstop as the other two masks).
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
    2021), applied to a batch of latent vectors `z` (shape (B, D), e.g.
    `z_patch_hist`). Directly penalizes representation collapse:

      - variance term: pushes the per-dimension std of z (across the batch) up
        toward `var_target`, via relu(target - std) -- a dimension whose std is
        already >= target contributes zero loss, one that's collapsed toward a
        constant gets pushed back up. This stops any single dimension (or a
        small correlated cluster of them) from soaking up most of the latent's
        variance while the rest flatline.
      - covariance term: pushes off-diagonal covariances between dimensions
        toward zero, so dimensions can't be redundant proxies for the same
        underlying direction -- without this, several *nominally* separate
        dimensions could still collectively behave like the one dominant PC
        we've been fighting (86.7%/67.7% variance concentration measured
        earlier), just spread across more raw coordinates.

    UNLIKE decoder conditioning (slide_code/tile_code): conditioning only ever
    removes the *requirement* for z to encode a nuisance direction, it never
    penalizes z for doing so anyway -- which is exactly why tile-conditioning
    alone didn't change anything. This is an ACTIVE penalty directly on z, so
    it discourages any one direction from dominating regardless of whether its
    origin is between-slide batch effect, within-slide position, or genuine
    biology -- it can't selectively target "just the bad stuff", but it also
    doesn't require correctly guessing what the nuisance variable even is
    (unlike a GRL classifier/regressor, which has to be told exactly what to
    penalize and has a track record of instability in this codebase).

    Requires B > 1 (a batch, not a single sample) -- std/covariance are
    undefined for B=1, so this returns 0 in that degenerate case.

    NOTE on `var_target`: the paper's default (1.0) assumes representations are
    roughly unit-scale already; `hist_enc`'s output has no such normalization
    built in, so its natural std may sit well above or below 1.0 depending on
    initialization/training. Check `z_patch_hist.std(dim=0)` early in a run --
    if the loss saturates at ~0 immediately, `var_target` is too low to do
    anything; if it stays large throughout, it may be fighting the other losses
    too hard and `vicreg_weight` should come down (or `var_target` down) rather
    than assuming this is working as intended.
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
    Phase-1 pretraining: hist_enc/hist_dec ONLY -- no graph machinery, no
    contrastive loss, no trans branch involved at all (the histology branch
    doesn't depend on cell-level graph structure, so this is a plain
    full-dataset MLP training loop).

    Rebuilt to match a known-good external baseline (a plain single-objective
    MLP autoencoder that preserved anatomical structure well where our original
    version of this function -- fixed LR, no validation, no early stopping --
    didn't; see conversation notes) as closely as possible:
      - PURE MSE, nothing else summed in -- avoids the "unprincipled combined
        loss" gradient-conflict problem discussed in conversation notes. If you
        want to try VICReg on top later, use `vicreg_regularizer` directly and
        add it deliberately -- it's not wired in here by default anymore, since
        the pure-MSE case alone already matched the working baseline.
      - AdamW with real weight decay, instead of unregularized Adam.
      - ReduceLROnPlateau on validation loss, instead of one fixed LR throughout.
      - a genuine held-out validation split (`val_frac`), early stopping
        (`patience`), and restoring the best-val-loss checkpoint at the end --
        instead of running a fixed epoch count and keeping whatever that
        happens to produce.

    Leave `model`'s `hist_condition_on_slide=True` (the default) -- see
    `DualAutoencoder`'s docstring for the measured effect of this flag.

    Meant to run BEFORE `train()`. Follow with `freeze_hist_encoder(model)` and
    `train(..., freeze_hist=True)` for the joint/contrastive phase, once this
    phase's output is confirmed to preserve real structure on its own.
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
    token_emb_all / cell_meta / local_contrast_weight / freeze_token: the token
    branch's cell-level analogue of hist_emb_all/patch_meta/contrast_weight/
    freeze_hist. `token_emb_all` and `cell_meta["xy"]`/`cell_meta["slide_id"]`
    must be row-aligned with `data.x` (see `build_cell_meta`) -- seed cells are
    looked up via `batch.n_id[:n_seed]` (PyG's NeighborLoader/`collate_fn` always
    puts seed nodes first and preserves their global ids in `n_id`, same
    convention `encode_full_graph_batched` relies on). Left at their defaults
    (None / 0.0), this reproduces the exact pre-token behavior, including for
    models built without a token branch at all (`model.token_enc is None`).

    patch_anchor_sim_threshold / local_anchor_sim_threshold: forwarded to
    `patch_level_infonce`'s `anchor_sim_threshold` for the patch-level and
    cell-level contrastive calls respectively -- see that function's docstring
    for the false-negative reasoning. Two separate thresholds (not one shared
    value) because pooled per-patch expression is smoother/more separable than
    raw per-cell expression, so the same cosine-similarity cutoff may not be
    equally meaningful at both scales. Both default to None (off), matching
    every other new-behavior flag in this file.

    edge_dropout_p / gene_dropout_p: denoising-style corruption applied ONLY to what
    `trans_enc` sees -- the reconstruction target (`x[:n_seed]` below) stays the clean,
    uncorrupted input, so this is a proper denoising objective (forces trans_enc to
    recover clean expression from a corrupted view) rather than noise for its own sake.
    Targets two different channels of the Slide_ID leakage seen in `z_trans_patch`:
    edge_dropout_p (torch_geometric.utils.dropout_edge on `edge_index`) attacks
    signal coming from local graph structure (e.g. slide-specific cell-density
    differences from preprocessing/filtering); gene_dropout_p (plain `F.dropout` on
    `x`) attacks over-reliance on any small set of especially slide/batch-discriminative
    genes. Neither touches the part of that leakage which reflects genuine biology
    (this dataset's 5 slides = 5 different patients, 3 tumor/2 normal, so Slide_ID is
    confounded 1:1 with patient identity and tumor status -- no amount of input
    corruption can separate a real biological signal from a technical one at this
    sample size). See conversation notes for the full reasoning.
    """
    model.train()
    if freeze_hist:
        # model.train() above just re-enabled BatchNorm updates on every submodule.
        # Pin back to eval mode ONLY the submodules that are actually frozen
        # (requires_grad=False on every param) -- checked directly rather than
        # assuming "freeze_hist=True" means both hist_enc AND hist_dec, since
        # `freeze_hist_encoder(model, freeze_decoder=False)` deliberately leaves
        # hist_dec trainable. Pinning a still-training hist_dec to .eval() would
        # freeze its BatchNorm running stats while its weights keep changing
        # underneath them -- wrong in the other direction.
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
        anchor_sim_threshold=patch_anchor_sim_threshold,
    )

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
    token_emb_all / cell_meta: the token branch's cell-level analogue of
      hist_emb_all/patch_meta, forwarded to `train_step` unchanged (see
      `build_cell_meta`). Leave both None (default) to train exactly as before,
      including for models with no token branch at all.

    patch_anchor_sim_threshold / local_anchor_sim_threshold: forwarded to
      `train_step`/`patch_level_infonce` unchanged -- excludes transcriptomically
      near-duplicate, non-matching patches/cells from the hard-negative set
      (cosine-similarity cutoff on `p_trans`/`p_trans_local`, e.g. 0.9), the
      same false-negative protection `neighbor_radius` already gives spatially
      close patches, just gated on expression similarity instead of distance.
      Both default to None (off).

    max_local_contrast_weight / local_warmup_epochs: own ramp schedule for the
      cell<->token contrastive term, reusing `contrast_weight_schedule` --
      separate from `max_contrast_weight`/`warmup_epochs` (the patch-level
      term's) since there's no reason the two should have to share one ramp.
      `local_warmup_epochs` defaults to `warmup_epochs` if not given.

    freeze_token / token_lr_mult: token-branch analogues of freeze_hist/
      hist_lr_mult -- same rationale (gentler LR on a branch that already went
      through its own Phase-1 pretraining, so the contrastive loss doesn't
      undo it).

    freeze_hist: if True, assumes `freeze_hist_encoder(model)` was already called
      (hist_enc/hist_dec requires_grad=False) -- the optimizer here only ever
      collects trainable (requires_grad=True) parameters, so frozen params are
      automatically excluded rather than needing special-casing beyond that.
      `train_step` separately re-pins hist_enc/hist_dec to .eval() each step (see
      its docstring) so BatchNorm running stats stay frozen too.

    hist_lr_mult: LR multiplier for hist_enc/hist_dec's param group, relative to
      `lr`. Only matters when freeze_hist=False (otherwise those params have
      requires_grad=False and never enter the optimizer at all). Use <1.0 so the
      contrastive/recon_hist losses reshape the histology tower more gently than
      trans_enc, which is training from scratch -- letting it move at full `lr`
      risks undoing the Slide_ID-leakage fix that Phase 1 pretraining bought
      (see `hist_condition_on_slide` docstring on DualAutoencoder), since
      loss_contrast has every incentive to exploit slide-level shortcuts in the
      histology latent once it's allowed to reshape it.

    weight_decay: applied uniformly to every trainable param group (trans_enc,
      trans_dec, pool, both projection heads, and hist_enc/hist_dec too if
      unfrozen). `pretrain_hist_autoencoder` already applies its own weight_decay
      to hist_enc/hist_dec in Phase 1 -- this is the equivalent lever for
      trans_enc, which previously had none at all here. trans_dec already
      defaults to condition_on_slide=True (see MLPDecoder), so the same
      conditioning-plus-weight-decay mechanism that dropped Slide_ID leakage in
      hist_enc is structurally already available to trans_enc; it was just
      missing the weight_decay half.

    edge_dropout_p / gene_dropout_p: forwarded to `train_step` -- see its
      docstring for the denoising-style mechanism and what each targets.
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

    Also reports median_rank/mrr (mean reciprocal rank) of the true match, off the
    same `sims` matrix as top1 -- top-1 over thousands of candidates is a harsh,
    all-or-nothing metric that can read as ~0 even when the model consistently
    places the true match near the top; rank-based stats surface that signal.

    Also returns patch-level embeddings for reuse downstream (leakage checks,
    spatial-autocorrelation checks, or as the actual joint representation to use):
    z_trans_patch, z_hist_patch, and z_joint_patch = concat(z_trans_patch, z_hist_patch),
    all row-aligned to `unique_patches` (also returned). These are the PRE-projection
    encoder outputs (not proj_trans/proj_hist's output) -- proj_trans/proj_hist exist
    specifically to serve the contrastive loss and are a narrower bottleneck (proj_dim
    < latent_dim); standard contrastive-learning practice (SimCLR etc.) is that the
    pre-projection encoder representation transfers better to downstream probes than
    the projection-head output used for the loss itself.
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

    Retrieval over the full cell population isn't feasible the way patch-level
    retrieval is -- an N x N similarity matrix over hundreds of thousands of
    cells would be 10^11+ entries. Instead this subsamples up to
    `max_cells_per_slide` cells PER SLIDE and computes top-1 retrieval within
    each slide's subsample separately (same reasoning as `PatchGroupedLoader`'s
    slide-stratified batching: pooling cells across slides into one candidate
    set would let the metric be satisfied by recognizing slide identity rather
    than genuine cell<->token correspondence).

    `cell_type_labels`: optional (n_cells,) array/tensor, row-aligned to `data`
    -- an INDEPENDENT cell-type/cluster assignment (e.g. leiden on a trans-only
    embedding), not derived from this model's own trans/token projections
    (that would be circular). If given, also reports cluster-level top-1:
    a "hit" if the retrieved cell shares the query cell's cluster label, even
    if it isn't the exact same cell. Strict exact-ID top-1 is a harsh metric
    when many cells are genuinely near-duplicates biologically (a large
    homogeneous region of similar cells makes picking the one true ID close to
    a coin flip among the crowd, regardless of embedding quality) -- see
    conversation notes. `neighbor_tolerant` already gives similar relief for
    spatial proximity; this gives the analogous relief for biological/cluster
    similarity regardless of physical distance.

    Returns full-dataset (unsampled) `z_trans_cell_full`/`z_token_cell_full` for
    reuse downstream (e.g. `evaluate_latent_leakage`), same convention as
    `retrieval_accuracy`'s `z_patch_trans_full`.
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

    mean_top1 = float(np.average(all_top1, weights=all_n))
    print(f"Cell<->token retrieval, n-weighted mean top-1 across slides: {mean_top1:.3f}")
    mean_cluster_top1 = None
    if all_cluster_top1:
        mean_cluster_top1 = float(np.average(all_cluster_top1, weights=all_n))
        print(f"Cell<->token retrieval, n-weighted mean CLUSTER-level top-1: {mean_cluster_top1:.3f}")
    for sid, r in results_per_slide.items():
        msg = f"  slide {sid}: top1={r['top1']:.3f}"
        if r["neighbor_tolerant"] is not None:
            msg += f" | neighbor_tolerant={r['neighbor_tolerant']:.3f}"
        if r["cluster_top1"] is not None:
            msg += f" | cluster_top1={r['cluster_top1']:.3f}"
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
                             tumor_prefix="CRC", max_iter=5000):
    """
    Tests whether an embedding captures real tumor-vs-normal biological signal --
    reframes part of the Slide_ID leakage problem as a feature, not purely a bug: with
    5 slides split 3 tumor (CRC_*) / 2 normal (Normal_*), one patient each, tumor status
    is confounded 1:1 with Slide_ID, so some of the Slide_ID leakage measured elsewhere
    in this pipeline (evaluate_latent_leakage) is plausibly real, desired biology, not a
    technical artifact to suppress.

    Two checks, both genuinely slide-disjoint (a slide's own patches never appear in
    both the fit set and the set they're evaluated on -- unlike a naive pooled
    cross_val_score, which would let a slide's patches leak across CV folds):

      - train_loso_acc: leave-one-TRAINING-slide-out, via sklearn's LeaveOneGroupOut so
        it works regardless of how many tumor/normal slides happen to be in the
        training pool for this particular held-out fold. Genuinely discriminative:
        does the classifier correctly separate tumor from normal on a training slide
        it wasn't fit on? Some folds are degenerate (e.g. holding a fold's training
        slides down to a single class when the removed slide was the only one of its
        class) -- those are skipped, not silently included; `n_valid_folds` reports how
        many contributed.

      - held_out_acc: fits on ALL training-slide patches (real tumor/normal labels),
        predicts on the TRUE held-out slide's patches (never seen at any stage) --
        genuine zero-shot generalization to a 5th, entirely unseen patient. Since the
        held-out slide's true label is CONSTANT (one slide = one class), this is "what
        fraction of its patches got classified as its own known true label", not a
        two-sided discriminative accuracy the way train_loso_acc is -- read the two
        together, not `held_out_acc` alone.
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
            clf_fold = LogisticRegression(max_iter=max_iter)
            clf_fold.fit(z_train[tr_idx], y_train[tr_idx])
            fold_scores.append(clf_fold.score(z_train[te_idx], y_train[te_idx]))
    train_loso_acc = float(np.mean(fold_scores)) if fold_scores else None

    held_out_acc = None
    if len(y_held) and len(np.unique(y_train)) >= 2:
        clf = LogisticRegression(max_iter=max_iter)
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
    initializations, and returns the BEST replicate's hist_enc/hist_dec state
    dicts -- plus a full per-replicate log (seed, stopped epoch, val MSE, Slide_ID
    leakage), so "best" is inspectable, not a black box, and so you can see how
    much run-to-run variance there actually is before trusting any single number.

    Motivated directly by an observed case: two Phase-1 runs on the IDENTICAL
    setup, differing only in the random init at model-construction time, gave
    Slide_ID leakage of 0.336 vs 0.906 -- a huge, previously unmeasured source of
    variance that had been silently driving whatever number a single earlier
    check happened to report.

    select_by: 'leakage' (default) picks the replicate with the lowest Slide_ID
      leakage -- directly targets the thing this whole investigation cares about,
      since leakage and reconstruction quality are different objectives and don't
      necessarily track each other. Pass 'val_loss' to select by reconstruction
      quality instead.

    Returns:
      best_template_state: dict with 'hist_enc'/'hist_dec' state dicts, ready to
        pass into `run_experiment`'s `model_template_state` argument
      replicate_log: list of per-replicate dicts (seed, stopped_epoch,
        best_val_mse, leakage_mean, leakage_chance) -- print/inspect this before
        trusting `best_template_state` blindly
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
    already phase-1-pretrained-and-frozen histology encoder (`model_template_state`,
    from `pretrain_hist_once`), evaluates it, and appends one row to a persistent
    CSV log at `log_path` (created with a header on first call) -- so every run's
    FULL config (not just whatever you think you're varying) and resulting
    metrics are stored together and survive kernel restarts.

    `config`: dict of the complete parameter set for this run. Every key is
      logged with a `cfg__` prefix, so the log is self-documenting -- you can
      always reconstruct exactly what a given run_id used without relying on
      memory or a separate notes file.

    `seed`: FIXED (default 0, same for every call) -- `trans_enc`/pooling/
      projection heads are freshly initialized here on every call, same as
      `hist_enc` was in `pretrain_hist_autoencoder` before `pretrain_hist_replicated`
      controlled for it. Without pinning this, a sweep over e.g.
      `max_contrast_weight` would be confounded by a different, uncontrolled
      random init at every sweep point -- the exact issue just found and fixed
      on the Phase-1 side, showing up again here. Keep this the SAME across every
      run in a given sweep so the swept parameter is the only thing that differs;
      only change it deliberately (e.g. to replicate-check the winning config the
      way `pretrain_hist_replicated` did for Phase 1).

    Metrics logged: `top1_retrieval`/`neighbor_tolerant_retrieval` (the primary
    task metric -- cross-modal patch retrieval quality) and
    `hist_leakage_mean`/`hist_leakage_chance` (via `evaluate_hist_leakage`, the
    de-confounding diagnostic this whole investigation has centered on).

    Returns (metrics_dict, trained_model) -- the model is NOT discarded, so you
    can inspect/plot a specific run's result immediately without reloading its
    checkpoint from disk.
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
    # CONFIG has gained new keys over the course of this project (e.g.
    # hist_condition_on_slide), so a log seeded from an older CONFIG's rows has
    # fewer columns than a fresh row does. Plain csv.DictWriter.writerow() with
    # write_header only gated on "does the file exist yet" doesn't notice that
    # mismatch -- it silently appends rows with a different field count than
    # whatever's already there, corrupting the file for any later pd.read_csv
    # (this is exactly what happened: "Expected 32 fields... saw 33"). Reading
    # the existing rows and doing a union-column concat instead means the file
    # is always internally consistent no matter how CONFIG has evolved between
    # runs -- older rows just get NaN in columns that didn't exist yet for them.
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

    Starts from:
      - a shared, ALREADY-pretrained token_enc/token_dec (`model_template_state`,
        from `pretrain_token_once`) -- reused across every sweep point so Phase-1
        variance never confounds the comparison, same rationale as
        `run_experiment`'s hist template.
      - hist_enc/hist_dec at random init, immediately frozen (`freeze_hist_encoder`)
        -- harmless as long as `config["max_contrast_weight"]` is 0.0 (the
        validated setting while patch-level alignment isn't being trained, see
        train_autoenc_token.py's CONFIG comment); `DualAutoencoder` still needs a
        hist_enc/hist_dec to exist structurally even when that branch isn't the
        thing being swept.

    `config`: dict of the complete parameter set for this run, logged in full
      with a `cfg__` prefix (same convention as `run_experiment`, so the log
      stays self-documenting no matter how the swept keys change over time).
      Expected keys: latent_dim, proj_dim, pool_type, encoder, n_epochs, lr,
      warmup_epochs, temperature, weight_reconstruction_by_patch_freq,
      max_contrast_weight (usually 0.0), patch_anchor_sim_threshold (usually
      None), freeze_token, token_lr_mult, max_local_contrast_weight,
      local_warmup_epochs, local_anchor_sim_threshold, hist_condition_on_slide,
      token_condition_on_slide.

    `cell_type_labels`: independent cluster/cell-type labels (e.g.
      `pd.factorize(trans_sub.obs["clusters"])[0]`), row-aligned to `data`/
      `cell_meta` -- forwarded to `token_retrieval_accuracy` for the
      cluster-tolerant top-1 metric (see that function's docstring: strict
      exact-cell top-1 is a harsh metric when many cells are genuine biological
      near-duplicates).

    `seed`: FIXED (default 0, same every call) -- trans_enc/pool/projection heads
      are freshly initialized here on every call. Keep the same across every run
      in one sweep so the swept config is the only thing that differs.

    `leakage_subsample`: cell-level Slide_ID leakage (`evaluate_latent_leakage`)
      runs cross-validated logistic regression, which gets slow at full dataset
      size (1M+ cells) run once per sweep point -- subsampled to this many cells
      (fixed seed) to keep sweep wall-clock bounded regardless of dataset size.

    Metrics logged: cell_top1 / cell_cluster_top1 (via `token_retrieval_accuracy`),
    token_leakage_mean/chance (via `evaluate_latent_leakage` on the token
    embedding, same de-confounding rationale `run_experiment` already applies to
    the hist branch), plus the LAST epoch's `local_frac_masked` and
    `contrast_local` straight from `train()`'s returned history -- a collapsed
    run shows up directly as a row in the sweep log (frac_masked near 1.0,
    contrast_local near 0.0) rather than only being visible by scrolling that
    run's stdout after the fact.

    Returns (metrics_dict, trained_model, history) -- history is train()'s full
    per-epoch record, so a suspicious final-epoch number can be traced back to
    exactly when it happened without rerunning.
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
                         max_patches_per_slide=None):
    """
    Runs the SVG-mask -> spatial-graph -> pairing -> standardize -> patch-metadata
    -> PyG-data-and-loader pipeline (the same steps that live inline in
    run_sweep.py / notebook cell 9), factored out into a reusable function so it
    can be run on an arbitrary SLICE of slides -- needed for leave-one-slide-out
    evaluation, where the exact same construction has to happen twice (once on
    the training slides, once on the held-out slide), with the held-out call
    reusing the training call's fitted feature selection and scalers rather
    than re-deriving its own.

    Expects f_hist_adata/f_trans_adata/f_trans_coord to ALREADY be filtered to
    whichever slides this call should use (same convention as the
    CONFIG["sample"] filtering already used elsewhere in this project, and
    already assumes the Slide_ID+Patch_ID / Slide_ID+id index-setting from the
    data-loading step has already been applied) -- this function itself does
    no slide filtering.

    svg_mask: if given, use this SVG selection instead of recomputing it from
      whatever slides are passed. REQUIRED for a held-out call: SVG selection
      must come from the training call, never recomputed on the held-out slide
      alone (a different, incomparable feature set, and it would leak the
      held-out slide's own spatial-variability pattern into what "counts" as a
      spatially variable gene in the first place).
    svg_scaler / hist_scaler: if given, .transform() (NOT .fit()) the data with
      these instead of fitting fresh ones on whatever's passed in -- same
      rationale, and exactly what standardize_modalities's own docstring
      already flagged as the right approach for a held-out slide, before this
      function existed to make it possible to actually do so. Fitting a
      separate scaler on the held-out slide alone would let its own mean/std
      leak into "normalization," defeating the point of holding it out.

    f_token_adata / token_scaler / local_neighbor_radius: token-branch
      (cell<->token) analogue of the hist args above, entirely optional --
      leave f_token_adata=None (default) to reproduce the original
      hist/trans-only behavior exactly, including the returned dict's keys.
      When given, f_token_adata should already be filtered/indexed the same
      way f_trans_adata is (Slide_ID+id index, "matched" filtering already
      applied -- see train_autoenc_token.py's data-loading section, which this
      mirrors); this function itself does no token-specific slide filtering.
      As with svg_scaler/hist_scaler, pass the TRAINING call's fitted
      token_scaler when building the held-out slice so its own mean/std can't
      leak into "normalization."

    Returns a dict with everything downstream code needs: svg_mask, svg_scaler,
    hist_scaler, X_svg, hist_emb, hist_emb_t, trans_sub, patch_meta,
    patch_meta_df, data, loader, n_slides, n_svg, hist_dim, and -- only when
    f_token_adata is given -- token_scaler, token_emb, token_emb_t, cell_meta,
    token_dim.
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
        keep_mask = keep_mask & has_token

    trans_sub = f_trans_adata[keep_mask].copy()
    conn_sub = conn[keep_mask][:, keep_mask]
    X_svg = trans_sub.X[:, svg_mask]
    X_svg = X_svg.toarray() if sp.issparse(X_svg) else np.asarray(X_svg)

    slide_codes, slide_uniques = pd.factorize(trans_sub.obs["Slide_ID"])
    n_slides = len(slide_uniques)

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
    run_tag: appended to run_id/checkpoint filename (as "loso_{slide}{run_tag}")
      to keep comparison runs on the same held_out_slide from colliding -- e.g.
      run_tag="_unfrozen" when testing freeze_hist=False against an existing
      frozen baseline for the same slide, so the new checkpoint doesn't silently
      overwrite the old one (same collision class as the run_id issue fixed in
      run_sweep.py's BATCH_TAG). Empty by default so existing single-regime runs
      keep their prior run_id/checkpoint naming unchanged.
    Trains the full pipeline (Phase 1 hist pretrain -> freeze -> Phase 3 joint
    training) on every slide EXCEPT held_out_slide, then evaluates cross-modal
    retrieval on held_out_slide alone -- tissue the model has never seen at any
    stage of training.

    This is the actual test of the "inductive" claim discussed in conversation:
    SAGEConv/TransformerConv are architecturally capable of running on unseen
    nodes, but nothing before this function ever exercised that -- every prior
    evaluation in this project (the whole max_contrast_weight sweep included)
    ran retrieval_accuracy on data the model had already been trained on.

    Reports retrieval_accuracy on BOTH the training slides (in-sample, for
    direct comparison) and the held-out slide (the actual generalization
    number) -- the GAP between them, not either number alone, is what tells
    you whether this generalizes.

    Deliberately NOT reported for the held-out slide:
      - hist_leakage: a Slide_ID-classification diagnostic. There's no other
        slide within a single held-out slide to classify against -- not just
        hard to compute here, genuinely meaningless in this setting.
      - hist_dec/trans_dec reconstruction quality: both decoders are
        conditioned on slide identity via a one-hot sized for exactly the
        TRAINING slides (n_slides = 4, not 5) -- the held-out slide has no
        valid slot in that one-hot at all, categorically, not merely unseen
        data. retrieval_accuracy is the right metric here specifically because
        it never touches either decoder: hist_enc, trans_enc, pooling, and the
        projection heads are all slide-agnostic.

    seed: fixed (default 0) for the same reason run_experiment pins it -- an
      uncontrolled random init for trans_enc/pooling/projections would
      confound a fold-to-fold comparison the same way it did in the
      max_contrast_weight sweep before that was caught and fixed.
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

    # Slide_ID leakage on the TRAINING slides only -- meaningful here (n_slides-1
    # classes to distinguish), unlike the held-out slide where there's nothing to
    # classify against (see docstring). Checked on all three of trans-only, hist-only,
    # and joint (the actual embedding this function exists to help produce) so a
    # frozen-vs-unfrozen comparison shows exactly which representation absorbs any
    # reopened leakage. Uses the numeric `patch_meta["slide_id"]` codes here (safe --
    # both z_*_patch and slide_id come from the SAME build_pipeline_data call, so the
    # factorization is internally consistent), restricted to `unique_patches` in case
    # any patches were dropped.
    train_slide_id_np = train_pipe["patch_meta"]["slide_id"].cpu().numpy()[train_eval["unique_patches"]]
    train_leak_trans = evaluate_latent_leakage(train_eval["z_trans_patch"], train_slide_id_np, cv=min(5, len(train_slides)))
    train_leak_hist = evaluate_latent_leakage(train_eval["z_hist_patch"], train_slide_id_np, cv=min(5, len(train_slides)))
    train_leak_joint = evaluate_latent_leakage(train_eval["z_joint_patch"], train_slide_id_np, cv=min(5, len(train_slides)))
    if verbose:
        print(f"  train-slide leakage -- trans: {train_leak_trans['leakage_mean']:.3f} | "
              f"hist: {train_leak_hist['leakage_mean']:.3f} | joint: {train_leak_joint['leakage_mean']:.3f} "
              f"(chance = {train_leak_hist['leakage_chance']:.3f})")

    # Held-out-vs-train leakage: does a classifier trained on trans/hist/joint embeddings
    # across ALL n_slides (train + held-out) separate the held-out slide's patches out as
    # their own obviously-distinct cluster, or do they blend in like just another training
    # slide would? Unlike the check above, this NEEDS string Slide_ID labels rather than
    # the numeric `patch_meta["slide_id"]` codes -- those codes are factorized separately
    # within each build_pipeline_data call (train_pipe sees 4 categories, held_pipe sees
    # only 1), so code 0 in one is NOT the same slide as code 0 in the other. patch_meta_df
    # carries the actual Slide_ID strings, aligned by the same patch_row indexing.
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
    cell-level cross-modal retrieval generalize to a slide the model never saw
    at any stage of training (not Phase 1b token pretraining, not Phase 3 joint
    training)?

    Same rationale as the patch-level version -- every cell-level retrieval
    number produced so far (the 18-config sweep in token_sweep.py included) was
    measured on data the model had already trained on. This is the actual test
    of whether SAGEConv/TransformerConv's inductive capability, and the trained
    token_enc, generalize to unseen tissue rather than just memorizing it.

    Mirrors train_autoenc_token.py's validated regime: hist_enc/hist_dec left
    at random init and frozen immediately (harmless as long as
    CONFIG["max_contrast_weight"]=0.0 -- patch-level alignment isn't being
    trained, see that script's CONFIG comment), token_enc/token_dec pretrained
    ONCE on the training slides only via `pretrain_token_autoencoder`, then
    Phase 3 trains the cell<->token contrastive term with whatever
    max_local_contrast_weight/local_warmup_epochs/local_anchor_sim_threshold
    CONFIG carries.

    CONFIG["max_patches_per_slide"] (optional, default None via .get()): forwarded to
      the TRAINING pipeline's PatchGroupedLoader to cap any one training slide's
      per-epoch batch share -- see that class's docstring. Motivated by a slide that's
      disproportionately large relative to the rest of the training pool (conversation
      notes): fully excluding such a slide measurably reduced token leakage and roughly
      doubled in-sample retrieval fit, but left held-out generalization flat while the
      train/held-out gap roughly doubled too (an overfitting signature, not a fix) --
      capping instead of excluding keeps that slide's data available across epochs
      (reshuffled each epoch, so a different subset is seen every time) without letting
      it dominate the gradient budget.

    f_hist_adata / f_trans_adata / f_trans_coord / f_token_adata: already
      loaded, concatenated, and index-set exactly the way
      train_autoenc_token.py's data-loading section does (Slide_ID+id /
      Slide_ID+Patch_ID index, f_token_adata already restricted to
      obs["matched"]) -- this function does no loading of its own, only the
      train/held-out slide split and the pipeline construction below it.

    Reports cell-level top1/cluster_top1 retrieval on BOTH the training slides
    (in-sample) and the held-out slide (the actual generalization number) --
    the GAP between them is what tells you whether this generalizes, same as
    the patch-level function.

    Deliberately NOT reported for the held-out slide: token_leakage. Same
    reason as the patch version's hist_leakage -- there's no other slide
    within a single held-out slide to classify against.

    leakage_subsample: `evaluate_latent_leakage` runs cross-validated logistic
      regression, too slow at full cell-population size (1M+ cells) -- capped
      the same way `run_token_experiment` caps it.

    seed: FIXED (default 0) for the same reason `run_token_experiment` pins it
      -- trans_enc/pooling/projection heads are freshly initialized here on
      every call; an uncontrolled random init would confound a fold-to-fold
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
    # max_patches_per_slide only matters for the TRAINING pipeline's loader (it caps
    # per-epoch batch share -- see PatchGroupedLoader's docstring); the held-out pipe's
    # own loader (built below) is never iterated over during training or eval, only its
    # "data" is used, so capping it would be a no-op left at CONFIG's default (None).
    train_pipe = build_pipeline_data(
        f_hist_train, f_trans_train, f_coord_train, CONFIG, svg_mask=gene_mask,
        f_token_adata=f_token_train, local_neighbor_radius=CONFIG.get("local_neighbor_radius"),
        max_patches_per_slide=CONFIG.get("max_patches_per_slide"), verbose=verbose,
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

    # Fail fast with a clear Python error instead of an opaque, asynchronously-reported
    # CUDA "index out of bounds" assert (ScatterGatherKernel) if train_step's
    # `token_emb_all[seed_global_ids]`/`cell_meta["slide_id"][seed_global_ids]` gathers
    # would go out of range -- both are indexed by `batch.n_id`, which ranges over
    # [0, data.num_nodes), so they must have exactly that many rows.
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
    # Guards the actual bug hit on the CRC_P1-held-out fold: patch_meta["slide_id"]
    # used to come from its own independent pd.factorize inside compute_patch_metadata,
    # which could assign codes >= n_slides (sized from the cell-side factorize alone)
    # when the two objects' slides appeared in a different order -- crashing
    # hist_dec's F.one_hot(slide_code, num_classes=n_slides). Now fixed at the source
    # (compute_patch_metadata takes slide_categories= to share one canonical mapping),
    # but this stays as a cheap, permanent guard against the same class of bug
    # reappearing anywhere upstream.
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