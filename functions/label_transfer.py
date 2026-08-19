"""
Patch-level label transfer from the CRC_P2 ground truth to the other Visium HD
samples, using two independent classifiers ("mlp" or "knn", see MODEL_TYPE):

  1. image            -- UNI histology patch embeddings
  2. transcriptomics   -- pseudobulk gene expression aggregated per patch from
                          the bins that fall inside that patch, projected into
                          a TruncatedSVD basis fit on the reference

Ground truth: data/CRC_P2/01_preprocessing/unimodel/metadata/filtered_WSI_patch_embeddings.h5ad,
obs['tissuetype'] (dominant class = tissuetype.split("_")[0], see
primary_tissuetype). Both the reference and every target sample are read
from their filtered_WSI_patch_embeddings.h5ad, so background/no-tissue
patches are excluded from training and prediction alike.

Usage
-----
As a library (data loaders, aggregation, model classes -- no side effects
on import):

    from functions import label_transfer as lt
    device = lt.get_device()
    ...

To actually run the pipeline and write results to
data/{sample}/02_baselines/label_transfer/{sample}_patch_predictions.h5ad:

    from functions import label_transfer as lt
    lt.main(min_compound_label_count=100, neoplasm_prior_weight_on_normal=0.0, model_type="knn")

or from the command line: `python functions/label_transfer.py`.
"""

from __future__ import annotations

import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from anndata.utils import make_index_unique
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_SAMPLE = "CRC_P2"
PATCH_SIZE = 224
# Per-sample output dir, mirroring the other baselines (e.g. 02_baselines/spagcn).
OUT_DIR_TEMPLATE = "{}/02_baselines/label_transfer"

# Not real samples / not eligible as label-transfer targets.
EXCLUDE_DIRS = {"new", "CRC_P_Mixed", "combined", REFERENCE_SAMPLE}

SPATIAL_DIR_TEMPLATE = "{}/output/binned_outputs/square_002um"
PATCH_METADATA_TEMPLATE = "{}/01_preprocessing/unimodel/metadata/filtered_WSI_patch_embeddings.h5ad"

# Defaults for main()'s keyword parameters below -- used by plain
# `python functions/label_transfer.py` (or `main()` with no arguments); pass
# min_compound_label_count=/neoplasm_prior_weight_on_normal=/model_type= to
# main() to override per call instead of editing these, e.g. from a notebook.

# Ground-truth label resolution: None collapses every (possibly compound,
# "_"-joined) tissuetype to its dominant component -- the 6 base classes.
# Set to an int to instead keep common border-region labels (e.g.
# "Veins_Neoplasm") as their own class whenever they occur at least this
# many times in the reference; rarer compounds still collapse to dominant.
# See primary_tissuetype for details.
MIN_COMPOUND_LABEL_COUNT: int | None = 100

# Biological prior: healthy-donor slides ("Normal_...", per get_disease_status)
# shouldn't contain neoplastic tissue, so Neoplasm's predicted probability can
# be suppressed before argmax on those slides only. None disables this
# entirely (default). 0.0 = hard mask (Neoplasm impossible on Normal slides);
# e.g. 0.1 instead soft-downweights it rather than ruling it out completely.
NEOPLASM_PRIOR_WEIGHT_ON_NORMAL: float | None = None

# Classifier used for both feature spaces: "mlp" (default, small GPU neural
# net) or "knn" (sklearn k-NN, CPU, no training loop -- simpler and handles
# rare classes naturally, at the cost of a less flexible decision boundary
# and less-calibrated confidence). See train_classifier.
MODEL_TYPE = "mlp"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_target_samples(data_dir: Path = DATA_DIR) -> list[str]:
    """Samples with UNI patch embeddings, excluding the reference and known exclusions."""
    samples = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir() or p.name in EXCLUDE_DIRS:
            continue
        if (data_dir / PATCH_METADATA_TEMPLATE.format(p.name)).exists():
            samples.append(p.name)
        else:
            print(f"Skipping {p.name}: {PATCH_METADATA_TEMPLATE.format(p.name)} not found.")
    return samples


def get_disease_status(sample: str) -> str:
    """"CRC_xxx" -> "CRC", "Normal_xxx" -> "Normal", else "Unknown"."""
    name = sample.lower()
    if name.startswith("crc"):
        return "CRC"
    if name.startswith("normal"):
        return "Normal"
    return "Unknown"


def load_patch_embeddings(sample: str, data_dir: Path = DATA_DIR) -> ad.AnnData:
    """UNI patch embeddings + patch metadata (Slide_ID, Patch_ID, X, Y, [tissuetype])."""
    return sc.read_h5ad(data_dir / PATCH_METADATA_TEMPLATE.format(sample))


def compute_harmony_image_embeddings(
    samples: list[str], data_dir: Path = DATA_DIR, n_comps: int = 50,
) -> dict[str, np.ndarray]:
    """Joint PCA + Harmony correction (keyed on Slide_ID) over the raw UNI image
    embeddings of every sample in `samples`, so the image classifier can be
    trained/predicted on a batch-corrected representation instead of raw .X.

    Returns {sample: harmonized array}, each row-aligned to
    load_patch_embeddings(sample) (same order, same length).
    """
    import harmonypy

    embs = {s: load_patch_embeddings(s, data_dir) for s in samples}
    combined = ad.concat(embs, label="Slide_ID", index_unique="-")
    sc.pp.pca(combined, n_comps=n_comps)

    # NOTE: harmonypy>=2.0 already returns Z_corr as (n_obs, n_pcs) -- do not
    # transpose (unlike scanpy's sce.pp.harmony_integrate, which still assumes
    # the pre-2.0 (n_pcs, n_obs) orientation and would silently produce a
    # wrongly-shaped result).
    harmony_out = harmonypy.run_harmony(combined.obsm["X_pca"].astype(np.float64), combined.obs, "Slide_ID")
    harmonized = harmony_out.Z_corr

    sizes = [embs[s].n_obs for s in samples]
    offsets = np.cumsum([0] + sizes)
    return {s: harmonized[offsets[i]:offsets[i + 1]] for i, s in enumerate(samples)}


def primary_tissuetype(tissuetype: pd.Series, min_count: int | None = None) -> pd.Series:
    """
    Resolve each (possibly compound, '_'-joined) tissuetype label to a
    training class.

    min_count=None (default): every label collapses to its dominant (first)
    component, e.g. "Veins_Neoplasm" -> "Veins" -- the 6 base classes.

    min_count=N: any raw label (compound or not) occurring at least N times
    in `tissuetype` is kept as its own class instead -- e.g. a common
    border type like "Veins_Neoplasm" survives as a distinct class; only
    the rarer tail of compound labels still collapses to its dominant
    component. Pure (non-compound) labels are unaffected either way, since
    splitting them on "_" is a no-op.
    """
    tissuetype = tissuetype.astype(str)
    dominant = tissuetype.str.split("_").str[0]
    if min_count is None:
        return dominant

    counts = tissuetype.value_counts()
    common = counts[counts >= min_count].index
    return tissuetype.where(tissuetype.isin(common), dominant)


def load_bin_counts(sample: str, data_dir: Path = DATA_DIR) -> ad.AnnData:
    h5_path = data_dir / SPATIAL_DIR_TEMPLATE.format(sample) / "filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    return adata


def load_tissue_positions(sample: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Bin pixel positions, indexed by barcode, restricted to in-tissue bins."""
    pos = pd.read_parquet(
        data_dir / SPATIAL_DIR_TEMPLATE.format(sample) / "spatial" / "tissue_positions.parquet"
    )
    pos = pos.set_index("barcode")
    return pos.loc[pos["in_tissue"] == 1, ["pxl_row_in_fullres", "pxl_col_in_fullres"]]


def assign_bins_to_patches(
    tissue_positions: pd.DataFrame, patch_obs: pd.DataFrame, patch_size: int = PATCH_SIZE
) -> pd.Series:
    """Map each bin barcode to the Patch_ID of the 224x224 patch it falls in.

    Patch top-left corners (patch_obs['X'], patch_obs['Y']) sit on a regular
    `patch_size`-pixel grid in the same pixel space as the bin coordinates
    (X <-> column/pxl_col_in_fullres, Y <-> row/pxl_row_in_fullres). Bins that
    don't land inside any extracted patch are dropped.
    """
    origin_x = int(patch_obs["X"].min())
    origin_y = int(patch_obs["Y"].min())

    patch_x = origin_x + patch_size * np.floor(
        (tissue_positions["pxl_col_in_fullres"].to_numpy() - origin_x) / patch_size
    ).astype(np.int64)
    patch_y = origin_y + patch_size * np.floor(
        (tissue_positions["pxl_row_in_fullres"].to_numpy() - origin_y) / patch_size
    ).astype(np.int64)

    bin_grid = pd.DataFrame(
        {"patch_x": patch_x, "patch_y": patch_y}, index=tissue_positions.index
    )
    patch_lookup = patch_obs.reset_index().set_index(["X", "Y"])["Patch_ID"]
    # avoid ambiguity if any (X, Y) pair repeats
    patch_lookup = patch_lookup[~patch_lookup.index.duplicated()]

    matched = bin_grid.join(patch_lookup, on=["patch_x", "patch_y"], how="inner")
    return matched["Patch_ID"]


def aggregate_expression_by_patch(bin_adata: ad.AnnData, bin_to_patch: pd.Series) -> ad.AnnData:
    """Sum raw counts of all bins assigned to the same patch -> patches x genes."""
    bin_to_patch = bin_to_patch[bin_to_patch.index.isin(bin_adata.obs_names)]
    patch_ids = bin_to_patch.unique()
    patch_index = {pid: i for i, pid in enumerate(patch_ids)}

    bin_pos = bin_adata.obs_names.get_indexer(bin_to_patch.index)
    patch_pos = bin_to_patch.map(patch_index).to_numpy()

    indicator = sparse.csr_matrix(
        (np.ones(len(bin_pos), dtype=np.float32), (patch_pos, bin_pos)),
        shape=(len(patch_ids), bin_adata.n_obs),
    )
    patch_expr = indicator @ bin_adata.X

    return ad.AnnData(
        X=sparse.csr_matrix(patch_expr),
        obs=pd.DataFrame(index=pd.Index(patch_ids, name="Patch_ID")),
        var=bin_adata.var.copy(),
    )


def stream_aggregate_expression_by_patch(
    sample: str,
    patch_obs: pd.DataFrame,
    gene_names: pd.Index,
    data_dir: Path = DATA_DIR,
    patch_size: float = PATCH_SIZE,
    chunk_bins: int = 2_000_000,
) -> pd.DataFrame:
    """
    Aggregate one sample's raw bin counts into patch-level pseudobulk,
    restricted to `gene_names`, streamed directly from the 10x H5 file in
    bin-chunks. The full bin x gene matrix is never materialized -- needed
    for the largest slides (CRC_P11mm: ~27.5M bins, 1B+ nonzeros across all
    18,085 genes) which would exceed available RAM via
    load_bin_counts()+aggregate_expression_by_patch(). Used for every
    TARGET sample; the reference doesn't need this (it's small, and needs
    every gene anyway to select the HVGs that become `gene_names`).

    Returns a (n_patches x len(gene_names)) DataFrame of summed raw counts,
    indexed by Patch_ID, in patch_obs's row order.
    """
    import h5py  # only this streaming path needs it

    h5_path = data_dir / SPATIAL_DIR_TEMPLATE.format(sample) / "filtered_feature_bc_matrix.h5"
    positions = load_tissue_positions(sample, data_dir)

    patch_ids_all = patch_obs["Patch_ID"].to_numpy()
    patch_pos_index = {pid: i for i, pid in enumerate(patch_ids_all)}
    accumulator = np.zeros((len(patch_ids_all), len(gene_names)), dtype=np.float32)

    with h5py.File(h5_path, "r") as f:
        mat = f["matrix"]
        n_genes, n_bins = (int(v) for v in mat["shape"][:])
        barcodes = mat["barcodes"][:].astype(str)
        # Raw gene names can contain duplicates; load_bin_counts() (via
        # adata.var_names_make_unique()) is what `gene_names` was ultimately
        # derived from, so uniquify identically here to match it up correctly.
        all_gene_names = make_index_unique(pd.Index(mat["features/name"][:].astype(str)))

        # original gene row -> column position in `gene_names` (-1 if not needed)
        gene_row_of_target = all_gene_names.get_indexer(gene_names)
        gene_lookup = np.full(n_genes, -1, dtype=np.int64)
        present = gene_row_of_target >= 0
        gene_lookup[gene_row_of_target[present]] = np.where(present)[0]

        indptr = mat["indptr"][:]  # small: n_bins + 1 values, column (bin) boundaries

        for start in range(0, n_bins, chunk_bins):
            end = min(start + chunk_bins, n_bins)
            lo, hi = int(indptr[start]), int(indptr[end])
            if hi == lo:
                continue

            # Only this chunk's slice of data/indices is actually read from disk.
            chunk_data = mat["data"][lo:hi].astype(np.float32)
            chunk_indices = mat["indices"][lo:hi]
            local_indptr = indptr[start : end + 1] - lo

            mapped_cols = gene_lookup[chunk_indices]
            keep = mapped_cols >= 0
            # per-column nnz counts after filtering, without an O(nnz) column-id array
            keep_cumsum = np.concatenate(([0], np.cumsum(keep)))
            filtered_indptr = keep_cumsum[local_indptr]

            chunk_csc = sparse.csc_matrix(
                (chunk_data[keep], mapped_cols[keep], filtered_indptr),
                shape=(len(gene_names), end - start),
            )
            # CSC -> CSR transpose is a relabel, not a copy -- cheap even though
            # it'd be expensive on the full (unchunked) matrix.
            chunk_csr = chunk_csc.T.tocsr()

            chunk_barcodes = barcodes[start:end]
            chunk_pos = positions.reindex(chunk_barcodes).dropna()
            if chunk_pos.empty:
                continue
            bin_to_patch = assign_bins_to_patches(chunk_pos, patch_obs, patch_size=patch_size)
            if bin_to_patch.empty:
                continue

            bin_pos_in_chunk = pd.Index(chunk_barcodes).get_indexer(bin_to_patch.index)
            patch_pos = bin_to_patch.map(patch_pos_index).to_numpy()

            indicator = sparse.csr_matrix(
                (np.ones(len(bin_pos_in_chunk), dtype=np.float32), (patch_pos, bin_pos_in_chunk)),
                shape=(len(patch_ids_all), end - start),
            )
            accumulator += np.asarray((indicator @ chunk_csr).todense())

    return pd.DataFrame(accumulator, index=pd.Index(patch_ids_all, name="Patch_ID"), columns=gene_names)


def fit_reference_svd(
    patch_expr: ad.AnnData, n_top_genes: int = 2000, n_components: int = 50
) -> tuple[TruncatedSVD, pd.Index, np.ndarray]:
    """Normalize, pick HVGs, and fit a TruncatedSVD basis -- reference only."""
    adata = patch_expr.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    hvg_genes = adata.var_names[adata.var["highly_variable"]]

    svd = TruncatedSVD(n_components=n_components, random_state=0)
    embedding = svd.fit_transform(adata[:, hvg_genes].X)
    return svd, hvg_genes, embedding


def project_svd(patch_expr: ad.AnnData, hvg_genes: pd.Index, svd: TruncatedSVD) -> np.ndarray:
    """Normalize + project a (possibly different-gene-set) patch expression matrix
    into the reference SVD space, aligning genes to hvg_genes (zero-filling any
    reference gene missing from this sample)."""
    adata = patch_expr.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    common = hvg_genes.intersection(adata.var_names)
    missing = hvg_genes.difference(adata.var_names)

    sub = adata[:, common].X
    if len(missing) > 0:
        pad = sparse.csr_matrix((adata.n_obs, len(missing)), dtype=np.float32)
        sub = sparse.hstack([sub, pad], format="csr")
        gene_order = list(common) + list(missing)
    else:
        gene_order = list(common)

    reorder = pd.Index(gene_order).get_indexer(hvg_genes)
    sub = sub[:, reorder]
    return svd.transform(sub)


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(
    X: np.ndarray,
    y: pd.Series,
    device: torch.device,
    model_type: str = "mlp",
    val_frac: float = 0.15,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    name: str = "",
    n_neighbors: int = 15,
) -> dict:
    """Fit a classifier. model_type="mlp" (default) is a small GPU MLP;
    model_type="knn" is a CPU sklearn KNeighborsClassifier -- simpler, no
    training loop, and naturally handles rare classes without the MLP's
    class-weighting, at the cost of a less flexible decision boundary and
    less-calibrated confidence. Both return the same dict shape, consumed
    identically by predict_classifier.
    """
    if model_type not in ("mlp", "knn"):
        raise ValueError(f"model_type must be 'mlp' or 'knn', got {model_type!r}")

    if sparse.issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float32)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y.to_numpy())

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    scaler = StandardScaler().fit(X[train_idx])
    X_scaled = scaler.transform(X).astype(np.float32)

    if model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        model.fit(X_scaled[train_idx], y_enc[train_idx])
        val_acc = model.score(X_scaled[val_idx], y_enc[val_idx])
        print(f"  [{name}] trained on {len(train_idx)} patches (knn, k={n_neighbors}), val accuracy = {val_acc:.3f}")
        return {
            "model_type": "knn",
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "val_accuracy": val_acc,
        }

    class_counts = np.bincount(y_enc[train_idx], minlength=len(label_encoder.classes_))
    class_weight = torch.tensor(
        1.0 / np.maximum(class_counts, 1), dtype=torch.float32, device=device
    )
    class_weight = class_weight * len(class_weight) / class_weight.sum()

    X_train = torch.tensor(X_scaled[train_idx], device=device)
    y_train = torch.tensor(y_enc[train_idx], dtype=torch.long, device=device)
    X_val = torch.tensor(X_scaled[val_idx], device=device)
    y_val = torch.tensor(y_enc[val_idx], dtype=torch.long, device=device)

    model = MLPClassifier(X.shape[1], len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    best_val_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  [{name}] trained on {len(train_idx)} patches, best val accuracy = {best_val_acc:.3f}")

    return {
        "model_type": "mlp",
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "val_accuracy": best_val_acc,
    }


def classes_containing(fit: dict, token: str) -> list[str]:
    """Fitted classes with `token` as one of their '_'-joined components --
    matches both a pure class ("Neoplasm") and any compound class it's part
    of ("Neoplasm_Connective Tissue"), so a prior stays correct regardless
    of whether primary_tissuetype's min_count kept compounds distinct."""
    return [c for c in fit["label_encoder"].classes_ if token in c.split("_")]


def predict_classifier(
    fit: dict,
    X: np.ndarray,
    device: torch.device,
    suppress_classes: list[str] | None = None,
    suppress_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted label strings, top-class confidence).

    suppress_classes: optional biological/domain prior -- down-weight these
    class(es)' probability before argmax/confidence (e.g. Neoplasm on a
    healthy-donor slide), then renormalize. suppress_weight=0.0 (default
    when suppress_classes is given) is a hard mask; e.g. 0.1 instead
    soft-downweights rather than ruling it out entirely.
    """
    if sparse.issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float32)
    X_scaled = fit["scaler"].transform(X).astype(np.float32)

    if fit["model_type"] == "knn":
        probs = fit["model"].predict_proba(X_scaled)
    else:
        model = fit["model"]
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_scaled, device=device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

    if suppress_classes:
        classes = list(fit["label_encoder"].classes_)
        idx = [classes.index(c) for c in suppress_classes if c in classes]
        if idx:
            probs = probs.copy()
            probs[:, idx] *= suppress_weight
            probs = probs / probs.sum(axis=1, keepdims=True)

    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    labels = fit["label_encoder"].inverse_transform(pred_idx)
    return labels, conf


def transcriptomics_embedding_for_sample(sample: str, patch_obs: pd.DataFrame, hvg_genes, svd) -> np.ndarray:
    """Aggregate bins -> patch pseudobulk -> SVD embedding, aligned to patch_obs order.
    Patches with no assigned bins get an all-zero embedding.

    Streamed directly from the 10x H5 file, restricted to hvg_genes, so the
    full bin x (all 18k genes) matrix is never materialized -- the largest
    slides have tens of millions of bins and would otherwise exceed
    available RAM (confirmed: CRC_P11mm is ~27.5M bins / 1B+ nonzeros)."""
    patch_expr_df = stream_aggregate_expression_by_patch(sample, patch_obs, hvg_genes)
    patch_expr = ad.AnnData(
        X=sparse.csr_matrix(patch_expr_df.to_numpy(dtype=np.float32)),
        obs=pd.DataFrame(index=patch_expr_df.index),
        var=pd.DataFrame(index=hvg_genes),
    )
    del patch_expr_df
    gc.collect()

    embedding = project_svd(patch_expr, hvg_genes, svd)
    embedding_df = pd.DataFrame(embedding, index=patch_expr.obs_names)
    del patch_expr
    # patch_expr is indexed by the bare Patch_ID (e.g. "patch_1.png"), while
    # patch_obs's own index is "{Slide_ID}_{Patch_ID}" -- join on the Patch_ID
    # column, keeping patch_obs's row order.
    return embedding_df.reindex(patch_obs["Patch_ID"], fill_value=0.0).to_numpy()


def main(
    min_compound_label_count: int | None = MIN_COMPOUND_LABEL_COUNT,
    neoplasm_prior_weight_on_normal: float | None = NEOPLASM_PRIOR_WEIGHT_ON_NORMAL,
    model_type: str = MODEL_TYPE,
    use_harmony: bool = False,
):
    """Run the full pipeline. All knobs default to the module-level
    constants above, but can be overridden per call -- e.g. from a notebook:

        from functions import label_transfer as lt
        lt.main(
            min_compound_label_count=100, neoplasm_prior_weight_on_normal=0.0, model_type="knn",
        )

    use_harmony: if True, the image classifier is trained/predicted on a joint
    PCA + Harmony-corrected embedding (see compute_harmony_image_embeddings)
    instead of the raw UNI .X embedding, to remove slide-to-slide batch effects
    before label transfer. Outputs are written to
    "{sample}_patch_predictions_harmony.h5ad" instead of the usual
    "{sample}_patch_predictions.h5ad", so both are kept side by side for
    comparison; the transcriptomics classifier is unaffected either way.
    """
    device = get_device()
    print(f"Using device: {device}")

    targets = discover_target_samples()
    print(f"Target samples: {targets}")

    harmony_emb = None
    if use_harmony:
        print("\nComputing joint PCA + Harmony correction (keyed on Slide_ID) for image embeddings...")
        harmony_emb = compute_harmony_image_embeddings([REFERENCE_SAMPLE] + targets)

    # ------------------------------------------------------------------
    # Reference (CRC_P2): ground-truth-labeled patches
    # ------------------------------------------------------------------
    print(f"\nLoading reference ({REFERENCE_SAMPLE}) patch embeddings + ground truth...")
    ref_emb = load_patch_embeddings(REFERENCE_SAMPLE)
    labeled = ref_emb.obs["tissuetype"].notna()
    y_ref = primary_tissuetype(ref_emb.obs.loc[labeled, "tissuetype"], min_count=min_compound_label_count)
    print(f"  {labeled.sum()}/{ref_emb.n_obs} patches have ground truth. Classes:\n"
          f"  {y_ref.value_counts().to_dict()}")

    print("\nTraining image-based classifier...")
    ref_img_source = harmony_emb[REFERENCE_SAMPLE] if use_harmony else ref_emb.X
    X_ref_img = ref_img_source[labeled.to_numpy()]
    img_fit = train_classifier(X_ref_img, y_ref, device, model_type=model_type, name="image")

    print("\nAggregating reference transcriptomics per patch + fitting SVD...")
    ref_bins = load_bin_counts(REFERENCE_SAMPLE)
    ref_positions = load_tissue_positions(REFERENCE_SAMPLE)
    ref_bin_to_patch = assign_bins_to_patches(ref_positions, ref_emb.obs)
    ref_patch_expr = aggregate_expression_by_patch(ref_bins, ref_bin_to_patch)
    del ref_bins, ref_positions, ref_bin_to_patch
    gc.collect()

    svd, hvg_genes, ref_svd_embedding = fit_reference_svd(ref_patch_expr)
    ref_svd_df = pd.DataFrame(ref_svd_embedding, index=ref_patch_expr.obs_names)
    X_ref_tx = ref_svd_df.reindex(ref_emb.obs.loc[labeled, "Patch_ID"], fill_value=0.0).to_numpy()
    del ref_patch_expr
    gc.collect()

    print("\nTraining transcriptomics-based classifier...")
    tx_fit = train_classifier(X_ref_tx, y_ref, device, model_type=model_type, name="transcriptomics")

    # ------------------------------------------------------------------
    # Target samples
    # ------------------------------------------------------------------
    for sample in targets:
        print(f"\n--- {sample} ---")
        emb = load_patch_embeddings(sample)

        apply_neoplasm_prior = (
            neoplasm_prior_weight_on_normal is not None and get_disease_status(sample) == "Normal"
        )
        if apply_neoplasm_prior:
            print(f"  disease_status=Normal: suppressing Neoplasm (weight={neoplasm_prior_weight_on_normal})")

        img_input = harmony_emb[sample] if use_harmony else emb.X
        img_suppress = classes_containing(img_fit, "Neoplasm") if apply_neoplasm_prior else None
        img_labels, img_conf = predict_classifier(
            img_fit, img_input, device,
            suppress_classes=img_suppress, suppress_weight=neoplasm_prior_weight_on_normal or 0.0,
        )

        tx_embedding = transcriptomics_embedding_for_sample(sample, emb.obs, hvg_genes, svd)
        tx_suppress = classes_containing(tx_fit, "Neoplasm") if apply_neoplasm_prior else None
        tx_labels, tx_conf = predict_classifier(
            tx_fit, tx_embedding, device,
            suppress_classes=tx_suppress, suppress_weight=neoplasm_prior_weight_on_normal or 0.0,
        )

        out = ad.AnnData(
            X=emb.X,
            obs=emb.obs.assign(
                predicted_tissuetype_image=img_labels,
                predicted_confidence_image=img_conf,
                predicted_tissuetype_transcriptomics=tx_labels,
                predicted_confidence_transcriptomics=tx_conf,
            ),
        )
        out.obsm["spatial"] = emb.obsm["spatial"]

        out_dir = DATA_DIR / OUT_DIR_TEMPLATE.format(sample)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_harmony" if use_harmony else ""
        out_path = out_dir / f"{sample}_patch_predictions{suffix}.h5ad"
        out.write_h5ad(out_path)
        print(f"  Saved -> {out_path}")
        agree = (img_labels == tx_labels).mean()
        print(f"  image/transcriptomics agreement: {agree:.1%}")
        del emb, out, tx_embedding
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
