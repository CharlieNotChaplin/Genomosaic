"""
Shared downstream evaluation metrics -- generic over both the label-transfer
outputs (functions/label_transfer.py) and the autoencoder embeddings
(functions/dual_autoencoder.py), so both consume one implementation instead
of duplicating sklearn-wrapper boilerplate.

Covers:
  - cluster_quality_metrics: Silhouette / Calinski-Harabasz / Davies-Bouldin --
    unsupervised, don't need ground truth, just an embedding + a partition
    (cluster assignments OR real categories, either works).
  - ari_between_labelings / compare_modalities_ari: adjusted Rand index, for
    "does this embedding's clustering agree with that one / with a reference."
  - classification_metrics: F1 (macro/weighted + per-class) plus precision/
    recall/accuracy, for anywhere true labels genuinely exist (e.g.
    evaluate_disease_signal's tumor/normal call, or CRC_P2's own held-out
    label-transfer accuracy) -- NOT valid on other slides' label-transfer
    predictions themselves, which have no independent ground truth to score
    against on this project's target slides.
  - expression_pearson_corr: per-gene/per-cell PCC between predicted and true
    expression -- for later, once a gene-expression-prediction task exists
    (e.g. DualAutoencoder.trans_dec's reconstruction vs. its real input).

Not included: SSIM. It measures local structural agreement between two 2D
IMAGES (luminance/contrast/structure terms over a sliding window), and
nothing in this project currently reconstructs pixel-level images -- add it
only once there's a concrete image-vs-image comparison to run it on.

"Cluster retrieval" isn't reimplemented here either -- it already exists as
the cluster_top1 metric in token_retrieval_accuracy (functions/dual_autoencoder.py):
a hit if the cross-modal nearest neighbor shares the query's independent
cluster label. This module's cluster_quality_metrics answers a different
question (is the clustering
itself well-separated), not that one.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, f1_score, precision_score, recall_score, accuracy_score,
)


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def cluster_quality_metrics(z, labels, subsample=10_000, seed=0):
    """
    Silhouette / Calinski-Harabasz / Davies-Bouldin on embedding `z` given a
    partition `labels` (cluster assignments from leiden/KMeans, OR real
    categories like tissue type -- any discrete grouping works). All three
    ask "does this embedding actually separate these groups," not "did we
    recover the RIGHT groups" -- that second question is what ARI
    (ari_between_labelings / compare_modalities_ari, below) answers instead.

    - silhouette: [-1, 1], higher is better. Needs pairwise distances (O(n^2)
      in `z`'s row count), so `z`/`labels` are subsampled to `subsample` rows
      when larger -- same rationale as this project's other O(n^2)-diagnostic
      subsampling (e.g. run_leave_one_slide_out_token's leakage_subsample).
    - calinski_harabasz: >=0, higher is better, unbounded, O(n) -- computed on
      the full data regardless of subsample (cheap).
    - davies_bouldin: >=0, LOWER is better -- the one metric here inverted
      relative to the other two, easy to misread when scanning a results table.

    Raises the same ValueError sklearn would if `labels` has fewer than 2
    distinct values, or any value with fewer than 2 members.
    """
    z = _to_numpy(z)
    labels = _to_numpy(labels)

    n = len(z)
    if n > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=subsample, replace=False)
        z_sil, labels_sil = z[idx], labels[idx]
    else:
        z_sil, labels_sil = z, labels

    return {
        "silhouette": float(silhouette_score(z_sil, labels_sil)),
        "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
        "davies_bouldin": float(davies_bouldin_score(z, labels)),
        "n_used_for_silhouette": int(len(z_sil)),
    }


def ari_between_labelings(labels_a, labels_b) -> float:
    """Thin, explicit wrapper around adjusted_rand_score -- exists so every ARI
    call in this project goes through one function instead of importing
    sklearn ad hoc per notebook cell (this replaces several near-identical
    inline ARI cells already in new_single_omics.ipynb)."""
    a = _to_numpy(labels_a).astype(str)
    b = _to_numpy(labels_b).astype(str)
    return float(adjusted_rand_score(a, b))


def compare_modalities_ari(cluster_labels: dict, reference_labels) -> pd.DataFrame:
    """
    ARI between a reference labeling (e.g. label-transfer predictions, or real
    ground truth) and EACH of several modality clusterings -- the direct
    "ARI between label transfer and single modalities plus combined modality"
    comparison.

    cluster_labels: {name: array-like} -- e.g. {"trans": leiden_trans,
      "hist": leiden_hist, "joint": leiden_joint}.
    reference_labels: array-like to compare each of the above against.

    Values may be plain arrays/tensors (assumed already row-aligned to
    reference_labels, same order) OR pandas Series -- if BOTH a given entry
    and reference_labels are Series, their indices are intersected first, so
    mismatched-but-overlapping id sets (e.g. different keep_masks per
    modality) don't need manual re-alignment beforehand. This generalizes the
    common-index pattern new_single_omics.ipynb's own ARI cells already use
    (get_basis/ari_vs_basis).

    Returns a DataFrame indexed by modality name, columns "ari" and "n" (cells
    actually compared -- relevant whenever index intersection narrows it).
    """
    rows = []
    for name, labels in cluster_labels.items():
        if isinstance(labels, pd.Series) and isinstance(reference_labels, pd.Series):
            common = labels.index.intersection(reference_labels.index)
            a, b = labels.loc[common], reference_labels.loc[common]
        else:
            a, b = labels, reference_labels
        rows.append({"modality": name, "ari": ari_between_labelings(a, b), "n": len(a)})
    return pd.DataFrame(rows).set_index("modality")


def classification_metrics(y_true, y_pred, average: str = "macro", labels=None) -> dict:
    """
    F1 (+ precision/recall/accuracy) for anywhere true labels genuinely exist
    -- e.g. evaluate_disease_signal's tumor/normal call (label = slide
    identity, not a guess), or CRC_P2's own held-out label-transfer accuracy.
    Do NOT use this on other slides' label-transfer predictions themselves:
    those predictions ARE the thing being evaluated, with no independent
    ground truth on this project's target slides to score them against.

    average="macro" (default): every class weighted equally regardless of
    size -- the right choice under class imbalance (e.g. 2 normal vs. 8 tumor
    slides, or the label-transfer reference's Neoplasm-heavy class
    distribution), since "weighted" would let the majority class dominate the
    score the same way plain accuracy already does -- defeating the point of
    reaching for F1 in the first place.

    Returns accuracy, f1/precision/recall at `average`, and per-class F1 (a
    single macro number can hide a class that's failing entirely; the
    per-class breakdown can't).
    """
    y_true = _to_numpy(y_true).astype(str)
    y_pred = _to_numpy(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "per_class_f1": dict(zip(labels, per_class_f1.tolist())),
    }


def expression_pearson_corr(pred_expr, true_expr, axis: str = "gene") -> pd.Series:
    """
    Pearson correlation coefficient between predicted and true expression --
    for LATER, once a gene-expression-prediction task exists (e.g.
    DualAutoencoder.trans_dec's reconstruction vs. the real SVG input it was
    trained to reconstruct). Not wired into anything yet.

    axis="gene" (default): one r per gene, correlating across cells/bins --
    "does this gene's expression pattern get reproduced." axis="cell": one r
    per cell/bin, correlating across genes -- "does this cell's expression
    profile shape get reproduced" (less common; sensitive to library-size-like
    scale effects unless expression is already normalized per cell).

    pred_expr/true_expr: (n_cells, n_genes) arrays/tensors/DataFrames, same
    shape and matching gene order (align genes yourself first, e.g. via
    label_transfer.project_svd's reindexing pattern, if they differ). Fully
    vectorized (no per-gene/per-cell Python loop) via standardize-then-dot,
    so it's cheap even at the cell axis with hundreds of thousands of rows.

    Returns NaN for any column with zero variance in either input (Pearson r
    is undefined there -- not a bug in this function).
    """
    pred_expr = _to_numpy(pred_expr)
    true_expr = _to_numpy(true_expr)
    if pred_expr.shape != true_expr.shape:
        raise ValueError(f"shape mismatch: pred {pred_expr.shape} vs true {true_expr.shape}")

    if axis == "cell":
        pred_expr, true_expr = pred_expr.T, true_expr.T
    elif axis != "gene":
        raise ValueError(f"axis must be 'gene' or 'cell', got {axis!r}")

    p = pred_expr - pred_expr.mean(axis=0, keepdims=True)
    t = true_expr - true_expr.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / denom
    return pd.Series(r, name=f"pearson_r_per_{axis}")
