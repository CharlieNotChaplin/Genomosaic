"""
Standalone port of evaluate_final_loso.ipynb's sections 7, 8 and 9, so they can
run unattended on the cluster (see submit_evaluate_final_loso_7_9.sbatch).
Shares its setup (UNITS/VARIANTS/SLIDES/run_tag/checkpoint_dir,
reconstruct_held_out_cells, cluster_embedding, build_slide_graph) with
evaluate_clustering_quality.py by importing from it, rather than copying it.

Outputs (all under --output-dir, default data/combined/03_integration/autoencoders/):
  section 7  final_loso_gene_expression_probe.csv
             One row per (unit, variant, held_out_slide, probe in {ridge, mlp}):
             a probe mapping the fold's FROZEN model.token_enc(histology tokens)
             output -> the real standardized gene matrix (SVG union or Xenium
             panel), fit on training-slide cells, scored on the held-out slide.
             median_r2 / median_pearson_r are the numbers to read.
  section 8  final_loso_cluster_annotation.csv
             One row per (unit, variant, held_out_slide, Leiden cluster on
             z_trans): the cell type whose canonical marker set scores highest in
             that cluster. Coarse sanity check, NOT a validated cell-type call
             (hand-curated marker dict, no atlas). assigned_score /
             margin_to_runner_up show how borderline each call is.
  section 9  final_loso_spatial_clusters_<slide>.png (one per --plot-slides)
             2 x 4 figure: columns nuclei+SVG, nuclei+Xenium panel, bins+SVG,
             bins+Xenium panel; row 1 raw Leiden clustering of z_trans, row 2
             the same clustering after refine_domains smoothing.

Differences from the notebook, all deliberate:
  - Section 7 caps cells per slide (--max-cells-per-slide, default 100000,
    0 = no cap; seeded uniform subsample, the same rows are used whether a
    slide is a training slide or the held-out slide). The notebook used every
    cell (~1.9M training rows x 3072 token dims = ~23 GB per copy) and timed
    out. The probe's input is only the encoder's latent, so this costs little.
  - Section 7 reads each slide's data once per unit and reuses it across all
    folds/variants (the notebook re-reconstructed every slide for every fold:
    243 times for nuclei), and standardizes/encodes slide-by-slide in batches
    instead of concatenating full-size copies.
  - Results are appended to the CSVs after every fold, and folds/slides already
    in the CSV are skipped on the next run (--overwrite to redo), so a job that
    hits its time limit loses at most one fold.
  - Section 8 adds assigned_score / margin_to_runner_up columns.

Usage (from 04_eval/):
    python -u evaluate_final_loso_7_9.py
    python -u evaluate_final_loso_7_9.py --sections 8 9
    python -u evaluate_final_loso_7_9.py --units nuclei --variants svg --slides CRC_P1 Normal_P3
    python -u evaluate_final_loso_7_9.py --sections 9 --plot-slides CRC_P2 CRC_P5 --smoothing-mode spagcn
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from functions.dual_autoencoder import DualAutoencoder, get_gene_mask, refine_domains
import evaluate_clustering_quality as ecq

BASEDIR = ecq.BASEDIR
UNITS, VARIANTS, SLIDES = ecq.UNITS, ecq.VARIANTS, ecq.SLIDES
DEFAULT_OUTPUT_DIR = BASEDIR / "combined/03_integration/autoencoders"

PROBE_CSV = "final_loso_gene_expression_probe.csv"
ANNOTATION_CSV = "final_loso_cluster_annotation.csv"

# Same dict as evaluate_final_loso.ipynb section 8 -- see its markdown for why this is a
# hand-curated sanity check rather than an atlas import.
CANONICAL_COLON_MARKERS = {
    "Epithelial (general)": ["EPCAM", "KRT8", "KRT18", "CDH1"],
    "Colonocyte / Enterocyte": ["CA2", "CA1", "SLC26A3", "CEACAM7"],
    "Goblet": ["MUC2", "TFF3", "FCGBP", "CLCA1"],
    "Stem / TA": ["LGR5", "OLFM4", "ASCL2"],
    "Enteroendocrine": ["CHGA", "CHGB", "CPE"],
    "Tuft": ["AVIL", "AZGP1"],
    "T cell": ["CD3D", "CD3E", "CD3G", "CD2"],
    "B cell": ["CD79A", "CD79B", "BANK1", "MS4A1"],
    "Plasma": ["DERL3"],
    "Myeloid / Macrophage": ["CD14", "C1QA", "C1QB", "C1QC", "CD163"],
    "Mast": ["CPA3", "CMA1"],
    "Dendritic": ["CLEC9A"],
    "Endothelial": ["AQP1", "CEACAM1"],
    "Fibroblast / Stromal": ["COL17A1", "COL19A1", "DPYSL3"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- shared helpers -------------------------------------------------------

def npz_path(unit, variant, slide):
    return ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, variant)}_embeddings.npz"


def ckpt_path(unit, variant, slide):
    return ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, variant)}.pt"


def done_keys(csv_path, key_cols):
    """(unit, variant, held_out_slide) tuples already present in an existing output CSV."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=key_cols)
    return set(map(tuple, df.drop_duplicates().to_numpy()))


def append_rows(csv_path, rows):
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def load_z_trans(unit, variant, slide, n_expected):
    """Held-out z_trans from a fold's npz, or None if missing / row count doesn't match the
    reconstructed cells (same safety check as evaluate_clustering_quality.load_embeddings)."""
    path = npz_path(unit, variant, slide)
    if not path.exists():
        return None
    z = np.load(path)["held_out_z_trans"]
    if z.shape[0] != n_expected:
        log(f"  [{unit}/{variant}/{slide}] row mismatch: {z.shape[0]} embeddings vs {n_expected} "
            f"reconstructed cells -- skipping")
        return None
    return z


# Leiden labels of z_trans per (unit, variant, slide), shared between sections 8 and 9.
_LEIDEN_CACHE = {}


def get_z_trans_clusters(unit, variant, slide, z_trans):
    key = (unit, variant, slide)
    if key not in _LEIDEN_CACHE:
        t0 = time.time()
        _LEIDEN_CACHE[key] = ecq.cluster_embedding(z_trans).astype(np.int32)
        log(f"  Leiden on z_trans [{unit}/{variant}/{slide}] n={z_trans.shape[0]}: "
            f"{len(np.unique(_LEIDEN_CACHE[key]))} clusters, {time.time() - t0:.0f}s")
    return _LEIDEN_CACHE[key]


# ---- section 7: gene expression probe -------------------------------------

class ProbeMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def fit_mlp_probe(z_train, y_train, device, val_frac=0.1, max_epochs=100, patience=8,
                  lr=1e-3, batch_size=4096, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(z_train.shape[0])
    n_val = int(len(idx) * val_frac)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    z_fit, y_fit = torch.from_numpy(z_train[fit_idx]).to(device), torch.from_numpy(y_train[fit_idx]).to(device)
    z_val, y_val = torch.from_numpy(z_train[val_idx]).to(device), torch.from_numpy(y_train[val_idx]).to(device)

    model = ProbeMLP(z_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val, best_state, epochs_no_improve = float("inf"), None, 0
    n_fit = z_fit.shape[0]
    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_fit, device=device)
        for start in range(0, n_fit, batch_size):
            batch_idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = F.mse_loss(model(z_fit[batch_idx]), y_fit[batch_idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(z_val), y_val).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, epochs_no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def score_predictions(y_true, y_pred):
    """median_r2/median_pearson_r are the numbers to read -- mean_r2 can blow up to enormous
    negative values on a single low-variance gene."""
    r2_per_gene = r2_score(y_true, y_pred, multioutput="raw_values")
    pearson_per_gene = np.array([
        pearsonr(y_true[:, g], y_pred[:, g])[0] if np.ptp(y_pred[:, g]) > 0 else np.nan
        for g in range(y_true.shape[1])
    ])
    return {
        "mean_r2": np.nanmean(r2_per_gene), "median_r2": np.nanmedian(r2_per_gene),
        "frac_genes_r2_pos": float(np.mean(r2_per_gene > 0)),
        "mean_pearson_r": np.nanmean(pearson_per_gene), "median_pearson_r": np.nanmedian(pearson_per_gene),
    }


def gene_panel_path_for(variant):
    panel = VARIANTS[variant]["gene_panel"]
    return None if panel is None else BASEDIR / "combined/01_preprocessing/gene_panels" / f"{panel}.json"


class Stats:
    """Column sums / sums of squares in float64, so a StandardScaler-equivalent (population
    mean/std, zero-variance columns left unscaled) can be assembled for any subset of slides
    without ever concatenating the full-size arrays."""

    def __init__(self, x):
        x64 = x.astype(np.float64)
        self.n, self.s, self.ss = x.shape[0], x64.sum(0), (x64 ** 2).sum(0)


def scaler_from(stats_list):
    n = sum(st.n for st in stats_list)
    s = sum(st.s for st in stats_list)
    ss = sum(st.ss for st in stats_list)
    mean = s / n
    scale = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    scale[scale < 1e-10] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build_probe_cache(unit, variants, slides, max_cells, seed=0):
    """Reads each slide ONCE: histology-token features + expression for every variant's gene
    set, for a (possibly capped) uniform random subset of that slide's paired cells. Returns
    {slide: {"token", "token_stats", "expr": {variant: array}, "expr_stats": {variant: Stats}}}."""
    u = UNITS[unit]
    cache, masks = {}, {}
    for i, s in enumerate(slides):
        t0 = time.time()
        recon = ecq.reconstruct_held_out_cells(unit, s)
        if recon is None:
            continue
        _, trans_sub = recon
        for v in variants:
            if v not in masks:
                masks[v] = get_gene_mask(trans_sub, gene_selection="svg", gene_panel_path=gene_panel_path_for(v))

        tok = sc.read_h5ad(BASEDIR / s / "01_preprocessing/unimodel/metadata" / u["token_filename"])
        tok.obs = tok.obs.set_index(tok.obs["Slide_ID"].astype(str) + "_" + tok.obs["id"].astype(str))
        tok = tok[tok.obs["matched"].astype(bool)]
        common = trans_sub.obs_names[trans_sub.obs_names.isin(tok.obs_names)]
        if len(common) == 0:
            continue
        if max_cells and len(common) > max_cells:
            rng = np.random.default_rng(seed + i)
            common = common[np.sort(rng.choice(len(common), size=max_cells, replace=False))]

        tok_sub = tok[common]
        token = np.concatenate(
            [np.asarray(tok_sub.obsm["X_token_footprint"]), np.asarray(tok_sub.obsm["X_token_halo"])], axis=1
        ).astype(np.float32)
        X = trans_sub[common].X
        expr = {}
        for v in variants:
            xv = X[:, masks[v]]
            expr[v] = (xv.toarray() if hasattr(xv, "toarray") else np.asarray(xv)).astype(np.float32)
        cache[s] = dict(token=token, token_stats=Stats(token), expr=expr,
                        expr_stats={v: Stats(e) for v, e in expr.items()})
        log(f"  cached {unit}/{s}: {token.shape[0]} cells x {token.shape[1]} token dims "
            f"({time.time() - t0:.0f}s)")
        del trans_sub, tok, tok_sub, X
    return cache, masks


def encode_tokens(model, token, mean, scale, device, batch=65536):
    """Standardize with the fold's token scaler and run the frozen token encoder, batch by batch."""
    out = []
    with torch.no_grad():
        for start in range(0, token.shape[0], batch):
            x = (token[start:start + batch] - mean) / scale
            out.append(model.token_enc(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def run_gene_expression_probe(unit, variant, held_out, cache, masks, device):
    """One fold's Ridge + MLP probe, both scored. Returns [ridge_row, mlp_row], or [] if the
    fold's checkpoint or slide data isn't available."""
    path = ckpt_path(unit, variant, held_out)
    if not path.exists() or held_out not in cache:
        return []
    train_slides = [s for s in SLIDES if s != held_out and s in cache]
    if not train_slides:
        return []

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    n_slides = ckpt["model_state_dict"]["trans_dec.net.0.weight"].shape[1] - cfg["latent_dim"]
    token_dim = cache[held_out]["token"].shape[1]
    model = DualAutoencoder(
        n_svg=int(masks[variant].sum()), latent_dim=cfg["latent_dim"], proj_dim=cfg["proj_dim"],
        n_slides=n_slides, encoder_type=cfg["encoder"], token_dim=token_dim,
        token_condition_on_slide=cfg.get("token_condition_on_slide", True),
        trans_condition_on_slide=cfg.get("trans_condition_on_slide", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    tok_mean, tok_scale = scaler_from([cache[s]["token_stats"] for s in train_slides])
    expr_mean, expr_scale = scaler_from([cache[s]["expr_stats"][variant] for s in train_slides])

    z_train = np.concatenate(
        [encode_tokens(model, cache[s]["token"], tok_mean, tok_scale, device) for s in train_slides], axis=0)
    y_train = np.concatenate(
        [(cache[s]["expr"][variant] - expr_mean) / expr_scale for s in train_slides], axis=0).astype(np.float32)
    z_held = encode_tokens(model, cache[held_out]["token"], tok_mean, tok_scale, device)
    y_held = ((cache[held_out]["expr"][variant] - expr_mean) / expr_scale).astype(np.float32)

    common_fields = {"unit": unit, "variant": variant, "held_out_slide": held_out,
                     "n_train_cells": z_train.shape[0], "n_held_cells": z_held.shape[0],
                     "n_genes": y_held.shape[1]}
    rows = []
    ridge = Ridge(alpha=10.0).fit(z_train, y_train)
    rows.append({**common_fields, "probe": "ridge", **score_predictions(y_held, ridge.predict(z_held))})

    mlp = fit_mlp_probe(z_train, y_train, device)
    with torch.no_grad():
        mlp_pred = np.concatenate([
            mlp(torch.from_numpy(z_held[i:i + 65536]).to(device)).cpu().numpy()
            for i in range(0, z_held.shape[0], 65536)
        ], axis=0)
    rows.append({**common_fields, "probe": "mlp", **score_predictions(y_held, mlp_pred)})
    return rows


def run_section_7(args, device):
    csv_path = args.output_dir / PROBE_CSV
    done = set() if args.overwrite else done_keys(csv_path, ["unit", "variant", "held_out_slide"])
    if args.overwrite and csv_path.exists():
        csv_path.unlink()

    for unit in args.units:
        todo = [(v, s) for v in args.variants for s in args.slides
                if ckpt_path(unit, v, s).exists() and (unit, v, s) not in done]
        if not todo:
            log(f"[7] {unit}: nothing to do")
            continue
        variants_needed = sorted({v for v, _ in todo}, key=list(VARIANTS).index)
        log(f"[7] {unit}: {len(todo)} folds to run; caching slide data (cap={args.max_cells_per_slide or 'none'})")
        cache, masks = build_probe_cache(unit, variants_needed, SLIDES, args.max_cells_per_slide)
        for v, s in todo:
            t0 = time.time()
            rows = run_gene_expression_probe(unit, v, s, cache, masks, device)
            if rows:
                append_rows(csv_path, rows)
                r = {row["probe"]: row for row in rows}
                log(f"[7] {unit}/{v}/{s}: ridge median_pearson_r={r['ridge']['median_pearson_r']:.3f} "
                    f"mlp median_pearson_r={r['mlp']['median_pearson_r']:.3f} ({time.time() - t0:.0f}s)")
            else:
                log(f"[7] {unit}/{v}/{s}: skipped (no checkpoint or slide data)")
        del cache
    log(f"[7] done -> {csv_path}")


# ---- section 8: cluster-type annotation -------------------------------------

def restrict_markers_to_available_genes(marker_dict, available_genes):
    available = set(available_genes)
    restricted = {}
    for cell_type, genes in marker_dict.items():
        kept = [g for g in genes if g in available]
        if kept:
            restricted[cell_type] = kept
        else:
            log(f"  '{cell_type}': none of {genes} present in this gene set -- dropped entirely")
    return restricted


def annotate_clusters_by_markers(trans_sub, cluster_labels, marker_dict):
    """Scores each cell type's marker set per cell (sc.tl.score_genes), averages within each
    Leiden cluster, assigns each cluster the highest-scoring type. Returns (cluster -> cell type,
    per-cluster score DataFrame)."""
    restricted = restrict_markers_to_available_genes(marker_dict, trans_sub.var_names)
    # lightweight AnnData sharing trans_sub.X -- score_genes only needs X and var_names
    adata = ad.AnnData(X=trans_sub.X, obs=pd.DataFrame(index=trans_sub.obs_names),
                       var=pd.DataFrame(index=trans_sub.var_names))
    for cell_type, genes in restricted.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"score__{cell_type}")

    score_cols = [f"score__{ct}" for ct in restricted]
    scores = pd.DataFrame(adata.obs[score_cols].values, columns=list(restricted.keys()))
    scores["cluster"] = np.asarray(cluster_labels)
    per_cluster = scores.groupby("cluster").mean()
    return per_cluster.idxmax(axis=1).to_dict(), per_cluster


def run_section_8(args):
    csv_path = args.output_dir / ANNOTATION_CSV
    done = set() if args.overwrite else done_keys(csv_path, ["unit", "variant", "held_out_slide"])
    if args.overwrite and csv_path.exists():
        csv_path.unlink()

    for unit in args.units:
        for slide in args.slides:  # slide-major: only one slide's expression matrix in RAM at a time
            variants = [v for v in args.variants if npz_path(unit, v, slide).exists() and (unit, v, slide) not in done]
            if not variants:
                continue
            recon = ecq.reconstruct_held_out_cells(unit, slide)
            if recon is None:
                continue
            cell_xy, trans_sub = recon
            for v in variants:
                z = load_z_trans(unit, v, slide, cell_xy.shape[0])
                if z is None:
                    continue
                labels = get_z_trans_clusters(unit, v, slide, z)
                assignment, per_cluster = annotate_clusters_by_markers(trans_sub, labels, CANONICAL_COLON_MARKERS)
                rows = []
                for cluster_id, cell_type in assignment.items():
                    ranked = np.sort(per_cluster.loc[cluster_id].to_numpy())[::-1]
                    rows.append({
                        "unit": unit, "variant": v, "held_out_slide": slide,
                        "cluster": cluster_id, "assigned_cell_type": cell_type,
                        "n_cells_in_cluster": int((labels == cluster_id).sum()),
                        "assigned_score": float(ranked[0]),
                        "margin_to_runner_up": float(ranked[0] - ranked[1]) if len(ranked) > 1 else np.nan,
                    })
                append_rows(csv_path, rows)
                log(f"[8] {unit}/{v}/{slide}: annotated {len(rows)} clusters")
            del trans_sub, cell_xy
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log("[8] assigned cell-type distribution (clusters pooled over everything annotated so far):\n"
            + df["assigned_cell_type"].value_counts().to_string())
    log(f"[8] done -> {csv_path}")


# ---- section 9: spatial cluster maps ------------------------------------------

def dot_size(n_units):
    """Marker area that shrinks with the number of plotted units, so dense slides do not merge into blobs."""
    return float(np.clip(60_000 / n_units, 0.05, 1.0))


def plot_cluster_panel(ax, cell_xy, labels, title):
    ax.scatter(cell_xy[:, 0], cell_xy[:, 1], c=pd.factorize(labels)[0], cmap="tab20",
               s=dot_size(len(cell_xy)), linewidths=0, rasterized=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\n(n={len(np.unique(labels))} clusters)", fontsize=9)


def run_section_9(args):
    panels = [(u, v) for u in ["nuclei", "bins"] for v in ["svg", "colon_xenium_panel"]]
    for slide in args.plot_slides:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for col, (unit, variant) in enumerate(panels):
            result = None
            if npz_path(unit, variant, slide).exists():
                recon = ecq.reconstruct_held_out_cells(unit, slide)
                if recon is not None:
                    cell_xy = recon[0]
                    z = load_z_trans(unit, variant, slide, cell_xy.shape[0])
                    if z is not None:
                        raw = get_z_trans_clusters(unit, variant, slide, z)
                        conn = ecq.build_slide_graph(unit, cell_xy)
                        result = (cell_xy, raw, refine_domains(raw, conn, mode=args.smoothing_mode))
            if result is None:
                for row in (0, 1):
                    axes[row, col].text(0.5, 0.5, f"{unit} / {variant}\nno data yet", ha="center", va="center",
                                        transform=axes[row, col].transAxes)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                continue
            cell_xy, raw, smoothed = result
            plot_cluster_panel(axes[0, col], cell_xy, raw, f"{unit} / {variant}")
            plot_cluster_panel(axes[1, col], cell_xy, smoothed, f"{unit} / {variant} (smoothed)")

        axes[0, 0].set_ylabel("before smoothing", fontsize=11)
        axes[1, 0].set_ylabel(f"after smoothing ({args.smoothing_mode})", fontsize=11)
        fig.suptitle(f"{slide}: Leiden clustering on z_trans, before vs. after spatial smoothing "
                     f"({args.smoothing_mode} mode)", fontsize=13)
        plt.tight_layout()
        out_path = args.output_dir / f"final_loso_spatial_clusters_{slide}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        log(f"[9] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", type=int, choices=[7, 8, 9], default=[8, 9, 7],
                        help="Which sections to run, in the order given. Default 8 9 7: the heavy GPU "
                             "section last, so a time-out costs the least.")
    parser.add_argument("--units", nargs="+", choices=list(UNITS), default=list(UNITS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--slides", nargs="+", choices=SLIDES, default=SLIDES,
                        help="Held-out slides for sections 7/8. Section 7 always caches all 9 slides "
                             "(needed as training data) regardless of this.")
    parser.add_argument("--max-cells-per-slide", type=int, default=100_000,
                        help="Section 7: cap on cells per slide (0 = no cap). See module docstring.")
    parser.add_argument("--plot-slides", nargs="+", choices=SLIDES, default=["CRC_P2"], help="Section 9 slides.")
    parser.add_argument("--smoothing-mode", choices=["hard", "spagcn"], default="hard", help="Section 9.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard existing section 7/8 CSVs instead of resuming from them.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}; sections {args.sections}; units {args.units}; variants {args.variants}")
    for section in args.sections:
        t0 = time.time()
        {7: lambda: run_section_7(args, device), 8: lambda: run_section_8(args), 9: lambda: run_section_9(args)}[section]()
        log(f"section {section} finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
