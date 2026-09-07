"""
Proof of concept: does the histology token embedding (z_token) actually predict
real gene expression on a slide the model never saw?

Fits two probes per fold, not just one: Ridge (linear) and a small MLP
(nonlinear). Ridge is the conservative default for "does this representation
carry information" questions -- limited capacity means it can't overfit its way
to a good-looking score, but that same limited capacity means a genuinely
nonlinear z_token->expression relationship would show up as a false negative
(low score) for Ridge alone. The MLP checks that: if it scores meaningfully
higher than Ridge, the earlier weak Ridge numbers were a linear-probe ceiling,
not evidence that z_token lacks the signal.

Context: run_loso.py's cell<->token contrastive alignment only optimizes z_token
and z_trans to be *relatively* closer for true pairs than for mismatched ones --
it never forces them to be literally interchangeable inputs into the same decoder
(the well-known "modality gap" in CLIP-style contrastive models). So rather than
feeding z_token into the already-trained trans_dec (which was only ever trained on
z_trans, and would need a fabricated slide_code for a genuinely held-out slide),
this fits a small, separate probe: Ridge regression z_token -> X_svg (the same
539-gene SVG-selected, standardized target trans_dec was trained to reconstruct),
fit on a fold's training-slide cells, evaluated on that fold's held-out slide.
This sidesteps both problems (no slide_code needed, and it explicitly learns any
offset between the two embedding spaces instead of assuming none exists).

Why this doesn't reuse run_loso.py's saved train_z_token/held_out_z_token arrays
directly (originally the plan): cell_token_embeddings.h5ad (the per-cell footprint
+halo token feature file token_preprocessing.py produces) has since been deleted
for CRC_P11mm/CRC_P3_SMAP/CRC_P4_SMAP (likely for disk space -- their raw
WSI_patch_token_embeddings source files are 23GB/9.6GB/4.6GB). Without that file
there's no way to know which rows of the saved (whole-training-pool) embedding
arrays belonged to those slides' cells, so their contribution can't be sliced
back out reliably -- reconstructing an exactly-matching population is a dead end.

Instead, this loads each fold's already-TRAINED model checkpoint (frozen, no
retraining) and runs its token_enc forward pass itself on whatever cells
currently have real token data:
  - training pool = every slide except the held-out one AND except the 3 slides
    above (permanently excluded here, not just for this fold) -- still 6-7
    slides' worth of real (token, expression) pairs per fold.
  - held-out side only works when the held-out slide itself still has token
    data -- true for 6 of the 9 evaluated folds (not CRC_P11mm/CRC_P3_SMAP/
    CRC_P4_SMAP as the held-out target; those 3 are skipped with a clear message).
No hist/patch pairing is needed for this (token_enc takes only the standardized
token feature vector, no slide_code) -- just a direct trans<->token id join,
independent of build_pairing/build_pipeline_data's heavier hist-matching path.
"""

import copy
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break
from functions.dual_autoencoder import DualAutoencoder, get_svg_union_mask

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "data").is_dir():
        basedir = _candidate / "data"
        break
else:
    raise RuntimeError("Could not find a 'data' directory in any ancestor of this script")

RUN_TAG = "_cap11000_oversample_normal"  # must match run_loso.py's RUN_TAG for this run
EVAL_SLIDES = ["CRC_P1", "CRC_P2", "CRC_P5", "CRC_P11mm", "CRC_P11mm_2_cut",
               "CRC_P3_SMAP", "CRC_P4_SMAP", "Normal_P3", "Normal_P5"]
RIDGE_ALPHA = 10.0

dir_transcriptomics = basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning"
dir_output = basedir / "combined/03_integration/autoencoders/002um"
checkpoint_dir = dir_output / "loso_token_checkpoints"

print("Loading combined transcriptomics data (backed -- 1.7M x 18085 is too big to load "
      "unbacked, only ever need a few hundred-gene column slice per slide at a time)...")
f_trans_adata = ad.read_h5ad(dir_transcriptomics / "count_area_filtered_adata.h5ad", backed="r")
svg_mask = get_svg_union_mask(f_trans_adata)
print(f"SVG union: {svg_mask.sum()} genes")

trans_global_id = f_trans_adata.obs["Slide_ID"].astype(str) + "_" + f_trans_adata.obs["id"].astype(str)
f_trans_adata.obs = f_trans_adata.obs.set_index(trans_global_id)

all_slides = sorted(f_trans_adata.obs["Slide_ID"].astype(str).unique())
token_paths = {s: basedir / s / "01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad" for s in all_slides}
slides_with_tokens = [s for s in all_slides if token_paths[s].exists()]
slides_missing_tokens = sorted(set(all_slides) - set(slides_with_tokens))
print(f"Slides with cell_token_embeddings.h5ad: {slides_with_tokens}")
print(f"Slides missing it (permanently excluded from the training pool here): {slides_missing_tokens}")


def load_token_features(slide):
    """Real (token_feature, X_svg) pairs for one slide's "matched" cells, row-aligned."""
    tok = sc.read_h5ad(token_paths[slide])
    tok_id = tok.obs["Slide_ID"].astype(str) + "_" + tok.obs["id"].astype(str)
    tok.obs = tok.obs.set_index(tok_id)
    tok = tok[tok.obs["matched"].astype(bool)].copy()

    trans_slide = f_trans_adata[f_trans_adata.obs["Slide_ID"].astype(str) == slide]
    common = trans_slide.obs_names.intersection(tok.obs_names)
    trans_sub = trans_slide[common]
    tok_sub = tok[common]

    token_feat = np.concatenate(
        [np.asarray(tok_sub.obsm["X_token_footprint"]), np.asarray(tok_sub.obsm["X_token_halo"])], axis=1
    ).astype(np.float32)
    X_svg = trans_sub.X[:, svg_mask]
    X_svg = X_svg.toarray() if sp.issparse(X_svg) else np.asarray(X_svg)
    return token_feat.astype(np.float32), X_svg.astype(np.float32)


print("Loading per-slide token features (once, reused across folds)...")
per_slide_data = {s: load_token_features(s) for s in slides_with_tokens}
for s, (tf, xs) in per_slide_data.items():
    print(f"  {s}: {tf.shape[0]} matched cells")


class ProbeMLP(nn.Module):
    """Single hidden layer -- enough to test "is the z_token->expression map
    nonlinear", not meant to be a tuned/production model."""
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def fit_mlp_probe(z_train, y_train, device, val_frac=0.1, max_epochs=100, patience=8,
                   lr=1e-3, batch_size=4096, seed=0):
    """Early-stopped on a held-back slice of the TRAINING cells only (never the fold's
    held-out slide) -- same val_frac/patience early-stopping idea as
    pretrain_token_autoencoder in functions/dual_autoencoder.py, applied here instead
    of a fixed epoch count so this doesn't just memorize training noise."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(z_train.shape[0])
    n_val = int(len(idx) * val_frac)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    z_fit = torch.from_numpy(z_train[fit_idx]).to(device)
    y_fit = torch.from_numpy(y_train[fit_idx]).to(device)
    z_val = torch.from_numpy(z_train[val_idx]).to(device)
    y_val = torch.from_numpy(y_train[val_idx]).to(device)

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
    """Per-gene R^2 (via sklearn, not hand-rolled ss_res/ss_tot) and Pearson r, plus
    the summary stats actually worth reading. Per-gene R^2 is NOT averaged as a
    headline number: any gene with genuinely low (not necessarily zero) variance in
    the held-out slide makes ss_res/ss_tot blow up to an enormous negative value for
    a merely-modest absolute miss -- confirmed on a smoke test where mean_r2 came out
    to -1.5 MILLION while median_r2 was a sane 0.014. median_r2 (robust to outlier
    genes) and Pearson r's mean/median (naturally bounded to [-1, 1], unaffected by a
    target's absolute variance) are the numbers to actually read; mean_r2 is kept only
    for completeness."""
    r2_per_gene = r2_score(y_true, y_pred, multioutput="raw_values")
    pearson_per_gene = np.array([
        pearsonr(y_true[:, g], y_pred[:, g])[0] if np.ptp(y_pred[:, g]) > 0 else np.nan
        for g in range(y_true.shape[1])
    ])
    return {
        "mean_r2": np.nanmean(r2_per_gene),
        "median_r2": np.nanmedian(r2_per_gene),
        "frac_genes_r2_pos": float(np.mean(r2_per_gene > 0)),
        "mean_pearson_r": np.nanmean(pearson_per_gene),
        "median_pearson_r": np.nanmedian(pearson_per_gene),
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for held_out_slide in EVAL_SLIDES:
    print(f"\n=== {held_out_slide} ===")
    if held_out_slide not in slides_with_tokens:
        print(f"  no cell_token_embeddings.h5ad for {held_out_slide} -- can't encode it as held-out, skipping "
              f"(its raw WSI_patch_token_embeddings source still exists, so this file COULD be regenerated "
              f"via token_preprocessing.py if this fold matters -- ask before doing that, it's a heavy job)")
        continue

    ckpt_path = checkpoint_dir / f"loso_token_{held_out_slide}{RUN_TAG}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # n_slides isn't saved in cfg directly (computed internally during the original
    # run) -- recovered from trans_dec's first layer instead: MLPDecoder concatenates
    # a slide one-hot onto z before its first Linear, so in_dim = latent_dim + n_slides.
    # NOTE: this recovery assumes trans_condition_on_slide=True (in_dim = latent_dim +
    # n_slides) -- every checkpoint that exists as of this parameter's introduction was
    # trained with it on, so cfg.get(..., True) below is safe for those. A future
    # checkpoint trained with trans_condition_on_slide=False would make this line wrong
    # (in_dim = latent_dim alone, so the subtraction gives n_slides=0) -- fix this
    # recovery to branch on cfg.get("trans_condition_on_slide", True) if/when that
    # ablation is actually run and reloaded here.
    n_slides = ckpt["model_state_dict"]["trans_dec.net.0.weight"].shape[1] - cfg["latent_dim"]
    model = DualAutoencoder(
        n_svg=int(svg_mask.sum()), latent_dim=cfg["latent_dim"], proj_dim=cfg["proj_dim"],
        n_slides=n_slides, encoder_type=cfg["encoder"],
        token_dim=per_slide_data[held_out_slide][0].shape[1],
        token_condition_on_slide=cfg.get("token_condition_on_slide", True),
        trans_condition_on_slide=cfg.get("trans_condition_on_slide", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    train_slides = [s for s in slides_with_tokens if s != held_out_slide]
    train_token = np.concatenate([per_slide_data[s][0] for s in train_slides], axis=0)
    train_expr = np.concatenate([per_slide_data[s][1] for s in train_slides], axis=0)
    held_token, held_expr = per_slide_data[held_out_slide]

    token_scaler = StandardScaler().fit(train_token)
    train_token_std = token_scaler.transform(train_token).astype(np.float32)
    held_token_std = token_scaler.transform(held_token).astype(np.float32)

    expr_scaler = StandardScaler().fit(train_expr)
    train_expr_std = expr_scaler.transform(train_expr).astype(np.float32)
    held_expr_std = expr_scaler.transform(held_expr).astype(np.float32)

    with torch.no_grad():
        z_train = model.token_enc(torch.from_numpy(train_token_std).to(device)).cpu().numpy()
        z_held = model.token_enc(torch.from_numpy(held_token_std).to(device)).cpu().numpy()

    print(f"  n_train={z_train.shape[0]} n_held={z_held.shape[0]}")
    common = {
        "held_out_slide": held_out_slide, "train_slides_used": ",".join(train_slides),
        "n_train_cells": z_train.shape[0], "n_held_cells": z_held.shape[0],
        "n_genes": held_expr_std.shape[1],
    }

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(z_train, train_expr_std)
    ridge_pred = ridge.predict(z_held)
    ridge_scores = score_predictions(held_expr_std, ridge_pred)
    rows.append({**common, "probe": "ridge", **ridge_scores})
    print(f"  [ridge] median R^2={ridge_scores['median_r2']:.4f} "
          f"({ridge_scores['frac_genes_r2_pos']:.0%} of genes R^2>0) | "
          f"median Pearson r={ridge_scores['median_pearson_r']:.4f} (mean {ridge_scores['mean_pearson_r']:.4f})")

    mlp = fit_mlp_probe(z_train, train_expr_std, device)
    with torch.no_grad():
        mlp_pred = mlp(torch.from_numpy(z_held).to(device)).cpu().numpy()
    mlp_scores = score_predictions(held_expr_std, mlp_pred)
    rows.append({**common, "probe": "mlp", **mlp_scores})
    print(f"  [mlp]   median R^2={mlp_scores['median_r2']:.4f} "
          f"({mlp_scores['frac_genes_r2_pos']:.0%} of genes R^2>0) | "
          f"median Pearson r={mlp_scores['median_pearson_r']:.4f} (mean {mlp_scores['mean_pearson_r']:.4f})")

summary = pd.DataFrame(rows).set_index(["held_out_slide", "probe"])
out_path = dir_output / "gene_expression_probe_log.csv"
summary.to_csv(out_path)
print(f"\n=== Summary (also written to {out_path}; mean_r2 not meaningful, see score_predictions docstring) ===")
print(summary.drop(columns=["train_slides_used"]))
if len(summary):
    for probe_name in summary.index.get_level_values("probe").unique():
        sub = summary.xs(probe_name, level="probe")
        print(f"\n[{probe_name}] across {len(sub)} folds: median R^2={sub['median_r2'].median():.4f}, "
              f"median Pearson r={sub['median_pearson_r'].median():.4f} "
              f"(mean Pearson r={sub['mean_pearson_r'].mean():.4f})")
