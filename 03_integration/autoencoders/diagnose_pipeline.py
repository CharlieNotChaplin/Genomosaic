"""
Fast, CPU-only, read-only diagnostics meant to run *alongside* the long
train_autoenc_token.py run (not instead of it) -- answers a few questions
without waiting ~8h on the training job:

1. Is training actually using the GPU? The last run's log printed "Device:
   cpu" despite the sbatch script requesting --gres=gpu:1 -- if that's still
   true, wall-clock time is CPU-bound and may drop substantially once fixed.
2. Does CRC_P11mm's larger tissue area (~11x11mm vs ~6.5x6.5mm for the other
   6 samples) give it disproportionate weight during training? PatchGroupedLoader
   builds one batch group per slide and does ceil(n_patches_in_slide /
   patches_per_batch) batches for it; nothing in train()/train_step() reweights
   across slides afterward (weight_reconstruction_by_patch_freq only rebalances
   cells within a patch relative to other patches on the SAME slide). This
   computes the actual per-slide share of every training epoch from real data.
3. Baseline (pre-training) Slide_ID separability in the raw modalities, via the
   same evaluate_latent_leakage() diagnostic run_sweep.py/run_loso.py already
   use. train_autoenc_token.py's eval step doesn't currently call it at all, so
   there's no leakage number yet for the trained token model to compare
   against -- this gives a pre-training reference point in the meantime.

Nothing here touches a GPU or trains anything. Run directly (login node, or a
plain CPU allocation):
    python -u diagnose_pipeline.py
"""
import time as _time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from autoencoder_functions_new import evaluate_latent_leakage

PATCHES_PER_BATCH = 64  # must match CONFIG["patches_per_batch"] in train_autoenc_token.py
LEAKAGE_SUBSAMPLE = 20000  # cap rows fed to LogisticRegression -- this is a fast sanity
                            # check, not the final reported metric, so keep it quick

_t0 = _time.time()
basedir = Path("../../data")

print("=" * 70)
print("1. GPU/CUDA check")
print("=" * 70)
print(f"torch: {torch.__version__} | torch.version.cuda: {torch.version.cuda}")
print(f"cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device count: {torch.cuda.device_count()} | device 0: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA device visible in THIS allocation/environment -- if train_autoenc_token.py's "
          "job sees the same thing, that explains the CPU wall-clock time. Check the sbatch "
          "partition/--gres actually lands on a GPU node, and that this conda env's torch build "
          "has CUDA support (`python -c \"import torch; print(torch.__version__)\"` should show "
          "a '+cuXXX' suffix, not '+cpu').")

print()
print("=" * 70)
print("2. Per-slide size / training-batch imbalance")
print("=" * 70)
dir_trans = basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning"
trans = ad.read_h5ad(dir_trans / "count_area_filtered_adata.h5ad", backed="r")
obs = trans.obs[["Slide_ID", "Patch_ID"]]

cell_counts = obs["Slide_ID"].value_counts()
patch_counts = obs.groupby("Slide_ID")["Patch_ID"].nunique()
na_rate = obs.groupby("Slide_ID")["Patch_ID"].apply(lambda s: s.isna().mean())
batches_per_epoch = np.ceil(patch_counts / PATCHES_PER_BATCH)

summary = pd.DataFrame({
    "n_cells": cell_counts,
    "cell_share": cell_counts / cell_counts.sum(),
    "n_patches": patch_counts,
    "patch_share": patch_counts / patch_counts.sum(),
    "batches_per_epoch": batches_per_epoch,
    "batch_share": batches_per_epoch / batches_per_epoch.sum(),
    "no_patch_match_rate": na_rate,
}).sort_values("n_cells", ascending=False)
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
print(f"\nTotal batches/epoch across all slides: {int(batches_per_epoch.sum())}")
print("batch_share = each slide's fraction of total gradient updates per epoch under "
      "PatchGroupedLoader. With no cross-slide reweighting anywhere in train()/train_step(), "
      "this IS the slide's influence on training -- directly comparable across slides.")

print()
print("=" * 70)
print("3. Per-slide token-match rate (from token_preprocessing.py's existing output)")
print("=" * 70)
for slide in sorted(obs["Slide_ID"].astype(str).unique()):
    p = basedir / slide / "01_preprocessing/unimodel/metadata/cell_token_embeddings.h5ad"
    if not p.exists():
        print(f"{slide:15s} -- no cell_token_embeddings.h5ad yet")
        continue
    tok = sc.read_h5ad(p, backed="r")
    matched = np.asarray(tok.obs["matched"]).astype(bool)
    print(f"{slide:15s} n={len(matched):7d}  matched={matched.mean():.1%}")

print()
print("=" * 70)
print("4. Baseline Slide_ID leakage in raw modalities (pre-training reference point)")
print("=" * 70)
print("train_autoenc_token.py's eval step never calls evaluate_latent_leakage/"
      "evaluate_hist_leakage -- these numbers are the only leakage reference that "
      "exists until that's added to the training script and re-run.")

dir_hist = basedir / "combined/01_preprocessing/unimodel/metadata"
hist = sc.read_h5ad(dir_hist / "filtered_WSI_patch_embeddings.h5ad")
hist_X = hist.X.toarray() if sp.issparse(hist.X) else np.asarray(hist.X)
hist_slide, hist_slide_names = pd.factorize(hist.obs["Slide_ID"].astype(str))

rng = np.random.default_rng(0)
if len(hist_slide) > LEAKAGE_SUBSAMPLE:
    sub = rng.choice(len(hist_slide), size=LEAKAGE_SUBSAMPLE, replace=False)
    hist_X_sub, hist_slide_sub = hist_X[sub], hist_slide[sub]
else:
    hist_X_sub, hist_slide_sub = hist_X, hist_slide
leak_hist_raw = evaluate_latent_leakage(hist_X_sub, hist_slide_sub, cv=5)
print(f"Raw UNI2h patch embeddings ({len(hist_slide_sub)}-row sample) -> Slide_ID leakage: "
      f"{leak_hist_raw['leakage_mean']:.3f} (chance={leak_hist_raw['leakage_chance']:.3f})")

# transcriptomics: reuse the existing global PCA already computed in the combined file
# (CustomBinning.ipynb's "Recalculating global SVG, HVG, PCA" cell) rather than
# recomputing -- this is the same representation trans_enc's contrastive term
# ultimately has to disentangle Slide_ID from.
if "X_pca" in trans.obsm:
    trans_pca = np.asarray(trans.obsm["X_pca"][:, :50])
    trans_slide, _ = pd.factorize(trans.obs["Slide_ID"].astype(str))
    sub = rng.choice(len(trans_slide), size=min(LEAKAGE_SUBSAMPLE, len(trans_slide)), replace=False)
    leak_trans_raw = evaluate_latent_leakage(trans_pca[sub], trans_slide[sub], cv=5)
    print(f"Raw transcriptomics PCA ({len(sub)}-row sample) -> Slide_ID leakage: "
          f"{leak_trans_raw['leakage_mean']:.3f} (chance={leak_trans_raw['leakage_chance']:.3f})")
else:
    print("No X_pca in combined count_area_filtered_adata.h5ad -- skipping (rerun the "
          "'Recalculating global SVG, HVG, PCA' cell in CustomBinning.ipynb first for this)")

print(f"\nTotal wall-clock time: {(_time.time() - _t0) / 60:.1f} min")
