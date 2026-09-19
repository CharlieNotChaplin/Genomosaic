"""
UMAP overview figures of the learned embeddings, per held-out slide.

Figure 1  final_loso_umap_<unit>_<variant>_<slide>_slides.png
    UMAP of z_trans, z_token and z_joint on a subsample of ALL training slides plus the held-out slide, coloured by
    slide ID (top) and tumor/normal status (bottom). Shows Slide_ID structure directly: whether the held-out slide
    lands inside the training slides' point cloud or forms its own island (the "leakage" numbers, visualized).
Figure 2  final_loso_umap_<unit>_<variant>_<slide>_domains.png
    Held-out slide only. Rows: UMAP of z_trans, of z_joint, and of PCA(50) of the raw highly variable genes (no
    encoder, for contrast). Columns: the reference labels (label transfer; the pathologist annotation on CRC_P2),
    the raw Leiden partition (k closest to 8) and the same partition after majority-vote spatial refinement.
    Refinement changes labels, not the embedding, so the UMAP coordinates are identical in the last two columns; what
    changes is how spatially scattered/mixed labels look on the fixed embedding.
Clusterings run on the FULL held-out embedding; only the plotted points are subsampled.

Usage (from 04_eval/; CPU, UMAP on ~30-40k points takes about a minute per panel row):
    python -u evaluate_umap_overview.py
    python -u evaluate_umap_overview.py --slides CRC_P2 Normal_P3 --variant svg
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_clustering_quality as ecq
import evaluate_k_matched_ari as km


def umap_coords(X, seed=0):
    import umap
    return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed, low_memory=True).fit_transform(np.asarray(X, dtype=np.float32))


def scatter(ax, xy, labels, title, legend=False, s=2):
    codes, uniques = pd.factorize(pd.Series(np.asarray(labels).astype(str)))
    cmap = plt.get_cmap("tab20")
    fixed = {"tumor": "#c0392b", "normal": "#2e86c1"}  # high-contrast pair for the tumor/normal panels
    colour = lambda i: fixed.get(str(uniques[i]), cmap(i % 20))
    ax.scatter(xy[:, 0], xy[:, 1], c=[colour(c) for c in codes], s=s, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if legend and len(uniques) <= 12:
        for i, u in enumerate(uniques):
            ax.scatter([], [], color=colour(i), s=12, label=str(u)[:22])
        ax.legend(fontsize=6, loc="best", frameon=False, markerscale=1.2)


def k8_partition(z, refine_conn, resolutions=(0.08, 0.12, 0.18, 0.25, 0.35, 0.5)):
    a = ad.AnnData(np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X", random_state=0)
    best = None
    for r in resolutions:
        sc.tl.leiden(a, resolution=r, random_state=0, flavor="igraph", n_iterations=2, key_added="l")
        lab = a.obs["l"].to_numpy().copy()
        k = len(np.unique(lab))
        if best is None or abs(k - 8) < abs(best[0] - 8):
            best = (k, lab)
    raw = best[1]
    return raw, ecq.refine_domains(raw, refine_conn, mode="hard"), best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=list(ecq.UNITS), default="nuclei")
    ap.add_argument("--variant", choices=list(ecq.VARIANTS), default="svg")
    ap.add_argument("--slides", nargs="+", choices=ecq.SLIDES, default=["CRC_P2", "CRC_P5", "Normal_P3"])
    ap.add_argument("--n-per-train-slide", type=int, default=3000)
    ap.add_argument("--n-held", type=int, default=15000)
    ap.add_argument("--output-dir", type=Path, default=ecq.BASEDIR / "combined/03_integration/autoencoders")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for slide in args.slides:
        path = ecq.checkpoint_dir(args.unit) / f"loso_token_{slide}{ecq.run_tag(args.unit, args.variant)}_embeddings.npz"
        if not path.exists():
            continue
        t0 = time.time()
        npz = np.load(path, allow_pickle=True)
        tr_slides = np.asarray(npz["train_slide_name"]).astype(str)
        zt_tr, zk_tr = npz["train_z_trans"], npz["train_z_token"]
        zt, zk = npz["held_out_z_trans"], npz["held_out_z_token"]
        tag = f"{args.unit}_{args.variant}_{slide}"

        # ---- Figure 1: slide identity / disease status on train + held-out ----
        keep = np.concatenate([np.sort(rng.choice(np.where(tr_slides == s)[0], size=min(args.n_per_train_slide, (tr_slides == s).sum()),
                                                  replace=False)) for s in np.unique(tr_slides)])
        held_idx = np.sort(rng.choice(len(zt), size=min(args.n_held // 3, len(zt)), replace=False))
        slide_lab = np.concatenate([tr_slides[keep], np.full(len(held_idx), f"{slide} (held out)")])
        status = np.array(["tumor" if s.startswith("CRC") else "normal" for s in slide_lab])
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for j, (name, tr, ho) in enumerate([("z_trans", zt_tr, zt), ("z_token", zk_tr, zk),
                                             ("z_joint", np.concatenate([zt_tr, zk_tr], 1), np.concatenate([zt, zk], 1))]):
            U = umap_coords(np.concatenate([tr[keep], ho[held_idx]]))
            scatter(axes[0, j], U, slide_lab, f"{name}: coloured by slide", legend=(j == 2))
            scatter(axes[1, j], U, status, f"{name}: coloured by tumor / normal", legend=(j == 2))
        fig.suptitle(f"{slide} held out ({args.variant}): training slides and held-out slide in embedding space", fontsize=12)
        plt.tight_layout()
        fig.savefig(args.output_dir / f"final_loso_umap_{tag}_slides.png", dpi=130)
        plt.close(fig)
        print(f"{tag}: slides figure ({time.time() - t0:.0f}s)", flush=True)

        # ---- Figure 2: held-out domains, before/after refinement ----
        recon = ecq.reconstruct_held_out_cells(args.unit, slide)
        if recon is None or recon[0].shape[0] != len(zt):
            print(f"{tag}: cannot align units, skipping domains figure", flush=True)
            continue
        xy, trans_sub = recon
        conn = ecq.build_slide_graph(args.unit, xy)
        refs, _ = km.build_references(slide, xy, conn)
        ref_name = "gt" if "gt" in refs else ("lt" if "lt" in refs else "sr")
        ref_title = {"gt": "pathologist annotation", "lt": "label transfer", "sr": "SpaceRanger clusters"}[ref_name]
        from sklearn.decomposition import PCA
        X = trans_sub.X[:, trans_sub.var["highly_variable"].to_numpy().astype(bool)]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        pcs = PCA(n_components=50, random_state=0, svd_solver="randomized").fit_transform(X - X.mean(0))
        del X, trans_sub
        plot_idx = np.sort(rng.choice(len(zt), size=min(args.n_held * 2, len(zt)), replace=False))
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        for i, (name, E) in enumerate([("z_trans", zt), ("z_joint", np.concatenate([zt, zk], 1)), ("PCA(50) of raw HVG", pcs)]):
            raw, refined, k = k8_partition(E, conn)
            U = umap_coords(E[plot_idx])
            scatter(axes[i, 0], U, refs[ref_name][plot_idx], f"{name}: {ref_title}", legend=True)
            scatter(axes[i, 1], U, raw[plot_idx], f"{name}: Leiden, k={k} (raw)")
            scatter(axes[i, 2], U, refined[plot_idx], f"{name}: after majority-vote refinement")
        fig.suptitle(f"{slide} held out ({args.variant}): partitions on a fixed UMAP of the embedding", fontsize=12)
        plt.tight_layout()
        fig.savefig(args.output_dir / f"final_loso_umap_{tag}_domains.png", dpi=130)
        plt.close(fig)
        print(f"{tag}: domains figure ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
