"""
Marker-score heatmap backing Table "Marker-based cluster annotation" (tab:07_cluster_annotation):
that table only shows each cluster's arg-max cell type, this shows the full cell-type x cluster
score matrix underneath it, so a reader can see how confident/ambiguous each call was.

Reuses evaluate_final_loso_7_9.py's exact scoring path (CANONICAL_COLON_MARKERS,
annotate_clusters_by_markers -- sc.tl.score_genes per cell, averaged per Leiden cluster) on one
representative fold, instead of the full multi-slide/-variant sweep that script runs for the CSV.

Usage (from 04_eval/):
    python plot_marker_score_heatmap.py
    python plot_marker_score_heatmap.py --unit nuclei --variant svg --slide CRC_P2
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_clustering_quality as ecq
import evaluate_final_loso_7_9 as ef7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", default="nuclei", choices=list(ecq.UNITS))
    ap.add_argument("--variant", default="svg", choices=list(ecq.VARIANTS))
    ap.add_argument("--slide", default="CRC_P2")
    ap.add_argument("--output", type=Path, default=Path("figures/07_marker_score_heatmap_CRC_P2.png"))
    args = ap.parse_args()

    print(f"reconstructing {args.unit}/{args.variant}/{args.slide} ...", flush=True)
    recon = ecq.reconstruct_held_out_cells(args.unit, args.slide)
    if recon is None:
        raise SystemExit(f"no token data for {args.unit}/{args.slide}")
    cell_xy, trans_sub = recon
    z = ef7.load_z_trans(args.unit, args.variant, args.slide, cell_xy.shape[0])
    if z is None:
        raise SystemExit(f"no z_trans embeddings for {args.unit}/{args.variant}/{args.slide}")
    labels = ef7.get_z_trans_clusters(args.unit, args.variant, args.slide, z)
    assignment, per_cluster = ef7.annotate_clusters_by_markers(trans_sub, labels, ef7.CANONICAL_COLON_MARKERS)
    print(f"{len(per_cluster)} clusters x {per_cluster.shape[1]} cell types")

    # z-score each cell type's column across clusters: score_genes' raw scale isn't comparable between
    # gene sets of different sizes, standardizing highlights *which* cluster is high for a given type.
    z_scores = (per_cluster - per_cluster.mean()) / per_cluster.std(ddof=0)
    order = per_cluster.max(axis=0).sort_values(ascending=False).index  # cell types, most-confident first
    z_scores = z_scores[order]

    n_clusters, n_types = z_scores.shape
    fig, ax = plt.subplots(figsize=(0.62 * n_types + 2, 0.42 * n_clusters + 1.5))
    # clip (not rescale) the colour limit to a robust percentile: one outlier cluster with an extreme
    # score for several types at once (small/degenerate clusters do this to score_genes' control-gene
    # normalization) would otherwise wash out the contrast for every other, well-behaved cluster.
    lim = float(np.clip(np.percentile(np.abs(z_scores.to_numpy()), 90), 1.5, None))
    im = ax.imshow(z_scores.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(n_types)); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters)); ax.set_yticklabels([f"{c} (n={int((labels == c).sum())})" for c in z_scores.index], fontsize=8)
    ax.set_xlabel("Marker-set cell type"); ax.set_ylabel("Leiden cluster")
    for i, cluster_id in enumerate(z_scores.index):
        j = list(order).index(assignment[cluster_id])
        # white-then-black double stroke so the box stays visible on both dark and light cells
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=3.2))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.4))
    fig.colorbar(im, ax=ax, label=f"Marker score (z-scored across clusters, clipped at $\\pm${lim:.1f})", shrink=0.8, extend="both")
    fig.suptitle(f"{args.slide}: {args.unit}, {args.variant} -- marker-set score per Leiden cluster", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
