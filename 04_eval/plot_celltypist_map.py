"""
Spatial cluster map for one fold, coloured by Leiden cluster and labelled with each cluster's
CellTypist-assigned cell type in a side legend (annotate_clusters_celltypist.py's atlas-based
annotation, not the hand-curated marker list of evaluate_final_loso_7_9.py section 8).

Re-clusters with the exact same call (embedding, resolution, seed) annotate_clusters_celltypist.py
used, so cluster IDs line up with its CSV without needing to re-run CellTypist itself.

Usage (from 04_eval/):
    python plot_celltypist_map.py
    python plot_celltypist_map.py --unit bins --variant svg --slide CRC_P2
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_clustering_quality as ecq


def dot_size(n_units):
    """Marker area that shrinks with the number of plotted units -- identical to
    evaluate_final_loso_7_9.py's own dot_size(), duplicated here rather than imported so this
    script does not pay for that module's heavy top-level imports (torch, etc.) just for one
    two-line helper."""
    return float(np.clip(60_000 / n_units, 0.05, 1.0)) * (2 / 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", default="nuclei", choices=list(ecq.UNITS))
    ap.add_argument("--variant", default="svg", choices=list(ecq.VARIANTS))
    ap.add_argument("--slide", default="CRC_P2")
    ap.add_argument("--model", default="Cells_Intestinal_Tract.pkl")
    ap.add_argument("--resolution", type=float, default=0.5, help="Must match the resolution "
                    "annotate_clusters_celltypist.py used, so cluster IDs line up with its CSV.")
    ap.add_argument("--annotation-csv", type=Path,
                    default=ecq.BASEDIR / "combined/03_integration/autoencoders/final_loso_cluster_annotation_celltypist.csv")
    ap.add_argument("--output", type=Path,
                    default=Path("figures/07_spatial_clusters_celltypist_CRC_P2.pdf"))
    args = ap.parse_args()

    ann = pd.read_csv(args.annotation_csv)
    ann = ann[(ann.unit == args.unit) & (ann.variant == args.variant) & (ann.held_out_slide == args.slide) &
              (ann.model == args.model)]
    if not len(ann):
        raise SystemExit(f"no CellTypist annotation rows for {args.unit}/{args.variant}/{args.slide}/{args.model} "
                          f"in {args.annotation_csv} -- has that fold finished?")
    cell_type_of = dict(zip(ann.cluster.astype(str), ann.assigned_cell_type))

    print(f"reconstructing {args.unit}/{args.variant}/{args.slide} ...", flush=True)
    recon = ecq.reconstruct_held_out_cells(args.unit, args.slide)
    if recon is None:
        raise SystemExit(f"no token data for {args.unit}/{args.slide}")
    cell_xy, trans_sub = recon
    npz = np.load(ecq.checkpoint_dir(args.unit) / f"loso_token_{args.slide}{ecq.run_tag(args.unit, args.variant)}_embeddings.npz")
    z = npz["held_out_z_trans"]
    if z.shape[0] != len(cell_xy):
        raise SystemExit(f"row mismatch: {z.shape[0]} embeddings vs {len(cell_xy)} reconstructed units")

    labels = ecq.cluster_embedding(z, resolution=args.resolution).astype(str)
    uniq = sorted(set(labels), key=int)  # numeric order (Leiden's own labels are strings: "0", "1", ..., "10", ...)
    codes = np.array([uniq.index(l) for l in labels])
    n_clusters = len(uniq)
    print(f"{n_clusters} clusters", flush=True)

    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=max(n_clusters - 1, 1))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(cell_xy[:, 0], cell_xy[:, 1], c=codes, cmap=cmap, norm=norm,
               s=dot_size(len(cell_xy)), linewidths=0, rasterized=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{args.slide}: {args.unit}, {args.variant} -- CellTypist-annotated Leiden clusters")

    counts = pd.Series(labels).value_counts()
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(norm(i)), markeredgewidth=0,
                   markersize=8, label=f"{cl}: {cell_type_of.get(cl, '?')} (n={counts.get(cl, 0)})")
        for i, cl in enumerate(uniq)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False,
              title="Cluster: CellTypist cell type", title_fontsize=9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
