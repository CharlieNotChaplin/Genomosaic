"""
UMAPs of the best-performing clustering configuration, chosen from the cluster-count-matched results.

A configuration is (unit, gene variant, embedding, partition), where the embedding may be spatially smoothed
(``z_trans_s2``: two rounds of neighbour averaging) and the partition may be the raw Leiden labels or labels refined
by ``refine_domains`` (``hard``, ``spagcn``). Each configuration is scored by its mean ARI over the Leiden runs with
7 to 10 clusters, averaged over slides, and only configurations that have a score on every slide with the reference
are compared. The best configuration is chosen once per criterion:
    sr   ARI against the SpaceRanger clusters
    lt   ARI against the label-transfer reference (CRC_P2 has none and is shown with its pathologist annotation)
    sr_hom, lt_hom   homogeneity instead of ARI (optional, --criteria). ARI punishes a cluster that is split into
                     two sub-clusters inside one reference class exactly like a merge of two classes; homogeneity only
                     asks whether every cluster lies inside one reference class. At a fixed number of clusters it cannot
                     be won by splitting more, so it is the fairer score for a coarse reference.
The ranking also lists NMI, homogeneity and completeness of every configuration next to the ARI.

Per criterion and slide, ``final_loso_best_<criterion>_umap_<slide>.png`` has three panels:
    1. UMAP of the unsmoothed embedding of a sample of the training slides and the held-out slide, coloured by slide
    2. UMAP of the configuration's embedding (smoothed if it is a smoothed one) on the held-out slide, coloured by its
       final clusters (the Leiden run with the cluster count closest to 8, refined if the configuration says so)
    3. the same coordinates coloured by the reference labels
The ranking of the top configurations is written to ``final_loso_best_configs.csv``.

Input: ``final_loso_kmatched_ari*.csv`` (evaluate_k_matched_ari.py) and the saved fold embeddings.

Usage (from 04_eval/; CPU, about one minute per UMAP):
    python -u evaluate_best_config_umap.py
    python -u evaluate_best_config_umap.py --criteria lt --slides CRC_P2 Normal_P3
    python -u evaluate_best_config_umap.py --rank-only
    python -u evaluate_best_config_umap.py --jobs 4              four figures at a time
"""

import argparse
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_clustering_quality as ecq
import evaluate_k_matched_ari as km
from evaluate_umap_overview import scatter, umap_coords

K_MIN, K_MAX, K_TARGET = 7, 10, 8
CRITERIA = {
    "sr": ("ari_sr", "SpaceRanger clusters"),
    "lt": ("ari_lt", "label transfer"),
    "sr_hom": ("hom_sr", "SpaceRanger clusters, homogeneity"),
    "lt_hom": ("hom_lt", "label transfer, homogeneity"),
}
DEFAULT_CRITERIA = ["sr", "lt"]
CONFIG_COLUMNS = ["unit", "variant", "embedding", "partition"]


def load_matched_results(base):
    """All k-matched result rows (shards and smoothing-only files) restricted to the 7 to 10 cluster runs."""
    files = sorted(base.glob("final_loso_kmatched_ari*.csv"))
    if not files:
        raise FileNotFoundError(f"No final_loso_kmatched_ari*.csv in {base}")
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = rows[rows["k_raw"].between(K_MIN, K_MAX) & (rows["variant"] != "pca_baseline")]
    return rows.rename(columns={"held_out_slide": "slide"})


def rank_configurations(rows, metric, smoothed_only=False):
    """
    Mean of ``metric`` over the matched-k runs per slide, then over slides, for every configuration that has a
    value on all slides that have the reference. Returns a DataFrame sorted by score, best first.
    """
    reference = metric.split("_")[1]
    columns = [metric] + [f"{m}_{reference}" for m in ("ari", "nmi", "hom", "com") if f"{m}_{reference}" != metric]
    per_slide = rows.dropna(subset=[metric]).groupby(CONFIG_COLUMNS + ["slide"])[columns].mean().reset_index()
    ranking = per_slide.groupby(CONFIG_COLUMNS).agg(
        score=(metric, "mean"), n_slides=(metric, "count"), **{c: (c, "mean") for c in columns[1:]}).reset_index()
    ranking = ranking[ranking["n_slides"] == ranking["n_slides"].max()]
    if smoothed_only:
        smoothed = ranking["embedding"].str.contains("_s[0-9]$") | (ranking["partition"] != "raw")
        ranking = ranking[smoothed]
    return ranking.sort_values("score", ascending=False).reset_index(drop=True)


def smooth_embedding(z, conn, hops):
    """Mean over each unit and its spatial neighbours, applied ``hops`` times."""
    adjacency = conn.tocsr() + sp.eye(conn.shape[0])
    adjacency = sp.diags(1.0 / np.asarray(adjacency.sum(1)).ravel()) @ adjacency
    for _ in range(hops):
        z = adjacency @ z
    return z


def split_embedding_name(name):
    """``z_trans_s2`` -> (``z_trans``, 2); an unsmoothed name gives 0 hops."""
    base, _, suffix = name.rpartition("_s")
    return (base, int(suffix)) if base and suffix.isdigit() else (name, 0)


def final_clusters(rows, config, slide, z, conn):
    """Leiden labels at the resolution whose cluster count is closest to 8, refined if the partition says so."""
    runs = rows[(rows[CONFIG_COLUMNS] == pd.Series(config)).all(axis=1) & (rows["slide"] == slide)]
    resolution = runs.assign(gap=(runs["k_raw"] - K_TARGET).abs()).sort_values(["gap", "resolution"]).iloc[0]["resolution"]
    adata = ad.AnnData(np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", random_state=0)
    sc.tl.leiden(adata, resolution=float(resolution), random_state=0, flavor="igraph", n_iterations=2, key_added="leiden")
    labels = adata.obs["leiden"].to_numpy()
    if config["partition"] != "raw":
        labels = ecq.refine_domains(labels, conn, mode=config["partition"])
    return labels, float(resolution)


def sample_indices(n, size, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(size, n), replace=False))


def plot_slide(rows, config, criterion, slide, args, output_dir):
    """Draw the three panels for one slide; returns False if the slide has no fold or reference."""
    unit, variant = config["unit"], config["variant"]
    npz_path = ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, variant)}_embeddings.npz"
    reconstruction = ecq.reconstruct_held_out_cells(unit, slide) if npz_path.exists() else None
    if reconstruction is None:
        return False
    xy, _ = reconstruction
    conn = ecq.build_slide_graph(unit, xy)
    references, _ = km.build_references(slide, xy, conn)
    reference_name = criterion.split("_")[0]
    reference_key = "gt" if reference_name == "lt" and slide == "CRC_P2" else reference_name
    if reference_key not in references:
        return False

    npz = np.load(npz_path, allow_pickle=True)
    parts = {"z_trans": (npz["train_z_trans"], npz["held_out_z_trans"]),
             "z_token": (npz["train_z_token"], npz["held_out_z_token"])}
    parts["z_joint"] = tuple(np.concatenate([parts["z_trans"][i], parts["z_token"][i]], axis=1) for i in (0, 1))
    base_name, hops = split_embedding_name(config["embedding"])
    train_z, held_z = parts[base_name]
    if held_z.shape[0] != xy.shape[0]:
        print(f"  {slide}: embedding rows do not match the reconstructed units -- skipped")
        return False

    shown = smooth_embedding(held_z, conn, hops) if hops else held_z
    labels, resolution = final_clusters(rows, config, slide, shown, conn)
    held_idx = sample_indices(len(held_z), args.n_held, seed=0)
    train_slides = np.asarray(npz["train_slide_name"]).astype(str)
    train_idx = np.concatenate([
        np.where(train_slides == s)[0][sample_indices((train_slides == s).sum(), args.n_per_train_slide, seed=1)]
        for s in np.unique(train_slides)
    ])
    slide_coords = umap_coords(np.concatenate([train_z[train_idx], held_z[held_idx]]))
    slide_names = train_slides[train_idx]
    cluster_coords = umap_coords(shown[held_idx])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n_train = len(train_idx)
    scatter(axes[0], slide_coords[:n_train], slide_names[:n_train], f"{base_name}, slide identity", legend=True)
    axes[0].scatter(slide_coords[n_train:, 0], slide_coords[n_train:, 1], c="black", s=2, linewidths=0,
                    rasterized=True, label=f"{slide} (held out)")
    handles, names = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, names, fontsize=6, loc="best", frameon=False, markerscale=1.2)
    n_clusters = len(np.unique(labels))
    scatter(axes[1], cluster_coords, labels[held_idx],
            f"{config['embedding']}, {config['partition']} clusters (k={n_clusters}, resolution {resolution:g})")
    reference_title = "pathologist annotation" if reference_key == "gt" else CRITERIA[criterion][1]
    scatter(axes[2], cluster_coords, references[reference_key][held_idx], reference_title, legend=True)
    fig.suptitle(f"{slide}: {unit}, {variant}; best configuration by ARI vs. {CRITERIA[criterion][1]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / f"final_loso_best_{criterion}_umap_{slide}.png", dpi=150)
    plt.close(fig)
    return True


def draw_task(task):
    """Worker entry point: ``task`` is (rows, config, criterion, slide, args); returns (criterion, slide, drawn)."""
    rows, config, criterion, slide, args = task
    return criterion, slide, plot_slide(rows, config, criterion, slide, args, args.output_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--criteria", nargs="+", choices=list(CRITERIA), default=DEFAULT_CRITERIA)
    ap.add_argument("--slides", nargs="+", choices=ecq.SLIDES, default=ecq.SLIDES)
    ap.add_argument("--units", nargs="+", choices=list(ecq.UNITS), default=list(ecq.UNITS),
                    help="units whose configurations are compared")
    ap.add_argument("--smoothed-only", action="store_true",
                    help="only compare configurations with a smoothed embedding or a refined partition")
    ap.add_argument("--top", type=int, default=5, help="number of ranked configurations to print and save")
    ap.add_argument("--rank-only", action="store_true", help="print the ranking and skip the figures")
    ap.add_argument("--n-held", type=int, default=15000, help="held-out units shown per UMAP")
    ap.add_argument("--n-per-train-slide", type=int, default=3000, help="units shown per training slide")
    ap.add_argument("--jobs", type=int, default=1,
                    help="figures drawn in parallel (one process each; the largest slides need several GB of memory)")
    ap.add_argument("--output-dir", type=Path, default=ecq.BASEDIR / "combined/03_integration/autoencoders")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_matched_results(args.output_dir)
    rows = rows[rows["unit"].isin(args.units)]
    ranked = []
    for criterion in args.criteria:
        metric, description = CRITERIA[criterion]
        ranking = rank_configurations(rows, metric, args.smoothed_only)
        top = ranking.head(args.top).assign(criterion=criterion, rank=lambda d: d.index + 1)
        ranked.append(top)
        print(f"\nTop {args.top} by {metric} ({description}), over {int(ranking['n_slides'].max())} slides:")
        extra = [c for c in top.columns if c[:4] in ("ari_", "nmi_", "hom_", "com_")]
        print(top[["rank"] + CONFIG_COLUMNS + ["score"] + extra].to_string(index=False, float_format="%.3f"))
    pd.concat(ranked).to_csv(args.output_dir / "final_loso_best_configs.csv", index=False)
    if args.rank_only:
        return

    tasks = []
    for top in ranked:
        criterion = top["criterion"].iloc[0]
        config = top.iloc[0][CONFIG_COLUMNS].to_dict()
        slides_with_reference = set(rows.loc[rows[CRITERIA[criterion][0]].notna(), "slide"])
        if criterion.startswith("lt"):
            slides_with_reference |= {"CRC_P2"}
        tasks += [(rows, config, criterion, slide, args) for slide in args.slides if slide in slides_with_reference]
    tasks.sort(key=lambda t: -km.SLIDE_SIZE[t[1]["unit"]].get(t[3], 0))

    if args.jobs <= 1:
        results = map(draw_task, tasks)
    else:
        os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or args.jobs) // args.jobs)))
        pool = ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(draw_task, tasks)
    for criterion, slide, drawn in results:
        print(f"{criterion}/{slide}: {'figure written' if drawn else 'skipped'}", flush=True)


if __name__ == "__main__":
    main()
