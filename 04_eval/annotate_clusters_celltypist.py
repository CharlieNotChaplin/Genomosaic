"""
Cluster annotation against published cell-type references (CellTypist models), as a stronger alternative to the
hand-curated marker list of evaluate_final_loso_7_9.py section 8.

For every fold, the transcriptomic embedding z_trans is clustered (Leiden, one resolution), CellTypist is run on the
held-out units' expression and each cluster gets the majority-voted cell type (celltypist's own over-clustering
majority vote, using OUR clusters as the over-clustering). One row per (unit, variant, slide, model, cluster).

NOT TESTED HERE: the `celltypist` package was not installed in the environment this was written in, and the
models must be downloaded once (internet needed; otherwise pass --model-path to a local .pkl). Install with
`pip install celltypist`. Read the caveats before using the output in the thesis:
  - CellTypist expects log1p(counts normalised to 10,000 per cell). This project's count_area_filtered_adata.h5ad is
    already normalize_total(1e4)+log1p (see cell_preprocessing.py); celltypist checks this and will raise if not.
  - The models were trained on single-cell data; nuclei-level Visium HD counts are sparse (median ~150-280 UMIs),
    so predictions are noisy and the majority vote per cluster is what carries the signal. Confidence scores
    (mean_conf) are stored so weak calls can be seen.
  - A model's label vocabulary is not the same as the marker list's; the two tables are not directly comparable.
  - --max-cells caps the units passed to CellTypist per fold (random subset, seeded) to keep memory and time in check.

Default model: Cells_Intestinal_Tract.pkl (gut atlas, Elmentaite et al.). Others can be added with --models.

Output: final_loso_cluster_annotation_celltypist.csv in --output-dir; folds already in the file are skipped.

Usage (from 04_eval/; CPU):
    python -u annotate_clusters_celltypist.py
    python -u annotate_clusters_celltypist.py --models Cells_Intestinal_Tract.pkl --variants svg --slides Normal_P3
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_clustering_quality as ecq


def load_model(name_or_path, allow_download=True):
    import celltypist
    from celltypist import models
    p = Path(name_or_path)
    if p.exists():
        return models.Model.load(str(p))
    if allow_download:
        models.download_models(model=[name_or_path], force_update=False)
    return models.Model.load(name_or_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", choices=list(ecq.UNITS), default=["nuclei"])
    ap.add_argument("--variants", nargs="+", choices=list(ecq.VARIANTS), default=["svg", "colon_xenium_panel"])
    ap.add_argument("--slides", nargs="+", choices=ecq.SLIDES, default=ecq.SLIDES)
    ap.add_argument("--models", nargs="+", default=["Cells_Intestinal_Tract.pkl"],
                    help="CellTypist model names (downloaded on first use) or paths to .pkl files")
    ap.add_argument("--resolution", type=float, default=0.5, help="Leiden resolution of the clusters that get annotated")
    ap.add_argument("--max-cells", type=int, default=100_000)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=ecq.BASEDIR / "combined/03_integration/autoencoders")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        import celltypist  # noqa: F401
    except ImportError:
        sys.exit("celltypist is not installed: pip install celltypist")
    models = {m: load_model(m, allow_download=not args.no_download) for m in args.models}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "final_loso_cluster_annotation_celltypist.csv"
    if args.overwrite:
        out.unlink(missing_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out, usecols=["unit", "variant", "held_out_slide"])
        done = set(map(tuple, d.drop_duplicates().to_numpy()))

    import celltypist
    rng = np.random.default_rng(0)
    for unit in args.units:
        for slide in args.slides:
            variants = [v for v in args.variants if (unit, v, slide) not in done and
                        (ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, v)}_embeddings.npz").exists()]
            if not variants:
                continue
            recon = ecq.reconstruct_held_out_cells(unit, slide)
            if recon is None:
                continue
            xy, trans_sub = recon
            for v in variants:
                t0 = time.time()
                z = np.load(ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, v)}_embeddings.npz")["held_out_z_trans"]
                if z.shape[0] != len(xy):
                    print(f"  {unit}/{v}/{slide}: row mismatch -- skipping", flush=True)
                    continue
                labels = ecq.cluster_embedding(z, resolution=args.resolution)
                idx = np.arange(len(z))
                if args.max_cells and len(z) > args.max_cells:
                    idx = np.sort(rng.choice(len(z), size=args.max_cells, replace=False))
                sub = ad.AnnData(X=trans_sub.X[idx], obs=pd.DataFrame(index=trans_sub.obs_names[idx]),
                                 var=pd.DataFrame(index=trans_sub.var_names))
                rows = []
                for mname, model in models.items():
                    pred = celltypist.annotate(sub, model=model, majority_voting=True, over_clustering=labels[idx])
                    obs = pred.predicted_labels
                    df = pd.DataFrame({"cluster": labels[idx], "majority": obs["majority_voting"].to_numpy(),
                                       "conf": pred.probability_matrix.max(axis=1).to_numpy()})
                    for c, g in df.groupby("cluster"):
                        top = g["majority"].value_counts()
                        rows.append({"unit": unit, "variant": v, "held_out_slide": slide, "model": mname, "cluster": c,
                                     "assigned_cell_type": top.index[0], "n_cells_annotated": len(g),
                                     "share_of_cluster_with_type": float(top.iloc[0] / len(g)), "mean_conf": float(g["conf"].mean())})
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"  {unit}/{v}/{slide}: {len(rows)} cluster annotations ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
