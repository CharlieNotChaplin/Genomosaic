"""
Cluster-count-matched, multi-reference evaluation of the held-out embeddings (and of the baselines, on the
SAME nuclei). Main comparison script for the thesis' domain-identification results.

Why it exists: evaluate_clustering_quality.py clusters at ONE fixed Leiden resolution (0.5, ~14 clusters), while the
baselines (SpaGCN, STAGATE, SpatialFusion) are run for k = 7..10. ARI against these references depends strongly on
the cluster count (few, unbalanced classes: on Normal_P3 the ARI of z_trans is 0.76 at k=2 and 0.11 at 14 clusters),
so that comparison is not fair. Here every embedding is clustered over a grid of Leiden resolutions (which also
gives ARI-vs-k curves) and the runs with 7 <= k <= 10 are the primary, cluster-count-matched result. NO
resolution is ever chosen using a reference -- make_thesis_tables.py averages all runs with 7 <= k <= 10, exactly
like the baselines are averaged over their k = 7..10.

References ("ari_*" / "nmi_*" columns; NaN where a reference does not exist for the slide):
  lt   image-based label transfer (nearest 8um grid cell per nucleus). Derived from HISTOLOGY tokens.
  lts  the same reference after majority-vote smoothing on the nucleus graph. Both this and a refined partition are
       spatially smooth, so their agreement is inflated by construction -- read it next to lt, not instead of it.
  sr   SpaceRanger's own transcriptomics-only 8um graph-based clusters (nearest bin per nucleus). Derived from
       EXPRESSION, so it favours expression-only methods. Only slides with a native square_008um SpaceRanger output
       have it (not CRC_P3_SMAP / CRC_P4_SMAP). A quick scanpy re-implementation was tried and does NOT reproduce
       SpaceRanger's clusters (k-means ARI 0.1-0.2 vs native), so it is not used; the only faithful option would be
       re-running SpaceRanger on those two slides.
  srk  SpaceRanger's k-means clustering with K equal to the number of clusters found (only for 7 <= k <= 10).
  gt   the pathologist annotation (dominant component per 8um grid cell) -- exists ONLY for CRC_P2. It is the one
       slide with real ground truth; CRC_P2 has no label-transfer reference (it is the classifier's own training
       slide), but its LOSO fold is a valid held-out test of the embedding.

Partitions (column "partition"; k_raw = clusters of the raw Leiden run, k = after refinement, which can merge clusters):
raw Leiden labels, "hard" (majority vote) and "spagcn" (conditional rule)
refinement of them via refine_domains. Refinement is computed for 7 <= k <= 10 and the default resolution 0.5 only.
Also per row: silhouette (Euclidean, in the embedding that was clustered, 10k subsample), and the label-free spatial
metrics chaos (mean distance to the nearest unit of the same cluster, lower = more compact) and pas (share of units
whose label differs from >= 6 of their 10 nearest spatial neighbours, lower = fewer isolated units); both reward
smoothness, so they favour refined partitions and spatially smooth methods.

Also per row: hom_*/com_*/pur_* -- homogeneity, completeness and purity against each reference (see hom_com_pur), which
separate "over-split" (low completeness) from "mixed" (low homogeneity) where ARI only gives one number. With --smooth-hops the
embeddings are additionally averaged over spatial neighbours before clustering (embedding names z_trans_s1, z_joint_s2, ...).
Outputs (in --output-dir):
  final_loso_kmatched_ari.csv           one row per (unit, variant, slide, embedding, resolution, partition); runs with
                                        --units other than nuclei alone add the units to the file names (..._ari_bins.csv),
                                        so nuclei and bin runs never overwrite each other
  final_loso_partition_agreement.csv    how much the z_trans / z_token / z_joint partitions (k closest to 8) agree with
                                        each other (ARI, NMI, homogeneity, completeness): does the joint embedding find
                                        structure the single modalities do not? Needs >= 2 embeddings.
  final_loso_baselines_ours_units.csv   (--baselines) SpaGCN / STAGATE / SpatialFusion re-scored on exactly our nuclei
                                        (the baselines have no bin-level output, so this is skipped for bins)
Folds already in a file are skipped (--overwrite to redo), so a job that hits its time limit can simply be resubmitted.

Usage (from 04_eval/; CPU only. Roughly 1 ms per nucleus per embedding: ~1 min for a 50k slide, ~10-15 min for the
785k-nucleus slide; the whole nuclei sweep (2 variants x 3 embeddings) is a few hours sequentially):
    python -u evaluate_k_matched_ari.py --embeddings z_trans z_token z_joint --baselines
    python -u evaluate_k_matched_ari.py --shard 0/5 ...   # ... --shard 4/5: five parallel processes (see submit_evaluate_kmatched_parallel.sbatch)
    python -u evaluate_k_matched_ari.py --slides Normal_P3 --variants svg --embeddings z_trans
    python -u evaluate_k_matched_ari.py --units bins --shard 0/5 ...   # bin level: about twice the units per slide
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
from scipy.spatial import cKDTree
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_clustering_quality as ecq
from functions import eval_metrics as em
from functions import label_transfer as lt

# Leiden resolutions. On these embeddings 7-10 clusters appear at roughly 0.1-0.4 (slide dependent), 0.5 gives ~14 (the
# default of the earlier analysis) and >= 1 gives dozens -- slow and irrelevant for matching, so the grid stops at 1.0.
DEFAULT_RESOLUTIONS = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]
# grid for the spatially smoothed embeddings: smoothing removes variance, so fewer clusters appear at a given resolution
DEFAULT_SMOOTH_RESOLUTIONS = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.1, 0.125, 0.15, 0.175,
                              0.2, 0.25, 0.3, 0.4, 0.5, 1.0]  # smoothing lowers the cluster count, and more so with more hops
# nuclei per slide (after QC), used only to balance --shard assignments
SLIDE_SIZE = {
    "nuclei": {"CRC_P11mm": 785828, "CRC_P3_SMAP": 295977, "CRC_P2": 224825, "CRC_P5": 176903, "CRC_P4_SMAP": 151437,
               "Normal_P5": 127584, "CRC_P1": 120451, "CRC_P11mm_2_cut": 89110, "Normal_P3": 57187},
    "bins": {"CRC_P11mm": 1538238, "CRC_P3_SMAP": 602274, "CRC_P2": 444991, "CRC_P4_SMAP": 440676, "CRC_P5": 412811,
             "Normal_P5": 285345, "CRC_P11mm_2_cut": 283676, "CRC_P1": 332795, "Normal_P3": 156510},
}
K_MIN, K_MAX = 7, 10
REFS = ["lt", "lts", "sr", "srk", "gt"]
OUT_NAME = "final_loso_kmatched_ari.csv"
BASELINE_DIRS = {"SpaGCN": ecq.REPO_ROOT / "02_baselines/spagcn/outputs",
                 "STAGATE": ecq.REPO_ROOT / "02_baselines/stagate/outputs",
                 "SpatialFusion": ecq.REPO_ROOT / "02_baselines/spatialfusion/outputs"}


# ---- references ------------------------------------------------------------------------------

def sr_base(slide):
    return ecq.BASEDIR / slide / "output/binned_outputs/square_008um"


def _nearest_bin_labels(pos_xy, cluster, xy):
    return cluster[cKDTree(pos_xy).query(xy)[1]]


def spaceranger_reference(slide, xy):
    """(labels, source) of SpaceRanger's graph-based clustering per unit, or (None, None)."""
    base = sr_base(slide)
    cl = base / "analysis/clustering/gene_expression_graphclust/clusters.csv"
    if cl.exists():
        pos = pd.read_parquet(base / "spatial/tissue_positions.parquet").set_index("barcode")
        c = pd.read_csv(cl).set_index("Barcode").Cluster
        pos = pos.loc[pos.index.intersection(c.index)]
        return _nearest_bin_labels(pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(),
                                   c.loc[pos.index].to_numpy(), xy), "native"
    return None, None


def spaceranger_kmeans_references(slide, xy):
    """{K: labels} for SpaceRanger's k-means clustering, K = 7..10 (native outputs only)."""
    base = sr_base(slide)
    out = {}
    pos = None
    for k in range(K_MIN, K_MAX + 1):
        cl = base / f"analysis/clustering/gene_expression_kmeans_{k}_clusters/clusters.csv"
        if not cl.exists():
            continue
        if pos is None:
            pos = pd.read_parquet(base / "spatial/tissue_positions.parquet").set_index("barcode")
        c = pd.read_csv(cl).set_index("Barcode").Cluster
        p = pos.loc[pos.index.intersection(c.index)]
        out[k] = _nearest_bin_labels(p[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(),
                                     c.loc[p.index].to_numpy(), xy)
    return out


def ground_truth_reference(slide, xy):
    """Pathologist annotation (dominant component per 8um grid cell) -- CRC_P2 only."""
    if slide != lt.REFERENCE_SAMPLE:
        return None
    g = lt.aggregate_bin_annotation_to_grid(slide, group_size=4)
    g = g.assign(gt=lt.primary_tissuetype(g["annotation"]))
    return lt.assign_nearest_grid_label(xy, g[["pxl_col", "pxl_row", "gt"]], label_col="gt")


def build_references(slide, xy, conn):
    refs, sr_source = {}, None
    grid = ecq.load_grid_reference(slide)
    if grid is not None:
        refs["lt"] = lt.assign_nearest_grid_label(xy, grid, label_col=ecq.LABEL_COL)
        refs["lts"] = ecq.refine_domains(refs["lt"], conn, mode="hard")
    sr, sr_source = spaceranger_reference(slide, xy)
    if sr is not None:
        refs["sr"] = sr
    gt = ground_truth_reference(slide, xy)
    if gt is not None:
        refs["gt"] = gt
    return refs, sr_source


# ---- metrics ------------------------------------------------------------------------------

def chaos_pas(xy, labels):
    labels = np.asarray(labels)
    d = np.zeros(len(labels))
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) >= 2:
            d[idx] = cKDTree(xy[idx]).query(xy[idx], k=2)[0][:, 1]
    nn = cKDTree(xy).query(xy, k=11)[1]
    return float(d.mean()), float(((labels[nn[:, 1:]] != labels[:, None]).sum(1) >= 6).mean())


def hom_com_pur(lab, ref):
    """Homogeneity (is every cluster inside ONE reference class -- a clean sub-cluster scores high), completeness (is every
    reference class kept inside ONE cluster -- splitting a class scores low) and purity (share of units in the majority
    reference class of their cluster; the trivial value is the share of the biggest reference class). ARI mixes all of this:
    it penalises splitting a class exactly like merging two."""
    h, c, _ = homogeneity_completeness_v_measure(ref, lab)
    cm = contingency_matrix(ref, lab, sparse=True)
    return float(h), float(c), float(cm.max(axis=0).toarray().sum() / len(lab))


def ref_scores(lab, refs, srk):
    """ari_/nmi_/hom_/com_/pur_ for every reference; srk (K-matched SpaceRanger k-means) chosen by the number of clusters."""
    out = {f"{p}_{r}": np.nan for r in REFS for p in ("ari", "nmi", "hom", "com", "pur")}
    for r, v in refs.items():
        out[f"ari_{r}"], out[f"nmi_{r}"] = em.ari_between_labelings(lab, v), em.nmi_between_labelings(lab, v)
        out[f"hom_{r}"], out[f"com_{r}"], out[f"pur_{r}"] = hom_com_pur(lab, v)
    k = len(np.unique(lab))
    if k in srk:
        out["ari_srk"], out["nmi_srk"] = em.ari_between_labelings(lab, srk[k]), em.nmi_between_labelings(lab, srk[k])
        out["hom_srk"], out["com_srk"], out["pur_srk"] = hom_com_pur(lab, srk[k])
    return out


def silhouette_of(z, lab):
    try:
        return em.cluster_quality_metrics(z, lab)["silhouette"]
    except Exception:
        return np.nan


def sweep(X, refs, srk, xy, conn, meta, keep_labels, resolutions):
    """Leiden over RESOLUTIONS on one embedding; returns rows and {resolution: raw labels} for k in range."""
    a = ad.AnnData(np.asarray(X, dtype=np.float32))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X", random_state=0)
    rows = []
    for r in resolutions:
        sc.tl.leiden(a, resolution=float(r), random_state=0, flavor="igraph", n_iterations=2, key_added="l")
        lab = a.obs["l"].to_numpy()
        k = len(np.unique(lab))
        interest = (K_MIN <= k <= K_MAX) or abs(r - 0.5) < 1e-9
        parts = {"raw": lab}
        if interest:
            parts["hard"] = ecq.refine_domains(lab, conn, mode="hard")
            parts["spagcn"] = ecq.refine_domains(lab, conn, mode="spagcn")
        if K_MIN <= k <= K_MAX:
            keep_labels[float(r)] = lab
        for pname, pl in parts.items():
            row = {**meta, "resolution": float(r), "k_raw": k, "k": len(np.unique(pl)), "partition": pname, "n": len(pl),
                   **ref_scores(pl, refs, srk)}
            if interest:
                row["silhouette"] = silhouette_of(X, pl)
                row["chaos"], row["pas"] = chaos_pas(xy, pl)
            rows.append(row)
    return rows


def agreement_rows(chosen, meta):
    """Pairwise agreement of the partitions of different embeddings (k closest to 8)."""
    rows = []
    names = list(chosen)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (ra, ka, la), (rb, kb, lb) = chosen[names[i]], chosen[names[j]]
            h, c, _ = homogeneity_completeness_v_measure(la, lb)  # h: b's clusters pure w.r.t. a; c: a's clusters kept whole in b
            rows.append({**meta, "emb_a": names[i], "emb_b": names[j], "k_a": ka, "k_b": kb,
                         "ari": em.ari_between_labelings(la, lb), "nmi": em.nmi_between_labelings(la, lb),
                         "homogeneity_of_b_given_a": h, "completeness_of_b_given_a": c})
    return rows


def score_baselines(slide, ids, xy, refs, srk, out_path):
    rows = []
    for name, d in BASELINE_DIRS.items():
        for f in sorted(d.glob(f"{slide}_cluster_*.parquet")):
            df = pd.read_parquet(f).set_index("id")
            for kind in [c for c in ("pred", "refined_pred") if c in df.columns]:
                lab = df[kind].reindex(ids)
                keep = lab.notna().to_numpy()
                l = lab[keep].to_numpy()
                r_keep = {k: v[keep] for k, v in refs.items()}
                s_keep = {k: v[keep] for k, v in srk.items()}
                ch, pas = chaos_pas(xy[keep], l)
                rows.append({"method": name, "label_kind": kind, "held_out_slide": slide, "k": int(f.stem.split("_")[-1]),
                             "n": int(keep.sum()), "chaos": ch, "pas": pas, **ref_scores(l, r_keep, s_keep)})
    if rows:
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)


def append(df_rows, path):
    if df_rows:
        pd.DataFrame(df_rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", choices=list(ecq.UNITS), default=["nuclei"])
    ap.add_argument("--variants", nargs="+", choices=list(ecq.VARIANTS), default=["svg", "colon_xenium_panel"])
    ap.add_argument("--slides", nargs="+", choices=ecq.SLIDES, default=ecq.SLIDES)
    ap.add_argument("--embeddings", nargs="+", choices=ecq.EMBEDDINGS, default=["z_trans", "z_token", "z_joint"])
    ap.add_argument("--pca-baseline", action="store_true", help="also cluster PCA(50) of the 2000 HVGs (no encoder)")
    ap.add_argument("--baselines", action="store_true", help="also re-score SpaGCN/STAGATE/SpatialFusion on our units")
    ap.add_argument("--baselines-only", action="store_true",
                    help="ONLY re-score the baselines on our units (implies --baselines): no embeddings are clustered. Use it "
                         "after adding a baseline's outputs, or when the baseline parquet files were not available in the first run.")
    ap.add_argument("--output-dir", type=Path, default=ecq.BASEDIR / "combined/03_integration/autoencoders")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resolutions", nargs="+", type=float, default=DEFAULT_RESOLUTIONS,
                    help="Leiden resolutions; keep 0.5 for the default-resolution comparison. Runs whose raw partition has "
                         "7-10 clusters are the cluster-count-matched result, so make sure the grid hits that range.")
    ap.add_argument("--smooth-hops", nargs="*", type=int, default=[1, 2, 3],
                    help="ALSO cluster spatially smoothed embeddings: each unit's embedding is averaged with its Delaunay/grid "
                         "neighbours, this many rounds (all listed depths are reported; none is picked with a reference). Rows "
                         "get embedding names like z_joint_s2. Pass no values to switch it off. 2 hops = the receptive field of the "
                         "two-layer graph encoder and is the depth to quote as the main one, decided in advance.")
    ap.add_argument("--smooth-embeddings", nargs="+", choices=ecq.EMBEDDINGS, default=["z_trans", "z_joint"])
    ap.add_argument("--smooth-variants", nargs="+", choices=list(ecq.VARIANTS), default=["svg"])
    ap.add_argument("--smooth-resolutions", nargs="+", type=float, default=DEFAULT_SMOOTH_RESOLUTIONS)
    ap.add_argument("--smooth-only", action="store_true",
                    help="run ONLY the smoothed-embedding sweeps (writes final_loso_kmatched_ari_smoothed*.csv), e.g. to add "
                         "them to folds that were already clustered without them")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="run only the I-th of N size-balanced groups of slides (0-based) and write shard-suffixed CSVs, so "
                         "N processes/array tasks can run in parallel without touching the same file; make_thesis_tables.py "
                         "reads all shards")
    args = ap.parse_args()
    args.baselines = args.baselines or args.baselines_only
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        load = [0] * n
        assign = {}
        size = {sl: sum(SLIDE_SIZE[u].get(sl, 1) for u in args.units) for sl in args.slides}
        for sl in sorted(args.slides, key=lambda x: -size[x]):  # greedy: biggest slide to least-loaded shard
            j = load.index(min(load))
            assign[sl] = j
            load[j] += size[sl]
        args.slides = [sl for sl in args.slides if assign[sl] == i]
        suffix = f".shard{i}of{n}"
        print(f"shard {i}/{n}: slides {args.slides}", flush=True)
    stem = "final_loso_kmatched_ari" + ("_smoothed" if args.smooth_only else "")
    unit_tag = "" if args.units == ["nuclei"] else "_" + "_".join(args.units)
    out = args.output_dir / f"{stem}{unit_tag}{suffix}.csv"
    out_agree = args.output_dir / f"final_loso_partition_agreement{unit_tag}{suffix}.csv"
    out_base = args.output_dir / f"final_loso_baselines_ours_units{suffix}.csv"
    if args.overwrite:
        for p in (out, out_agree, *([out_base] if "nuclei" in args.units else [])):
            p.unlink(missing_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out, usecols=["unit", "variant", "held_out_slide"])
        done = set(map(tuple, d.drop_duplicates().to_numpy()))
    base_done = set(pd.read_csv(out_base, usecols=["held_out_slide"]).held_out_slide) if out_base.exists() else set()

    for unit in args.units:
        for slide in args.slides:
            todo = [v for v in (args.smooth_variants if args.smooth_only else args.variants)
                    if not args.baselines_only and (unit, v, slide) not in done and
                    (ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, v)}_embeddings.npz").exists()]
            need_base = args.baselines and unit == "nuclei" and slide not in base_done
            if not todo and not need_base:
                continue
            recon = ecq.reconstruct_held_out_cells(unit, slide)
            if recon is None:
                continue
            xy, trans_sub = recon
            conn = ecq.build_slide_graph(unit, xy)
            refs, sr_source = build_references(slide, xy, conn)
            if not refs:
                print(f"  {slide}: no reference of any kind -- skipping", flush=True)
                continue
            srk = spaceranger_kmeans_references(slide, xy)
            print(f"{slide}: n={len(xy)}, references {sorted(refs)}, sr_source={sr_source}, "
                  f"k-means K={sorted(srk)}", flush=True)
            if need_base:
                score_baselines(slide, trans_sub.obs["id"].astype(str).to_numpy(), xy, refs, srk, out_base)
            pca = None
            if args.pca_baseline and todo:
                from sklearn.decomposition import PCA
                X = trans_sub.X[:, trans_sub.var["highly_variable"].to_numpy().astype(bool)]
                X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                pca = PCA(n_components=50, random_state=0, svd_solver="randomized").fit_transform(X - X.mean(0))
                del X
            del trans_sub
            for v in todo:
                t0 = time.time()
                z = np.load(ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, v)}_embeddings.npz")
                zt, zk = z["held_out_z_trans"], z["held_out_z_token"]
                if zt.shape[0] != len(xy):
                    print(f"  {unit}/{v}/{slide}: row mismatch {zt.shape[0]} vs {len(xy)} -- skipping", flush=True)
                    continue
                embs = {"z_trans": zt, "z_token": zk, "z_joint": np.concatenate([zt, zk], 1)}
                rows, chosen = [], {}
                for e in ([] if args.smooth_only else args.embeddings):
                    kept = {}
                    rows += sweep(embs[e], refs, srk, xy, conn, dict(unit=unit, variant=v, held_out_slide=slide,
                                                                      embedding=e, sr_source=sr_source), kept, args.resolutions)
                    if len(kept) < 2:
                        print(f"  WARNING {unit}/{v}/{slide}/{e}: only {len(kept)} Leiden run(s) with 7-10 clusters -- "
                              f"add resolutions between the ones that gave fewer/more clusters", flush=True)
                    if kept:
                        r_best = min(kept, key=lambda r: (abs(len(np.unique(kept[r])) - 8), r))
                        chosen[e] = (r_best, len(np.unique(kept[r_best])), kept[r_best])
                if args.smooth_hops and v in args.smooth_variants:
                    A = conn.tocsr() + sp.eye(conn.shape[0])
                    A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A  # mean over the unit itself and its neighbours
                    for e in args.smooth_embeddings:
                        X = embs[e]
                        for hop in range(1, max(args.smooth_hops) + 1):
                            X = A @ X
                            if hop not in args.smooth_hops:
                                continue
                            kept = {}
                            rows += sweep(X, refs, srk, xy, conn, dict(unit=unit, variant=v, held_out_slide=slide,
                                                                        embedding=f"{e}_s{hop}", sr_source=sr_source),
                                          kept, args.smooth_resolutions)
                            if len(kept) < 2:
                                print(f"  WARNING {unit}/{v}/{slide}/{e}_s{hop}: only {len(kept)} Leiden run(s) with 7-10 clusters", flush=True)
                if pca is not None and v == todo[0]:
                    rows += sweep(pca, refs, srk, xy, conn, dict(unit=unit, variant="pca_baseline", held_out_slide=slide,
                                                                  embedding="pca50_hvg", sr_source=sr_source), {}, args.resolutions)
                append(rows, out)
                if len(chosen) >= 2 and not args.smooth_only:
                    append(agreement_rows(chosen, dict(unit=unit, variant=v, held_out_slide=slide)), out_agree)
                print(f"  {unit}/{v}/{slide}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
