"""
How well do the two independent-ish references agree with each other, at the level of the native 8um bins?
  label transfer (LT)   image-based classifier predictions per 8um bin (raw and majority-vote smoothed columns of
                        <slide>_grid_predictions.parquet)
  SpaceRanger (SR)      SpaceRanger's own transcriptomics-only clustering of the same bins: graph-based clusters and
                        k-means with K = 8 (analysis/clustering/), computed per slide
Both are joined by barcode, so no spatial lookup is involved. Slides without a native square_008um output (the two SMAP
slides) have no SR clusters, and CRC_P2 has no LT predictions (it is the classifier's training slide), so 7 slides remain.
Output: <output-dir>/reference_agreement_sr_vs_lt.csv (ARI, NMI per slide); tables via make_thesis_tables.py.

Usage (from 04_eval/): python -u evaluate_reference_agreement.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from functions import eval_metrics as em

BASEDIR = REPO_ROOT / "data"
SLIDES = ["CRC_P1", "CRC_P11mm", "CRC_P11mm_2_cut", "CRC_P5", "CRC_P6_5mm", "Normal_P3", "Normal_P5"]
LT_RAW, LT_SMOOTH = "predicted_tissuetype_grid_image", "predicted_tissuetype_grid_image_smoothed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=BASEDIR / "combined/03_integration/autoencoders")
    args = ap.parse_args()
    rows = []
    for s in SLIDES:
        pp = BASEDIR / s / "02_baselines/label_transfer" / f"{s}_grid_predictions.parquet"
        cols = set(pq.ParquetFile(pp).schema.names)
        lt = pd.read_parquet(pp, columns=["bin_id", LT_RAW] + ([LT_SMOOTH] if LT_SMOOTH in cols else [])).set_index("bin_id")
        base = BASEDIR / s / "output/binned_outputs/square_008um/analysis/clustering"
        graph = pd.read_csv(base / "gene_expression_graphclust/clusters.csv").set_index("Barcode").Cluster.rename("graph")
        km8 = pd.read_csv(base / "gene_expression_kmeans_8_clusters/clusters.csv").set_index("Barcode").Cluster.rename("km8")
        j = lt.join(graph, how="inner").join(km8, how="inner")
        r = {"slide": s, "n_bins": len(j), "k_graph": int(j.graph.nunique()), "n_lt_classes": int(j[LT_RAW].nunique())}
        for ref_name, col in (("lt", LT_RAW), ("lts", LT_SMOOTH)):
            if col not in j:
                continue
            for sr_name in ("graph", "km8"):
                r[f"ari_{sr_name}_{ref_name}"] = em.ari_between_labelings(j[sr_name].to_numpy(), j[col].to_numpy())
                r[f"nmi_{sr_name}_{ref_name}"] = em.nmi_between_labelings(j[sr_name].to_numpy(), j[col].to_numpy())
        rows.append(r)
        print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "reference_agreement_sr_vs_lt.csv", index=False)


if __name__ == "__main__":
    main()
