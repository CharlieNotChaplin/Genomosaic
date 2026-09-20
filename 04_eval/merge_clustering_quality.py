"""
Merge the output of evaluate_clustering_quality.py (clustering_quality.csv) into the two notebook-style result files
the thesis tables read, without re-running everything else:

  final_loso_clustering_quality.csv           one row per (unit, variant, slide, embedding, reference_kind): the raw
                                              (unrefined) Leiden partition -> Table "Clustering quality of the
                                              transcriptomic embedding" (evaluate_final_loso.ipynb, section 3)
  final_loso_clustering_quality_smoothed.csv  the same plus the label refinement per label_kind raw/hard/spagcn ->
                                              Table "Effect of spatial refinement" (section 4)

For every (unit, variant, held_out_slide) that occurs in the new file, its old rows are REPLACED; all other rows are kept.
This is how bins results are added once their folds have finished (evaluate_clustering_quality.py was run for them on
the cluster), and how the bins semantic-exclusion arm is added later; nothing else has to be recomputed.

Usage (from 04_eval/):
    python merge_clustering_quality.py --new <dir-with-clustering_quality.csv>
    python merge_clustering_quality.py --new <dir> --dry-run          # print what would change
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "data/combined/03_integration/autoencoders"
KEY = ["unit", "variant", "held_out_slide"]


def replace_rows(old, new):
    """`old` without the (unit, variant, slide) combinations present in `new`, plus `new`; columns follow `old`."""
    new = new[[c for c in old.columns if c in new.columns]]
    hit = old.set_index(KEY).index.isin(new.set_index(KEY).index)
    return pd.concat([old[~hit], new], ignore_index=True), int(hit.sum())


def coverage(df):
    return df.groupby(["unit", "variant"]).held_out_slide.nunique().to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True, help="Directory (or file) holding clustering_quality.csv")
    ap.add_argument("--base", type=Path, default=BASE, help="Directory of the final_loso_* files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_path = args.new / "clustering_quality.csv" if args.new.is_dir() else args.new
    new = pd.read_csv(new_path)
    if "label_kind" not in new:
        raise SystemExit(f"{new_path} has no label_kind column -- is it the output of evaluate_clustering_quality.py?")
    print(f"new results: {len(new)} rows, {coverage(new)}")

    for name, frame in [("final_loso_clustering_quality.csv", new[new.label_kind == "raw"].drop(columns="label_kind")),
                        ("final_loso_clustering_quality_smoothed.csv", new)]:
        path = args.base / name
        old = pd.read_csv(path)
        merged, n_replaced = replace_rows(old, frame)
        print(f"{name}: {len(old)} rows -> {len(merged)} ({n_replaced} replaced); folds per (unit, variant): {coverage(merged)}")
        if not args.dry_run:
            path.rename(path.with_suffix(".csv.bak"))  # keep the previous version next to it
            merged.to_csv(path, index=False)


if __name__ == "__main__":
    main()
