"""
Combines the per-held-out-slide LOSO-token logs (loso_token_log_{slide}.csv, one per
array task of the final LOSO sbatch scripts) into a single table for analysis.

Kept separate from run_loso.py itself: each array task writes only its own file, so
concurrent tasks never write to one shared log, and merging is a distinct, run-agnostic step done once after every fold
you care about has finished -- rerun it any time as more folds complete, it
doesn't need all N to be done.
"""

import sys
from pathlib import Path
import pandas as pd

# Robust to invocation directory: walks up from this file's own location for
# a `data/` sibling, rather than assuming a fixed relative depth from CWD
# (broke when this file moved into scripts/, one level deeper than before).
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "data").is_dir():
        basedir = _candidate / "data"
        break
else:
    raise RuntimeError("Could not find a 'data' directory in any ancestor of this script")
dir_output = basedir / "combined/03_integration/autoencoders/002um"

out_path = dir_output / "loso_token_log_combined.csv"
# Excludes out_path itself: it matches this same glob once it exists, so a second
# run would otherwise feed its own prior output back in as an "input" and duplicate
# every row already merged (caught when CRC_P11mm's rerun produced 15 rows instead
# of 8 -- 7 stale rows from the first merge's own output, re-ingested).
logs = sorted(p for p in dir_output.glob("loso_token_log_*.csv") if p != out_path)
if not logs:
    sys.exit(f"No loso_token_log_*.csv files found under {dir_output}")

print(f"Found {len(logs)} per-slide log(s):")
for p in logs:
    print(f"  {p.name}")

combined = pd.concat([pd.read_csv(p) for p in logs], ignore_index=True, sort=False)
combined.to_csv(out_path, index=False)
print(f"\nWrote combined table ({len(combined)} row(s)) to {out_path}")

cols = ["held_out_slide", "train_cell_top1_retrieval", "held_out_cell_top1_retrieval",
        "train_cell_cluster_top1_retrieval", "held_out_cell_cluster_top1_retrieval",
        "combined_token_leakage_mean"]
cols = [c for c in cols if c in combined.columns]
if cols:
    print("\nSummary:")
    print(combined[cols].to_string(index=False))
