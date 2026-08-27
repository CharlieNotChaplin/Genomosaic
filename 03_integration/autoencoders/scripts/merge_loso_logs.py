"""
Combines the per-held-out-slide LOSO-token logs (loso_token_log_{slide}.csv, one per
array task from submit_run_loso_array.sbatch) into a single table for analysis.

Kept separate from run_loso.py itself: each array task writes only its own file
(see run_loso.py's docstring for why a shared log would race under concurrent
writers), so merging is a distinct, run-agnostic step done once after every fold
you care about has finished -- rerun it any time as more folds complete, it
doesn't need all N to be done.
"""

import sys
from pathlib import Path
import pandas as pd

basedir = Path("../../data")
dir_output = basedir / "combined/03_integration/autoencoders/002um"

logs = sorted(dir_output.glob("loso_token_log_*.csv"))
if not logs:
    sys.exit(f"No loso_token_log_*.csv files found under {dir_output}")

print(f"Found {len(logs)} per-slide log(s):")
for p in logs:
    print(f"  {p.name}")

combined = pd.concat([pd.read_csv(p) for p in logs], ignore_index=True, sort=False)
out_path = dir_output / "loso_token_log_combined.csv"
combined.to_csv(out_path, index=False)
print(f"\nWrote combined table ({len(combined)} row(s)) to {out_path}")

cols = ["held_out_slide", "train_cell_top1_retrieval", "held_out_cell_top1_retrieval",
        "train_cell_cluster_top1_retrieval", "held_out_cell_cluster_top1_retrieval",
        "combined_token_leakage_mean"]
cols = [c for c in cols if c in combined.columns]
if cols:
    print("\nSummary:")
    print(combined[cols].to_string(index=False))
