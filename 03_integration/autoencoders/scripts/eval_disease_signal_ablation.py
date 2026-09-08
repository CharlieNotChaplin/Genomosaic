"""
Throwaway script: runs evaluate_disease_signal per fold against the ablation run's
already-saved LOSO embeddings (loso_ablation/*_embeddings.npz), one fold at a time
-- each npz is ~840MB (train_z_trans/train_z_token for ~1.9M cells across 8 training
slides + held_out_z_trans/held_out_z_token for the held-out slide), so loading all
available folds at once would be several GB; this loads, evaluates, and discards one
fold before moving to the next to keep peak memory to roughly one file's worth.

Only pulls train_z_trans/train_slide_name/held_out_z_trans out of each npz (not the
token variants) -- evaluate_disease_signal is checked against z_trans here, matching
the embedding used for the retrieval eval throughout this project.

Usage:
    python eval_disease_signal_ablation.py
"""
import gc
import sys
from pathlib import Path

import numpy as np

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break
from functions.dual_autoencoder import evaluate_disease_signal

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "data").is_dir():
        basedir = _candidate / "data"
        break
else:
    raise RuntimeError("Could not find a 'data' directory in any ancestor of this script")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="loso_ablation", help="subfolder under .../002um/ holding the embeddings")
parser.add_argument("--run-tag", default="_slideloaderablation_oldcap_cap11000_oversample_normal")
args = parser.parse_args()

ablation_dir = basedir / "combined/03_integration/autoencoders/002um" / args.dir
RUN_TAG = args.run_tag

npz_paths = sorted(ablation_dir.glob(f"loso_token_*{RUN_TAG}_embeddings.npz"))
if not npz_paths:
    raise RuntimeError(f"No embeddings found under {ablation_dir} matching {RUN_TAG!r}")

rows = []
for p in npz_paths:
    held_out_slide = p.name[len("loso_token_"):-len(f"{RUN_TAG}_embeddings.npz")]
    print(f"=== {held_out_slide} ===")

    npz = np.load(p, allow_pickle=True)  # train_slide_name is a string object array
    z_train = npz["train_z_trans"]
    train_slide_name = npz["train_slide_name"]
    z_held = npz["held_out_z_trans"]
    npz.close()

    held_slide_name = np.full(z_held.shape[0], held_out_slide)

    result = evaluate_disease_signal(z_train, train_slide_name, z_held, held_slide_name)
    result["held_out_slide"] = held_out_slide
    rows.append(result)
    print(f"  train_loso_acc={result['train_loso_acc']:.4f} train_loso_f1={result['train_loso_f1']:.4f} "
          f"(n_valid_folds={result['n_valid_folds']})")
    label = "tumor" if result["held_out_true_label"] == 1 else "normal"
    print(f"  held_out_acc={result['held_out_acc']:.4f} (true label: {label}, "
          f"i.e. fraction classified as its own known label)")

    del npz, z_train, train_slide_name, z_held, held_slide_name
    gc.collect()

print("\n=== Summary ===")
tumor_rows = [r for r in rows if r["held_out_true_label"] == 1]
normal_rows = [r for r in rows if r["held_out_true_label"] == 0]
if tumor_rows:
    print(f"Tumor held-out folds (n={len(tumor_rows)}): mean held_out_acc="
          f"{np.mean([r['held_out_acc'] for r in tumor_rows]):.4f}")
if normal_rows:
    for r in normal_rows:
        print(f"  {r['held_out_slide']}: held_out_acc={r['held_out_acc']:.4f}")
    print(f"Normal held-out folds (n={len(normal_rows)}): mean held_out_acc="
          f"{np.mean([r['held_out_acc'] for r in normal_rows]):.4f}")
print(f"Mean train_loso_acc across all {len(rows)} folds: {np.mean([r['train_loso_acc'] for r in rows]):.4f}")
print(f"Mean train_loso_f1 across all {len(rows)} folds: {np.mean([r['train_loso_f1'] for r in rows]):.4f}")
