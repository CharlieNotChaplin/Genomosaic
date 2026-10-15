"""
Disease-signal validation with and without class re-weighting.

The tumor-versus-normal probe (functions.dual_autoencoder.evaluate_disease_signal) is heavily imbalanced: when a
normal slide is held out, the training pool has ONE normal slide against EIGHT tumor slides, and the held-out normal
slides were classified as normal only 16-51% of the time. This script asks whether that is only the probe's class
imbalance: it refits the same probe on frozen embeddings with class_weight=None (as in the notebook) and
class_weight="balanced", side by side.

It is a diagnostic of the probe, not a tuned result: no weight or threshold is chosen using the held-out slide, and
with two normal slides in the whole cohort the held-out normal numbers remain single-slide observations.
Per fold it also reports the mean predicted tumor probability on the held-out slide (a threshold-free view) and the
share of held-out units predicted tumor.

Data: the saved *_embeddings.npz of each fold (train_z_trans / train_z_token / train_slide_name / held_out_*), no model
forward pass. Units are subsampled per slide (--max-per-slide, seeded) so the leave-one-training-slide-out logistic
regressions stay tractable; 0 = no cap.

Output (--output-dir): final_loso_disease_signal_weighted.csv, one row per (unit, variant, held_out_slide, embedding,
class_weight). Folds already in the file are skipped (--overwrite to redo).

Usage (from 04_eval/; CPU only):
    python -u scripts/evaluate_disease_signal_weighted.py
    python -u scripts/evaluate_disease_signal_weighted.py --units nuclei --variants svg --embeddings z_trans
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_clustering_quality as ecq
from functions.dual_autoencoder import evaluate_disease_signal

OUT_NAME = "final_loso_disease_signal_weighted.csv"


def subsample_by_slide(z, slides, cap, seed):
    if not cap:
        return z, slides
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(slides):
        idx = np.where(slides == s)[0]
        keep.append(np.sort(rng.choice(idx, size=min(cap, len(idx)), replace=False)))
    keep = np.concatenate(keep)
    return z[keep], slides[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", choices=list(ecq.UNITS), default=["nuclei"])
    ap.add_argument("--variants", nargs="+", choices=list(ecq.VARIANTS), default=list(ecq.VARIANTS))
    ap.add_argument("--slides", nargs="+", choices=ecq.SLIDES, default=ecq.SLIDES)
    ap.add_argument("--embeddings", nargs="+", choices=["z_trans", "z_token", "z_joint"], default=["z_trans", "z_joint"])
    ap.add_argument("--max-per-slide", type=int, default=10_000)
    ap.add_argument("--output-dir", type=Path, default=ecq.BASEDIR / "combined/03_integration/autoencoders")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / OUT_NAME
    if args.overwrite:
        out.unlink(missing_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out, usecols=["unit", "variant", "held_out_slide"])
        done = set(map(tuple, d.drop_duplicates().to_numpy()))

    for unit in args.units:
        for variant in args.variants:
            for slide in args.slides:
                path = ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, variant)}_embeddings.npz"
                if (unit, variant, slide) in done or not path.exists():
                    continue
                t0 = time.time()
                npz = np.load(path, allow_pickle=True)  # train_slide_name is a pickled object array
                train_slides = np.asarray(npz["train_slide_name"]).astype(str)
                held_slides = np.full(npz["held_out_z_trans"].shape[0], slide)
                embs = {"z_trans": (npz["train_z_trans"], npz["held_out_z_trans"]),
                        "z_token": (npz["train_z_token"], npz["held_out_z_token"]),
                        "z_joint": (np.concatenate([npz["train_z_trans"], npz["train_z_token"]], 1),
                                    np.concatenate([npz["held_out_z_trans"], npz["held_out_z_token"]], 1))}
                rows = []
                for emb in args.embeddings:
                    tr, held = embs[emb]
                    tr, tr_slides = subsample_by_slide(tr, train_slides, args.max_per_slide, seed=0)
                    rng = np.random.default_rng(1)
                    if args.max_per_slide and len(held) > args.max_per_slide:
                        held = held[np.sort(rng.choice(len(held), size=args.max_per_slide, replace=False))]
                    held_s = np.full(len(held), slide)
                    y_train = np.array([1 if s.startswith("CRC") else 0 for s in tr_slides])
                    for cw in (None, "balanced"):
                        res = evaluate_disease_signal(tr, tr_slides, held, held_s, class_weight=cw)
                        clf = LogisticRegression(max_iter=5000, class_weight=cw).fit(tr, y_train)
                        p = clf.predict_proba(held)[:, 1]
                        rows.append({"unit": unit, "variant": variant, "held_out_slide": slide, "embedding": emb,
                                     "class_weight": "none" if cw is None else cw,
                                     "n_train_units": len(tr), "n_held_units": len(held),
                                     "n_normal_train_units": int((y_train == 0).sum()), **res,
                                     "held_out_mean_p_tumor": float(p.mean()),
                                     "held_out_frac_pred_tumor": float((p >= 0.5).mean())})
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                r = {(x["embedding"], x["class_weight"]): x["held_out_acc"] for x in rows}
                print(f"{unit}/{variant}/{slide}: held_out_acc " +
                      ", ".join(f"{k[0]}/{k[1]}={v:.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
