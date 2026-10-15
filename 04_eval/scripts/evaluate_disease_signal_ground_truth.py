"""
Per-unit disease-signal probe against the real pathologist ground truth, for CRC_P2 only.

evaluate_disease_signal (functions/dual_autoencoder.py, used by step_disease/evaluate_disease_signal_weighted.py)
trains a tumor-vs-normal logistic probe on the other training slides, labeling every unit of a slide with that
slide's own tumor/normal status (1 if the Slide_ID starts with "CRC", else 0). held_out_acc then only asks
"what fraction of the held-out slide's units got the slide's own single label" -- for a CRC slide this treats
stroma, muscle and immune units as if they were tumor too, so it cannot tell whether the embedding actually
separates tumor tissue from non-tumor tissue WITHIN a slide.

CRC_P2 is the one slide with a real pathologist annotation (ground_truth_reference, reused from
evaluate_k_matched_ari.py -- same dominant-tissue-type-per-8um-grid-cell logic used everywhere else in this
thesis for CRC_P2), used for evaluation. None of the other 9 training slides has one, so training uses the
same per-unit convention one level down: each training unit's own label-transfer prediction (nearest 8um
grid cell's predicted_tissuetype_grid_image, the same "primary pseudo-label source" this project uses
everywhere else a real annotation isn't available), not the coarse "every unit of a CRC_ slide is diseased"
label the first version of this script used. That coarse label has exactly the same flaw described above,
just moved from the evaluation side to the training side -- fixed here too, not just evaluation.
Diseased = "Neoplasm", normal = everything else (Connective Tissue, Non-neoplastic Epithelium, Smooth
Muscle, Vessel, Veins), both for training (label transfer) and evaluation (ground truth).

Data: the saved loso_token_CRC_P2..._embeddings.npz of each (unit, variant) fold (train_z_trans/z_token,
train_slide_name, held_out_z_trans/z_token), no model forward pass. Ground truth comes from
data/CRC_P2/ground_truth/8um_squares_annotation.csv; training labels come from each training slide's own
{slide}_grid_predictions.parquet (functions.label_transfer), both via functions.label_transfer.

Output (--output-dir): final_loso_disease_signal_ground_truth.csv, one row per (unit, variant, embedding,
class_weight). Folds already in the file are skipped (--overwrite to redo).

Usage (from 04_eval/; CPU only):
    python -u scripts/evaluate_disease_signal_ground_truth.py
    python -u scripts/evaluate_disease_signal_ground_truth.py --units nuclei --variants svg --embeddings z_trans
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_clustering_quality as ecq
from evaluate_disease_signal_weighted import subsample_by_slide
from evaluate_k_matched_ari import ground_truth_reference
from functions import label_transfer as lt

OUT_NAME = "final_loso_disease_signal_ground_truth.csv"
DISEASED_LABEL = "Neoplasm"


def per_unit_labels(slide, cell_xy):
    """1 where the nearest 8um grid cell's dominant annotation is Neoplasm, 0 for every other
    tissue type (Connective Tissue, Non-neoplastic Epithelium, Smooth Muscle, Vessel, Veins).
    ground_truth_reference always returns a label (nearest-grid assignment, no NaNs), so every
    unit gets used."""
    gt = ground_truth_reference(slide, cell_xy)
    return (np.asarray(gt) == DISEASED_LABEL).astype(int)


def train_labels_from_label_transfer(unit, train_slides):
    """Per-unit label-transfer Neoplasm/not label for every training unit, aligned to
    `train_slides`' own row order (the saved npz has no per-unit id/xy for the training pool, only
    for the held-out slide, so this re-derives it from each training slide's own data instead).

    -1 for a unit whose slide has no {slide}_grid_predictions.parquet yet -- excluded from training
    by the caller, not silently treated as "normal".

    Verifies each slide's rows form one contiguous block in `train_slides` (true by construction --
    merge_anndatas concatenates sample-by-sample -- but checked, not assumed: a silent misalignment
    here would corrupt every training label without raising anything)."""
    labels = np.full(len(train_slides), -1, dtype=int)
    for slide in pd.unique(train_slides):
        idx = np.flatnonzero(train_slides == slide)
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise RuntimeError(f"{slide}'s rows are not contiguous in train_slide_name -- the "
                                f"row-order assumption this relies on no longer holds")
        grid_ref = ecq.load_grid_reference(slide)
        if grid_ref is None:
            print(f"    [{unit}] no label-transfer grid predictions for {slide} -- its {len(idx)} "
                  f"units are excluded from training")
            continue
        cell_xy, _ = ecq.reconstruct_held_out_cells(unit, slide)
        if len(cell_xy) != len(idx):
            # mismatch, not necessarily misorder: the merged transcriptomics file can have been
            # regenerated since this fold's npz was written, drifting its unit count slightly (seen
            # directly: CRC_P5 was off by 23/115473 units). A *count* mismatch can't be safely
            # realigned, so exclude the slide rather than risk a silent off-by-one; a genuine
            # non-contiguous block above still raises, since that indicates something structural,
            # not routine drift.
            print(f"    [{unit}] {slide}: {len(cell_xy)} units reconstructed now vs {len(idx)} in "
                  f"this fold's train_slide_name (data drift since this fold was trained) -- "
                  f"excluded from training, can't safely realign")
            continue
        tissue = lt.assign_nearest_grid_label(cell_xy, grid_ref, label_col=ecq.LABEL_COL)
        labels[idx] = (np.asarray(tissue) == DISEASED_LABEL).astype(int)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", choices=list(ecq.UNITS), default=list(ecq.UNITS))
    ap.add_argument("--variants", nargs="+", choices=list(ecq.VARIANTS), default=list(ecq.VARIANTS))
    ap.add_argument("--embeddings", nargs="+", choices=["z_trans", "z_token", "z_joint"],
                     default=["z_trans", "z_token", "z_joint"])
    ap.add_argument("--slide", default=lt.REFERENCE_SAMPLE, choices=[lt.REFERENCE_SAMPLE],
                     help="only CRC_P2 has a pathologist annotation to build per-unit ground truth from")
    ap.add_argument("--max-train-per-slide", type=int, default=10_000,
                     help="training units per slide (0 = no cap); mirrors evaluate_disease_signal_weighted.py")
    ap.add_argument("--output-dir", type=Path, default=ecq.BASEDIR / "combined/03_integration/autoencoders")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / OUT_NAME
    if args.overwrite:
        out.unlink(missing_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out, usecols=["unit", "variant", "embedding", "class_weight"])
        done = set(map(tuple, d.drop_duplicates().to_numpy()))

    recon = None  # (cell_xy, y_true) -- same for every variant of a unit, computed once per unit
    recon_unit = None
    train_labels = None  # per-unit label-transfer labels for the training pool -- same cache strategy
    train_labels_unit = None
    for unit in args.units:
        for variant in args.variants:
            path = ecq.checkpoint_dir(unit) / f"loso_token_{args.slide}{ecq.run_tag(unit, variant)}_embeddings.npz"
            if not path.exists():
                print(f"  [{unit}/{variant}] no fold for {args.slide} -- skipping")
                continue
            todo = [e for e in args.embeddings if (unit, variant, e, "none") not in done
                    or (unit, variant, e, "balanced") not in done]
            if not todo:
                print(f"{unit}/{variant}: nothing to do")
                continue

            t0 = time.time()
            npz = np.load(path, allow_pickle=True)  # train_slide_name is a pickled object array
            held_n = npz["held_out_z_trans"].shape[0]
            if recon_unit != unit:
                cell_xy, trans_sub = ecq.reconstruct_held_out_cells(unit, args.slide)
                if cell_xy.shape[0] != held_n:
                    print(f"  [{unit}] row mismatch: {cell_xy.shape[0]} reconstructed vs {held_n} embedded -- skipping")
                    continue
                y_true = per_unit_labels(args.slide, cell_xy)
                recon, recon_unit = (cell_xy, y_true), unit
                print(f"  [{unit}] {args.slide}: {len(y_true):,} units, "
                      f"{y_true.sum():,} diseased (Neoplasm), {len(y_true) - y_true.sum():,} normal")
            cell_xy, y_true = recon

            train_slides = np.asarray(npz["train_slide_name"]).astype(str)
            embs = {"z_trans": (npz["train_z_trans"], npz["held_out_z_trans"]),
                    "z_token": (npz["train_z_token"], npz["held_out_z_token"]),
                    "z_joint": (np.concatenate([npz["train_z_trans"], npz["train_z_token"]], 1),
                                np.concatenate([npz["held_out_z_trans"], npz["held_out_z_token"]], 1))}

            if train_labels_unit != unit or len(train_labels) != len(train_slides):
                train_labels = train_labels_from_label_transfer(unit, train_slides)
                train_labels_unit = unit
                n_unlabeled = int((train_labels == -1).sum())
                print(f"  [{unit}] training pool: {len(train_labels):,} units, "
                      f"{(train_labels == 1).sum():,} diseased, {(train_labels == 0).sum():,} normal"
                      + (f", {n_unlabeled:,} excluded (no label-transfer file)" if n_unlabeled else ""))
            labeled = train_labels != -1

            rows = []
            for emb in todo:
                tr, held = embs[emb]
                tr, tr_slides = subsample_by_slide(tr[labeled], train_slides[labeled], args.max_train_per_slide, seed=0)
                y_train, _ = subsample_by_slide(train_labels[labeled], train_slides[labeled], args.max_train_per_slide, seed=0)
                for cw in (None, "balanced"):
                    if (unit, variant, emb, "none" if cw is None else cw) in done:
                        continue
                    clf = LogisticRegression(max_iter=5000, class_weight=cw).fit(tr, y_train)
                    p = clf.predict_proba(held)[:, 1]
                    y_pred = (p >= 0.5).astype(int)

                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                    recall = tp / (tp + fn) if (tp + fn) else np.nan        # sensitivity on Neoplasm
                    specificity = tn / (tn + fp) if (tn + fp) else np.nan   # recall on normal
                    precision = tp / (tp + fp) if (tp + fp) else np.nan
                    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan

                    rows.append({
                        "unit": unit, "variant": variant, "embedding": emb,
                        "class_weight": "none" if cw is None else cw,
                        "n_train_units": len(tr), "n_held_units": len(held),
                        "n_gt_diseased": int(y_true.sum()), "n_gt_normal": int(len(y_true) - y_true.sum()),
                        "accuracy": float((y_pred == y_true).mean()),
                        "balanced_accuracy": float(np.nanmean([recall, specificity])),
                        "f1_macro": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
                        "precision_diseased": float(precision), "recall_diseased": float(recall),
                        "specificity_normal": float(specificity), "auc": float(auc),
                        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                    })
            if rows:
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                r = {(x["embedding"], x["class_weight"]): x["accuracy"] for x in rows}
                print(f"{unit}/{variant}: accuracy " +
                      ", ".join(f"{k[0]}/{k[1]}={v:.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
