"""
Runs the parts of the final leave-one-slide-out evaluation that are still missing, for the chosen units and variants.

Steps (``--steps``), each of which skips what is already in its result file unless ``--overwrite`` is given:
    quality   clustering quality and label refinement against the label-transfer reference
              (evaluate_clustering_quality.py, merged into final_loso_clustering_quality[_smoothed].csv)
    disease   tumor-versus-normal probe on z_trans (final_loso_disease_signal.csv)
    weighted  the same probe with and without class weights (evaluate_disease_signal_weighted.py)
    sections  gene-expression probe, cluster annotation and the spatial cluster maps (evaluate_final_loso_7_9.py)
    umap      UMAPs of the best configurations (evaluate_best_config_umap.py; needs the k-matched results)

A failing step is reported and the next one still runs. Run it after the folds of the chosen variants have finished
and after any running job that writes the same files (for example the bin-level clustering-quality job).

Usage (from 04_eval/; CPU only):
    python -u run_final_evaluation.py                                  bins, all three variants, every step
    python -u run_final_evaluation.py --variants svg_semantic_excl_raw_0.2 --steps quality disease sections
    python -u run_final_evaluation.py --units nuclei bins              all six variants (finished parts are skipped)
    python -u run_final_evaluation.py --units nuclei bins --overwrite  recompute everything
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_clustering_quality as ecq
from evaluate_disease_signal_weighted import subsample_by_slide
from functions.dual_autoencoder import evaluate_disease_signal

BASE = ecq.BASEDIR / "combined/03_integration/autoencoders"
STEPS = ["quality", "disease", "weighted", "sections", "umap"]
DISEASE_FILE = BASE / "final_loso_disease_signal.csv"
DISEASE_KEY = ["unit", "variant", "held_out_slide"]


def run(script, *arguments):
    """Run one evaluation script in this directory; returns True if it exited normally."""
    command = [sys.executable, "-u", str(HERE / script), *map(str, arguments)]
    print("\n$", " ".join(command), flush=True)
    code = subprocess.run(command, cwd=HERE).returncode
    if code:
        print(f"!!!!! {script} failed with exit code {code}", flush=True)
    return code == 0


def embeddings_path(unit, variant, slide):
    return ecq.checkpoint_dir(unit) / f"loso_token_{slide}{ecq.run_tag(unit, variant)}_embeddings.npz"


def has_label_transfer(slide):
    return (ecq.BASEDIR / slide / "02_baselines/label_transfer" / f"{slide}_grid_predictions.parquet").exists()


def finished_slides(unit, variant):
    """Slides whose fold for this unit and variant has finished."""
    return [s for s in ecq.SLIDES if embeddings_path(unit, variant, s).exists()]


def done_folds(path):
    if not path.exists():
        return set()
    return set(map(tuple, pd.read_csv(path, usecols=DISEASE_KEY).drop_duplicates().to_numpy()))


def step_quality(units, variants, overwrite):
    done = done_folds(BASE / "final_loso_clustering_quality.csv")
    for unit in units:
        for variant in variants:
            slides = [s for s in finished_slides(unit, variant) if has_label_transfer(s)]
            todo = slides if overwrite else [s for s in slides if (unit, variant, s) not in done]
            if not todo:
                print(f"quality {unit}/{variant}: nothing to do")
                continue
            work = BASE / "final_eval_tmp" / f"quality_{unit}_{variant}"
            if run("evaluate_clustering_quality.py", "--units", unit, "--variants", variant, "--slides", *todo,
                   "--smooth-reference", "--output-dir", work):
                run("merge_clustering_quality.py", "--new", work)


def step_disease(units, variants, overwrite, max_per_slide):
    done = done_folds(DISEASE_FILE)
    for unit in units:
        for variant in variants:
            for slide in finished_slides(unit, variant):
                if not overwrite and (unit, variant, slide) in done:
                    continue
                npz = np.load(embeddings_path(unit, variant, slide), allow_pickle=True)
                train_slides = np.asarray(npz["train_slide_name"]).astype(str)
                train_z, train_slides = subsample_by_slide(npz["train_z_trans"], train_slides, max_per_slide, seed=0)
                held = npz["held_out_z_trans"]
                if max_per_slide and len(held) > max_per_slide:
                    held = held[np.sort(np.random.default_rng(1).choice(len(held), size=max_per_slide, replace=False))]
                result = evaluate_disease_signal(train_z, train_slides, held, np.full(len(held), slide))
                row = pd.DataFrame([{**result, "unit": unit, "variant": variant, "held_out_slide": slide}])
                if DISEASE_FILE.exists():
                    old = pd.read_csv(DISEASE_FILE)
                    row = pd.concat([old, row], ignore_index=True).drop_duplicates(DISEASE_KEY, keep="last")
                row.to_csv(DISEASE_FILE, index=False)
                print(f"disease {unit}/{variant}/{slide}: held_out_acc {result['held_out_acc']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", choices=list(ecq.UNITS), default=["bins"])
    ap.add_argument("--variants", nargs="+", choices=list(ecq.VARIANTS), default=list(ecq.VARIANTS))
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--overwrite", action="store_true", help="recompute folds that are already in the result files")
    ap.add_argument("--umap-jobs", type=int, default=4, help="UMAP figures drawn in parallel")
    ap.add_argument("--disease-max-per-slide", type=int, default=20_000,
                    help="units per slide used by the disease probe of the new folds; 0 uses all units")
    args = ap.parse_args()

    if "quality" in args.steps:
        step_quality(args.units, args.variants, args.overwrite)
    if "disease" in args.steps:
        step_disease(args.units, args.variants, args.overwrite, args.disease_max_per_slide)
    if "weighted" in args.steps:
        run("evaluate_disease_signal_weighted.py", "--units", *args.units, "--variants", *args.variants,
            *(["--overwrite"] if args.overwrite else []))
    if "sections" in args.steps:
        overwrite = ["--overwrite"] if args.overwrite else []
        run("evaluate_final_loso_7_9.py", "--sections", 8, 7, "--units", *args.units, "--variants", *args.variants, *overwrite)
        run("evaluate_final_loso_7_9.py", "--sections", 9, *overwrite)
    if "umap" in args.steps:
        run("evaluate_best_config_umap.py", "--jobs", args.umap_jobs)


if __name__ == "__main__":
    main()
