"""
Trains the dual autoencoder on the entire cohort (no held-out slide), with the settings decided
after the ablation studies (``config/training_config.json`` -- the same file ``run_loso.py`` reads),
and saves the result to a dedicated ``weights/`` folder.

This is the deployable final model: ``run_loso.py``'s folds exist only to validate generalization
(Chapter 6) and are never meant to be used for inference, since each one deliberately withholds a
slide. This script is their counterpart -- a script form of ``03_integration/train_autoencoder.ipynb``
(same ``functions.training_pipeline`` building blocks: load_slide_data -> build_pipeline ->
build_model -> pretrain_token_branch -> train_joint -> evaluate_embeddings -> save_run), so it can
run headless via sbatch instead of a notebook kernel.

Applies the ``--unit`` preset on top of the decided configuration exactly like ``run_loso.py`` does,
and overrides only ``output_name`` -- WITHOUT writing back to ``--config`` (unlike the notebook's own
``save_config()`` call), so running this never changes what ``run_loso.py`` or the notebook read.

Output: ``data/combined/03_integration/autoencoders/<bin_size>/weights/``
    dual_ae_checkpoint.pt   weights, config, scalers, gene names, history
    dual_ae_latents.npz     embeddings with unit ids and slide names
    gene_panel.txt, training_config.json (this run's own config, not --config), metrics.json

Usage (from 03_integration/scripts/):
    python train_final_model.py --unit nuclei
    python train_final_model.py --unit bins
    python train_final_model.py --unit nuclei --dry-run
"""

import argparse
import gc
import json
import sys
from pathlib import Path

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break

import torch

from functions.training_pipeline import (
    CONFIG_PATH, apply_unit_preset, build_model, build_pipeline, data_paths, evaluate_embeddings,
    load_config, load_slide_data, pretrain_token_branch, save_run, train_joint,
)

OUTPUT_NAME = "weights"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=["nuclei", "bins"], default="nuclei",
                     help="unit-specific settings applied on top of the decided configuration")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH, help="JSON configuration file to read settings from")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the resolved settings and output path, then exit")
    args = ap.parse_args()

    config = apply_unit_preset(load_config(args.config), args.unit)
    config["output_name"] = OUTPUT_NAME  # kept local to this run; never written back to --config
    output_dir = data_paths(config)["output"]

    if args.dry_run:
        print(json.dumps(config, indent=2))
        print(f"\nunit: {args.unit}\noutput: {output_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\n=== 1. Load ===")
    data = load_slide_data(config)
    print(f"Slides: {data['slides']}")
    print(f"{data['trans'].n_obs:,} units, {data['hist'].n_obs:,} histology patches, "
          f"{data['token'].n_obs:,} token embeddings")

    print("\n=== 2. Prepare ===")
    pipe = build_pipeline(data, config)
    del data
    gc.collect()
    print(f"{pipe['n_svg']} genes, token dimension {pipe['token_dim']}, {pipe['n_slides']} slides")
    print(pipe["trans_sub"].obs["Slide_ID"].value_counts().to_string())

    print("\n=== 3. Model ===")
    model = build_model(config, pipe, device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n=== 4. Pretrain token branch ===")
    pretrain_info = pretrain_token_branch(model, pipe, config, device)
    print(pretrain_info)

    print("\n=== 5. Joint training ===")
    history = train_joint(model, pipe, config, device)

    print("\n=== 6. Evaluate (in-sample -- every slide was part of training) ===")
    evaluation = evaluate_embeddings(model, pipe, device)
    print(evaluation["metrics"])

    print("\n=== 7. Save ===")
    output_dir = save_run(output_dir, model, pipe, config, history, evaluation, pretrain_info)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
