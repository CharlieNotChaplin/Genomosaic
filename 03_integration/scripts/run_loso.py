"""
Leave-one-slide-out (LOSO) run of the dual autoencoder.

Trains on all selected slides except ``--held_out_slide`` and reports cell-to-token retrieval on the
training slides (in-sample) and on the held-out slide.

Settings are resolved in this order:
    1. the JSON configuration (``--config``, default ``config/training_config.json``)
    2. the unit preset (``--unit nuclei`` or ``--unit bins``)
    3. explicit flags

Every key of the configuration is a flag, with underscores written as hyphens, for example
``--edge-dropout-p 0.1``, ``--no-freeze-token``, ``--local-anchor-sim-threshold none``,
``--oversample-slides '{}'`` or ``--slides CRC_P1 CRC_P2``. ``--help`` lists all of them with their
defaults; ``--dry-run`` prints the resolved settings and the output files without loading any data.

Outputs are written to ``data/combined/03_integration/autoencoders/<bin_size>/``:
    loso_token_log_<slide><tag>.csv
    loso_token_checkpoints/loso_token_<slide><tag>.pt, ..._gene_panel.txt and ..._embeddings.npz
``<tag>`` lists the settings that differ from the reference configuration
(``functions.training_pipeline.run_tag``).

Examples:
    python run_loso.py --held_out_slide CRC_P1
    python run_loso.py --held_out_slide CRC_P1 --unit bins
    python run_loso.py --held_out_slide CRC_P1 --gene-panel colon_xenium
    python run_loso.py --held_out_slide CRC_P1 --local-anchor-sim-threshold 0.2 --local-anchor-sim-source raw
"""

import argparse
import json
import sys
from pathlib import Path

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break

import torch

from functions.dual_autoencoder import run_leave_one_slide_out_token, set_seed
from functions.training_pipeline import (
    CONFIG_HELP, CONFIG_PATH, UNIT_PRESETS, apply_unit_preset, data_paths, load_config,
    load_slide_data, run_tag,
)

LIST_KEYS = {"slides": str, "exclude_slides": str, "num_neighbors": int}
CHOICES = {
    "encoder": ["transconv", "sage"],
    "gene_selection": ["svg", "hvg", "intersection"],
    "local_anchor_sim_source": ["projected", "raw"],
    "local_positive_mode": ["exclude", "attract"],
}


def _parse_value(text):
    """Parse a flag value as JSON (numbers, objects); ``none`` gives None."""
    if text.lower() == "none":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"cannot parse {text!r} as a number, JSON value or 'none'") from error


def _str_or_none(text):
    """Parse a string flag value; ``none`` gives None."""
    return None if text.lower() == "none" else text


def build_parser(config):
    """Argument parser with one flag per configuration key; unset flags are absent from the namespace."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held_out_slide", "--held-out-slide", dest="held_out_slide", required=True,
                        help="slide that is left out of training")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="JSON configuration file")
    parser.add_argument("--unit", choices=sorted(UNIT_PRESETS),
                        help="apply the unit-specific settings of nuclei or bins on top of the configuration")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved settings and output files, then exit")
    for key, default in config.items():
        flag = "--" + key.replace("_", "-")
        help_text = f"{CONFIG_HELP.get(key, key)} (default: {json.dumps(default)})"
        options = dict(dest=key, default=argparse.SUPPRESS, help=help_text)
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, **options)
        elif key in LIST_KEYS:
            parser.add_argument(flag, nargs="+", type=LIST_KEYS[key], **options)
        elif key in CHOICES:
            parser.add_argument(flag, choices=CHOICES[key], **options)
        elif isinstance(default, str):
            parser.add_argument(flag, type=str, **options)
        elif key == "gene_panel":
            parser.add_argument(flag, type=_str_or_none, **options)
        else:
            parser.add_argument(flag, type=_parse_value, **options)
    return parser


def resolve_config(args, base_config):
    """Apply the unit preset and the explicit flags to the configuration from the JSON file."""
    config = apply_unit_preset(base_config, args.unit) if args.unit else dict(base_config)
    config.update({key: value for key, value in vars(args).items() if key in base_config})
    config["num_neighbors"] = tuple(config["num_neighbors"])
    return config


def print_result(metrics, held_out_slide, log_path, checkpoint_file):
    """Print the retrieval, pretraining and leakage summary of one fold."""
    train_cluster = metrics["train_cell_cluster_top1_retrieval"]
    held_cluster = metrics["held_out_cell_cluster_top1_retrieval"]
    train_top1 = metrics["train_cell_top1_retrieval"]
    held_top1 = metrics["held_out_cell_top1_retrieval"]
    print("\n=== Result ===")
    print(f"Training slides, cluster top-1 retrieval:  {train_cluster:.4f} (exact top-1 {train_top1:.4f})")
    print(f"{held_out_slide}, cluster top-1 retrieval:  {held_cluster:.4f} (exact top-1 {held_top1:.4f})")
    print(f"Gap (train - held-out): cluster top-1 {train_cluster - held_cluster:.4f}, exact top-1 {train_top1 - held_top1:.4f}")
    print(f"Token pretraining: best validation MSE {metrics['token_pretrain_best_val_mse']:.5f} "
          f"(stopped at epoch {metrics['token_pretrain_stopped_epoch']})")
    print(f"Token leakage, training slides: {metrics['train_token_leakage_mean']:.4f} "
          f"(chance {metrics['train_token_leakage_chance']:.4f})")
    print(f"Token leakage, held-out vs training: {metrics['combined_token_leakage_mean']:.4f} "
          f"(chance {metrics['combined_token_leakage_chance']:.4f})")
    print(f"Last epoch: local_frac_masked {metrics['final_local_frac_masked']:.4f} "
          f"(above 0.5 indicates collapse), local_frac_extra_positive {metrics['final_local_frac_extra_positive']:.4f}, "
          f"contrast_local {metrics['final_contrast_local']:.4f}")
    print(f"\nLog: {log_path}\nEmbeddings: {checkpoint_file}")


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    base_config = load_config(pre_parser.parse_known_args()[0].config)

    args = build_parser(base_config).parse_args()
    config = resolve_config(args, base_config)
    held_out = args.held_out_slide
    if held_out in config["exclude_slides"]:
        raise ValueError(f"{held_out} is held out and excluded at the same time")

    tag = run_tag(config)
    paths = data_paths(config)
    output_dir = paths["output"].parent
    log_path = output_dir / f"loso_token_log_{held_out}{tag}.csv"
    checkpoint_dir = output_dir / "loso_token_checkpoints"
    embeddings_file = checkpoint_dir / f"loso_token_{held_out}{tag}_embeddings.npz"

    if args.dry_run:
        print(json.dumps(config, indent=2))
        print(f"\nheld-out slide: {held_out}\nrun tag: {tag}\nlog: {log_path}\ncheckpoints: {checkpoint_dir}")
        return

    run_config = dict(config, gene_panel_path=None)
    if config["gene_panel"]:
        panel_path = paths["gene_panels"] / f"{config['gene_panel']}.json"
        if not panel_path.is_file():
            raise FileNotFoundError(f"No gene panel named {config['gene_panel']!r} at {panel_path}")
        run_config["gene_panel_path"] = panel_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"])
    data = load_slide_data(config)
    if held_out not in data["slides"]:
        raise ValueError(f"{held_out} has no data or no token embeddings; available slides: {data['slides']}")
    print(f"Slides: {data['slides']}")

    # The callee frees the full tables after splitting, which only works if this frame no longer references them.
    handoff = [data.pop("hist"), data.pop("trans"), data.pop("token")]
    print(f"\n=== Leave-one-slide-out: holding out {held_out} ===\n")
    metrics, _, _ = run_leave_one_slide_out_token(
        held_out, handoff.pop(0), handoff.pop(0), data["coord"], handoff.pop(0), run_config, device,
        log_path=log_path, checkpoint_dir=checkpoint_dir, seed=config["seed"], verbose=True, run_tag=tag,
    )
    print_result(metrics, held_out, log_path, embeddings_file)


if __name__ == "__main__":
    main()
