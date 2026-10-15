"""
Building blocks for training the dual autoencoder on all slides.

Used by ``03_integration/train_autoencoder.ipynb`` and by the leave-one-slide-out
script ``03_integration/scripts/run_loso.py``. All settings live in one JSON file
(``CONFIG_PATH``); the key names are the ones ``functions.dual_autoencoder`` reads.

Typical order:
    config = load_config()
    data = load_slide_data(config)
    pipe = build_pipeline(data, config)
    model = build_model(config, pipe, device)
    pretrain_token_branch(model, pipe, config, device)
    history = train_joint(model, pipe, config, device)
    evaluation = evaluate_embeddings(model, pipe, device)
    save_run(output_dir, model, pipe, config, history, evaluation)
"""

import json
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from functions.dual_autoencoder import (
    DualAutoencoder,
    build_pipeline_data,
    evaluate_latent_leakage,
    freeze_token_encoder,
    get_gene_mask,
    pretrain_token_autoencoder,
    set_seed,
    token_retrieval_accuracy,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = next((d / "data" for d in (REPO_ROOT, *REPO_ROOT.parents) if (d / "data").is_dir()), REPO_ROOT / "data")
CONFIG_PATH = REPO_ROOT / "03_integration/config/training_config.json"
TUPLE_KEYS = ("num_neighbors",)

UNIT_PRESETS = {
    "nuclei": dict(
        bin_size="002um", dir_cellseg="cell_seg_stardist", token_filename="cell_token_embeddings.h5ad",
        n_neighs=6, spatial_graph_delaunay=True, spatial_graph_grid=False,
        max_cells_per_slide=300000, oversample_slides={"Normal_P3": 172000, "Normal_P5": 383000},
    ),
    "bins": dict(
        bin_size="008um", dir_cellseg="grid_bin", token_filename="bin_token_embeddings.h5ad",
        n_neighs=8, spatial_graph_delaunay=False, spatial_graph_grid=True,
        max_cells_per_slide=610000, oversample_slides={"Normal_P3": 470000, "Normal_P5": 856000},
    ),
}

CONFIG_HELP = {
    "slides": "slides to use; none selects all slides that have token data",
    "exclude_slides": "slides removed from the data entirely",
    "output_name": "output directory name of the all-slide training run",
    "seed": "random seed of the model initialisation",
    "bin_size": "unit resolution directory, 002um for nuclei and 008um for bins",
    "dir_cellseg": "segmentation or binning directory, cell_seg_stardist or grid_bin",
    "token_filename": "per-slide token embedding file",
    "gene_selection": "genes used as input: svg, hvg or intersection (ignored if gene_panel is set)",
    "gene_panel": "name of an external gene panel in gene_panels/ (for example colon_xenium); none uses gene_selection",
    "latent_dim": "latent dimension of both encoders",
    "proj_dim": "dimension of the contrastive projection heads",
    "encoder": "transcriptomics graph encoder: transconv or sage",
    "token_condition_on_slide": "condition the token decoder on the slide",
    "trans_condition_on_slide": "condition the transcriptomics decoder on the slide",
    "n_neighs": "neighbours of the fallback and grid graphs",
    "spatial_graph_delaunay": "build the spatial graph by Delaunay triangulation",
    "spatial_graph_grid": "build the spatial graph on the regular bin lattice",
    "num_neighbors": "neighbours sampled per hop by the loader",
    "cells_per_batch": "seed units per batch",
    "max_cells_per_slide": "cap on a slide's units per epoch; none disables",
    "oversample_slides": "JSON mapping slide to units per epoch after resampling; {} disables",
    "neighbor_radius": "radius of the patch metadata; not used by the contrastive loss",
    "temperature": "softmax temperature of the contrastive loss",
    "local_neighbor_radius": "spatial exclusion radius in pixels for negatives; none disables",
    "local_anchor_sim_threshold": "cosine similarity above which non-matching pairs are excluded; none disables",
    "local_anchor_sim_source": "features the similarity is computed on: projected or raw",
    "local_positive_mode": "exclude flagged pairs from the negatives or attract them as extra positives",
    "weight_reconstruction_by_patch_freq": "weight the reconstruction loss by inverse local density",
    "edge_dropout_p": "fraction of graph edges dropped per step",
    "gene_dropout_p": "fraction of gene entries masked per step",
    "weight_decay": "AdamW weight decay of the joint training",
    "n_epochs": "epochs of the joint training",
    "lr": "learning rate of the joint training",
    "warmup_epochs": "warm-up epochs of the learning schedule",
    "skip_token_pretrain": "skip the token autoencoder pretraining",
    "freeze_token": "freeze the token encoder and decoder during joint training",
    "token_pretrain_epochs": "maximum epochs of the token pretraining",
    "token_pretrain_lr": "learning rate of the token pretraining",
    "token_pretrain_weight_decay": "weight decay of the token pretraining",
    "token_pretrain_val_frac": "validation fraction of the token pretraining",
    "token_pretrain_patience": "early-stopping patience of the token pretraining",
    "token_lr_mult": "learning-rate multiplier of the token branch during joint training",
    "max_local_contrast_weight": "final weight of the cell-to-token contrastive loss",
    "local_warmup_epochs": "epochs over which the contrastive weight ramps up",
}


def apply_unit_preset(config, unit):
    """Return a copy of ``config`` with the unit-specific settings of ``UNIT_PRESETS[unit]``."""
    if unit not in UNIT_PRESETS:
        raise ValueError(f"unit must be one of {sorted(UNIT_PRESETS)}, got {unit!r}")
    return {**config, **UNIT_PRESETS[unit]}


def run_tag(config):
    """
    Suffix that distinguishes the output files of runs with different settings, for example
    ``_edrop0.1_gdrop0.1_wd1e-05_cap300000_oversample_normal``. Only settings that differ
    from the reference values (transconv encoder, temperature 0.05, spatial radius 100, ...) appear.
    """
    parts = ["bins"] if config["dir_cellseg"] == "grid_bin" else []
    if config["freeze_token"]:
        parts.append("frozen_token")
    if config["skip_token_pretrain"]:
        parts.append("nopretrain")
    if config["edge_dropout_p"] > 0 or config["gene_dropout_p"] > 0:
        parts.append(f"edrop{config['edge_dropout_p']}_gdrop{config['gene_dropout_p']}")
    if config["local_neighbor_radius"] != 100:
        parts.append(f"localrad{config['local_neighbor_radius']}")
    if config["local_positive_mode"] != "exclude":
        parts.append(f"posmode_{config['local_positive_mode']}")
    if config["local_anchor_sim_threshold"] is not None:
        parts.append(f"anchorsim{config['local_anchor_sim_threshold']}")
        if config["local_anchor_sim_source"] != "projected":
            parts.append(f"simsrc_{config['local_anchor_sim_source']}")
    if not config["weight_reconstruction_by_patch_freq"]:
        parts.append("noweight")
    if config["weight_decay"] > 0:
        parts.append(f"wd{config['weight_decay']}")
    if config["encoder"] != "transconv":
        parts.append(f"enc_{config['encoder']}")
    if config["temperature"] != 0.05:
        parts.append(f"temp{config['temperature']}")
    if config["gene_panel"]:
        parts.append(f"panel_{config['gene_panel']}")
    elif config["gene_selection"] != "svg":
        parts.append(config["gene_selection"])
    if config["exclude_slides"]:
        parts.append("excl_" + "_".join(config["exclude_slides"]))
    if config["max_cells_per_slide"] is not None:
        parts.append(f"cap{config['max_cells_per_slide']}")
    if config["oversample_slides"]:
        parts.append("oversample_normal")
    return "_" + "_".join(parts) if parts else ""


def load_config(path=CONFIG_PATH):
    """Read the training configuration from JSON and restore tuple-valued entries."""
    config = json.loads(Path(path).read_text())
    for key in TUPLE_KEYS:
        config[key] = tuple(config[key])
    return config


def save_config(config, path=CONFIG_PATH):
    """Write the training configuration to JSON, replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def data_paths(config, data_dir=DATA_DIR):
    """Input and output directories implied by ``bin_size``, ``dir_cellseg`` and ``output_name``."""
    preprocessing = data_dir / "combined/01_preprocessing"
    return {
        "histology": preprocessing / "unimodel/metadata",
        "transcriptomics": preprocessing / config["dir_cellseg"] / config["bin_size"] / "custom_binning",
        "gene_panels": preprocessing / "gene_panels",
        "output": data_dir / "combined/03_integration/autoencoders" / config["bin_size"] / config["output_name"],
    }


def _index_by_slide_and_id(adata, id_column):
    """Replace ``obs`` index by ``<Slide_ID>_<id_column>``, which must be unique across slides."""
    key = adata.obs["Slide_ID"].astype(str) + "_" + adata.obs[id_column].astype(str)
    if not key.is_unique:
        raise ValueError(f"Slide_ID + {id_column} is not unique")
    adata.obs = adata.obs.set_index(key)


def load_slide_data(config, data_dir=DATA_DIR):
    """
    Load the inputs of the configured slides.

    Only slides that have token embeddings are kept, since a slide without them would
    let the model separate slides by missing values. ``config["slides"]`` selects slides
    (None: all), ``config["exclude_slides"]`` removes slides.

    Returns a dict with ``hist``, ``trans``, ``coord`` (None for bins), ``token`` and ``slides``.
    """
    paths = data_paths(config, data_dir)
    trans_file = ad.read_h5ad(paths["transcriptomics"] / "count_area_filtered_adata.h5ad", backed="r")
    available = sorted(trans_file.obs["Slide_ID"].astype(str).unique())
    requested = config["slides"] or available
    slides = [s for s in requested if s in available and s not in config["exclude_slides"]]

    token_parts = {}
    for slide in slides:
        path = data_dir / slide / "01_preprocessing/unimodel/metadata" / config["token_filename"]
        if path.exists():
            token_parts[slide] = sc.read_h5ad(path)
        else:
            print(f"No token data for {slide}: {path}")
    slides = [s for s in slides if s in token_parts]
    if not slides:
        raise RuntimeError("None of the selected slides has token data")

    in_selection = trans_file.obs["Slide_ID"].astype(str).isin(slides).to_numpy()
    trans = trans_file.to_memory() if in_selection.all() else trans_file[in_selection].to_memory()
    trans_file.file.close()
    _index_by_slide_and_id(trans, "id")

    hist = sc.read_h5ad(paths["histology"] / "filtered_WSI_patch_embeddings.h5ad")
    hist = hist[hist.obs["Slide_ID"].astype(str).isin(slides).to_numpy()].copy()
    _index_by_slide_and_id(hist, "Patch_ID")

    geometry_path = paths["transcriptomics"] / "nuclei_geometry.parquet"
    coord = None
    if geometry_path.exists():
        coord = gpd.read_parquet(geometry_path)
        coord = coord[coord["Slide_ID"].astype(str).isin(slides)].copy()

    token = ad.concat([token_parts[s] for s in slides], join="outer")
    _index_by_slide_and_id(token, "id")
    token = token[token.obs["matched"].astype(bool).to_numpy()].copy()

    return {"hist": hist, "trans": trans, "coord": coord, "token": token, "slides": slides}


def build_pipeline(data, config, verbose=True):
    """
    Select genes, build the spatial graph, pair units with histology patches and tokens,
    standardise the features and create the slide-stratified loader.

    Returns the dictionary of ``functions.dual_autoencoder.build_pipeline_data``.
    """
    panel = config["gene_panel"]
    panel_path = data_paths(config)["gene_panels"] / f"{panel}.json" if panel else None
    gene_mask = get_gene_mask(data["trans"], gene_selection=config["gene_selection"], gene_panel_path=panel_path)
    return build_pipeline_data(
        data["hist"], data["trans"], data["coord"], config, svg_mask=gene_mask,
        f_token_adata=data["token"], local_neighbor_radius=config["local_neighbor_radius"],
        max_cells_per_slide=config["max_cells_per_slide"], oversample_slides=config["oversample_slides"],
        verbose=verbose,
    )


def build_model(config, pipe, device):
    """Create the dual autoencoder with dimensions taken from the prepared data."""
    set_seed(config["seed"])
    model = DualAutoencoder(
        n_svg=pipe["n_svg"], token_dim=pipe["token_dim"],
        latent_dim=config["latent_dim"], proj_dim=config["proj_dim"],
        n_slides=pipe["n_slides"], encoder_type=config["encoder"],
        token_condition_on_slide=config["token_condition_on_slide"],
        trans_condition_on_slide=config["trans_condition_on_slide"],
    )
    return model.to(device)


def pretrain_token_branch(model, pipe, config, device):
    """
    Pretrain the token encoder and decoder as an autoencoder with early stopping.

    Returns ``{"best_val_mse", "stopped_epoch"}``; a skipped pretraining gives NaN and -1.
    """
    if config["skip_token_pretrain"]:
        return {"best_val_mse": float("nan"), "stopped_epoch": -1}
    return pretrain_token_autoencoder(
        model, pipe["token_emb_t"], pipe["cell_meta"],
        epochs=config["token_pretrain_epochs"], lr=config["token_pretrain_lr"],
        weight_decay=config["token_pretrain_weight_decay"],
        val_frac=config["token_pretrain_val_frac"], patience=config["token_pretrain_patience"],
        device=device,
    )


def train_joint(model, pipe, config, device):
    """
    Joint training with reconstruction of both modalities and the cell-to-token contrastive loss.

    Returns the per-epoch history (list of dicts with the loss terms).
    """
    if config["freeze_token"]:
        freeze_token_encoder(model, freeze_decoder=True)
    return train(
        model, pipe["loader"], pipe["density_weight"],
        n_epochs=config["n_epochs"], lr=config["lr"], warmup_epochs=config["warmup_epochs"],
        temperature=config["temperature"],
        weight_reconstruction=config["weight_reconstruction_by_patch_freq"],
        weight_decay=config["weight_decay"],
        edge_dropout_p=config["edge_dropout_p"], gene_dropout_p=config["gene_dropout_p"],
        device=device,
        token_emb_all=pipe["token_emb_t"], cell_meta=pipe["cell_meta"],
        max_local_contrast_weight=config["max_local_contrast_weight"],
        local_warmup_epochs=config["local_warmup_epochs"],
        freeze_token=config["freeze_token"], token_lr_mult=config["token_lr_mult"],
        local_anchor_sim_threshold=config["local_anchor_sim_threshold"],
        local_anchor_sim_source=config["local_anchor_sim_source"],
        local_positive_mode=config["local_positive_mode"],
    )


def evaluate_embeddings(model, pipe, device, leakage_subsample=20_000, seed=0):
    """
    Encode all units and measure cell-to-token retrieval and slide leakage.

    Every slide was part of training, so the scores are in-sample and only show whether
    training worked; they say nothing about generalisation (see the leave-one-slide-out runs).

    Returns ``{"metrics", "z_trans", "z_token"}``.
    """
    cell_meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in pipe["cell_meta"].items()}
    cluster_labels = pd.factorize(pipe["trans_sub"].obs["clusters"])[0]
    retrieval = token_retrieval_accuracy(
        model, pipe["data"], pipe["token_emb_t"], cell_meta, device=device, cell_type_labels=cluster_labels,
    )
    z_trans, z_token = retrieval["z_trans_cell_full"], retrieval["z_token_cell_full"]

    rng = np.random.default_rng(seed)
    subset = rng.choice(len(z_token), size=min(leakage_subsample, len(z_token)), replace=False)
    leakage = evaluate_latent_leakage(z_token[subset], pipe["cell_meta"]["slide_id"].cpu().numpy()[subset])
    metrics = {
        "cell_top1_retrieval": retrieval["mean_top1"],
        "cell_cluster_top1_retrieval": retrieval["mean_cluster_top1"],
        "token_leakage_mean": leakage["leakage_mean"],
        "token_leakage_chance": leakage["leakage_chance"],
    }
    return {"metrics": metrics, "z_trans": z_trans, "z_token": z_token}


def save_run(output_dir, model, pipe, config, history, evaluation, pretrain_info=None):
    """
    Write everything needed to reuse the trained model into ``output_dir``:
    ``dual_ae_checkpoint.pt`` (weights, config, scalers, gene names, history),
    ``dual_ae_latents.npz`` (embeddings with unit ids and slide names), ``gene_panel.txt``,
    ``training_config.json`` and ``metrics.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trans_sub = pipe["trans_sub"]
    gene_names = trans_sub.var_names[pipe["svg_mask"]].tolist()
    torch.save(
        {
            "model_state_dict": model.state_dict(), "config": config, "history": history,
            "metrics": evaluation["metrics"], "pretrain": pretrain_info, "gene_names": gene_names,
            "svg_scaler": pipe["svg_scaler"], "hist_scaler": pipe["hist_scaler"],
            "token_scaler": pipe["token_scaler"],
        },
        output_dir / "dual_ae_checkpoint.pt",
    )
    np.savez(
        output_dir / "dual_ae_latents.npz",
        z_trans=evaluation["z_trans"], z_token=evaluation["z_token"],
        unit_id=trans_sub.obs_names.to_numpy(), slide=trans_sub.obs["Slide_ID"].astype(str).to_numpy(),
    )
    (output_dir / "gene_panel.txt").write_text("\n".join(gene_names) + "\n")
    save_config(config, output_dir / "training_config.json")
    (output_dir / "metrics.json").write_text(json.dumps(evaluation["metrics"], indent=2) + "\n")
    return output_dir
