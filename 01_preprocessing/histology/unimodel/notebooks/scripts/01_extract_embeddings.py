#!/usr/bin/env python3
import argparse
import gc
import glob
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# already processed samples, ignored if --sample is set
SKIP_SAMPLES = ["new", "CRC_P1", "CRC_P2", "CRC_P5", "Normal_P3", "Normal_P5"]

# defaults relative to this file, independent of the working directory
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[4] / "data"  # Genomosaic/data
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parents[1] / "weights"  # unimodel/weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patch and token features using UNI model."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        # one subfolder per sample
        # needs {sample}/01_preprocessing/unimodel/patches/*.png
        # needs combined/01_preprocessing/unimodel/metadata/WSI_patch_metadata.csv
        # both created by 01_patch_creation_uni.ipynb
        help="Path to base data directory.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=str(DEFAULT_WEIGHTS_DIR),
        # must contain UNI2-h checkpoint pytorch_model.bin, no auto-download
        help="Path to directory containing model weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        # patches per forward pass, lower if CUDA out of memory
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        # CPU workers for loading PNGs, only affects speed
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        # single Slide_ID, None = all folders except SKIP_SAMPLES
        help="Slide_ID of a single sample to process. If omitted, all valid "
        "samples under --data_dir are processed.",
    )
    return parser.parse_args()


def check_for_patches(basedir: Path, skip_samples, only_sample=None):
    # single sample or all non-skipped samples
    if only_sample is not None:
        sample_dirs = [
            p for p in basedir.iterdir() if p.is_dir() and p.name == only_sample
        ]
        if not sample_dirs:
            raise FileNotFoundError(
                f"Sample '{only_sample}' not found as a directory under {basedir}"
            )
    else:
        sample_dirs = [
            p
            for p in basedir.iterdir()
            if p.is_dir() and p.name != "combined" and p.name not in skip_samples
        ]

    valid_patch_dirs = []
    for sample_dir in sample_dirs:
        patch_dir = sample_dir / "01_preprocessing" / "unimodel" / "patches"
        if patch_dir.is_dir() and any(patch_dir.glob("*.png")):
            valid_patch_dirs.append(patch_dir)
        else:
            print(f"Skipping {sample_dir.name}: No patch PNGs found at {patch_dir}")

    if not valid_patch_dirs:
        raise FileNotFoundError(
            f"No valid sample patch directories found inside {basedir}"
        )

    print(f"Found {len(valid_patch_dirs)} valid sample patch directories.")
    return valid_patch_dirs


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.transform = transform
        self.patch_ids = [os.path.basename(img) for img in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, -1  # dummy label


@torch.no_grad()
def extract_patch_and_token_features(model, dataloader, device, num_prefix_tokens):
    all_patch_embeddings = []
    all_token_embeddings = []

    for batch, _ in dataloader:
        batch = batch.to(device)
        feats = model.forward_features(batch)

        # pooled embedding per patch
        patch_emb = model.forward_head(feats, pre_logits=True)
        all_patch_embeddings.append(patch_emb.cpu())

        # drop cls/register tokens, keep spatial tokens (fp16 to save memory)
        patch_tokens = feats[:, num_prefix_tokens:, :].to(torch.float16)
        all_token_embeddings.append(patch_tokens.cpu())

    patch_feats_tensor = torch.cat(all_patch_embeddings, dim=0)
    token_feats_tensor = torch.cat(all_token_embeddings, dim=0)

    return patch_feats_tensor, token_feats_tensor


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model
    weights_path = Path(args.weights_dir) / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model(**timm_kwargs)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    # cls + register tokens come first, then the token grid
    num_prefix_tokens = getattr(
        model, "num_prefix_tokens", 1 + timm_kwargs.get("reg_tokens", 0)
    )
    grid_h, grid_w = model.patch_embed.grid_size  # (16, 16)
    n_tokens = grid_h * grid_w
    token_patch_px = timm_kwargs["patch_size"]  # 14

    # paths
    basedir = Path(args.data_dir)
    dir_combined = basedir / "combined/01_preprocessing/unimodel/metadata"
    f_metadata_h5ad = "WSI_patch_metadata.h5ad"
    f_metadata_csv = "WSI_patch_metadata.csv"
    f_token_h5ad = "WSI_patch_token_embeddings.h5ad"

    master_csv_path = dir_combined / f_metadata_csv
    combined_patch_h5ad_path = dir_combined / f_metadata_h5ad

    patch_directories = check_for_patches(basedir, SKIP_SAMPLES, only_sample=args.sample)

    if not master_csv_path.exists():
        raise FileNotFoundError(f"Master metadata CSV file not found at {master_csv_path}")

    print(f"Loading master spatial metadata from {master_csv_path}...")
    master_metadata_df = pd.read_csv(master_csv_path, dtype={"Slide_ID": str, "Patch_ID": str, "X": float, "Y": float})

    all_patch_adata_list = []
    processed_sample_names = []

    # per-sample processing
    for dir_path in patch_directories:
        sample_name = dir_path.parents[2].name
        print(f"\n================ Processing Sample: {sample_name} ================")

        metadata_dir = dir_path.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # patch positions of current sample only
        sample_metadata_df = master_metadata_df[master_metadata_df["Slide_ID"] == sample_name]
        if sample_metadata_df.empty:
            print(f"Skipping {sample_name}: No matching metadata found in master CSV.")
            continue
        sample_positions = sample_metadata_df[["Patch_ID", "X", "Y"]]

        test_dataset = PatchDataset(root_dir=dir_path, transform=transform)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # keep patch order
            num_workers=args.num_workers,
        )

        patch_feats, token_feats = extract_patch_and_token_features(
            model, test_dataloader, device, num_prefix_tokens
        )

        # patch-level adata, X/Y attached per patch
        obs_patch = pd.DataFrame({"Slide_ID": sample_name, "Patch_ID": test_dataset.patch_ids})
        obs_patch = obs_patch.merge(sample_positions, on="Patch_ID", how="left")
        if len(obs_patch) != len(patch_feats):
            print(f"Warning: Row count mismatch in patch merge ({len(patch_feats)} -> {len(obs_patch)})!")
        obs_patch.index = sample_name + "_" + obs_patch["Patch_ID"]

        adata_patch = ad.AnnData(X=patch_feats.numpy(), obs=obs_patch)
        adata_patch.obsm["spatial"] = obs_patch[["X", "Y"]].values

        patch_h5ad_path = metadata_dir / f_metadata_h5ad
        adata_patch.write(patch_h5ad_path)
        print(f" Saved Patch AnnData: {patch_h5ad_path}")
        all_patch_adata_list.append(adata_patch)

        # token-level adata (one row per token)
        token_matrix = token_feats.reshape(-1, token_feats.shape[-1]).numpy()

        # token index within patch -> grid row/col
        token_ids = np.tile(np.arange(n_tokens), len(test_dataset.patch_ids))
        obs_tokens = pd.DataFrame({
            "Slide_ID": sample_name,
            "Patch_ID": np.repeat(test_dataset.patch_ids, n_tokens),
            "Token_ID": token_ids.astype(np.uint16),
            "Token_row": (token_ids // grid_w).astype(np.uint8),
            "Token_col": (token_ids % grid_w).astype(np.uint8),
        })
        obs_tokens = obs_tokens.merge(sample_positions, on="Patch_ID", how="left")

        # token position = patch origin + grid offset in px
        obs_tokens["Token_X"] = obs_tokens["X"] + obs_tokens["Token_col"] * token_patch_px
        obs_tokens["Token_Y"] = obs_tokens["Y"] + obs_tokens["Token_row"] * token_patch_px
        obs_tokens.index = sample_name + "_" + obs_tokens["Patch_ID"] + "_token" + obs_tokens["Token_ID"].astype(str)

        adata_token = ad.AnnData(X=token_matrix, obs=obs_tokens)
        adata_token.obsm["spatial"] = obs_tokens[["Token_X", "Token_Y"]].values

        # tokens are only stored per sample (no combined file)
        token_h5ad_path = metadata_dir / f_token_h5ad
        adata_token.write(token_h5ad_path, compression="gzip")
        print(f" Saved Sample Token AnnData: {token_h5ad_path}")

        processed_sample_names.append(sample_name)

        # free memory
        del patch_feats, token_feats, token_matrix, obs_tokens, adata_token
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # combined patch file: append new samples, replace re-processed ones
    if all_patch_adata_list:
        print("\nWriting master combined Patch AnnData...")
        new_patch_adata = ad.concat(all_patch_adata_list, join="outer", index_unique=None)
        if combined_patch_h5ad_path.exists():
            existing_patch_adata = ad.read_h5ad(combined_patch_h5ad_path)
            stale_mask = existing_patch_adata.obs["Slide_ID"].astype(str).isin(processed_sample_names)
            if stale_mask.any():
                print(f"Dropping {stale_mask.sum()} pre-existing rows for re-processed sample(s) "
                      f"{sorted(set(processed_sample_names))} before appending.")
                existing_patch_adata = existing_patch_adata[~stale_mask].copy()
            new_patch_adata = ad.concat([existing_patch_adata, new_patch_adata], join="outer", index_unique=None)
        new_patch_adata.write(combined_patch_h5ad_path)

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
