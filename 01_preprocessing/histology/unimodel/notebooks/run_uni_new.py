#!/usr/bin/env python3
import argparse
import gc
import glob
import os
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patch and token features using UNI model."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../../data",
        help="Path to base data directory.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="../weights",
        help="Path to directory containing model weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global seed set to: {seed}")


def check_for_patches(basedir: Path, skip_samples=None):
    if skip_samples is None:
        skip_samples = ["new", "CRC_P1", "CRC_P2", "CRC_P5", "Normal_P3", "Normal_P5"]

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
        return image, -1


@torch.no_grad()
def extract_patch_and_token_features(model, dataloader, device, num_prefix_tokens):
    all_patch_embeddings = []
    all_token_embeddings = []

    for batch, _ in dataloader:
        batch = batch.to(device)
        feats = model.forward_features(batch)

        patch_emb = model.forward_head(feats, pre_logits=True)
        all_patch_embeddings.append(patch_emb.cpu())

        patch_tokens = feats[:, num_prefix_tokens:, :].to(torch.float16)
        all_token_embeddings.append(patch_tokens.cpu())

    patch_feats_tensor = torch.cat(all_patch_embeddings, dim=0)
    token_feats_tensor = torch.cat(all_token_embeddings, dim=0)

    return patch_feats_tensor, token_feats_tensor


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # Model Setup
    # ============================================================
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(exist_ok=True, parents=True)
    weights_path = weights_dir / "pytorch_model.bin"

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

    num_prefix_tokens = getattr(
        model, "num_prefix_tokens", 1 + timm_kwargs.get("reg_tokens", 0)
    )
    grid_h, grid_w = model.patch_embed.grid_size
    n_tokens = grid_h * grid_w
    token_patch_px = timm_kwargs["patch_size"]

    # ============================================================
    # Paths & Setup
    # ============================================================
    basedir = Path(args.data_dir)
    skip_samples = ["new", "CRC_P1", "CRC_P2", "CRC_P5", "Normal_P3", "Normal_P5"]

    f_metadata_h5ad = "WSI_patch_metadata.h5ad"
    f_metadata_csv = "WSI_patch_metadata.csv"

    master_csv_path = basedir / "combined/01_preprocessing/unimodel/metadata" / f_metadata_csv

    patch_directories = check_for_patches(basedir, skip_samples)

    if not master_csv_path.exists():
        raise FileNotFoundError(f"Master metadata CSV file not found at {master_csv_path}")

    print(f"Loading master spatial metadata from {master_csv_path}...")
    master_metadata_df = pd.read_csv(master_csv_path)

    # Ensure metadata data types match expectations
    master_metadata_df['Slide_ID'] = master_metadata_df['Slide_ID'].astype(str)
    master_metadata_df['Patch_ID'] = master_metadata_df['Patch_ID'].astype(str)
    master_metadata_df['X'] = master_metadata_df['X'].astype(float)
    master_metadata_df['Y'] = master_metadata_df['Y'].astype(float)

    master_metadata_df['match_id'] = master_metadata_df['Slide_ID'] + '_' + master_metadata_df['Patch_ID']

    dir_combined_meta = basedir / "combined" / "01_preprocessing" / "unimodel" / "metadata"
    dir_combined_meta.mkdir(parents=True, exist_ok=True)

    combined_patch_h5ad_path = dir_combined_meta / f_metadata_h5ad
    combined_token_h5ad_path = dir_combined_meta / "WSI_patch_token_embeddings.h5ad"

    all_patch_adata_list = []
    all_token_adata_list = []

    # ============================================================
    # Main Processing Loop
    # ============================================================
    for dir_path in patch_directories:
        sample_name = dir_path.parents[2].name
        print(f"\n================ Processing Sample: {sample_name} ================")

        metadata_dir = dir_path.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        sample_metadata_df = master_metadata_df[master_metadata_df['Slide_ID'] == sample_name].copy()
        if sample_metadata_df.empty:
            print(f"Skipping {sample_name}: No matching metadata found in master CSV.")
            continue

        test_dataset = PatchDataset(root_dir=dir_path, transform=transform)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )

        # Feature Extraction
        patch_feats, token_feats = extract_patch_and_token_features(
            model, test_dataloader, device, num_prefix_tokens
        )

        # --------------------------------------------------------
        # PART I: Patch-Level AnnData
        # --------------------------------------------------------
        df_patch = pd.DataFrame(patch_feats.numpy())
        df_patch["Slide_ID"] = sample_name
        df_patch["Patch_ID"] = test_dataset.patch_ids
        df_patch['match_id'] = df_patch['Slide_ID'].astype(str) + '_' + df_patch['Patch_ID'].astype(str)

        init_len = len(df_patch)
        df_patch = df_patch.merge(sample_metadata_df[['match_id', 'X', 'Y']], on='match_id', how='left')
        if len(df_patch) != init_len:
            print(f"Warning: Row count mismatch in patch merge ({init_len} -> {len(df_patch)})!")

        meta_cols = ["Slide_ID", "Patch_ID", "X", "Y"]
        feat_cols = [c for c in df_patch.columns if c not in meta_cols + ["match_id"]]

        adata_patch = ad.AnnData(X=df_patch[feat_cols].values, obs=df_patch[meta_cols].copy())
        adata_patch.obsm["spatial"] = df_patch[["X", "Y"]].values
        adata_patch.obs.index = df_patch["Slide_ID"].astype(str) + "_" + df_patch["Patch_ID"].astype(str)

        # Save sample patch file
        patch_h5ad_path = metadata_dir / f_metadata_h5ad
        adata_patch.write(patch_h5ad_path)
        print(f" Saved Patch AnnData: {patch_h5ad_path}")
        all_patch_adata_list.append(adata_patch)

        # --------------------------------------------------------
        # PART II: Token-Level AnnData
        # --------------------------------------------------------
        token_matrix = token_feats.reshape(-1, token_feats.shape[-1]).numpy()

        t_sample_ids, t_patch_ids, t_ids, t_rows, t_cols = [], [], [], [], []
        for p_id in test_dataset.patch_ids:
            for t in range(n_tokens):
                t_sample_ids.append(sample_name)
                t_patch_ids.append(p_id)
                t_ids.append(t)
                t_rows.append(t // grid_w)
                t_cols.append(t % grid_w)

        obs_tokens = pd.DataFrame({
            "Slide_ID": t_sample_ids,
            "Patch_ID": t_patch_ids,
            "Token_ID": np.array(t_ids, dtype=np.uint16),
            "Token_row": np.array(t_rows, dtype=np.uint8),
            "Token_col": np.array(t_cols, dtype=np.uint8),
        })

        obs_tokens["match_id"] = obs_tokens["Slide_ID"].astype(str) + "_" + obs_tokens["Patch_ID"].astype(str)
        obs_tokens = obs_tokens.merge(sample_metadata_df[["match_id", "X", "Y"]], on="match_id", how="left").drop(columns="match_id")

        obs_tokens["Token_X"] = obs_tokens["X"] + obs_tokens["Token_col"] * token_patch_px
        obs_tokens["Token_Y"] = obs_tokens["Y"] + obs_tokens["Token_row"] * token_patch_px
        obs_tokens.index = obs_tokens["Slide_ID"].astype(str) + "_" + obs_tokens["Patch_ID"].astype(str) + "_token" + obs_tokens["Token_ID"].astype(str)

        adata_token = ad.AnnData(X=token_matrix, obs=obs_tokens)
        adata_token.obsm["spatial"] = obs_tokens[["Token_X", "Token_Y"]].values

        # Save sample token file
        token_h5ad_path = metadata_dir / "WSI_patch_token_embeddings.h5ad"
        adata_token.write(token_h5ad_path, compression="gzip")
        print(f" Saved Sample Token AnnData: {token_h5ad_path}")
        all_token_adata_list.append(adata_token)

        # Cleanup intermediate arrays
        del patch_feats, token_feats, token_matrix, df_patch, obs_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # Master Files Generation
    # ============================================================
    if all_patch_adata_list:
        print("\nWriting master combined Patch AnnData...")
        new_patch_adata = ad.concat(all_patch_adata_list, join='outer', index_unique=None)
        if combined_patch_h5ad_path.exists():
            existing_patch_adata = ad.read_h5ad(combined_patch_h5ad_path)
            combined_patch_adata = ad.concat([existing_patch_adata, new_patch_adata], join='outer', index_unique=None)
            combined_patch_adata.write(combined_patch_h5ad_path)
        else:
            new_patch_adata.write(combined_patch_h5ad_path)

    if all_token_adata_list:
        print("Writing master combined Token AnnData...")
        new_token_adata = ad.concat(all_token_adata_list, join='outer', index_unique=None)
        if combined_token_h5ad_path.exists():
            existing_token_adata = ad.read_h5ad(combined_token_h5ad_path)
            combined_token_adata = ad.concat([existing_token_adata, new_token_adata], join='outer', index_unique=None)
            combined_token_adata.write(combined_token_h5ad_path, compression="gzip")
        else:
            new_token_adata.write(combined_token_h5ad_path, compression="gzip")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
