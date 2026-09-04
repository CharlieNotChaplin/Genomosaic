"""
Throwaway script version of BinPreprocessing.ipynb's per-sample steps (load ->
QC -> clustering -> patch-assignment), so the bin pipeline can be driven
per-sample from run_bin_pipeline.sbatch instead of running the notebook by
hand for every sample. The notebook stays the source of truth for anything
interactive (UMI-cutoff diagnostic plots, ad-hoc inspection) -- this script
just applies umi_cutoff directly, no plotting.

Does NOT merge across samples (needs every sample's output to exist first) --
run BinPreprocessing.ipynb's merge cells (14-17) by hand once this script
(and get_svg.py --unit bin) have both been run for every sample.

Usage:
    python bin_preprocessing.py --sample CRC_P1
"""
import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path("../..").resolve()))
from functions.label_transfer import load_grid_bin_counts, aggregate_expression_to_grid, assign_bins_to_patches
from functions.merge import read_h5ad_with_null_uns_healing, sanitize_uns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample", required=True, help="Sample name, e.g. CRC_P1")
parser.add_argument("--group-size", type=int, default=4,
                     help="2um bins per 8um bin side, forwarded to aggregate_expression_to_grid (fallback loader only)")
parser.add_argument("--umi-cutoff", type=float, default=80,
                     help="same default BinPreprocessing.ipynb / the cell path start from")
args = parser.parse_args()

sample = args.sample
dir_binning = "grid_bin"
bin_size = "008um"

basedir = Path("../../data")
dir_result_output = basedir / sample / "01_preprocessing" / dir_binning / bin_size
dir_custom_binning = dir_result_output / "custom_binning"
dir_result_output.mkdir(parents=True, exist_ok=True)
dir_custom_binning.mkdir(parents=True, exist_ok=True)

###############################################################
###################### Load bin counts #########################
###############################################################

# Same runtime check main_grid() already uses: native square_008um output if
# present, else exact 4x4 aggregation from square_002um.
grid_h5_path = basedir / sample / "output/binned_outputs/square_008um/filtered_feature_bc_matrix.h5"
if grid_h5_path.exists():
    print(f"{sample}: native square_008um output found")
    adata = load_grid_bin_counts(sample, data_dir=basedir)
else:
    print(f"{sample}: no native square_008um output -- aggregating from square_002um instead")
    adata = aggregate_expression_to_grid(sample, group_size=args.group_size, data_dir=basedir)

# Neither loader sets Slide_ID or a canonical, non-sample-prefixed obs id --
# normalize both here so obs schema/obs_names match the cell path's convention
# regardless of which loader ran (see BinPreprocessing.ipynb cell 5 for why).
adata.obs["Slide_ID"] = sample
adata.obs["id"] = [f"grid_{r}_{c}" for r, c in zip(adata.obs["array_row"], adata.obs["array_col"])]
adata.obs_names = adata.obs["id"].values
adata.var_names_make_unique()
print(f"{sample}: {adata.n_obs} bins, {adata.n_vars} genes")

###############################################################
###################### QC (UMI count only) #####################
###############################################################

sc.pp.calculate_qc_metrics(adata, inplace=True)
mask_count = adata.obs["total_counts"] > args.umi_cutoff
count_area_filtered_adata = adata[mask_count].copy()
count_area_filtered_adata.var_names_make_unique()
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)
print(f"{mask_count.sum()}/{adata.n_obs} bins kept (total_counts > {args.umi_cutoff})")
del adata
gc.collect()

###############################################################
###################### Clustering ##############################
###############################################################

sc.pp.normalize_total(count_area_filtered_adata, inplace=True)
sc.pp.log1p(count_area_filtered_adata)
sc.pp.highly_variable_genes(count_area_filtered_adata, flavor="seurat", n_top_genes=2000)
sc.pp.pca(count_area_filtered_adata)
sc.pp.neighbors(count_area_filtered_adata)
# flavor="igraph": default leidenalg backend is impractically slow at bin scale
# (systematically more nodes than the cell path) -- same convention already
# used elsewhere in this codebase.
sc.tl.leiden(count_area_filtered_adata, resolution=0.2, key_added="clusters",
             flavor="igraph", n_iterations=2, directed=False)

count_area_filtered_adata.write(dir_custom_binning / "count_area_filtered_adata.h5ad")
print(f"{sample}: wrote {dir_custom_binning / 'count_area_filtered_adata.h5ad'}")

###############################################################
###################### Patch assignment #########################
###############################################################

dir_histology = basedir / sample / "01_preprocessing/unimodel/metadata"
patch_file = dir_histology / "filtered_WSI_patch_embeddings.h5ad"
if not patch_file.exists():
    print(f"{sample}: no filtered_WSI_patch_embeddings.h5ad yet -- skipping patch assignment.")
else:
    adata_patches = sc.read_h5ad(patch_file)
    bin_adata = read_h5ad_with_null_uns_healing(dir_custom_binning / "count_area_filtered_adata.h5ad")

    tissue_positions = pd.DataFrame(
        {"pxl_col_in_fullres": bin_adata.obsm["spatial"][:, 0],
         "pxl_row_in_fullres": bin_adata.obsm["spatial"][:, 1]},
        index=bin_adata.obs_names,
    )
    patch_ids = assign_bins_to_patches(tissue_positions, adata_patches.obs)
    bin_adata.obs["Patch_ID"] = patch_ids.reindex(bin_adata.obs_names)

    n_missing = bin_adata.obs["Patch_ID"].isna().sum()
    print(f"{sample}: {n_missing}/{bin_adata.n_obs} bins got no Patch_ID (outside any extracted patch)")

    stray_nones = sanitize_uns(bin_adata)
    if stray_nones:
        print(f"{sample}: sanitized stray null uns entries before saving: {stray_nones}")
    bin_adata.write(dir_custom_binning / "count_area_filtered_adata.h5ad")

print(f"{sample}: bin preprocessing done.")
