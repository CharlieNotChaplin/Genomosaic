"""
Throwaway script version of BinPreprocessing.ipynb's per-sample steps (load ->
patch-assignment -> QC -> clustering), so the bin pipeline can be driven
per-sample from run_bin_pipeline.sbatch instead of running the notebook by
hand for every sample. The notebook stays the source of truth for anything
interactive (UMI-cutoff diagnostic plots, ad-hoc inspection) -- this script
just applies umi_cutoff directly, no plotting.

Patch assignment runs before QC/clustering (not after): a bin with no
histology counterpart -- e.g. tissue Visium detected on-slide but never
scanned into this sample's WSI, such as a slide holding several tissue pieces
where only one was imaged (CRC_P11mm_2_cut) -- would otherwise get clustered
together with completely unrelated tissue, skewing this sample's own Leiden
clusters. Dropping those bins first means QC/clustering only ever sees the
imaged region.

Does NOT merge across samples (needs every sample's output to exist first) --
run BinPreprocessing.ipynb's merge cells by hand once this script (and
get_svg.py --unit bin) have both been run for every sample.

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
from functions.merge import sanitize_uns

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
###################### Patch assignment #########################
###############################################################

dir_histology = basedir / sample / "01_preprocessing/unimodel/metadata"
patch_file = dir_histology / "filtered_WSI_patch_embeddings.h5ad"
if not patch_file.exists():
    print(f"{sample}: no filtered_WSI_patch_embeddings.h5ad yet -- skipping patch assignment (all bins kept).")
else:
    adata_patches = sc.read_h5ad(patch_file)
    tissue_positions = pd.DataFrame(
        {"pxl_col_in_fullres": adata.obsm["spatial"][:, 0],
         "pxl_row_in_fullres": adata.obsm["spatial"][:, 1]},
        index=adata.obs_names,
    )
    patch_ids = assign_bins_to_patches(tissue_positions, adata_patches.obs)
    adata.obs["Patch_ID"] = patch_ids.reindex(adata.obs_names)

    n_missing = adata.obs["Patch_ID"].isna().sum()
    print(f"{sample}: {n_missing}/{adata.n_obs} bins got no Patch_ID (outside any extracted patch) -- dropping them")
    # A bin with no Patch_ID can never be paired for the histology-transcriptomics
    # objective this bin-level output exists for -- drop it here, before QC/clustering,
    # so it can't dilute either.
    adata = adata[adata.obs["Patch_ID"].notna()].copy()

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

# fixed target_sum (not the per-sample median) so every sample lands on the same scale
# before log1p -- see functions/label_transfer.py's own normalize_total(target_sum=1e4)
sc.pp.normalize_total(count_area_filtered_adata, target_sum=1e4, inplace=True)
sc.pp.log1p(count_area_filtered_adata)
sc.pp.highly_variable_genes(count_area_filtered_adata, flavor="seurat", n_top_genes=2000)
sc.pp.pca(count_area_filtered_adata)
sc.pp.neighbors(count_area_filtered_adata)
# flavor="igraph": default leidenalg backend is impractically slow at bin scale
# (systematically more nodes than the cell path) -- same convention already
# used elsewhere in this codebase.
sc.tl.leiden(count_area_filtered_adata, resolution=0.2, key_added="clusters",
             flavor="igraph", n_iterations=2, directed=False)

stray_nones = sanitize_uns(count_area_filtered_adata)
if stray_nones:
    print(f"{sample}: sanitized stray null uns entries before saving: {stray_nones}")
count_area_filtered_adata.write(dir_custom_binning / "count_area_filtered_adata.h5ad")
print(f"{sample}: wrote {dir_custom_binning / 'count_area_filtered_adata.h5ad'}")

print(f"{sample}: bin preprocessing done.")
