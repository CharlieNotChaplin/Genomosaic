"""
Throwaway script version of CustomBinning.ipynb's per-sample steps (load
StarDist nuclei polygons -> spatial-join to 2um bins -> sum counts per
nucleus -> QC -> clustering -> nucleus geometry/centroid parquet -> patch
assignment), so the per-cell pipeline can be driven per-sample from
run_cell_pipeline.sbatch instead of running the notebook by hand for every
sample. Mirrors bin_preprocessing.py's role for the bin-level path.

Assumes NucleiSegmentation.py has already produced polys_FullSection.pckl for
this sample -- segmentation itself is a separate, expensive step, not part of
this script (same assumption CustomBinning.ipynb itself makes).

Does NOT compute Moran's I / spatially-variable genes -- that's get_svg.py's
job (--unit cell, the default), run separately afterward so the two heavy
steps (clustering vs. permutation testing) can be resourced/retried
independently, same split as the bin path's bin_preprocessing.py + get_svg.py.
get_svg.py's own cell-path logic reloads nuclei_geometry.parquet and derives
obsm['spatial'] itself, so this script doesn't need to (unlike the notebook's
cell 35, which only existed to feed the notebook's own inline SVG cell).

Does NOT merge across samples (needs every sample's output to exist first) --
run CustomBinning.ipynb's merge cells (49-51) by hand once this script (and
get_svg.py) have both been run for every sample.

Usage:
    python cell_preprocessing.py --sample CRC_P1
"""
import argparse
import gc
import pickle
import sys
from pathlib import Path

import anndata as ad
import geopandas as gpd
import pandas as pd
import scanpy as sc
from scipy import sparse
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path("../..").resolve()))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample", required=True, help="Sample name, e.g. CRC_P1")
parser.add_argument("--nuclei-area-cutoff", type=float, default=1000,
                     help="same default as CustomBinning.ipynb -- drop nuclei with "
                          "area >= this (likely mis-segmented aggregates)")
parser.add_argument("--umi-cutoff", type=float, default=80,
                     help="same default as CustomBinning.ipynb / the bin path")
args = parser.parse_args()

sample = args.sample
bin_size = "002um"
dir_cellseg = "cell_seg_stardist"
patch_size = 224

basedir = Path("../../data")
dir_base_output = basedir / sample / f"output/binned_outputs/square_{bin_size}"
dir_result_output = basedir / sample / "01_preprocessing" / dir_cellseg / bin_size
dir_custom_binning = dir_result_output / "custom_binning"
dir_result_output.mkdir(parents=True, exist_ok=True)
dir_custom_binning.mkdir(parents=True, exist_ok=True)

###############################################################
###################### Load StarDist nuclei polygons ############
###############################################################

polys_path = dir_result_output / "polys_FullSection.pckl"
if not polys_path.exists():
    raise FileNotFoundError(
        f"{polys_path} not found -- run NucleiSegmentation.py for {sample} first "
        f"(cell segmentation is a separate step, not part of this script)."
    )
with open(polys_path, "rb") as f:
    polys = pickle.load(f)

geometries = [
    Polygon([(y, x) for x, y in zip(polys["coord"][n][0], polys["coord"][n][1])])
    for n in range(len(polys["coord"]))
]
gdf = gpd.GeoDataFrame(geometry=geometries)
gdf["id"] = [f"ID_{i + 1}" for i in range(len(gdf))]
del polys, geometries
gc.collect()
print(f"{sample}: {len(gdf)} StarDist nuclei polygons loaded")

###############################################################
###################### Load 2um bin counts #######################
###############################################################

adata = sc.read_10x_h5(dir_base_output / "filtered_feature_bc_matrix.h5")
tissue_positions = pd.read_parquet(
    dir_base_output / "spatial/tissue_positions.parquet",
    columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres", "in_tissue"],
).set_index("barcode")
adata.obs = adata.obs.join(tissue_positions, how="left")
adata.obs["pxl_row_in_fullres"] = adata.obs["pxl_row_in_fullres"].astype("float32")
adata.obs["pxl_col_in_fullres"] = adata.obs["pxl_col_in_fullres"].astype("float32")
adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()

gdf_coordinates = gpd.GeoDataFrame(
    adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]],
    geometry=gpd.points_from_xy(adata.obs["pxl_col_in_fullres"], adata.obs["pxl_row_in_fullres"]),
    crs=None,
)
del tissue_positions
gc.collect()
print(f"{sample}: {adata.n_obs} 2um bins loaded")

###############################################################
###################### Spatial join: bins -> nucleus #############
###############################################################

result_spatial_join = gpd.sjoin(gdf_coordinates, gdf, how="left", predicate="within")
result_spatial_join["is_within_polygon"] = ~result_spatial_join["index_right"].isna()
# Bins that landed in >1 overlapping nucleus polygon are ambiguous -- drop them
# (matches CustomBinning.ipynb's own tie-breaking: exclude, don't guess).
overlapping_barcodes = result_spatial_join.index[result_spatial_join.index.duplicated()].unique()
valid_mask = result_spatial_join["is_within_polygon"] & ~result_spatial_join.index.isin(overlapping_barcodes)
barcodes_in_one_polygon = result_spatial_join[valid_mask]

filtered_adata = adata[adata.obs_names.isin(barcodes_in_one_polygon.index)].copy()
filtered_adata.obs = filtered_adata.obs.join(barcodes_in_one_polygon[["id", "is_within_polygon"]], how="left")
filtered_adata.obs["Slide_ID"] = sample
filtered_adata.write(dir_custom_binning / "filtered_custom_bins.h5ad")
print(f"{sample}: {filtered_adata.n_obs}/{adata.n_obs} bins uniquely assigned to a nucleus")

del adata, gdf_coordinates, result_spatial_join, barcodes_in_one_polygon
gc.collect()

###############################################################
###################### Sum counts per nucleus #####################
###############################################################

groupby_object = filtered_adata.obs.groupby(["id"], observed=True)
counts = filtered_adata.X
n_groups = groupby_object.ngroups
n_genes = counts.shape[1]

summed_counts = sparse.lil_matrix((n_groups, n_genes))
polygon_id = []
row = 0
for polygon_key, idx_ in groupby_object.indices.items():
    summed_counts[row] = counts[idx_].sum(0)
    row += 1
    polygon_id.append(polygon_key)
summed_counts = summed_counts.tocsr()

obs_df = pd.DataFrame(polygon_id, columns=["id"], index=polygon_id)
obs_df = obs_df.merge(gdf[["id", "geometry"]], on="id", how="left")

grouped_filtered_adata = ad.AnnData(X=summed_counts, obs=obs_df, var=filtered_adata.var)
grouped_filtered_adata.obs["Slide_ID"] = sample

geodata = grouped_filtered_adata.obs["geometry"]
grouped_filtered_adata.obs = grouped_filtered_adata.obs.drop(columns=["geometry"])
grouped_filtered_adata.write(dir_custom_binning / "nuclei_grouped_UMI_count.h5ad")
grouped_filtered_adata.obs["geometry"] = geodata
print(f"{sample}: wrote nuclei_grouped_UMI_count.h5ad ({grouped_filtered_adata.n_obs} nuclei)")

del groupby_object, n_groups, polygon_id, summed_counts, obs_df, filtered_adata
gc.collect()

###############################################################
###################### QC (nucleus area + UMI count) ##############
###############################################################

gdf["area"] = gdf["geometry"].area
sc.pp.calculate_qc_metrics(grouped_filtered_adata, inplace=True)

n_total = grouped_filtered_adata.n_obs
mask_area = grouped_filtered_adata.obs["id"].isin(gdf[gdf["area"] < args.nuclei_area_cutoff].id)
mask_count = grouped_filtered_adata.obs["total_counts"] > args.umi_cutoff
count_area_filtered_adata = grouped_filtered_adata[mask_area & mask_count, :].copy()
count_area_filtered_adata.var_names_make_unique()
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)
print(f"{sample}: {count_area_filtered_adata.n_obs}/{n_total} nuclei kept "
      f"(area<{args.nuclei_area_cutoff} & total_counts>{args.umi_cutoff})")
del grouped_filtered_adata
gc.collect()

###############################################################
###################### Clustering ##################################
###############################################################

# fixed target_sum (not the per-sample median) so every sample lands on the same scale
# before log1p -- see functions/label_transfer.py's own normalize_total(target_sum=1e4)
sc.pp.normalize_total(count_area_filtered_adata, target_sum=1e4, inplace=True)
sc.pp.log1p(count_area_filtered_adata)
sc.pp.highly_variable_genes(count_area_filtered_adata, flavor="seurat", n_top_genes=2000)
sc.pp.pca(count_area_filtered_adata)
sc.pp.neighbors(count_area_filtered_adata)
sc.tl.leiden(count_area_filtered_adata, resolution=0.2, key_added="clusters",
             flavor="igraph", n_iterations=2, directed=False)  # default leidenalg backend is
             # orders of magnitude slower at this scale -- same convention already used in
             # BinPreprocessing.ipynb/bin_preprocessing.py, run_loso.py, new_single_omics.ipynb

geodata = count_area_filtered_adata.obs["geometry"]
count_area_filtered_adata.obs = count_area_filtered_adata.obs.drop(columns=["geometry"])
count_area_filtered_adata.write(dir_custom_binning / "count_area_filtered_adata.h5ad")
count_area_filtered_adata.obs["geometry"] = geodata
print(f"{sample}: wrote count_area_filtered_adata.h5ad ({count_area_filtered_adata.n_obs} nuclei, "
      f"{count_area_filtered_adata.obs['clusters'].nunique()} clusters)")

###############################################################
###################### Nucleus geometry/centroid parquet #########
###############################################################

nuclei_geometry = gpd.GeoDataFrame(
    count_area_filtered_adata.obs[["id", "geometry", "Slide_ID"]].copy(), geometry="geometry"
)
nuclei_geometry.index = count_area_filtered_adata.obs.index
nuclei_geometry.index.name = None
nuclei_geometry["centroid"] = nuclei_geometry.geometry.centroid
nuclei_geometry["interior_centroid"] = nuclei_geometry.geometry.representative_point()
nuclei_geometry.to_parquet(dir_custom_binning / "nuclei_geometry.parquet")
print(f"{sample}: wrote nuclei_geometry.parquet")

del gdf, nuclei_geometry
gc.collect()

###############################################################
###################### Patch assignment #############################
###############################################################

def assign_cells_to_patches(hist_data, trans_data, slide_id, box_size=224):
    """Maps each nucleus to the patch it overlaps most (by area), for one slide.
    Ported from CustomBinning.ipynb's assign_cells_to_patches, scoped to a
    single sample (called per-sample here rather than batched)."""
    core_data_hist = hist_data[hist_data["Slide_ID"] == slide_id]
    if core_data_hist.empty:
        print(f"No box data for {slide_id}")
        return {}

    patch_geometries = [
        box(patch["X"], patch["Y"], patch["X"] + box_size, patch["Y"] + box_size)
        for _, patch in core_data_hist.iterrows()
    ]
    patches_gdf = gpd.GeoDataFrame(core_data_hist[["Patch_ID"]], geometry=patch_geometries, crs=None)

    trans_data_subset = trans_data[trans_data.obs["Slide_ID"] == slide_id]
    transcr_gdf = gpd.GeoDataFrame(
        trans_data_subset.obs.reset_index(drop=True),
        geometry=trans_data_subset.obs.reset_index(drop=True)["geometry"],
        crs=None,
    )
    transcr_gdf["composite_id"] = transcr_gdf["Slide_ID"].astype(str) + "_" + transcr_gdf["id"].astype(str)

    overlaps = gpd.overlay(transcr_gdf, patches_gdf, how="intersection")
    if overlaps.empty:
        print(f"Warning: no spatial overlaps found between nuclei and patches for {slide_id}.")
        return {}

    overlaps["intersection_area"] = overlaps.geometry.area
    overlaps["cell_area"] = overlaps.groupby("composite_id")["intersection_area"].transform("sum")
    overlaps["percent_covered"] = overlaps["intersection_area"] / overlaps["cell_area"]
    best_matches = overlaps.sort_values("percent_covered", ascending=False).drop_duplicates(subset="composite_id")
    best_matches["id"] = best_matches["id"].astype(str)

    return dict(zip(best_matches["id"], best_matches["Patch_ID"]))


dir_histology = basedir / sample / "01_preprocessing/unimodel/metadata"
patch_file = dir_histology / "filtered_WSI_patch_embeddings.h5ad"
if not patch_file.exists():
    print(f"{sample}: no filtered_WSI_patch_embeddings.h5ad yet -- skipping patch assignment.")
else:
    adata_patches = sc.read_h5ad(patch_file)
    nuclei_geometry = gpd.read_parquet(dir_custom_binning / "nuclei_geometry.parquet")
    nuclei_adata = sc.read_h5ad(dir_custom_binning / "count_area_filtered_adata.h5ad")
    nuclei_adata.obs = nuclei_adata.obs.merge(nuclei_geometry, on=["Slide_ID", "id"], how="left")

    mapping = assign_cells_to_patches(adata_patches.obs, nuclei_adata, sample, patch_size)
    nuclei_adata.obs["Patch_ID"] = nuclei_adata.obs["id"].astype(str).map(mapping)

    n_missing = nuclei_adata.obs["Patch_ID"].isna().sum()
    print(f"{sample}: {n_missing}/{nuclei_adata.n_obs} nuclei got no Patch_ID (no overlapping patch)")

    geom_cols = [c for c in ["geometry", "centroid", "interior_centroid"] if c in nuclei_adata.obs.columns]
    nuclei_adata.obs = nuclei_adata.obs.drop(columns=geom_cols)
    nuclei_adata.write(dir_custom_binning / "count_area_filtered_adata.h5ad")
    print(f"{sample}: wrote {dir_custom_binning / 'count_area_filtered_adata.h5ad'} with Patch_ID")

print(f"{sample}: cell preprocessing done.")
