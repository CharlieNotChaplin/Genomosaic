"""
Throwaway script: regenerates ONLY filtered_custom_bins.h5ad and
nuclei_grouped_UMI_count.h5ad (accidentally deleted) -- the load -> spatial-join
-> sum-counts-per-nucleus prefix of cell_preprocessing.py, truncated right after
the second file is written. Both are deterministic functions of the StarDist
pickle + raw 2um SpaceRanger counts alone (no QC/clustering/patch-assignment
input), so this reproduces them exactly without needing to redo anything after.

Usage:
    python regenerate_grouping_files.py --sample CRC_P1
"""
import argparse
import gc
import pickle
from pathlib import Path

import anndata as ad
import geopandas as gpd
import pandas as pd
import scanpy as sc
from scipy import sparse
from shapely.geometry import Polygon

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample", required=True, help="Sample name, e.g. CRC_P1")
args = parser.parse_args()

sample = args.sample
bin_size = "002um"
dir_cellseg = "cell_seg_stardist"

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
    raise FileNotFoundError(f"{polys_path} not found -- run NucleiSegmentation.py for {sample} first.")
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

print(f"{sample}: done (filtered_custom_bins.h5ad + nuclei_grouped_UMI_count.h5ad regenerated).")
