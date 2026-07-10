import anndata
import gc
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import scanpy as sc
import squidpy as sq

from csbdeep.utils import normalize
from matplotlib.colors import ListedColormap
from scipy import sparse
from shapely.geometry import Polygon, Point
from stardist.models import StarDist2D
from tifffile import imread, imwrite

# Variables (names)
sample = "CRC_P2"  # Adapt to requirements
bin_size = "002um"
dir_cellseg = "cell_seg_stardist" # chosen cell segmentation method
f_wsi = 'Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf'  # Adjust filename as needed

# Directories
basedir = Path("../../data")
dir_base_input = basedir / sample / "input" 
dir_base_output = basedir / sample / f"output/binned_outputs/square_{bin_size}"
dir_result_output = basedir / sample / "01_preprocessing" / dir_cellseg / bin_size
dir_custom_binning = dir_result_output / "custom_binning"
dir_result_output.mkdir(parents=True, exist_ok=True)
dir_custom_binning.mkdir(parents=True, exist_ok=True)


# test
print("Loading coordinates....")
geos = gpd.read_parquet(basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning/nuclei_geometry.parquet")
count_area_filtered_adata = sc.read_h5ad(basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning/count_area_filtered_adata.h5ad")

geos = geos[geos["Slide_ID"] == sample]
count_area_filtered_adata = count_area_filtered_adata[count_area_filtered_adata.obs["Slide_ID"] == sample]

geos_aligned = geos.set_index('id').reindex(count_area_filtered_adata.obs['id'])
assert geos_aligned['centroid'].isnull().sum() == 0, "Some nucleus IDs didn't match!"
centroids = geos_aligned['centroid']

del geos, geos_aligned
gc.collect()

count_area_filtered_adata.obsm['spatial'] = np.column_stack([centroids.x.values, centroids.y.values])
count_area_filtered_adata.write(dir_custom_binning / "count_area_filtered_adata.h5ad")

# Sanity check for centroids
count_area_filtered_adata.obsm["spatial"] = np.asarray(count_area_filtered_adata.obsm["spatial"])

print("Building spatial neighborhood graph...")
# Build spatial neighborhood graph
sq.gr.spatial_neighbors(count_area_filtered_adata, 
                        coord_type="generic", 
                        delaunay=True,)

print("Building spatial autocorr...")
sq.gr.spatial_autocorr(count_area_filtered_adata, 
mode="moran", 
genes=count_area_filtered_adata.var_names,
n_perms=1000,
n_jobs=20,)


# Store moran I result into var, aligned by gene name
moran_res = count_area_filtered_adata.uns['moranI']
print("Got morans I")

count_area_filtered_adata.var['moranI'] = moran_res['I'].reindex(count_area_filtered_adata.var_names)
count_area_filtered_adata.var['moranI_pval_fdr'] = moran_res['pval_norm_fdr_bh'].reindex(count_area_filtered_adata.var_names)

# Define spatially variable genes by threshold
count_area_filtered_adata.var['spatially_variable'] = (
    (count_area_filtered_adata.var['moranI_pval_fdr'] < 0.05) &
    (count_area_filtered_adata.var['moranI'] > 0.1)   # adjust threshold based on your I distribution
)
print(count_area_filtered_adata.var['spatially_variable'].sum(), "spatially variable genes for person ", sample)