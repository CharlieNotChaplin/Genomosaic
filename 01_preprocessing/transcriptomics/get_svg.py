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
import sys

from csbdeep.utils import normalize
from matplotlib.colors import ListedColormap
from scipy import sparse
from shapely.geometry import Polygon, Point
from stardist.models import StarDist2D
from tifffile import imread, imwrite

sys.path.insert(0, str(Path("../..").resolve()))
from functions.merge import read_h5ad_with_null_uns_healing, sanitize_uns


###############################################################
###################### Variables ##############################
###############################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True, help="Sample name")
args = parser.parse_args()


# Variables (names)
sample = args.sample
bin_size = "002um"
dir_cellseg = "cell_seg_stardist" # chosen cell segmentation method

print("Working on sample ", sample)

# Directories
basedir = Path("../../data")
dir_base_input = basedir / sample / "input" 
dir_base_output = basedir / sample / f"output/binned_outputs/square_{bin_size}"
dir_result_output = basedir / sample / "01_preprocessing" / dir_cellseg / bin_size
dir_custom_binning = dir_result_output / "custom_binning"
dir_result_output.mkdir(parents=True, exist_ok=True)
dir_custom_binning.mkdir(parents=True, exist_ok=True)


###############################################################
###################### Loading Files ##########################
###############################################################

# load files
nuclei_geometry = gpd.read_parquet(dir_custom_binning / "nuclei_geometry.parquet")
# read_h5ad_with_null_uns_healing instead of a plain sc.read_h5ad: this is the
# exact file type that broke merge_anndatas with an unreadable null-encoded
# uns entry (see functions/merge.py) -- auto-heals in place if this particular
# file predates the write-side sanitize_uns fix below.
count_area_filtered_adata = read_h5ad_with_null_uns_healing(dir_custom_binning / "count_area_filtered_adata.h5ad")
count_area_filtered_adata.var_names_make_unique()

# get centroids if they dont exist yet
if "centroid" not in nuclei_geometry.columns:
    nuclei_geometry["centroid"] = nuclei_geometry.geometry.centroid
    nuclei_geometry.to_parquet(dir_custom_binning / "nuclei_geometry.parquet")

if "interior_centroid" not in nuclei_geometry.columns:
    nuclei_geometry["interior_centroid"] = nuclei_geometry.geometry.representative_point()
    nuclei_geometry.to_parquet(dir_custom_binning / "nuclei_geometry.parquet")

# align geomtery nuclei ids with count_area_filtered_adata nuclei order
geometry_aligned = nuclei_geometry.set_index('id').reindex(count_area_filtered_adata.obs['id'])
assert geometry_aligned['centroid'].isnull().sum() == 0, "Some nucleus IDs didn't match!"
centroids = geometry_aligned['centroid']

# store centroid coords in obsm.spatial in adata
count_area_filtered_adata.obsm['spatial'] = np.column_stack([centroids.x.values, centroids.y.values])

# save RAM
del nuclei_geometry, geometry_aligned, centroids
gc.collect()


###############################################################
###################### Getting SVGs ###########################
###############################################################


# Sanity check for centroids
count_area_filtered_adata.obsm["spatial"] = np.asarray(count_area_filtered_adata.obsm["spatial"])

print("Building spatial neighborhood graph...")
# Build spatial neighborhood graph
sq.gr.spatial_neighbors(count_area_filtered_adata, 
                        coord_type="generic", 
                        # coord_type="grid", 
                        delaunay=True,)

# based on neighborhood graph: get spatial autocorr
print("Building spatial autocorr...")
sq.gr.spatial_autocorr(count_area_filtered_adata, 
                       mode="moran", 
                       genes=count_area_filtered_adata.var_names,
                       n_perms=1000,
                       n_jobs=20,) # adapt to computing capabilities of machine


# store moran I result into var, aligned by gene name
moran_res = count_area_filtered_adata.uns['moranI']
print("Got morans I")

count_area_filtered_adata.var['moranI'] = moran_res['I'].reindex(count_area_filtered_adata.var_names)
count_area_filtered_adata.var['moranI_pval_fdr'] = moran_res['pval_sim_fdr_bh'].reindex(count_area_filtered_adata.var_names)

# define spatially variable genes by threshold
count_area_filtered_adata.var['spatially_variable'] = (
    (count_area_filtered_adata.var['moranI_pval_fdr'] < 0.05) &
    (count_area_filtered_adata.var['moranI'] > 0.1)   # adjust threshold based on I distr.
)
print(count_area_filtered_adata.var['spatially_variable'].sum(), "spatially variable genes for person ", sample)

# sq.gr.spatial_neighbors (delaunay=True, no radius) leaves uns['spatial_neighbors']['params']
# with None values (e.g. 'radius', 'transform') that newer anndata writes as a "null"-encoded
# entry older anndata versions can't read back -- strip those before saving so this file stays
# readable regardless of which environment reads it later (see functions/merge.py:sanitize_uns).
stray_nones = sanitize_uns(count_area_filtered_adata)
if stray_nones:
    print("Sanitized stray null uns entries before saving:", stray_nones)

count_area_filtered_adata.write(dir_result_output / "custom_binning/count_area_filtered_adata_test_svg.h5ad")
print("unique var_names in count_area_filtered_adata: ", count_area_filtered_adata.var_names.is_unique)