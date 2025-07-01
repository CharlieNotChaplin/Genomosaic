# PROVIDED BY 10X GENOMICS — MODIFIED
# Segments nuclei using StarDist and maps them to Visium HD barcodes

import sys
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc
import polars as pl
import pickle
import imageio.v3 as iio
from pathlib import Path


from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap
import pyarrow

from csbdeep.io import save_tiff_imagej_compatible
from stardist import export_imagej_rois


def main(args):

    Path("output").mkdir(parents=True, exist_ok=True)

    # Load image (supports .png, .jpg, .tif)
    img = iio.imread(args.image)
    print("Image loaded.")

    # Crop and save if coordinates exist
    if args.rmax is not None:
        cropped_img = img[args.rmin:args.rmax, args.cmin:args.cmax]
        print(f"Image cropped to: {args.rmin}:{args.rmax}, {args.cmin}:{args.cmax}")
        save_tiff_imagej_compatible(args.out + '/img_Stardist.tif', cropped_img, axes='YXC')

    # Load pretrained StarDist model
    model = StarDist2D.from_pretrained('2D_versatile_he')
    print("Model loaded.")

    # Normalize image
    img = normalize(img, 5, 95)

    # Dynamically adjust block size
    # h, w = img.shape[:2]
    # block_size = (
    #     min(4096, h),
    #     min(4096, w),
    #     img.shape[2] if img.ndim == 3 else 1
    #)

    # outcomment the following if block size is adjusted dynamically (see above)
    block_size =4096
   
    # Run segmentation
    labels, polys = model.predict_instances_big(
        img,
        axes='YXC',
        block_size=block_size,
        prob_thresh=0.01,
        nms_thresh=0.001,
        min_overlap=128,
        context=128,
        normalizer=None,
        n_tiles=(4, 4, 1)
    )

    print(f"Detected objects: {np.max(labels)}")

    # Save raw segmentation results
    with open(args.out + '/labels_FullSection.pckl', 'wb') as f:
        pickle.dump(labels, f)
    with open(args.out + '/polys_FullSection.pckl', 'wb') as f:
        pickle.dump(polys, f)

    # If a crop was specified, export ImageJ ROIs for that region
    if args.rmax is not None:
        polys2 = polys
        index = []
        for nuclei in range(len(polys2['coord'])):
            if np.all(polys2['coord'][nuclei][0] > args.rmin) and np.all(polys2['coord'][nuclei][0] < args.rmax) and np.all(polys2['coord'][nuclei][1] > args.cmin) and np.all(polys2['coord'][nuclei][1] < args.cmax):
                polys2['coord'][nuclei][0] -= args.rmin
                polys2['coord'][nuclei][1] -= args.cmin
                index.append(nuclei)
        
        # Save masks for Subset
        export_imagej_rois(args.out + '/img_rois_Stardist_Subset.zip', polys2['coord'][index])

    # Convert polygons to GeoDataFrame
    geometries = []

    # Iterating through each nuclei in the 'polys' DataFrame 
    for nuclei in range(len(polys['coord'])):
        coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]
        geometries.append(Polygon(coords))

    # Creating a GeoDataFrame using the Polygon geometries 
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [f"ID_{i+1}" for i in gdf.index]

    # Load Visium HD spatial data
    raw_h5_file = args.srdir + "/filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(raw_h5_file)
    
    # Load spatial coordinates   
    tissue_position_file = args.srdir + "/spatial/tissue_positions.parquet"
  
    df_tissue_positions = pd.read_parquet(tissue_position_file).set_index("barcode")
    df_tissue_positions["index"] = df_tissue_positions.index
    
    # Adding the tissue positions to the Metadata
    adata.obs = pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)

    # Convert tissue coordinates to GeoDataFrame
    geometry = [
        Point(xy) for xy in zip(
            df_tissue_positions["pxl_col_in_fullres"],
            df_tissue_positions["pxl_row_in_fullres"],
        )
    ]
    gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry)

    # Spatial join to match barcodes to nuclei polygons
    result_spatial_join = gpd.sjoin(gdf_coordinates, gdf, how="left", predicate="within")
    result_spatial_join["is_within_polygon"] = ~result_spatial_join["index_right"].isna()
    barcodes_in_overlapping = pd.unique(result_spatial_join[result_spatial_join.duplicated(subset=["index"])]["index"])
    result_spatial_join["is_overlap"] = result_spatial_join["index"].isin(barcodes_in_overlapping)

    Result = result_spatial_join.loc[result_spatial_join['is_within_polygon']]

    # Save result as csv
    result_spatial_join.loc[result_spatial_join['is_within_polygon']][['index', 'id', 'is_within_polygon', 'is_overlap']].to_csv(
        args.out + "/Nuclei_Barcode_Map.csv"
    )

    print("Total barcodes:", len(gdf_coordinates))
    print("Total nuclei:", len(gdf))
    print("Matches found (within):", result_spatial_join['is_within_polygon'].sum())
    print("Result shape:", Result.shape)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Path to image (.png, .tif, etc.)')
    parser.add_argument('-r1', '--rmin', type=int, help='row min')
    parser.add_argument('-r2', '--rmax', type=int, help='row max')
    parser.add_argument('-c1', '--cmin', type=int, help='column min')
    parser.add_argument('-c2', '--cmax', type=int, help='column max')
    parser.add_argument('-s', '--srdir', type=str, help='Path to spaceranger output directory')
    parser.add_argument('-o', '--out', type=str, help='Output directory')
    args = parser.parse_args()
    main(args)
