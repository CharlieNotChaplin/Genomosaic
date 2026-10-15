"""
Per-sample transcriptomics preprocessing, either per cell (StarDist nuclei) or
per 8um grid bin. Script version of CustomBinning.ipynb (--unit cell) and
BinPreprocessing.ipynb (--unit bin); the notebooks stay the place for
interactive plots and diagnostics.

cell: load nuclei polygons -> spatial-join 2um bins to nuclei -> sum counts per
      nucleus -> QC (area + UMI) -> clustering -> nucleus geometry parquet ->
      patch assignment
      needs polys_FullSection.pckl from NucleiSegmentation.py
bin:  load 8um bins -> patch assignment (bins without patch are dropped) ->
      QC (UMI) -> clustering
      patch assignment comes first so tissue that was never imaged does not
      distort the clusters

Does NOT compute spatially variable genes (get_svg.py --unit ...) and does NOT
merge samples (merge cells in the notebooks, needs every sample first).

Usage:
    python preprocess.py --sample CRC_P1 --unit cell
    python preprocess.py --sample CRC_P1 --unit bin
"""
import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

# repo root = first parent containing functions/ and data/
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "functions").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from functions.merge import sanitize_uns
from functions.transcriptomics import normalize_and_cluster, load_grid_bins, add_patch_ids_to_nuclei

BASEDIR = REPO_ROOT / "data"
PATCH_SIZE = 224
UNITS = {
    "cell": {"bin_size": "002um", "dir_binning": "cell_seg_stardist"},
    "bin": {"bin_size": "008um", "dir_binning": "grid_bin"},
}


def write_count_area(adata, path):
    # geometry column cannot be written to h5ad
    stray_nones = sanitize_uns(adata)
    if stray_nones:
        print(f"sanitized stray null uns entries before saving: {stray_nones}")
    geodata = adata.obs["geometry"] if "geometry" in adata.obs else None
    if geodata is not None:
        adata.obs = adata.obs.drop(columns=["geometry"])
    adata.write(path)
    if geodata is not None:
        adata.obs["geometry"] = geodata


def load_patch_obs(sample):
    # None if the histology embeddings of this sample are not filtered yet
    patch_file = BASEDIR / sample / "01_preprocessing/unimodel/metadata/filtered_WSI_patch_embeddings.h5ad"
    if not patch_file.exists():
        print(f"{sample}: no filtered_WSI_patch_embeddings.h5ad yet -- skipping patch assignment.")
        return None
    return sc.read_h5ad(patch_file).obs


###############################################################
###################### Cell path ###############################
###############################################################

def load_and_group_nuclei(sample, dir_result_output, dir_custom_binning, dir_base_output):
    """Nuclei polygons + 2um bins -> raw counts summed per nucleus.
    Writes filtered_custom_bins.h5ad and nuclei_grouped_UMI_count.h5ad."""
    import pickle

    import anndata as ad
    import geopandas as gpd
    from scipy import sparse
    from shapely.geometry import Polygon

    # StarDist polygons
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

    # 2um bins with pixel positions
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

    # spatial join bin -> nucleus
    result_spatial_join = gpd.sjoin(gdf_coordinates, gdf, how="left", predicate="within")
    result_spatial_join["is_within_polygon"] = ~result_spatial_join["index_right"].isna()
    # bins in >1 overlapping nucleus are ambiguous, dropped
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

    # sum counts per nucleus
    groupby_object = filtered_adata.obs.groupby(["id"], observed=True)
    counts = filtered_adata.X
    summed_counts = sparse.lil_matrix((groupby_object.ngroups, counts.shape[1]))
    polygon_id = []
    for row, (polygon_key, idx_) in enumerate(groupby_object.indices.items()):
        summed_counts[row] = counts[idx_].sum(0)
        polygon_id.append(polygon_key)
    summed_counts = summed_counts.tocsr()

    obs_df = pd.DataFrame(polygon_id, columns=["id"], index=polygon_id)
    obs_df = obs_df.merge(gdf[["id", "geometry"]], on="id", how="left")

    grouped = ad.AnnData(X=summed_counts, obs=obs_df, var=filtered_adata.var)
    grouped.obs["Slide_ID"] = sample
    write_count_area(grouped, dir_custom_binning / "nuclei_grouped_UMI_count.h5ad")
    print(f"{sample}: wrote nuclei_grouped_UMI_count.h5ad ({grouped.n_obs} nuclei)")

    del groupby_object, summed_counts, polygon_id, obs_df, filtered_adata
    gc.collect()
    return grouped, gdf


def run_cell(sample, dirs, args):
    import geopandas as gpd

    dir_custom_binning = dirs["custom_binning"]
    grouped, gdf = load_and_group_nuclei(sample, dirs["result"], dir_custom_binning, dirs["base_output"])

    # QC: nucleus area + UMI count
    gdf["area"] = gdf["geometry"].area
    sc.pp.calculate_qc_metrics(grouped, inplace=True)
    n_total = grouped.n_obs
    mask_area = grouped.obs["id"].isin(gdf[gdf["area"] < args.nuclei_area_cutoff].id)
    mask_count = grouped.obs["total_counts"] > args.umi_cutoff
    adata = grouped[mask_area & mask_count, :].copy()
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    print(f"{sample}: {adata.n_obs}/{n_total} nuclei kept "
          f"(area<{args.nuclei_area_cutoff} & total_counts>{args.umi_cutoff})")
    del grouped
    gc.collect()

    normalize_and_cluster(adata)
    path_adata = dir_custom_binning / "count_area_filtered_adata.h5ad"
    write_count_area(adata, path_adata)
    print(f"{sample}: wrote count_area_filtered_adata.h5ad ({adata.n_obs} nuclei, "
          f"{adata.obs['clusters'].nunique()} clusters)")

    # geometry + centroids for get_svg.py
    nuclei_geometry = gpd.GeoDataFrame(adata.obs[["id", "geometry", "Slide_ID"]].copy(), geometry="geometry")
    nuclei_geometry.index = adata.obs.index
    nuclei_geometry.index.name = None
    nuclei_geometry["centroid"] = nuclei_geometry.geometry.centroid
    nuclei_geometry["interior_centroid"] = nuclei_geometry.geometry.representative_point()
    nuclei_geometry.to_parquet(dir_custom_binning / "nuclei_geometry.parquet")
    print(f"{sample}: wrote nuclei_geometry.parquet")
    del gdf
    gc.collect()

    # patch assignment, written back into the same file
    patch_obs = load_patch_obs(sample)
    if patch_obs is None:
        return
    nuclei_adata = add_patch_ids_to_nuclei(sc.read_h5ad(path_adata), nuclei_geometry, patch_obs, sample, PATCH_SIZE)
    nuclei_adata.write(path_adata)
    print(f"{sample}: wrote {path_adata} with Patch_ID")


###############################################################
###################### Bin path ################################
###############################################################

def run_bin(sample, dirs, args):
    from functions.label_transfer import assign_bins_to_patches

    adata = load_grid_bins(sample, BASEDIR, group_size=args.group_size)
    print(f"{sample}: {adata.n_obs} bins, {adata.n_vars} genes")

    # patch assignment before QC/clustering, bins without patch are dropped
    patch_obs = load_patch_obs(sample)
    if patch_obs is not None:
        tissue_positions = pd.DataFrame(
            {"pxl_col_in_fullres": adata.obsm["spatial"][:, 0],
             "pxl_row_in_fullres": adata.obsm["spatial"][:, 1]},
            index=adata.obs_names,
        )
        adata.obs["Patch_ID"] = assign_bins_to_patches(tissue_positions, patch_obs).reindex(adata.obs_names)
        n_missing = adata.obs["Patch_ID"].isna().sum()
        print(f"{sample}: {n_missing}/{adata.n_obs} bins got no Patch_ID (outside any extracted patch) -- dropping them")
        adata = adata[adata.obs["Patch_ID"].notna()].copy()

    # QC: UMI count only (all bins have the same area)
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    mask_count = adata.obs["total_counts"] > args.umi_cutoff
    filtered = adata[mask_count].copy()
    filtered.var_names_make_unique()
    sc.pp.calculate_qc_metrics(filtered, inplace=True)
    print(f"{mask_count.sum()}/{adata.n_obs} bins kept (total_counts > {args.umi_cutoff})")
    del adata
    gc.collect()

    normalize_and_cluster(filtered)
    path_adata = dirs["custom_binning"] / "count_area_filtered_adata.h5ad"
    write_count_area(filtered, path_adata)
    print(f"{sample}: wrote {path_adata}")


###############################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True, help="Sample name, e.g. CRC_P1")
    parser.add_argument("--unit", choices=list(UNITS), required=True,
                        help="cell (StarDist nuclei, 2um bins) or bin (8um grid bins)")
    parser.add_argument("--umi-cutoff", type=float, default=80,
                        help="keep cells/bins with total_counts > this")
    parser.add_argument("--nuclei-area-cutoff", type=float, default=1000,
                        help="cell only: drop nuclei with area >= this (likely mis-segmented aggregates)")
    parser.add_argument("--group-size", type=int, default=4,
                        help="bin only: 2um bins per 8um bin side, only used if no native square_008um output exists")
    args = parser.parse_args()

    cfg = UNITS[args.unit]
    sample = args.sample
    dirs = {
        "base_output": BASEDIR / sample / f"output/binned_outputs/square_{cfg['bin_size']}",
        "result": BASEDIR / sample / "01_preprocessing" / cfg["dir_binning"] / cfg["bin_size"],
    }
    dirs["custom_binning"] = dirs["result"] / "custom_binning"
    dirs["custom_binning"].mkdir(parents=True, exist_ok=True)

    print(f"Working on sample {sample} (unit={args.unit})")
    if args.unit == "cell":
        run_cell(sample, dirs, args)
    else:
        run_bin(sample, dirs, args)
    print(f"{sample}: {args.unit} preprocessing done.")


if __name__ == "__main__":
    main()
