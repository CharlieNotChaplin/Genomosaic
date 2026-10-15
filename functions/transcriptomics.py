"""
Shared helpers for the transcriptomics preprocessing (cell and bin path):
loading 8um grid bins, normalisation + clustering, and the global gene
selection recomputed after merging all samples.
"""

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_and_cluster(adata, resolution=0.2, n_top_genes=2000):
    # fixed target_sum (not the per-sample median) so every sample lands on the same scale
    # before log1p, otherwise differing depth becomes a per-sample batch effect
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    # igraph backend, default leidenalg is far too slow at this scale
    sc.tl.leiden(adata, resolution=resolution, key_added="clusters",
                 flavor="igraph", n_iterations=2, directed=False)


def native_grid_is_consistent(sample, data_dir, tol=(12, 20)):
    # native square_008um should have ~16x fewer in-tissue bins than square_002um
    # fails if square_002um was cropped to a smaller region (e.g. CRC_P11mm_2_cut)
    spatial = "output/binned_outputs/square_{}um/spatial/tissue_positions.parquet"
    n8 = (pd.read_parquet(data_dir / sample / spatial.format("008"), columns=["in_tissue"])["in_tissue"] == 1).sum()
    n2 = (pd.read_parquet(data_dir / sample / spatial.format("002"), columns=["in_tissue"])["in_tissue"] == 1).sum()
    ratio = n2 / n8 if n8 else float("nan")
    ok = tol[0] <= ratio <= tol[1]
    if not ok:
        print(f"{sample}: native square_008um covers a different region than square_002um "
              f"(n_002um/n_008um={ratio:.2f}, expected ~16) -- aggregating from square_002um instead")
    return ok


def load_grid_bins(sample, data_dir, group_size=4, check_region=False):
    """
    8um grid bins of one sample, with the same obs schema regardless of loader:
    native square_008um output if present, else exact aggregation from square_002um.
    check_region=True also falls back to square_002um if the native grid covers a
    different region than square_002um (see native_grid_is_consistent).
    """
    from functions.label_transfer import load_grid_bin_counts, aggregate_expression_to_grid

    grid_h5_path = data_dir / sample / "output/binned_outputs/square_008um/filtered_feature_bc_matrix.h5"
    use_native = grid_h5_path.exists() and (not check_region or native_grid_is_consistent(sample, data_dir))
    if use_native:
        print(f"{sample}: native square_008um output found")
        adata = load_grid_bin_counts(sample, data_dir=data_dir)
    else:
        print(f"{sample}: aggregating from square_002um")
        adata = aggregate_expression_to_grid(sample, group_size=group_size, data_dir=data_dir)

    # neither loader sets Slide_ID or an un-prefixed id
    adata.obs["Slide_ID"] = sample
    adata.obs["id"] = [f"grid_{r}_{c}" for r, c in zip(adata.obs["array_row"], adata.obs["array_col"])]
    adata.obs_names = adata.obs["id"].values
    adata.var_names_make_unique()
    return adata


def recompute_global_gene_selection(merged, i_thresh=0.1, fdr_thresh=0.05):
    """
    Global HVG/SVG selection and PCA on a merged (all samples) AnnData.
    Needs varm['highly_variable_per_sample'] (and varm['moranI_per_sample'] for SVGs,
    only present once get_svg.py has been run for every sample and the merge repeated).
    """
    # union of per-sample HVGs
    hvg_per_sample = np.nan_to_num(merged.varm["highly_variable_per_sample"], nan=0.0).astype(bool)
    merged.var["highly_variable"] = hvg_per_sample.any(axis=1)
    merged.var["highly_variable_n_samples"] = hvg_per_sample.sum(axis=1)

    # union of per-sample SVGs
    if "moranI_per_sample" in merged.varm:
        svg_per_sample = (merged.varm["moranI_pval_per_sample"] < fdr_thresh) & (merged.varm["moranI_per_sample"] > i_thresh)
        svg_per_sample = np.nan_to_num(svg_per_sample, nan=0.0).astype(bool)
        merged.var["spatially_variable"] = svg_per_sample.any(axis=1)
        merged.var["spatially_variable_n_samples"] = svg_per_sample.sum(axis=1)
    else:
        print("No moranI_per_sample yet -- run get_svg.py for every sample, then repeat the merge.")
        merged.var["spatially_variable"] = False
        merged.var["spatially_variable_n_samples"] = 0

    # batch-aware HVGs, PCA on those
    sc.pp.highly_variable_genes(merged, flavor="seurat", n_top_genes=2000, batch_key="Slide_ID")
    sc.pp.pca(merged, use_highly_variable=True)

    hvg = merged.var["highly_variable"]
    svg = merged.var["spatially_variable"]
    merged.var["gene_selection"] = "none"
    merged.var.loc[hvg, "gene_selection"] = "HVG"
    merged.var.loc[svg, "gene_selection"] = "SVG"
    merged.var.loc[hvg & svg, "gene_selection"] = "HVG_and_SVG"

    print(merged.var["gene_selection"].value_counts())
    print("TOTAL SVG union count:", svg.sum())
    return merged


def assign_cells_to_patches(patch_obs, nuclei_adata, slide_id, box_size=224):
    """
    Maps each nucleus to the histology patch it overlaps most (by area), for one slide.
    patch_obs: obs of filtered_WSI_patch_embeddings.h5ad (Slide_ID, Patch_ID, X, Y of top left corner)
    nuclei_adata: needs obs Slide_ID, id and geometry (nucleus polygon in pixels)
    returns {nucleus id: Patch_ID}
    """
    import geopandas as gpd
    from shapely.geometry import box

    # one polygon per patch
    slide_patches = patch_obs[patch_obs["Slide_ID"] == slide_id]
    if slide_patches.empty:
        print(f"No box data for {slide_id}")
        return {}
    patch_geometries = [
        box(patch["X"], patch["Y"], patch["X"] + box_size, patch["Y"] + box_size)
        for _, patch in slide_patches.iterrows()
    ]
    patches_gdf = gpd.GeoDataFrame(slide_patches[["Patch_ID"]], geometry=patch_geometries, crs=None)

    # nuclei of this slide as geodataframe
    nuclei_obs = nuclei_adata.obs[nuclei_adata.obs["Slide_ID"] == slide_id].reset_index(drop=True)
    nuclei_gdf = gpd.GeoDataFrame(nuclei_obs, geometry=nuclei_obs["geometry"], crs=None)
    # nucleus ids repeat across slides
    nuclei_gdf["composite_id"] = nuclei_gdf["Slide_ID"].astype(str) + "_" + nuclei_gdf["id"].astype(str)

    # overlap of every nucleus with every patch it touches
    overlaps = gpd.overlay(nuclei_gdf, patches_gdf, how="intersection")
    if overlaps.empty:
        print(f"Warning: no spatial overlaps found between nuclei and patches for {slide_id}.")
        return {}
    overlaps["intersection_area"] = overlaps.geometry.area
    overlaps["cell_area"] = overlaps.groupby("composite_id")["intersection_area"].transform("sum")
    overlaps["percent_covered"] = overlaps["intersection_area"] / overlaps["cell_area"]

    # keep the patch with the largest share per nucleus
    best_matches = overlaps.sort_values("percent_covered", ascending=False).drop_duplicates(subset="composite_id")
    best_matches["id"] = best_matches["id"].astype(str)
    n_non_unique = best_matches["id"].count() - best_matches["id"].nunique()
    if n_non_unique > 0:
        print(f"Warning: {n_non_unique} nuclei have duplicate matches after filtering for {slide_id}.")

    return dict(zip(best_matches["id"], best_matches["Patch_ID"]))


def add_patch_ids_to_nuclei(nuclei_adata, nuclei_geometry, patch_obs, sample, box_size=224):
    """
    Adds obs['Patch_ID'] to the (filtered) nuclei AnnData of one sample, geometry columns
    are merged in for the overlap and removed again so the result can be written to h5ad.
    nuclei_geometry: nuclei_geometry.parquet as geodataframe
    """
    nuclei_adata.obs = nuclei_adata.obs.merge(nuclei_geometry, on=["Slide_ID", "id"], how="left")
    if "Patch_ID" in nuclei_adata.obs.columns:
        print(f"Dropping stale Patch_ID column ({nuclei_adata.obs['Patch_ID'].notna().sum()} non-null values) before reassigning.")
        nuclei_adata.obs = nuclei_adata.obs.drop(columns=["Patch_ID"])

    mapping = assign_cells_to_patches(patch_obs, nuclei_adata, sample, box_size)
    nuclei_adata.obs["Patch_ID"] = nuclei_adata.obs["id"].astype(str).map(mapping)
    n_missing = nuclei_adata.obs["Patch_ID"].isna().sum()
    print(f"{sample}: {n_missing}/{nuclei_adata.n_obs} nuclei got no Patch_ID (no overlapping patch)")

    geom_cols = [c for c in ["geometry", "centroid", "interior_centroid"] if c in nuclei_adata.obs.columns]
    nuclei_adata.obs = nuclei_adata.obs.drop(columns=geom_cols)
    return nuclei_adata
