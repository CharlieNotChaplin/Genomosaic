"""
Shared helpers for merging per-sample AnnData/parquet files (custom-binned
transcriptomics, UNI patch embeddings, label-transfer predictions, ...) into
one combined file under data/combined/..., mirroring each sample's own
subfolder layout.

Consolidates what used to be two near-identical, notebook-local functions
(`merge_anndatas` in CustomBinning.ipynb, `merge_patch_embeddings` in
02_downstream_analysis_uni.ipynb) into one. The important behavior both
shared, and that this preserves: per-sample embeddings (PCA, UMAP, ...) are
NOT stacked into the merged AnnData as if they were one joint,
cross-sample-comparable embedding -- each was fit independently per slide,
so they're relabeled to f"{key}_per_sample_raw" instead. Call
sc.pp.pca(merged, ...) / sc.tl.umap(merged, ...) separately for a real joint
embedding.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc


def merge_anndatas(
    samples: list[str],
    path_template: str,
    filename: str,
    basedir,
    index_col: str | None = None,
    obs_name_mode: str = "prefix",
    per_sample_var_cols: list[str] | None = None,
    auto_detect_var_cols: bool = False,
    reference_only_var_cols: set[str] = frozenset(),
    per_sample_uns_moran_key: str | None = None,
    per_sample_pca_key: str | None = "X_pca",
    additional_per_sample_obsm_keys: tuple[str, ...] = (),
    **path_kwargs,
) -> ad.AnnData:
    """
    Merge per-sample AnnData files into one combined AnnData, tagging each
    row with Slide_ID, and writing the result alongside the per-sample
    files (same path_template, with sample="combined").

    Parameters
    ----------
    samples : list of str
        Sample directory names to merge, in the order they're concatenated.
    path_template : str
        Format string for each sample's file, filled in with `basedir`,
        `sample`, `filename` (itself formatted with `sample` first, so a
        per-sample filename like "{sample}_patch_predictions.h5ad" works),
        plus any extra **path_kwargs (e.g. bin_size=...). Example:
        "{basedir}/{sample}/01_preprocessing/unimodel/metadata/{filename}".
        The combined output is written to the same template with
        sample="combined".
    filename : str
        Filename, optionally containing "{sample}" (formatted per-sample
        before path_template is filled in).
    basedir : str or Path
        Root data directory.
    index_col : str, optional
        Only used when obs_name_mode="prefix". obs column used to build
        unique obs_names as "{value}_{sample}"; if not given, obs_names are
        built as "{sample}_{obs_name}" instead.
    obs_name_mode : str
        "prefix" (default): prefix/suffix obs_names with the sample name (see
        index_col) -- use when raw obs_names can collide across samples (e.g.
        bin barcodes). "unique": just call obs_names_make_unique() -- use
        when obs_names are already sample-scoped (e.g. patch filenames
        already unique enough that de-duplication alone is sufficient).
    per_sample_var_cols : list of str, optional
        var columns to preserve per-sample as varm[f"{col}_per_sample"]
        matrices (aligned to the merged feature order) instead of being
        collapsed by the concat.
    auto_detect_var_cols : bool
        If True and per_sample_var_cols is None, use the union of var
        columns across all samples (minus reference_only_var_cols).
    reference_only_var_cols : set of str
        var columns to exclude from auto-detection (e.g. columns that
        describe the feature itself, identical across samples).
    per_sample_uns_moran_key : str, optional
        uns key holding a per-sample Moran's I (SVG) result table, computed
        before merging. None (default) skips this.
    per_sample_pca_key : str, optional
        obsm key holding a per-sample PCA embedding (default "X_pca"). Its
        loadings (varm["PCs"]) and variance ratio (uns["pca"]["variance_ratio"]),
        if present, are captured per-sample too. Set to None to skip PCA
        handling entirely.
    additional_per_sample_obsm_keys : tuple of str
        Further obsm keys that were fit independently per sample (e.g.
        "X_umap") and should just be relabeled to f"{key}_per_sample_raw",
        without the loadings/variance-ratio bookkeeping PCA gets.

    Returns
    -------
    AnnData
    """
    adatas, sample_ids = [], []

    for sample in samples:
        fname = filename.format(sample=sample)
        file_path = Path(path_template.format(basedir=basedir, sample=sample, filename=fname, **path_kwargs))
        if not file_path.exists():
            print(f"Skipping {sample}: {file_path} not found.")
            continue

        adata = sc.read_h5ad(file_path)
        if "Slide_ID" not in adata.obs.columns:
            adata.obs["Slide_ID"] = sample

        if obs_name_mode == "prefix":
            if index_col is None:
                adata.obs_names = [f"{sample}_{idx}" for idx in adata.obs_names]
            else:
                if index_col not in adata.obs.columns:
                    raise ValueError(f"Column {index_col} not found in obs of {file_path}")
                adata.obs_names = [f"{val}_{sample}" for val in adata.obs[index_col]]
            adata.var_names_make_unique()
        elif obs_name_mode == "unique":
            adata.obs_names_make_unique()
        else:
            raise ValueError(f"obs_name_mode must be 'prefix' or 'unique', got {obs_name_mode!r}")

        adatas.append(adata)
        sample_ids.append(sample)

    if not adatas:
        raise ValueError(f"No files named {filename!r} found to merge.")

    # Auto-detect which var columns to preserve per-sample, if not explicitly given
    if per_sample_var_cols is None and auto_detect_var_cols:
        all_var_cols = set()
        for a in adatas:
            all_var_cols.update(a.var.columns)
        per_sample_var_cols = sorted(all_var_cols - set(reference_only_var_cols))
        print(f"Auto-detected per-sample var columns to preserve: {per_sample_var_cols}")
    elif per_sample_var_cols is None:
        per_sample_var_cols = []

    all_features = sorted(set().union(*[set(a.var_names) for a in adatas]))

    # Preserve per-sample var-level stats (HVG, means, dispersions, Moran's I summary cols, etc.)
    var_matrices = {}
    for col in per_sample_var_cols:
        mat = pd.DataFrame(np.nan, index=all_features, columns=sample_ids, dtype=float)
        for a, sid in zip(adatas, sample_ids):
            if col in a.var.columns:
                vals = a.var[col].values
                if vals.dtype == bool:
                    vals = vals.astype(float)  # avoid bool+NaN -> object dtype on write
                mat.loc[a.var_names, sid] = vals
            else:
                print(f"Warning: column '{col}' not found in var of sample {sid}")
        var_matrices[col] = mat

    # Preserve per-sample Moran's I (SVG) full result table, computed before merging
    moran_I_matrix = None
    moran_pval_matrix = None
    if per_sample_uns_moran_key is not None:
        has_moran = [per_sample_uns_moran_key in a.uns for a in adatas]
        if any(has_moran):
            moran_I_matrix = pd.DataFrame(np.nan, index=all_features, columns=sample_ids, dtype=float)
            moran_pval_matrix = pd.DataFrame(np.nan, index=all_features, columns=sample_ids, dtype=float)
            for a, sid in zip(adatas, sample_ids):
                if per_sample_uns_moran_key not in a.uns:
                    print(f"Warning: '{per_sample_uns_moran_key}' not found in uns of sample {sid}")
                    continue
                moran_res = a.uns[per_sample_uns_moran_key]
                moran_I_matrix.loc[moran_res.index, sid] = moran_res["I"].values
                pval_col = "pval_sim_fdr_bh" if "pval_sim_fdr_bh" in moran_res.columns else "pval_norm_fdr_bh"
                moran_pval_matrix.loc[moran_res.index, sid] = moran_res[pval_col].values

    # Preserve per-sample PCA (embedding + loadings + variance ratio), BEFORE concatenation
    pca_loadings_per_sample = {}
    pca_variance_ratio_per_sample = {}
    has_pca = per_sample_pca_key is not None and any(per_sample_pca_key in a.obsm for a in adatas)

    if has_pca:
        for a, sid in zip(adatas, sample_ids):
            if "PCs" in a.varm:
                pca_loadings_per_sample[sid] = pd.DataFrame(a.varm["PCs"], index=a.var_names)
            if "pca" in a.uns and "variance_ratio" in a.uns["pca"]:
                pca_variance_ratio_per_sample[sid] = np.asarray(a.uns["pca"]["variance_ratio"])

    # Concat actual data (X, obs, obsm['spatial'], obsm[per_sample_pca_key], etc.)
    # Note: obsp (e.g. spatial_connectivities) and uns are dropped by default here
    adata_merged = ad.concat(adatas, join="outer", merge="first")

    for col, mat in var_matrices.items():
        adata_merged.varm[f"{col}_per_sample"] = mat.reindex(adata_merged.var_names).values

    if moran_I_matrix is not None:
        adata_merged.varm["moranI_per_sample"] = moran_I_matrix.reindex(adata_merged.var_names).values
        adata_merged.varm["moranI_pval_per_sample"] = moran_pval_matrix.reindex(adata_merged.var_names).values

    adata_merged.uns["sample_order"] = sample_ids  # column order reference for all *_per_sample matrices

    # Relabel per-sample PCA so it can never be mistaken for a joint embedding
    if has_pca:
        if per_sample_pca_key in adata_merged.obsm:
            adata_merged.obsm[f"{per_sample_pca_key}_per_sample_raw"] = adata_merged.obsm.pop(per_sample_pca_key)

        if "PCs" in adata_merged.varm:
            del adata_merged.varm["PCs"]

        adata_merged.uns["pca_loadings_per_sample"] = {
            sid: df.reindex(adata_merged.var_names).values for sid, df in pca_loadings_per_sample.items()
        }
        adata_merged.uns["pca_variance_ratio_per_sample"] = pca_variance_ratio_per_sample

    # Relabel any further per-sample embeddings (e.g. UMAP) the same way, minus
    # the loadings/variance-ratio bookkeeping that's PCA-specific.
    for key in additional_per_sample_obsm_keys:
        if key in adata_merged.obsm:
            adata_merged.obsm[f"{key}_per_sample_raw"] = adata_merged.obsm.pop(key)

    if has_pca or additional_per_sample_obsm_keys:
        print(
            "Preserved per-sample embedding(s) under obsm[...'_per_sample_raw'] "
            "(filter by obs['Slide_ID'] to recover one sample's own coordinates)"
            + (", uns['pca_loadings_per_sample'], and uns['pca_variance_ratio_per_sample']" if has_pca else "")
            + ". Call sc.pp.pca(adata_merged, ...) / sc.tl.umap(adata_merged, ...) "
            "separately for a real joint, cross-sample-comparable embedding."
        )

    out_fname = filename.format(sample="combined")
    out_path = Path(path_template.format(basedir=basedir, sample="combined", filename=out_fname, **path_kwargs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_merged.write(out_path)
    print(f"Merged {len(adatas)}/{len(samples)} sample(s) -> {out_path}")

    return adata_merged


def merge_parquet(
    samples: list[str],
    path_template: str,
    filename: str,
    basedir,
    **path_kwargs,
) -> gpd.GeoDataFrame:
    """Merge per-sample GeoParquet files into one, tagging each row with
    Slide_ID. path_template/filename/basedir/**path_kwargs follow the same
    convention as merge_anndatas (combined output uses sample="combined")."""
    parts = []

    for sample in samples:
        fname = filename.format(sample=sample)
        file_path = Path(path_template.format(basedir=basedir, sample=sample, filename=fname, **path_kwargs))
        if not file_path.exists():
            print(f"Skipping {sample}: {file_path} not found.")
            continue
        df = gpd.read_parquet(file_path)
        if "Slide_ID" not in df.columns:
            df["Slide_ID"] = sample
        parts.append(df)

    if not parts:
        raise ValueError(f"No files named {filename!r} found to merge.")

    merged = pd.concat(parts, ignore_index=True)

    out_fname = filename.format(sample="combined")
    out_path = Path(path_template.format(basedir=basedir, sample="combined", filename=out_fname, **path_kwargs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Merged {len(parts)}/{len(samples)} sample(s) -> {out_path}")

    return merged
