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


def sanitize_uns(adata: ad.AnnData) -> list[str]:
    """
    WRITE-SIDE fix: recursively strip Python `None` values out of `adata.uns`
    (in place), so the file `.write()`s cleanly readable by any anndata
    version. Newer anndata versions serialize a `None` using a "null"
    IOSpec encoding that OLDER anndata versions have no reader registered
    for -- e.g. `sc.pp.log1p(adata)` leaves `uns["log1p"]["base"] = None`
    (natural log marker), and `sq.gr.spatial_neighbors(..., delaunay=True)`
    leaves `uns["spatial_neighbors"]["params"]["radius"] = None` (unused
    since delaunay doesn't take a radius) -- both inert bookkeeping, not
    data anything actually reads back out. Call this right before `.write()`
    in whatever script produces a file (e.g. `get_svg.py`), rather than
    patching files after the fact -- see `_heal_null_uns_entries` for the
    read-side fallback this is paired with, for files that predate this.

    Returns the list of dotted-path keys removed (e.g.
    ["log1p.base", "spatial_neighbors.params.radius"]), so callers can log
    what happened. Only descends into nested dicts (uns commonly holds
    DataFrames/arrays too, which are left untouched regardless of content).
    """
    removed = []

    def _strip(d, prefix):
        for key in list(d.keys()):
            value = d[key]
            path = f"{prefix}.{key}" if prefix else key
            if value is None:
                removed.append(path)
                del d[key]
            elif isinstance(value, dict):
                _strip(value, path)

    _strip(adata.uns, "")
    return removed


def _heal_null_uns_entries(file_path: Path) -> list[str]:
    """
    READ-SIDE fallback, paired with `sanitize_uns`: for a file written
    before that sanitizer existed (or by some other script it wasn't added
    to), patches AWAY the same class of stray null-encoded `uns` entries
    directly in the HDF5 file on disk, in place, using plain h5py (stable
    across anndata versions, unlike anndata's own internal read machinery --
    deliberately not using anndata here, since the whole point is this file
    is currently unreadable BY anndata).

    Scoped to `uns` only, and only entries anndata itself marked with
    `encoding-type: "null"` -- never touches X/obs/var/obsm, and never
    guesses at what "looks like" a problem. `merge_anndatas` doesn't use
    anything under `uns` except the one key `per_sample_uns_moran_key`
    names (see module docstring: uns is otherwise dropped by the merge
    anyway), so removing inert None entries elsewhere in `uns` can't affect
    any data this pipeline actually consumes.

    Returns the list of HDF5 paths removed (empty if nothing needed fixing,
    e.g. if the read failure turns out to have a different cause -- callers
    should treat an empty return as "don't retry, re-raise the original error").
    """
    import h5py

    removed = []
    with h5py.File(file_path, "r+") as f:
        if "uns" not in f:
            return removed

        def _strip(group):
            for key in list(group.keys()):
                item = group[key]
                if isinstance(item, h5py.Group):
                    _strip(item)
                elif item.attrs.get("encoding-type") == "null":
                    removed.append(item.name)
                    del group[key]

        _strip(f["uns"])

    return removed


try:
    from anndata._io.specs.registry import IORegistryError as _AnndataIORegistryError
except ImportError:
    # private path -- if it ever moves, degrade to catching any read failure
    # rather than hard-erroring on import (the "encoding_type='null'" message
    # check below still keeps this narrowly scoped to the null-encoding case).
    _AnndataIORegistryError = Exception


def read_h5ad_with_null_uns_healing(file_path: Path):
    """
    `sc.read_h5ad(file_path)`, but on the specific "can't read a null-encoded
    uns entry" failure (see `_heal_null_uns_entries`'s docstring), patches the
    file in place and retries once, instead of crashing over a piece of `uns`
    that's typically not even read (`merge_anndatas` only ever reads one
    named uns key). Prints what it did, since silently rewriting a file on
    disk as a side effect of a *read* call is the kind of thing that should
    be visible, not a surprise. Any other failure (or a retry that still
    fails) propagates normally.

    Public (not `_`-prefixed): this fixes the exact same failure for ANY
    script reading these files, not just `merge_anndatas` -- e.g. `get_svg.py`
    reads `count_area_filtered_adata.h5ad` directly and would hit the
    identical crash on an unsanitized file, so it uses this too.
    """
    try:
        return sc.read_h5ad(file_path)
    except _AnndataIORegistryError as e:
        if "encoding_type='null'" not in str(e):
            raise  # not the specific issue this fallback handles -- don't mask other bugs

        print(f"  {file_path.name}: hit a stray null-encoded uns entry anndata can't read "
              f"-- attempting to auto-heal in place...")
        removed = _heal_null_uns_entries(file_path)
        if not removed:
            raise  # nothing to fix -- the read failure must have another cause
        print(f"  {file_path.name}: removed {removed}, retrying read...")
        return sc.read_h5ad(file_path)


def _read_and_prepare_samples(
    samples: list[str],
    path_template: str,
    filename: str,
    basedir,
    index_col: str | None,
    obs_name_mode: str,
    path_kwargs: dict,
) -> tuple[list[ad.AnnData], list[str]]:
    """Read each sample's raw per-sample file and relabel obs_names/Slide_ID.
    Split out of `merge_anndatas` so `merge_mode="incremental"` can call this
    on just the newly-requested samples instead of all of them."""
    adatas, sample_ids = [], []

    for sample in samples:
        fname = filename.format(sample=sample)
        file_path = Path(path_template.format(basedir=basedir, sample=sample, filename=fname, **path_kwargs))
        if not file_path.exists():
            print(f"Skipping {sample}: {file_path} not found.")
            continue

        adata = read_h5ad_with_null_uns_healing(file_path)
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

    return adatas, sample_ids


def _build_merged_from_adatas(
    adatas: list[ad.AnnData],
    sample_ids: list[str],
    per_sample_var_cols: list[str] | None,
    auto_detect_var_cols: bool,
    reference_only_var_cols: set[str],
    per_sample_uns_moran_key: str | None,
    per_sample_pca_key: str | None,
    additional_per_sample_obsm_keys: tuple[str, ...],
) -> ad.AnnData:
    """Everything `merge_anndatas` used to do between reading files and
    writing the result to disk: build one merged AnnData (with per-sample
    var stats / Moran's I / PCA preserved, never collapsed by the concat)
    from an already-loaded, already-relabeled list of per-sample AnnDatas.
    Split out so `merge_mode="incremental"` can call this on just the new
    samples, then hand the result to `_extend_combined` to fold into the
    existing combined file, instead of rebuilding everything from scratch.
    Callers are expected to have already checked `adatas` is non-empty."""
    if per_sample_var_cols is None and auto_detect_var_cols:
        all_var_cols = set()
        for a in adatas:
            all_var_cols.update(a.var.columns)
        per_sample_var_cols = sorted(all_var_cols - set(reference_only_var_cols))
        print(f"Auto-detected per-sample var columns to preserve: {per_sample_var_cols}")
    elif per_sample_var_cols is None:
        per_sample_var_cols = []

    all_features = sorted(set().union(*[set(a.var_names) for a in adatas]))

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

    moran_I_matrix = None
    moran_pval_matrix = None
    if per_sample_uns_moran_key is not None:
        if any(per_sample_uns_moran_key in a.uns for a in adatas):
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

    pca_loadings_per_sample = {}
    pca_variance_ratio_per_sample = {}
    has_pca = per_sample_pca_key is not None and any(per_sample_pca_key in a.obsm for a in adatas)

    if has_pca:
        for a, sid in zip(adatas, sample_ids):
            if "PCs" in a.varm:
                pca_loadings_per_sample[sid] = pd.DataFrame(a.varm["PCs"], index=a.var_names)
            if "pca" in a.uns and "variance_ratio" in a.uns["pca"]:
                pca_variance_ratio_per_sample[sid] = np.asarray(a.uns["pca"]["variance_ratio"])

    # Note: obsp (e.g. spatial_connectivities) and uns are dropped by default here
    adata_merged = ad.concat(adatas, join="outer", merge="first")

    for col, mat in var_matrices.items():
        adata_merged.varm[f"{col}_per_sample"] = mat.reindex(adata_merged.var_names).values

    if moran_I_matrix is not None:
        adata_merged.varm["moranI_per_sample"] = moran_I_matrix.reindex(adata_merged.var_names).values
        adata_merged.varm["moranI_pval_per_sample"] = moran_pval_matrix.reindex(adata_merged.var_names).values

    adata_merged.uns["sample_order"] = sample_ids  # column order reference for all *_per_sample matrices

    if has_pca:
        if per_sample_pca_key in adata_merged.obsm:
            adata_merged.obsm[f"{per_sample_pca_key}_per_sample_raw"] = adata_merged.obsm.pop(per_sample_pca_key)

        if "PCs" in adata_merged.varm:
            del adata_merged.varm["PCs"]

        adata_merged.uns["pca_loadings_per_sample"] = {
            sid: df.reindex(adata_merged.var_names).values for sid, df in pca_loadings_per_sample.items()
        }
        adata_merged.uns["pca_variance_ratio_per_sample"] = pca_variance_ratio_per_sample

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

    return adata_merged


def _reindex_to_features(arr: np.ndarray, src_var_names, target_var_names, columns=None) -> pd.DataFrame:
    """Reindex a var_names-indexed array (rows = features) from `src_var_names`
    to `target_var_names` (as a DataFrame -- callers that just want the raw
    array back can call `.values` on the result), filling any feature
    `target` has that `src` didn't with NaN. Shared by `_extend_combined`'s
    two per-sample reconciliation passes (the `*_per_sample` varm matrices,
    and the `pca_loadings_per_sample` per-sample arrays) -- same operation,
    just applied to a multi-sample matrix vs. one array at a time."""
    return pd.DataFrame(arr, index=src_var_names, columns=columns).reindex(target_var_names)


def _extend_combined(existing: ad.AnnData, addition: ad.AnnData) -> ad.AnnData:
    """
    Fold a newly-merged AnnData (covering samples not yet in `existing`,
    built by `_build_merged_from_adatas` the exact same way `existing` was)
    into `existing`, WITHOUT losing or overwriting `existing`'s per-sample
    bookkeeping -- a plain `ad.concat([existing, addition])` alone would
    silently let `addition`'s varm/uns entries clobber `existing`'s (both use
    the same key names, e.g. varm["moranI_per_sample"]), discarding whichever
    one came second.

    Every f"..._per_sample" varm matrix and the `pca_loadings_per_sample` uns
    dict get reindexed to the (possibly now-larger, if `addition` introduced
    genes `existing` didn't have) concatenated feature set and column-unioned
    by sample, so a sample merged in an earlier call keeps its stats exactly
    as they were, and the new sample's stats sit alongside them --
    `uns["sample_order"]` is extended the same way. `pca_variance_ratio_per_sample`
    is per-sample scalars/arrays (not feature-indexed), so it's a plain dict union.
    """
    combined = ad.concat([existing, addition], join="outer", merge="first")

    combined.uns["sample_order"] = list(existing.uns.get("sample_order", [])) + \
        list(addition.uns.get("sample_order", []))

    per_sample_varm_keys = {k for k in existing.varm.keys() if k.endswith("_per_sample")} | \
                            {k for k in addition.varm.keys() if k.endswith("_per_sample")}
    for key in per_sample_varm_keys:
        pieces = []
        for src, sids in ((existing, existing.uns.get("sample_order", [])),
                           (addition, addition.uns.get("sample_order", []))):
            if key in src.varm:
                pieces.append(_reindex_to_features(src.varm[key], src.var_names, combined.var_names, columns=sids))
            else:
                pieces.append(pd.DataFrame(np.nan, index=combined.var_names, columns=sids))
        combined.varm[key] = pd.concat(pieces, axis=1).values

    if "pca_loadings_per_sample" in existing.uns or "pca_loadings_per_sample" in addition.uns:
        merged_loadings = {}
        for src in (existing, addition):
            for sid, arr in src.uns.get("pca_loadings_per_sample", {}).items():
                merged_loadings[sid] = _reindex_to_features(arr, src.var_names, combined.var_names).values
        combined.uns["pca_loadings_per_sample"] = merged_loadings

    if "pca_variance_ratio_per_sample" in existing.uns or "pca_variance_ratio_per_sample" in addition.uns:
        merged_variance = dict(existing.uns.get("pca_variance_ratio_per_sample", {}))
        merged_variance.update(addition.uns.get("pca_variance_ratio_per_sample", {}))
        combined.uns["pca_variance_ratio_per_sample"] = merged_variance

    return combined


def _resolve_incremental(out_path: Path, samples: list[str], read_existing, get_slide_ids):
    """
    Shared merge_mode="incremental" resolution logic for `merge_anndatas` and
    `merge_parquet`: does the existing combined output exist, which of
    `samples` aren't in it yet, and is there anything new to do at all.
    `read_existing`/`get_slide_ids` abstract over the one real difference
    between the two formats (how to load the file, how to pull Slide_ID
    values out of it) so this isn't duplicated between them.

    Returns (existing, samples_to_read, existing_slides):
      - out_path doesn't exist yet: (None, samples, None) -- caller falls
        back to a full merge over everything.
      - it exists but every requested sample is already in it:
        (existing, [], existing_slides) -- caller should treat this as a
        no-op and hand `existing` straight back.
      - it exists and some samples are new: (existing, samples_to_read,
        existing_slides) -- caller reads/merges just samples_to_read and
        folds the result into `existing`.
    """
    if not out_path.exists():
        print(f"merge_mode='incremental' but {out_path} doesn't exist yet -- doing a full merge instead.")
        return None, samples, None

    existing = read_existing(out_path)
    existing_slides = set(str(s) for s in get_slide_ids(existing))
    samples_to_read = [s for s in samples if s not in existing_slides]

    if not samples_to_read:
        print(f"All {len(samples)} requested sample(s) already present in {out_path} -- nothing to do.")
    else:
        print(f"Incremental merge: {len(samples_to_read)} new sample(s) to add "
              f"({sorted(samples_to_read)}), {len(existing_slides)} already present.")

    return existing, samples_to_read, existing_slides


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
    merge_mode: str = "full",
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
        With merge_mode="incremental", this is the full set you WANT in the
        combined file -- whichever of them are already present are skipped,
        not re-read.
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
    merge_mode : str
        "full" (default): read and merge every sample in `samples` from
        scratch, exactly as before -- overwrites the existing combined file
        if there is one. "incremental": read the existing combined output
        (same path this call would write to) if it exists, work out which
        of `samples` aren't in its Slide_ID column yet, and read/merge ONLY
        those -- then fold the result into the existing combined data
        (`_extend_combined`) rather than rebuilding everything. Falls back
        to a full merge if no combined output exists yet. Meant for the
        common case of adding one or two newly-processed slides to an
        already-large combined file without re-reading everyone else's
        (potentially many-GB) per-sample files.

    Returns
    -------
    AnnData
    """
    out_fname = filename.format(sample="combined")
    out_path = Path(path_template.format(basedir=basedir, sample="combined", filename=out_fname, **path_kwargs))

    if merge_mode not in ("full", "incremental"):
        raise ValueError(f"merge_mode must be 'full' or 'incremental', got {merge_mode!r}")

    existing = None
    existing_slides = None
    samples_to_read = samples
    if merge_mode == "incremental":
        existing, samples_to_read, existing_slides = _resolve_incremental(
            out_path, samples,
            read_existing=lambda p: sc.read_h5ad(p, backed="r"),
            get_slide_ids=lambda a: a.obs["Slide_ID"],
        )
        if existing is not None:
            if not samples_to_read:
                return existing.to_memory()
            existing = existing.to_memory()

    adatas, sample_ids = _read_and_prepare_samples(
        samples_to_read, path_template, filename, basedir, index_col, obs_name_mode, path_kwargs,
    )
    if not adatas:
        if existing is not None:
            print("None of the new samples' files were found -- returning the existing combined data unchanged.")
            return existing
        raise ValueError(f"No files named {filename!r} found to merge.")

    adata_merged = _build_merged_from_adatas(
        adatas, sample_ids, per_sample_var_cols, auto_detect_var_cols, reference_only_var_cols,
        per_sample_uns_moran_key, per_sample_pca_key, additional_per_sample_obsm_keys,
    )

    if existing is not None:
        adata_merged = _extend_combined(existing, adata_merged)
        print(f"Merged {len(adatas)} new sample(s) into the existing {len(existing_slides)} -> {out_path}")
    else:
        print(f"Merged {len(adatas)}/{len(samples)} sample(s) -> {out_path}")

    stray_nones = sanitize_uns(adata_merged)
    if stray_nones:
        print(f"  sanitized {len(stray_nones)} stray null uns entries before writing: {stray_nones}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_merged.write(out_path)

    return adata_merged


def merge_parquet(
    samples: list[str],
    path_template: str,
    filename: str,
    basedir,
    merge_mode: str = "full",
    **path_kwargs,
) -> gpd.GeoDataFrame:
    """Merge per-sample GeoParquet files into one, tagging each row with
    Slide_ID. path_template/filename/basedir/**path_kwargs follow the same
    convention as merge_anndatas (combined output uses sample="combined").

    merge_mode: "full" (default) rebuilds from every sample in `samples`,
    same as before. "incremental" reads the existing combined output (if
    any), skips samples whose Slide_ID is already in it, reads/concats only
    the new ones onto the existing rows, and writes that back -- see
    `merge_anndatas`'s merge_mode docstring for the full rationale. No
    per-sample bookkeeping to preserve here (unlike merge_anndatas), so this
    is just "read existing + new rows, concat, write" -- no separate extend
    helper needed.
    """
    if merge_mode not in ("full", "incremental"):
        raise ValueError(f"merge_mode must be 'full' or 'incremental', got {merge_mode!r}")

    out_fname = filename.format(sample="combined")
    out_path = Path(path_template.format(basedir=basedir, sample="combined", filename=out_fname, **path_kwargs))

    existing = None
    existing_slides = None
    samples_to_read = samples
    if merge_mode == "incremental":
        existing, samples_to_read, existing_slides = _resolve_incremental(
            out_path, samples, read_existing=gpd.read_parquet, get_slide_ids=lambda df: df["Slide_ID"],
        )
        if existing is not None and not samples_to_read:
            return existing

    new_parts = []
    for sample in samples_to_read:
        fname = filename.format(sample=sample)
        file_path = Path(path_template.format(basedir=basedir, sample=sample, filename=fname, **path_kwargs))
        if not file_path.exists():
            print(f"Skipping {sample}: {file_path} not found.")
            continue
        df = gpd.read_parquet(file_path)
        if "Slide_ID" not in df.columns:
            df["Slide_ID"] = sample
        new_parts.append(df)

    parts = ([existing] if existing is not None else []) + new_parts
    if not parts:
        raise ValueError(f"No files named {filename!r} found to merge.")

    merged = pd.concat(parts, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Merged {len(new_parts)}/{len(samples_to_read)} new sample(s) -> {out_path}"
          + (f" ({len(existing_slides)} already present, kept)" if existing is not None else ""))

    return merged
