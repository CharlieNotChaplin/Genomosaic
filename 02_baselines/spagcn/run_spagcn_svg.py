"""
Run SpaGCN spatial-domain clustering for one sample, using the
Moran's-I-annotated ("svg") count_area_filtered_adata produced by
01_preprocessing/transcriptomics/get_svg.py, plus the sample's .btf WSI.

Layout follows tutorial.ipynb's per-sample convention:
    data/{sample}/02_baselines/spagcn/data/       intermediate files (adj, l, res, prepped adata)
    data/{sample}/02_baselines/spagcn/results/    final clustering + plots

Prerequisite: get_svg.py has already been run for --sample, i.e.
    data/{sample}/01_preprocessing/{cellseg-dir}/{bin-size}/custom_binning/count_area_filtered_adata.h5ad
already has var['spatially_variable'] / obsm['spatial'] (get_svg.py annotates and
writes back to this same file, it does not produce a separate one).

Split into two stages (matching get_res.py / get_clustering.py) because
search_res + train are the expensive steps and this is meant to run as a
SLURM job with a time limit:
    --stage search   build adj matrix from svg centroids + histology image,
                      find l (spg.search_l) and res (spg.search_res), write
                      them to disk
    --stage train    load l/res from disk, train SpaGCN, refine, save results
    --stage all      run both in one process (fine for smaller samples)

Usage:
    python run_spagcn_svg.py --sample CRC_P1 --n-clusters 19 --stage all
"""
import argparse
import gc
import math
import os
import random
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import scanpy as sc
import scipy.sparse as sp
import SpaGCN as spg
import torch
from sklearn.decomposition import PCA

# SpaGCN (SpaGCN.py, util.py) still uses the old `.A` shorthand for sparse -> dense
# conversion, which scipy removed in favor of `.toarray()`. Restore it globally
# rather than patching vendored code in site-packages.
if not hasattr(sp.spmatrix, "A"):
    sp.spmatrix.A = property(lambda self: self.toarray())


def log_rss(tag):
    # ru_maxrss is the process's PEAK resident set size SO FAR (cumulative,
    # never decreases), in KB on Linux. Cheap (a single syscall), so left in
    # permanently rather than stripped after debugging -- added 2026-09-11
    # after estimate_peak_ram_gb's 3x and then 4x formulas both undershot for
    # several samples in ways that didn't fit a clean per-matrix-size model,
    # and the failures traced back to somewhere inside find_l's calculate_p
    # calls specifically (before clf.train() ever runs) -- narrower, direct
    # checkpoints here beat inferring the cause indirectly from which --mem
    # values happened to OOM.
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[mem] {tag}: peak RSS so far = {gb:.1f} GB", flush=True)


# Same math as the original spg.calculate_p(adj, l), but writing into one
# reused buffer in place instead of letting numpy allocate a fresh full-size
# n x n array at every operator (adj**2, /(2l^2), negate, exp each
# independently allocated one in the original one-liner). Measured directly
# (2026-09-11, CRC_P6_5mm, 22.7GB matrix, via log_rss checkpoints): the naive
# version -- even rewritten with named intermediates and explicit `del`
# between them -- peaked at 114.1GB (~5x the matrix size) for just two calls,
# not the ~3x a simple object-count would suggest. `del` didn't help because
# Python evaluates an expression's RHS (allocating the new array) before the
# old variable's `del` on the next line runs, so old and new are briefly
# resident together regardless of how the code is split; a jump between two
# sequential calls also showed the allocator wasn't cleanly reusing freed
# blocks between them. This needs at most 2x the matrix size (adj + the one
# reused buffer) total -- verified numerically identical to the original.
#
# Patched at the module level (spg.util.calculate_p, not just spg.calculate_p)
# because spg.util.search_l calls calculate_p as a bare name from within its
# own module up to a dozen+ times per bisection search (one call per Run),
# and Python resolves that at call time via search_l's own module globals --
# patching only the flat spg.calculate_p alias would leave every one of
# search_l's internal calls on the original, unpatched, 5x-peak version.
_calc_p_buf = [None]


def _calculate_p_low_mem(adj, l):
    buf = _calc_p_buf[0]
    if buf is None or buf.shape != adj.shape:
        buf = np.empty(adj.shape, dtype=np.float32)
        _calc_p_buf[0] = buf
    np.square(adj, out=buf)
    buf /= (2 * (l ** 2))
    np.negative(buf, out=buf)
    np.exp(buf, out=buf)
    return float(np.mean(buf.sum(axis=1))) - 1


spg.util.calculate_p = _calculate_p_low_mem
spg.calculate_p = _calculate_p_low_mem


# Same inefficiency as calculate_p above, one step further down the pipeline:
# vendored SpaGCN.train's own `adj_exp=np.exp(-1*(adj**2)/(2*(self.l**2)))`
# builds a second full n x n array via the same chained-expression pattern,
# on top of the adj matrix already resident. Confirmed directly (2026-09-17,
# CRC_P3_SMAP, ~350GB matrix, 900G job): after the calculate_p fix above
# already reduced find_l's own peak to the expected ~1x-per-call, the very
# next thing to blow the memory budget was this exact line, in a plain
# MemoryError requesting one more matrix-sized allocation (326GiB) partway
# through the first n_clusters' resolution search.
#
# spg.util.search_res calls clf.train() itself, up to ~10+ times per
# resolution bisection (a fresh SpaGCN() instance each time) -- patching the
# class method (rather than a free function, as calculate_p needed) is
# enough here because util.py does `from .SpaGCN import SpaGCN`, binding the
# same class object; there's no separate bare-name copy of `train` to miss
# the way calculate_p's was.
#
# adj_exp has to stay resident as self.adj_exp for predict() to use right
# after, unlike calculate_p's buffer, which is scratch-only -- but every real
# caller in this pipeline (search_res's internal loop, then run_train)
# creates one throwaway SpaGCN() instance, calls train() then predict(), then
# discards it before the next one is created. So handing out a single reused
# module-level buffer as self.adj_exp is safe: the previous instance's
# predict() has always already consumed it before the next train() call
# would overwrite it.
_train_adj_exp_buf = [None]


def _train_low_mem(self, adata, adj,
                    num_pcs=50,
                    lr=0.005,
                    max_epochs=2000,
                    weight_decay=0,
                    opt="admin",
                    init_spa=True,
                    init="louvain",  # louvain or kmeans
                    n_neighbors=10,  # for louvain
                    n_clusters=None,  # for kmeans
                    res=0.4,  # for louvain
                    tol=1e-3):
    self.num_pcs = num_pcs
    self.res = res
    self.lr = lr
    self.max_epochs = max_epochs
    self.weight_decay = weight_decay
    self.opt = opt
    self.init_spa = init_spa
    self.init = init
    self.n_neighbors = n_neighbors
    self.n_clusters = n_clusters
    self.res = res
    self.tol = tol
    assert adata.shape[0] == adj.shape[0] == adj.shape[1]
    pca = PCA(n_components=self.num_pcs)
    if sp.issparse(adata.X):
        pca.fit(adata.X.A)
        embed = pca.transform(adata.X.A)
    else:
        pca.fit(adata.X)
        embed = pca.transform(adata.X)
    ###------------------------------------------###
    if self.l is None:
        raise ValueError('l should not be set before fitting the model!')
    buf = _train_adj_exp_buf[0]
    if buf is None or buf.shape != adj.shape:
        buf = np.empty(adj.shape, dtype=np.float32)
        _train_adj_exp_buf[0] = buf
    np.square(adj, out=buf)
    buf /= (2 * (self.l ** 2))
    np.negative(buf, out=buf)
    np.exp(buf, out=buf)
    adj_exp = buf
    # ----------Train model----------
    self.model = spg.simple_GC_DEC(embed.shape[1], embed.shape[1])
    self.model.fit(embed, adj_exp, lr=self.lr, max_epochs=self.max_epochs, weight_decay=self.weight_decay,
                    opt=self.opt, init_spa=self.init_spa, init=self.init, n_neighbors=self.n_neighbors,
                    n_clusters=self.n_clusters, res=self.res, tol=self.tol)
    self.embed = embed
    self.adj_exp = adj_exp


spg.SpaGCN.train = _train_low_mem


def _calculate_adj_matrix_patched(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    # Same fix already applied once in this repo's local SpaGCN_package/SpaGCN/calculate_adj.py
    # (git commit 85f879a, Nov 2025) but never carried into a fresh `pip install SpaGCN` on the
    # cluster: stock SpaGCN's calculate_adj_matrix does a plain np.mean over each nucleus's beta
    # window with no guard for an out-of-bounds/empty/all-background crop. One such nucleus makes
    # c0/c1/c2 contain a NaN, which poisons np.var(c0) (etc.) for every nucleus at once -> the
    # whole z coordinate becomes NaN -> every entry of adj becomes NaN.
    if histology:
        assert (x_pixel is not None) and (y_pixel is not None) and (image is not None)
        assert (len(x) == len(x_pixel)) and (len(y) == len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        beta_half = round(beta / 2)
        g = []
        for i in range(len(x_pixel)):
            max_x = image.shape[0]
            max_y = image.shape[1]
            nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                        max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
            if nbs.size == 0 or np.isnan(nbs).all():
                g.append([0, 0, 0])
            else:
                g.append(np.nanmean(np.nanmean(nbs, axis=0), axis=0))
        c0 = np.array([i[0] for i in g])
        c1 = np.array([i[1] for i in g])
        c2 = np.array([i[2] for i in g])
        print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = (c4 * z_scale).tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    return spg.pairwise_distance(X)


spg.calculate_adj_matrix = _calculate_adj_matrix_patched

import warnings

warnings.filterwarnings("ignore")


def load_sample_adata(dir_custom_binning, dir_combined_adata, sample):
    """
    Load the per-nucleus adata for `sample`: from its own
    count_area_filtered_adata.h5ad if that still exists, otherwise filtered out
    of the combined multi-sample file (obs['Slide_ID'] == sample). The
    per-sample files this pipeline originally read from have since been
    deleted for every sample -- verified (2026-09-08) that the combined file's
    per-slide obsm['spatial'] coordinates are still in that slide's own native
    full-res WSI pixel space (CRC_P1's x/y range matches the old per-sample
    file to within a few px, the small difference being extra nuclei present
    in the combined file's less-filtered version) -- so no rescaling is needed
    either way, just picking the right rows out.
    """
    per_sample_path = dir_custom_binning / "count_area_filtered_adata.h5ad"
    if per_sample_path.exists():
        adata = sc.read_h5ad(per_sample_path)
        source = per_sample_path
    else:
        if not dir_combined_adata.exists():
            raise FileNotFoundError(f"Neither {per_sample_path} nor {dir_combined_adata} exist")
        print(f"{per_sample_path} not found, loading {sample} out of {dir_combined_adata} instead", flush=True)
        combined = sc.read_h5ad(dir_combined_adata, backed="r")
        if "Slide_ID" not in combined.obs:
            raise ValueError(f"{dir_combined_adata} has no obs['Slide_ID'] to select {sample!r} by")
        mask = (combined.obs["Slide_ID"] == sample).values
        if mask.sum() == 0:
            raise ValueError(f"No rows with Slide_ID == {sample!r} in {dir_combined_adata}")
        adata = combined[mask].to_memory()
        source = dir_combined_adata

    if "spatially_variable" not in adata.var or "spatial" not in adata.obsm:
        raise ValueError(
            f"{source} has no var['spatially_variable']/obsm['spatial'] for sample {sample!r} -- "
            f"run get_svg.py --sample {sample} first"
        )
    return adata


def build_adj_and_centroids(dir_custom_binning, dir_combined_adata, dir_base_input, dir_base_output,
                             dir_spagcn_data, dir_spagcn_results, sample, beta, alpha,
                             force_recompute=False, skip_adj_disk=False):
    """
    Returns (adata, adj) -- the same objects the adjacency graph was built
    from, so a caller that already holds them in memory (run_spagcn_range.py)
    doesn't have to round-trip them through disk to use them.

    skip_adj_disk=True never writes adj.npy at all (nuclei_centroids_svg.h5ad
    still is -- it's a few hundred MB at most, not the multi-GB/TB item).
    Only safe when the whole build->search->train sequence for this sample
    runs in one continuous process (run_spagcn_range.py): run_spagcn_svg.py's
    own --stage search/--stage train CLI runs those as separate SLURM jobs,
    which can only share state through disk, so it always leaves this False.
    """
    nuclei_centroids_path = dir_spagcn_data / "nuclei_centroids_svg.h5ad"
    adj_path = dir_spagcn_data / "adj.npy"

    if not skip_adj_disk and nuclei_centroids_path.exists() and adj_path.exists():
        if not force_recompute:
            print(f"Found existing {nuclei_centroids_path.name} and adj.npy, skipping prep "
                  "(pass --force-recompute, or delete them, to force recompute)")
            return None, None
        print(f"--force-recompute set: deleting existing {nuclei_centroids_path.name} and adj.npy")
        nuclei_centroids_path.unlink()
        adj_path.unlink()

    # .btf (BigTIFF) for the original CRC_P*/Normal_P* samples, .tif/.tiff for newer
    # ones (e.g. CRC_P3_SMAP/CRC_P4_SMAP ship "HE.tif" instead) -- both are read the
    # same way via pyvips, so just widen the search rather than assuming one extension.
    wsi_candidates = sorted(set(
        p for pattern in ("*.btf", "*.tif", "*.tiff") for p in dir_base_input.glob(pattern)
    ))
    if len(wsi_candidates) != 1:
        raise RuntimeError(f"Expected exactly one .btf/.tif/.tiff in {dir_base_input}, found {wsi_candidates}")
    f_wsi = wsi_candidates[0]
    print("Using WSI:", f_wsi)

    # obsm['spatial'] already holds nuclei centroid pixel coords (set by get_svg.py
    # from nuclei_geometry.parquet), so we don't need geopandas here.
    adata = load_sample_adata(dir_custom_binning, dir_combined_adata, sample)
    adata.var_names_make_unique()
    if adata.obs["id"].is_unique:
        adata.obs = adata.obs.set_index("id")
    else:
        raise ValueError("Nucleus IDs in count_area_filtered_adata_test_svg.h5ad are not unique.")

    spatial = np.asarray(adata.obsm["spatial"])
    adata.obs["x_pixel"] = spatial[:, 0]
    adata.obs["y_pixel"] = spatial[:, 1]

    # --- derive array coords (x_array/y_array) from tissue_positions, needed later for refine() ---
    tissue_positions = pd.read_parquet(dir_base_output / "spatial/tissue_positions.parquet")

    array_x_min, array_y_min = tissue_positions["array_col"].min(), tissue_positions["array_row"].min()
    array_x_max, array_y_max = tissue_positions["array_col"].max(), tissue_positions["array_row"].max()

    # Fit the pixel -> array affine transform via least squares over EVERY bin,
    # not just the 4 extreme grid corners (min/max array_row combined with
    # min/max array_col): tissue often doesn't fill its full rectangular
    # array_row/array_col bounding box (e.g. an irregular or cut tissue shape),
    # so one specific corner combination may not exist as an actual row in
    # tissue_positions.parquet at all -- CRC_P11mm_2_cut hit exactly this,
    # raising IndexError on an empty filtered lookup. Using every bin needs no
    # specific corner to exist and is strictly more robust regardless (fits
    # from thousands of points instead of 4).
    pixel_pts = tissue_positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    array_pts = tissue_positions[["array_col", "array_row"]].to_numpy()

    A = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
    params, _, _, _ = np.linalg.lstsq(A, array_pts, rcond=None)

    px, py = adata.obs["x_pixel"].to_numpy(), adata.obs["y_pixel"].to_numpy()
    pixel_matrix = np.vstack([px, py, np.ones_like(px)]).T
    array_coords = pixel_matrix @ params

    adata.obs["x_array"] = np.clip(np.floor(array_coords[:, 0]).astype(int), array_x_min, array_x_max)
    adata.obs["y_array"] = np.clip(np.floor(array_coords[:, 1]).astype(int), array_y_min, array_y_max)

    del tissue_positions
    gc.collect()

    # --- crop WSI to tissue bounding box and load into memory ---
    tp2 = pd.read_parquet(dir_base_output / "spatial/tissue_positions.parquet")
    xmin = max(0, math.floor(tp2["pxl_col_in_fullres"].min()))
    xmax = max(0, math.ceil(tp2["pxl_col_in_fullres"].max()))
    ymin = max(0, math.floor(tp2["pxl_row_in_fullres"].min()))
    ymax = max(0, math.ceil(tp2["pxl_row_in_fullres"].max()))
    del tp2
    gc.collect()

    print("Cropping WSI to bounding box:", xmin, xmax, ymin, ymax)
    image = pyvips.Image.new_from_file(str(f_wsi), access="sequential")
    cropped = image.crop(xmin, ymin, xmax - xmin, ymax - ymin)
    del image

    numpy_array = np.ndarray(
        buffer=cropped.write_to_memory(),
        dtype=np.uint8,
        shape=[cropped.height, cropped.width, cropped.bands],
    )
    # RGB -> BGR, matching what cv2.cvtColor(..., cv2.COLOR_RGB2BGR) would do
    img = numpy_array[:, :, ::-1] if cropped.bands == 3 else numpy_array
    del numpy_array, cropped
    gc.collect()

    x_pixel = [round(x - xmin) for x in adata.obs["x_pixel"].tolist()]
    y_pixel = [round(y - ymin) for y in adata.obs["y_pixel"].tolist()]

    print("Building adjacency matrix using histology image...")
    adj = spg.calculate_adj_matrix(
        x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel,
        image=img, beta=beta, alpha=alpha, histology=True,
    )
    del img
    gc.collect()
    log_rss("build_adj_and_centroids: after calculate_adj_matrix")

    dir_spagcn_data.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(nuclei_centroids_path)
    if skip_adj_disk:
        print(f"Wrote {nuclei_centroids_path} (adj.npy skipped -- kept in memory only)")
    else:
        np.save(adj_path, adj.astype(np.float32))
        print(f"Wrote {nuclei_centroids_path} and {adj_path}")
    log_rss("build_adj_and_centroids: return")
    return adata, adj


def prep_adata_for_search(adata):
    """
    Prefilter in place, matching load_prepped_adata_and_adj's prep steps --
    factored out so a caller that already has adata in memory
    (run_spagcn_range.py) can prep it directly instead of round-tripping
    through disk first. Mutates and returns `adata`; pass a copy in if the
    original (unfiltered) adata is still needed afterward.

    Does NOT normalize/log1p: count_area_filtered_adata.h5ad (both the
    per-sample and combined versions) is already sc.pp.normalize_total
    (target_sum=1e4) + sc.pp.log1p'd by cell_preprocessing.py before it's
    ever written to disk -- confirmed directly in that script (2026-09-10).
    Calling normalize_per_cell/log1p again here was a real double-transform
    of every SpaGCN run before this fix (not just a harmless idempotent
    no-op): normalize_per_cell is scanpy's older name for normalize_total,
    called here with a DIFFERENT (default) target_sum than the 1e4 already
    applied upstream, so it isn't even a clean double-log1p, it's
    log1p(renormalize(log1p(normalize_total(raw)))) feeding SpaGCN's PCA --
    a materially different, wrong distribution. Only spg.prefilter_genes/
    prefilter_specialgenes (which drop genes, not rescale values) are safe
    to reapply here regardless of upstream state.
    """
    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    return adata


def load_prepped_adata_and_adj(dir_spagcn_data):
    adata = sc.read_h5ad(dir_spagcn_data / "nuclei_centroids_svg.h5ad")
    adj = np.load(dir_spagcn_data / "adj.npy", mmap_mode="r")
    prep_adata_for_search(adata)
    return adata, adj


def find_l(dir_spagcn_data, adj, p, l_start, l_end, l_tol, l_max_run):
    # Diagnose the search range ourselves before calling spg.search_l: SpaGCN's own
    # diagnostic prints ("l not found, try smaller/bigger ...") go to a block-buffered
    # stdout and can be lost if the process later dies, leaving only the confusing
    # downstream "l should not be set" ValueError with no explanation. flush=True here
    # so this always lands in the log regardless of buffering.
    #
    # spg.calculate_p is monkeypatched (near the top of this file) to a
    # low-memory in-place version -- these two calls, and every one of
    # spg.search_l's own internal calls just below, all go through it.
    log_rss("find_l: start (adj already resident)")
    p_low = spg.calculate_p(adj, l_start)
    log_rss("find_l: after calculate_p(l_start)")
    p_high = spg.calculate_p(adj, l_end)
    log_rss("find_l: after calculate_p(l_end)")
    print(f"Searching for l (p target = {p}); p(l={l_start}) = {p_low:.4f}, "
          f"p(l={l_end}) = {p_high:.4f}", flush=True)
    if p_low > p + l_tol:
        raise RuntimeError(
            f"p(l={l_start}) = {p_low:.4f} already exceeds target p={p} -- this sample's "
            f"adjacency is denser than the default range assumes, pass a smaller --l-start"
        )
    if p_high < p - l_tol:
        raise RuntimeError(
            f"p(l={l_end}) = {p_high:.4f} is still below target p={p} -- "
            f"pass a larger --l-end"
        )

    l = spg.search_l(p, adj, start=l_start, end=l_end, tol=l_tol, max_run=l_max_run)
    if l is None:
        raise RuntimeError(
            f"spg.search_l did not converge within max_run={l_max_run} even though "
            f"p={p} is bracketed by [{p_low:.4f}, {p_high:.4f}] -- try --l-max-run higher "
            f"or --l-tol looser"
        )
    with open(dir_spagcn_data / "l.txt", "w") as f:
        f.write(str(l))
    print("l =", l, flush=True)
    return l


def find_res(dir_spagcn_data, adata, adj, l, n_clusters, seed):
    print(f"Searching for resolution giving {n_clusters} clusters...", flush=True)
    res = spg.search_res(
        adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05,
        max_epochs=20, r_seed=seed, t_seed=seed, n_seed=seed,
    )
    with open(dir_spagcn_data / f"res_cluster_{n_clusters}.txt", "w") as f:
        f.write(str(res))
    print("l =", l, " res =", res)
    return res


def run_search(dir_spagcn_data, n_clusters, p, seed, l_start, l_end, l_tol, l_max_run):
    adata, adj = load_prepped_adata_and_adj(dir_spagcn_data)
    l = find_l(dir_spagcn_data, adj, p, l_start, l_end, l_tol, l_max_run)
    find_res(dir_spagcn_data, adata, adj, l, n_clusters, seed)


def run_train(dir_spagcn_data, dir_spagcn_results, n_clusters, seed, svg_genes_only,
               adata=None, adj=None):
    """
    If adata/adj are not given, loads them fresh from disk, matching the
    original behavior exactly. If given (run_spagcn_range.py's in-memory
    path), operates on a COPY of adata rather than the caller's object --
    required because this function's own prefilter/subset/normalize/log1p
    steps are destructive, and it's called once per n_clusters value against
    what would otherwise be the same shared, already-mutated adata on the
    second call onward.
    """
    if adata is None:
        adata = sc.read_h5ad(dir_spagcn_data / "nuclei_centroids_svg.h5ad")
    else:
        adata = adata.copy()
    if adj is None:
        adj = np.load(dir_spagcn_data / "adj.npy", mmap_mode="r")

    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)

    if svg_genes_only:
        n_before = adata.n_vars
        adata = adata[:, adata.var["spatially_variable"].fillna(False)].copy()
        print(f"Restricted to {adata.n_vars}/{n_before} spatially variable genes")

    # NOT normalize_per_cell/log1p here: count_area_filtered_adata.h5ad is
    # already normalize_total(target_sum=1e4) + log1p'd by
    # cell_preprocessing.py before it's ever written -- see
    # prep_adata_for_search's docstring for the full explanation. Doing it
    # again here was a real double-transform of every SpaGCN run before this
    # fix, not a harmless no-op.

    with open(dir_spagcn_data / "l.txt") as f:
        l = float(f.read().strip())
    with open(dir_spagcn_data / f"res_cluster_{n_clusters}.txt") as f:
        res = float(f.read().strip())

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clf = spg.SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, _ = clf.predict()
    adata.obs["pred"] = pd.Categorical(y_pred)

    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(),
                               dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"] = pd.Categorical(refined_pred)

    dir_spagcn_results.mkdir(parents=True, exist_ok=True)
    suffix = "_svggenes" if svg_genes_only else ""
    out_path = dir_spagcn_results / f"results_cluster_{n_clusters}{suffix}.h5ad"
    adata.write_h5ad(out_path)
    print(f"Wrote {out_path}")

    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C",
                  "#15821E", "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236",
                  "#AF5F3C", "#93796C", "#F9BD3F", "#DAB370", "#877F6C", "#268785"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for domains in ["pred", "refined_pred"]:
        num_celltype = len(adata.obs[domains].unique())
        adata.uns[domains + "_colors"] = list((plot_color * (num_celltype // len(plot_color) + 1))[:num_celltype])
        ax = sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title=domains,
                            show=False, size=100000 / adata.shape[0])
        ax.set_aspect("equal", "box")
        ax.axes.invert_yaxis()
        plt.savefig(dir_spagcn_results / f"{domains}_cluster_{n_clusters}{suffix}.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--bin-size", default="002um")
    parser.add_argument("--cellseg-dir", default="cell_seg_stardist")
    parser.add_argument("--n-clusters", type=int, required=True,
                         help="Expected number of spatial domains (check the sample's Web Summary)")
    parser.add_argument("--svg-genes-only", action="store_true",
                         help="Restrict training to var['spatially_variable'] genes from get_svg.py "
                              "(does not affect adj/l/res search, which is purely spatial)")
    parser.add_argument("--stage", choices=["search", "train", "all"], default="all")
    parser.add_argument("--beta", type=int, default=49)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--l-start", type=float, default=0.01, help="spg.search_l lower bound")
    parser.add_argument("--l-end", type=float, default=1000, help="spg.search_l upper bound")
    parser.add_argument("--l-tol", type=float, default=0.01, help="spg.search_l tolerance on p")
    parser.add_argument("--l-max-run", type=int, default=100, help="spg.search_l bisection step cap")
    parser.add_argument("--force-recompute", action="store_true",
                         help="delete and rebuild cached nuclei_centroids_svg.h5ad/adj.npy even if present")
    args = parser.parse_args()

    basedir = Path("../../data")
    dir_sample = basedir / args.sample
    dir_base_input = dir_sample / "input"
    dir_base_output = dir_sample / f"output/binned_outputs/square_{args.bin_size}"
    dir_custom_binning = dir_sample / "01_preprocessing" / args.cellseg_dir / args.bin_size / "custom_binning"
    dir_combined_adata = (basedir / "combined" / "01_preprocessing" / args.cellseg_dir / args.bin_size
                           / "custom_binning" / "count_area_filtered_adata.h5ad")
    dir_spagcn_data = dir_sample / "02_baselines/spagcn/data"
    dir_spagcn_results = dir_sample / "02_baselines/spagcn/results"

    if args.stage in ("search", "all"):
        build_adj_and_centroids(dir_custom_binning, dir_combined_adata, dir_base_input, dir_base_output,
                                 dir_spagcn_data, dir_spagcn_results, args.sample, args.beta, args.alpha,
                                 force_recompute=args.force_recompute)
        run_search(dir_spagcn_data, args.n_clusters, args.p, args.seed,
                   args.l_start, args.l_end, args.l_tol, args.l_max_run)

    if args.stage in ("train", "all"):
        run_train(dir_spagcn_data, dir_spagcn_results, args.n_clusters, args.seed, args.svg_genes_only)


if __name__ == "__main__":
    main()
