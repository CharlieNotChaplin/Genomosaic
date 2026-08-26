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
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import scanpy as sc
import scipy.sparse as sp
import SpaGCN as spg
import torch

# SpaGCN (SpaGCN.py, util.py) still uses the old `.A` shorthand for sparse -> dense
# conversion, which scipy removed in favor of `.toarray()`. Restore it globally
# rather than patching vendored code in site-packages.
if not hasattr(sp.spmatrix, "A"):
    sp.spmatrix.A = property(lambda self: self.toarray())


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


def build_adj_and_centroids(dir_custom_binning, dir_base_input, dir_base_output,
                             dir_spagcn_data, dir_spagcn_results, sample, beta, alpha):
    nuclei_centroids_path = dir_spagcn_data / "nuclei_centroids_svg.h5ad"
    adj_path = dir_spagcn_data / "adj.npy"

    if nuclei_centroids_path.exists() and adj_path.exists():
        print(f"Found existing {nuclei_centroids_path.name} and adj.npy, skipping prep "
              "(delete them to force recompute)")
        return

    # get_svg.py annotates and writes back to the same count_area_filtered_adata.h5ad
    # it reads (var['spatially_variable'], var['moranI'], obsm['spatial']) -- there is
    # no separate "*_svg.h5ad" file on disk for any processed sample.
    svg_path = dir_custom_binning / "count_area_filtered_adata.h5ad"
    if not svg_path.exists():
        raise FileNotFoundError(f"{svg_path} not found")

    btf_candidates = sorted(dir_base_input.glob("*.btf"))
    if len(btf_candidates) != 1:
        raise RuntimeError(f"Expected exactly one .btf in {dir_base_input}, found {btf_candidates}")
    f_wsi = btf_candidates[0]
    print("Using WSI:", f_wsi)

    # obsm['spatial'] already holds nuclei centroid pixel coords (set by get_svg.py
    # from nuclei_geometry.parquet), so we don't need geopandas here.
    adata = sc.read_h5ad(svg_path)
    if "spatially_variable" not in adata.var or "spatial" not in adata.obsm:
        raise ValueError(
            f"{svg_path} has no var['spatially_variable']/obsm['spatial'] -- "
            f"run get_svg.py --sample {sample} first"
        )
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

    tissue_pos_array = tissue_positions[
        ((tissue_positions["array_col"] == array_x_min) | (tissue_positions["array_col"] == array_x_max)) &
        ((tissue_positions["array_row"] == array_y_min) | (tissue_positions["array_row"] == array_y_max))
    ]

    def get_pxl_value(tissue_pos, col_val, row_val):
        row = tissue_pos.loc[(tissue_pos["array_row"] == row_val) & (tissue_pos["array_col"] == col_val)]
        return row["pxl_col_in_fullres"].iloc[0], row["pxl_row_in_fullres"].iloc[0]

    px_tl, py_tl = get_pxl_value(tissue_pos_array, array_x_min, array_y_max)
    px_tr, py_tr = get_pxl_value(tissue_pos_array, array_x_max, array_y_max)
    px_bl, py_bl = get_pxl_value(tissue_pos_array, array_x_min, array_y_min)
    px_br, py_br = get_pxl_value(tissue_pos_array, array_x_max, array_y_min)

    array_pts = np.array([[array_x_min, array_y_min], [array_x_max, array_y_min],
                           [array_x_min, array_y_max], [array_x_max, array_y_max]])
    pixel_pts = np.array([[px_bl, py_bl], [px_br, py_br], [px_tl, py_tl], [px_tr, py_tr]])

    A = np.hstack([pixel_pts, np.ones((4, 1))])
    params, _, _, _ = np.linalg.lstsq(A, array_pts, rcond=None)

    px, py = adata.obs["x_pixel"].to_numpy(), adata.obs["y_pixel"].to_numpy()
    pixel_matrix = np.vstack([px, py, np.ones_like(px)]).T
    array_coords = pixel_matrix @ params

    adata.obs["x_array"] = np.clip(np.floor(array_coords[:, 0]).astype(int), array_x_min, array_x_max)
    adata.obs["y_array"] = np.clip(np.floor(array_coords[:, 1]).astype(int), array_y_min, array_y_max)

    del tissue_positions, tissue_pos_array
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

    dir_spagcn_data.mkdir(parents=True, exist_ok=True)
    np.save(adj_path, adj.astype(np.float32))
    adata.write_h5ad(nuclei_centroids_path)
    print(f"Wrote {nuclei_centroids_path} and {adj_path}")


def run_search(dir_spagcn_data, n_clusters, p, seed, l_start, l_end, l_tol, l_max_run):
    adata = sc.read_h5ad(dir_spagcn_data / "nuclei_centroids_svg.h5ad")
    adj = np.load(dir_spagcn_data / "adj.npy", mmap_mode="r")
    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    # Diagnose the search range ourselves before calling spg.search_l: SpaGCN's own
    # diagnostic prints ("l not found, try smaller/bigger ...") go to a block-buffered
    # stdout and can be lost if the process later dies, leaving only the confusing
    # downstream "l should not be set" ValueError with no explanation. flush=True here
    # so this always lands in the log regardless of buffering.
    p_low = spg.calculate_p(adj, l_start)
    p_high = spg.calculate_p(adj, l_end)
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

    print(f"Searching for resolution giving {n_clusters} clusters...", flush=True)
    res = spg.search_res(
        adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05,
        max_epochs=20, r_seed=seed, t_seed=seed, n_seed=seed,
    )
    with open(dir_spagcn_data / f"res_cluster_{n_clusters}.txt", "w") as f:
        f.write(str(res))
    print("l =", l, " res =", res)


def run_train(dir_spagcn_data, dir_spagcn_results, n_clusters, seed, svg_genes_only):
    adata = sc.read_h5ad(dir_spagcn_data / "nuclei_centroids_svg.h5ad")
    adj = np.load(dir_spagcn_data / "adj.npy", mmap_mode="r")
    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)

    if svg_genes_only:
        n_before = adata.n_vars
        adata = adata[:, adata.var["spatially_variable"].fillna(False)].copy()
        print(f"Restricted to {adata.n_vars}/{n_before} spatially variable genes")

    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

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
    args = parser.parse_args()

    basedir = Path("../../data")
    dir_sample = basedir / args.sample
    dir_base_input = dir_sample / "input"
    dir_base_output = dir_sample / f"output/binned_outputs/square_{args.bin_size}"
    dir_custom_binning = dir_sample / "01_preprocessing" / args.cellseg_dir / args.bin_size / "custom_binning"
    dir_spagcn_data = dir_sample / "02_baselines/spagcn/data"
    dir_spagcn_results = dir_sample / "02_baselines/spagcn/results"

    if args.stage in ("search", "all"):
        build_adj_and_centroids(dir_custom_binning, dir_base_input, dir_base_output,
                                 dir_spagcn_data, dir_spagcn_results, args.sample, args.beta, args.alpha)
        run_search(dir_spagcn_data, args.n_clusters, args.p, args.seed,
                   args.l_start, args.l_end, args.l_tol, args.l_max_run)

    if args.stage in ("train", "all"):
        run_train(dir_spagcn_data, dir_spagcn_results, args.n_clusters, args.seed, args.svg_genes_only)


if __name__ == "__main__":
    main()
