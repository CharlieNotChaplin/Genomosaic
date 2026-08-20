"""
Distills the raw per-patch ViT token grid (256 tokens x 1536-dim per 224px
patch, ~9GB/slide) down to two compact, per-CELL feature vectors:

  - footprint: area-weighted average of the token embeddings overlapping a
    cell's segmented nucleus polygon.
  - halo: area-weighted average of the token embeddings overlapping a small
    ring around the nucleus (nucleus dilated by `halo_px`, minus the nucleus
    itself) -- captures immediate membrane/cytoplasm/microenvironment context
    that StarDist's nuclear-only segmentation misses.

Kept as two separate vectors (rather than one blended average) so the model
can treat "this cell's own tissue" and "immediately around this cell" as
distinct signals -- see conversation notes (TRIPLEX's target/neighbor split).

Why per-cell distillation at all: the raw token file is one row per token
(256/patch), which is only tractable to keep on disk per-slide. Once every
cell has been resolved to its own footprint+halo vector, the result is one
row per CELL (same order of magnitude as the existing transcriptomics/SVG
data), cheap enough to load whole across every slide at once.

Coordinate system: cell centroids (`trans_adata.obsm['spatial']`), nucleus
polygons (`nuclei_geometry.parquet`), and token positions (`Token_X`/`Token_Y`
in the token h5ad) all live in the same per-slide WSI pixel space -- verified
directly against this project's data (see conversation notes). Patch linkage
uses the existing `Patch_ID` string (e.g. "patch_123.png"), which is shared
between the transcriptomics obs and the token obs for a given slide.

Usage:
    python token_preprocessing.py --slide CRC_P1
"""
import argparse
from pathlib import Path

import anndata as ad
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box

TOKEN_SIZE_PX = 14.0
TOKENS_PER_SIDE = 16


def _candidate_token_range(minx, miny, maxx, maxy, origin_x, origin_y, margin_tokens=1):
    """Bounding-box prefilter: which (col, row) token grid cells could possibly
    overlap a region with these pixel bounds, given the patch's token-grid
    origin. `margin_tokens` pads by extra token-widths to be safe against
    off-by-one rounding at the boundary -- cheap (a handful of extra shapely
    intersection checks that will just come back as zero-area)."""
    col_lo = int(np.floor((minx - origin_x) / TOKEN_SIZE_PX)) - margin_tokens
    col_hi = int(np.floor((maxx - origin_x) / TOKEN_SIZE_PX)) + margin_tokens
    row_lo = int(np.floor((miny - origin_y) / TOKEN_SIZE_PX)) - margin_tokens
    row_hi = int(np.floor((maxy - origin_y) / TOKEN_SIZE_PX)) + margin_tokens
    col_lo, col_hi = max(0, col_lo), min(TOKENS_PER_SIDE - 1, col_hi)
    row_lo, row_hi = max(0, row_lo), min(TOKENS_PER_SIDE - 1, row_hi)
    return col_lo, col_hi, row_lo, row_hi


def compute_positional_bias(token_h5ad_path, chunk_size=50_000, verbose=True):
    """
    Returns bias_grid, shape (TOKENS_PER_SIDE, TOKENS_PER_SIDE, token_dim):
    the mean token embedding at each (row, col) position within a patch,
    minus the grand mean across every token on this slide.

    Real, confirmed artifact, not a hypothesis: each 224px patch is run
    through UNI2h independently, so tokens near a patch's edge (row/col 0 or
    15) systematically lack the surrounding tissue context that interior
    tokens have. This shows up as a real per-position bias that survives
    averaging over 100k+ tokens (i.e. it isn't just content variance
    cancelling out unevenly), strongest at the edges and smoothly varying
    across the patch -- and was shown directly (see conversation notes) to
    drive leiden cluster assignment as a function of a cell's position
    *within its own patch*, independent of actual tissue content: cluster
    composition swings from one leiden cluster to a totally different one
    across a single patch's width, reproducibly across thousands of
    unrelated patches. That's what produced the "striping" tied to the
    224px patch grid seen in cell-level clustering.

    Computed fresh per-slide from that slide's own raw token grid (not a
    hardcoded correction from some other slide) and applied by subtracting
    bias_grid[row, col] from each token before it ever gets pooled into a
    cell's footprint/halo vector. Streamed in chunks so the full
    (n_tokens x token_dim) array is never held in memory at once.
    """
    f = h5py.File(token_h5ad_path, "r")
    token_patch_codes = f["obs/Patch_ID/codes"][:]
    token_X = f["obs/Token_X"][:]
    token_Y = f["obs/Token_Y"][:]
    token_dim = f["X"].shape[1]
    n_tokens = f["X"].shape[0]

    order = np.argsort(token_patch_codes, kind="stable")
    sorted_codes = token_patch_codes[order]
    uniq_codes, start_idx = np.unique(sorted_codes, return_index=True)
    groups = np.split(order, start_idx[1:])

    tok_col_all = np.zeros(n_tokens, dtype=np.int16)
    tok_row_all = np.zeros(n_tokens, dtype=np.int16)
    for code, g in zip(uniq_codes, groups):
        g = np.sort(g)
        x, y = token_X[g], token_Y[g]
        ox, oy = x.min(), y.min()
        tok_col_all[g] = np.round((x - ox) / TOKEN_SIZE_PX).astype(np.int16)
        tok_row_all[g] = np.round((y - oy) / TOKEN_SIZE_PX).astype(np.int16)

    grid_sum = np.zeros((TOKENS_PER_SIDE, TOKENS_PER_SIDE, token_dim), dtype=np.float64)
    grid_count = np.zeros((TOKENS_PER_SIDE, TOKENS_PER_SIDE), dtype=np.int64)
    grand_sum = np.zeros(token_dim, dtype=np.float64)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        if verbose:
            print(f"  [positional bias] tokens {start}/{n_tokens}")
        # read+sum in float32 (source is float16 on disk) -- only the running
        # accumulators need float64 precision, not each transient chunk, which
        # matters here: a float64 chunk at the old 200k chunk_size was ~3.7GB,
        # enough to OOM on top of whatever else is already resident
        chunk = f["X"][start:end].astype(np.float32)
        grand_sum += chunk.sum(axis=0, dtype=np.float64)
        rows, cols = tok_row_all[start:end], tok_col_all[start:end]
        for r in range(TOKENS_PER_SIDE):
            for c in range(TOKENS_PER_SIDE):
                m = (rows == r) & (cols == c)
                if m.any():
                    grid_sum[r, c] += chunk[m].sum(axis=0, dtype=np.float64)
                    grid_count[r, c] += m.sum()
        del chunk

    f.close()
    grand_mean = grand_sum / n_tokens
    grid_mean = grid_sum / grid_count[:, :, None]
    bias_grid = (grid_mean - grand_mean).astype(np.float32)
    if verbose:
        dev_norms = np.linalg.norm(bias_grid.reshape(-1, token_dim), axis=1).reshape(TOKENS_PER_SIDE, TOKENS_PER_SIDE)
        print(f"  [positional bias] deviation norm range: {dev_norms.min():.2f} - {dev_norms.max():.2f}")
    return bias_grid


def compute_cell_token_features(trans_adata_slide, nuclei_geom_slide, token_h5ad_path,
                                 patch_col="Patch_ID", halo_px=15.0, margin_tokens=1,
                                 positional_bias=None, verbose=True):
    """
    Returns (footprint, halo, matched), each row-aligned to
    `trans_adata_slide.obs_names`:
      footprint: (n_cells, token_dim) float32 -- area-weighted tokens under the nucleus
      halo:      (n_cells, token_dim) float32 -- area-weighted tokens in the halo ring
      matched:   (n_cells,) bool -- False for cells whose patch has no token data
                 (e.g. dropped/filtered patch) or whose nucleus id wasn't found
                 in `nuclei_geom_slide`

    `positional_bias`: optional (TOKENS_PER_SIDE, TOKENS_PER_SIDE, token_dim)
    array from `compute_positional_bias` -- if given, subtracted from every
    token's embedding (by its (row, col) position within its own patch)
    before it contributes to any footprint/halo average. See
    `compute_positional_bias`'s docstring for why this matters.
    """
    f = h5py.File(token_h5ad_path, "r")
    token_patch_codes = f["obs/Patch_ID/codes"][:]
    token_patch_cats = f["obs/Patch_ID/categories"].asstr()[:]
    token_X = f["obs/Token_X"][:]
    token_Y = f["obs/Token_Y"][:]
    token_dim = f["X"].shape[1]

    # tokens are stored contiguously per patch (verified against this project's
    # files) -- group into contiguous row-ranges so each patch's 256 tokens are
    # read from disk with a single slice, not 256 individual row reads
    order = np.argsort(token_patch_codes, kind="stable")
    sorted_codes = token_patch_codes[order]
    uniq_codes, start_idx = np.unique(sorted_codes, return_index=True)
    groups = np.split(order, start_idx[1:])
    patch_token_rows = {}
    for code, g in zip(uniq_codes, groups):
        g_sorted = np.sort(g)
        assert g_sorted[-1] - g_sorted[0] + 1 == len(g_sorted), \
            f"token rows for patch {token_patch_cats[code]} aren't contiguous -- distillation assumes they are"
        patch_token_rows[token_patch_cats[code]] = (int(g_sorted[0]), int(g_sorted[-1]) + 1)

    nuclei = nuclei_geom_slide.set_index(nuclei_geom_slide["id"].astype(str))["geometry"]

    cell_ids = trans_adata_slide.obs["id"].astype(str).values
    patch_ids = trans_adata_slide.obs[patch_col].astype(str).values
    n_cells = len(cell_ids)

    footprint = np.zeros((n_cells, token_dim), dtype=np.float32)
    halo = np.zeros((n_cells, token_dim), dtype=np.float32)
    matched = np.zeros(n_cells, dtype=bool)

    cell_idx_by_patch = pd.Series(np.arange(n_cells)).groupby(patch_ids).apply(lambda s: s.values).to_dict()

    n_patches = len(cell_idx_by_patch)
    for pi, (patch_id, cell_rows) in enumerate(cell_idx_by_patch.items()):
        if verbose and pi % 2000 == 0:
            print(f"  [token distillation] patch {pi}/{n_patches}")

        row_range = patch_token_rows.get(patch_id)
        if row_range is None:
            continue  # this patch has no token data

        lo, hi = row_range
        tok_x = token_X[lo:hi]
        tok_y = token_Y[lo:hi]
        tok_emb = f["X"][lo:hi].astype(np.float32)
        origin_x, origin_y = tok_x.min(), tok_y.min()
        tok_col = np.round((tok_x - origin_x) / TOKEN_SIZE_PX).astype(int)
        tok_row = np.round((tok_y - origin_y) / TOKEN_SIZE_PX).astype(int)
        if positional_bias is not None:
            tok_emb = tok_emb - positional_bias[tok_row, tok_col]

        for ci in cell_rows:
            cid = cell_ids[ci]
            if cid not in nuclei.index:
                continue
            geom = nuclei.loc[cid]
            if geom is None or geom.is_empty:
                continue

            halo_poly = geom.buffer(halo_px)
            halo_ring = halo_poly.difference(geom)

            minx, miny, maxx, maxy = halo_poly.bounds
            col_lo, col_hi, row_lo, row_hi = _candidate_token_range(
                minx, miny, maxx, maxy, origin_x, origin_y, margin_tokens
            )
            cand = np.nonzero(
                (tok_col >= col_lo) & (tok_col <= col_hi) & (tok_row >= row_lo) & (tok_row <= row_hi)
            )[0]
            if len(cand) == 0:
                continue
            matched[ci] = True

            fw = np.empty(len(cand))
            hw = np.empty(len(cand))
            for j, ti in enumerate(cand):
                tbox = shapely_box(tok_x[ti], tok_y[ti], tok_x[ti] + TOKEN_SIZE_PX, tok_y[ti] + TOKEN_SIZE_PX)
                fw[j] = tbox.intersection(geom).area
                hw[j] = tbox.intersection(halo_ring).area

            if fw.sum() > 0:
                w = fw / fw.sum()
                footprint[ci] = (tok_emb[cand] * w[:, None]).sum(axis=0)
            else:
                # fallback: nucleus fell entirely between candidate token boxes
                # (shouldn't happen given the margin, but stay defensive) --
                # use the single nearest token by center distance
                cx, cy = geom.centroid.x, geom.centroid.y
                d2 = (tok_x[cand] + TOKEN_SIZE_PX / 2 - cx) ** 2 + (tok_y[cand] + TOKEN_SIZE_PX / 2 - cy) ** 2
                footprint[ci] = tok_emb[cand[np.argmin(d2)]]

            if hw.sum() > 0:
                w = hw / hw.sum()
                halo[ci] = (tok_emb[cand] * w[:, None]).sum(axis=0)
            else:
                # ring fully covered by the footprint itself (dense tissue) --
                # neutral fallback rather than an artificial zero vector
                halo[ci] = footprint[ci]

    f.close()
    return footprint, halo, matched


def run(slide, basedir=Path("../../data"), halo_px=15.0, out_name="cell_token_embeddings.h5ad",
        token_filename="WSI_patch_token_embeddings_filtered.h5ad", correct_positional_bias=True):
    trans_path = basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning/count_area_filtered_adata.h5ad"
    nuclei_path = basedir / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning/nuclei_geometry.parquet"
    token_path = basedir / slide / "01_preprocessing/unimodel/metadata" / token_filename
    out_path = basedir / slide / "01_preprocessing/unimodel/metadata" / out_name

    print(f"[{slide}] loading transcriptomics (backed) + nuclei geometry...")
    trans_adata = ad.read_h5ad(trans_path, backed="r")
    trans_slide = trans_adata[trans_adata.obs["Slide_ID"] == slide].to_memory()

    nuclei = gpd.read_parquet(nuclei_path)
    if "Slide_ID" in nuclei.columns:
        nuclei = nuclei[nuclei["Slide_ID"] == slide]

    positional_bias = None
    if correct_positional_bias:
        print(f"[{slide}] computing per-(row,col) positional bias from {token_path.name}...")
        positional_bias = compute_positional_bias(token_path)

    print(f"[{slide}] {trans_slide.n_obs} cells, {len(nuclei)} nucleus polygons, "
          f"distilling from {token_path.name} (halo_px={halo_px}, "
          f"positional_bias_correction={correct_positional_bias})...")
    footprint, halo, matched = compute_cell_token_features(
        trans_slide, nuclei, token_path, halo_px=halo_px, positional_bias=positional_bias,
    )
    print(f"[{slide}] matched {matched.sum()}/{len(matched)} cells to token data "
          f"({matched.mean():.1%})")

    # X left empty (0 vars) -- footprint/halo live in obsm only, so neither is
    # stored twice (an earlier version of this script set X=footprint AND
    # obsm["X_token_footprint"]=footprint, silently doubling the footprint
    # array's storage for no reason).
    out = ad.AnnData(
        X=np.empty((len(matched), 0), dtype=np.float32),
        obs=pd.DataFrame({
            "Slide_ID": trans_slide.obs["Slide_ID"].values,
            "Patch_ID": trans_slide.obs["Patch_ID"].values,
            "id": trans_slide.obs["id"].values,
            "matched": matched,
        }, index=trans_slide.obs_names),
    )
    out.obsm["X_token_footprint"] = footprint
    out.obsm["X_token_halo"] = halo
    out.uns["halo_px"] = halo_px
    out.uns["token_size_px"] = TOKEN_SIZE_PX
    out.uns["positional_bias_corrected"] = correct_positional_bias

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_path)
    print(f"[{slide}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", required=True, help="Slide_ID, e.g. CRC_P1")
    parser.add_argument("--halo-px", type=float, default=15.0,
                         help="halo ring dilation in pixels. Was 6.0 (nucleus[~21px]+halo "
                              "~33px wide, only ~2-3 tokens[14px] contributing per cell -- too "
                              "few to average out per-cell noise, confirmed via conversation-notes "
                              "diagnostics: raw pre-autoencoder footprint/halo features already show "
                              "fine-scale spatial speckle in cluster assignment). 15.0 widens the "
                              "window to ~51px (~3.6 tokens) for better SNR, at the cost of pulling "
                              "in more neighboring-cell signal (median nearest-cell spacing ~27px, "
                              "so halo rings now commonly extend into neighboring cells' territory "
                              "-- an accepted tradeoff, not an oversight)")
    parser.add_argument("--basedir", type=Path, default=Path("../../data"))
    parser.add_argument("--no-bias-correction", action="store_true",
                         help="skip the per-(row,col) positional-bias correction (on by default) -- "
                              "e.g. to reproduce the older, uncorrected cell_token_embeddings.h5ad "
                              "for comparison")
    args = parser.parse_args()
    run(args.slide, basedir=args.basedir, halo_px=args.halo_px,
        correct_positional_bias=not args.no_bias_correction)
