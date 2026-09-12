"""
Distills the raw per-patch ViT token grid (256 tokens x 1536-dim per 224px
patch, ~9GB/slide) down to two compact, per-UNIT feature vectors, where unit
is either a segmented nucleus ("nuclei") or a fixed-size spatial bin ("bins"):

  - footprint: area-weighted average of the token embeddings overlapping the
    unit's own footprint (a nucleus polygon, or a bin's known axis-aligned
    square).
  - halo: area-weighted average of the token embeddings overlapping a small
    ring around the unit (footprint dilated by `halo_px`, minus the
    footprint itself) -- captures immediate membrane/cytoplasm/microenvironment
    context that the footprint alone misses.

Kept as two separate vectors (rather than one blended average) so the model
can treat "this unit's own tissue" and "immediately around this unit" as
distinct signals -- see conversation notes (TRIPLEX's target/neighbor split).

Why per-unit distillation at all: the raw token file is one row per token
(256/patch), which is only tractable to keep on disk per-slide. Once every
unit has been resolved to its own footprint+halo vector, the result is one
row per unit (same order of magnitude as the existing transcriptomics/SVG
data), cheap enough to load whole across every slide at once.

Nuclei ("--unit nuclei") use exact polygon-overlap-area weighting
(compute_cell_token_features), needed since a nucleus is an irregular shape.
Bins ("--unit bins") use functions.label_transfer's exact
rectangle-overlap-area weighting (aggregate_tokens_to_grid_footprint_halo) --
a bin has no polygon, just a known center and fixed physical size, so both
the bin and a raw token are axis-aligned squares. Both branches write output
matching the same contract (obsm['X_token_footprint']/['X_token_halo'], obs
Slide_ID/Patch_ID/id/matched) so a later autoencoder-side integration is a
path change, not a code change -- see functions/dual_autoencoder.py's
build_pairing.

Coordinate system: cell/bin centroids (`trans_adata.obsm['spatial']`),
nucleus polygons (`nuclei_geometry.parquet`), and token positions
(`Token_X`/`Token_Y` in the token h5ad) all live in the same per-slide WSI
pixel space -- verified directly against this project's data (see
conversation notes). Patch linkage uses the existing `Patch_ID` string (e.g.
"patch_123.png"), which is shared between the transcriptomics obs and the
token obs for a given slide.

Usage:
    python token_preprocessing.py --unit nuclei --slide CRC_P1
    python token_preprocessing.py --unit bins --slide CRC_P1
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box

TOKEN_SIZE_PX = 14.0
TOKENS_PER_SIDE = 16
GROUP_SIZE = 4  # 2um x 4 = 8um, matches BinPreprocessing.ipynb's bin_size="008um"


def _find_data_dir():
    """Walks up from this file's own location for a `data/` sibling, robust to
    invocation directory -- rather than assuming a fixed relative depth from CWD."""
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "data").is_dir():
            return candidate / "data"
    raise RuntimeError("Could not find a 'data' directory in any ancestor of this script")


for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "functions").is_dir():
        sys.path.insert(0, str(_candidate))
        break
from functions.label_transfer import (
    aggregate_tokens_to_grid_footprint_halo,
    compute_token_positional_bias as compute_positional_bias,
    microns_per_pixel,
    resolve_token_h5ad_path,
)


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


def _write_output(slide, out_path, obs_names, slide_col, patch_col, id_col, matched, footprint, halo, extra_uns):
    out = ad.AnnData(
        X=np.empty((len(matched), 0), dtype=np.float32),
        obs=pd.DataFrame({
            "Slide_ID": slide_col,
            "Patch_ID": patch_col,
            "id": id_col,
            "matched": matched,
        }, index=obs_names),
    )
    out.obsm["X_token_footprint"] = footprint
    out.obsm["X_token_halo"] = halo
    for k, v in extra_uns.items():
        out.uns[k] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_path)
    print(f"[{slide}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def run_nuclei(slide, basedir, halo_px=15.0, out_name="cell_token_embeddings.h5ad",
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
    _write_output(
        slide, out_path, trans_slide.obs_names,
        trans_slide.obs["Slide_ID"].values, trans_slide.obs["Patch_ID"].values, trans_slide.obs["id"].values,
        matched, footprint, halo,
        extra_uns={
            "halo_px": halo_px,
            "token_size_px": TOKEN_SIZE_PX,
            "positional_bias_corrected": correct_positional_bias,
        },
    )


def run_bins(slide, basedir, halo_px=15.0, out_name="bin_token_embeddings.h5ad",
             trans_filename="count_area_filtered_adata.h5ad", correct_positional_bias=True):
    trans_path = basedir / "combined/01_preprocessing/grid_bin/008um/custom_binning" / trans_filename
    out_path = basedir / slide / "01_preprocessing/unimodel/metadata" / out_name
    token_path = resolve_token_h5ad_path(slide, data_dir=basedir)

    trans_adata = ad.read_h5ad(trans_path, backed="r")
    bin_slide = trans_adata[trans_adata.obs["Slide_ID"].astype(str) == slide].to_memory()
    if bin_slide.n_obs == 0:
        raise ValueError(f"No bins found for Slide_ID={slide!r} in {trans_path} -- run "
                          f"BinPreprocessing.ipynb (load+QC+merge) for this slide first.")

    grid_um = GROUP_SIZE * 2.0
    # Exact half-side of the bin's own known physical square footprint -- no fudge
    # factor needed (unlike main_grid's own radius_factor-tuned circular radius,
    # which was calibrated for a soft classifier feature, not an exact boundary).
    half_side_px = grid_um / (2 * microns_per_pixel(slide, data_dir=basedir))

    positional_bias = None
    if correct_positional_bias:
        print(f"[{slide}] computing per-(row,col) positional bias from {token_path.name}...")
        positional_bias = compute_positional_bias(token_path)

    print(f"[{slide}] {bin_slide.n_obs} bins, half_side_px={half_side_px:.2f} "
          f"halo_px={halo_px}, positional_bias_correction={correct_positional_bias}")

    footprint, halo, matched = aggregate_tokens_to_grid_footprint_halo(
        slide, bin_slide.obsm["spatial"], half_side_px, halo_px, data_dir=basedir,
        positional_bias=positional_bias,
    )
    print(f"[{slide}] matched {matched.sum()}/{len(matched)} bins ({matched.mean():.1%})")

    _write_output(
        slide, out_path, bin_slide.obs_names,
        bin_slide.obs["Slide_ID"].values, bin_slide.obs["Patch_ID"].values, bin_slide.obs["id"].values,
        matched, footprint, halo,
        extra_uns={
            "half_side_px": half_side_px,
            "halo_px": halo_px,
            "positional_bias_corrected": correct_positional_bias,
        },
    )


def run(unit, slide, basedir=None, halo_px=15.0, out_name=None, correct_positional_bias=True):
    basedir = basedir or _find_data_dir()
    kwargs = {"out_name": out_name} if out_name else {}
    if unit == "nuclei":
        run_nuclei(slide, basedir, halo_px=halo_px, correct_positional_bias=correct_positional_bias, **kwargs)
    elif unit == "bins":
        run_bins(slide, basedir, halo_px=halo_px, correct_positional_bias=correct_positional_bias, **kwargs)
    else:
        raise ValueError(f"unit must be 'nuclei' or 'bins', got {unit!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", required=True, choices=["nuclei", "bins"],
                         help="distill per-nucleus (cell_token_embeddings.h5ad) or per-bin "
                              "(bin_token_embeddings.h5ad) footprint+halo features")
    parser.add_argument("--slide", required=True, help="Slide_ID, e.g. CRC_P1")
    parser.add_argument("--halo-px", type=float, default=15.0,
                         help="halo ring dilation in pixels, same meaning/default for both units "
                              "(nucleus[~21px]+halo ~33px wide would only pull in ~2-3 tokens[14px] "
                              "per cell -- too few to average out per-cell noise, confirmed via "
                              "conversation-notes diagnostics: raw pre-autoencoder footprint/halo "
                              "features already show fine-scale spatial speckle in cluster "
                              "assignment. 15.0 widens the window to ~51px (~3.6 tokens) for better "
                              "SNR, at the cost of pulling in more neighboring-unit signal -- an "
                              "accepted tradeoff, not an oversight)")
    parser.add_argument("--basedir", type=Path, default=None)
    parser.add_argument("--out-name", default=None,
                         help="override the output filename (default: cell_token_embeddings.h5ad "
                              "for --unit nuclei, bin_token_embeddings.h5ad for --unit bins)")
    parser.add_argument("--no-bias-correction", action="store_true",
                         help="skip the per-(row,col) positional-bias correction (on by default "
                              "for both units) -- e.g. to reproduce the older, uncorrected "
                              "cell_token_embeddings.h5ad / bin_token_embeddings.h5ad for comparison")
    args = parser.parse_args()
    run(args.unit, args.slide, basedir=args.basedir, halo_px=args.halo_px, out_name=args.out_name,
        correct_positional_bias=not args.no_bias_correction)
