"""
Bin-level analogue of token_preprocessing.py: distills raw ViT tokens into
per-8um-bin footprint+halo features, using functions.label_transfer's
aggregate_tokens_to_grid_footprint_halo -- exact rectangle-overlap-area
weighting (a bin and a raw token are both known axis-aligned squares) rather
than token_preprocessing.py's compute_cell_token_features (exact
polygon-overlap-area weighting, needed there since a nucleus is an irregular
shape). Same underlying idea (area-weighted contribution, not a coarser
membership test), specialized to rectangles since a bin has no polygon, just
a known center and fixed physical size.

Output matches cell_token_embeddings.h5ad's contract exactly (obsm
['X_token_footprint']/['X_token_halo'], obs Slide_ID/Patch_ID/id/matched) so
a later autoencoder-side integration is a path change, not a code change --
see functions/dual_autoencoder.py's build_pairing.

Usage:
    python bin_token_preprocessing.py --slide CRC_P1
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


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
from functions.label_transfer import aggregate_tokens_to_grid_footprint_halo, microns_per_pixel

GROUP_SIZE = 4  # 2um x 4 = 8um, matches BinPreprocessing.ipynb's bin_size="008um"


def run(slide, basedir=_find_data_dir(), halo_width_px=15.0,
        out_name="bin_token_embeddings.h5ad",
        trans_filename="count_area_filtered_adata.h5ad"):
    trans_path = basedir / "combined/01_preprocessing/grid_bin/008um/custom_binning" / trans_filename
    out_path = basedir / slide / "01_preprocessing/unimodel/metadata" / out_name

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
    print(f"[{slide}] {bin_slide.n_obs} bins, half_side_px={half_side_px:.2f} "
          f"halo_width_px={halo_width_px}")

    footprint, halo, matched = aggregate_tokens_to_grid_footprint_halo(
        slide, bin_slide.obsm["spatial"], half_side_px, halo_width_px, data_dir=basedir,
    )
    print(f"[{slide}] matched {matched.sum()}/{len(matched)} bins ({matched.mean():.1%})")

    out = ad.AnnData(
        X=np.empty((len(matched), 0), dtype=np.float32),
        obs=pd.DataFrame({
            "Slide_ID": bin_slide.obs["Slide_ID"].values,
            "Patch_ID": bin_slide.obs["Patch_ID"].values,
            "id": bin_slide.obs["id"].values,
            "matched": matched,
        }, index=bin_slide.obs_names),
    )
    out.obsm["X_token_footprint"] = footprint
    out.obsm["X_token_halo"] = halo
    out.uns["half_side_px"] = half_side_px
    out.uns["halo_width_px"] = halo_width_px

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_path)
    print(f"[{slide}] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", required=True, help="Slide_ID, e.g. CRC_P1")
    parser.add_argument("--halo-width-px", type=float, default=15.0,
                         help="same default as token_preprocessing.py's --halo-px, for direct comparability")
    parser.add_argument("--basedir", type=Path, default=_find_data_dir())
    args = parser.parse_args()
    run(args.slide, basedir=args.basedir, halo_width_px=args.halo_width_px)
