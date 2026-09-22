"""
UMI-cutoff sensitivity across ALL slides and BOTH units (nuclei and 8um bins) -- replaces the single-subset analysis
in the evaluation chapter (Section "UMI-count cutoff sensitivity").

Three criteria, each applied per slide and unit to the total UMI count of every unit that has at least one UMI
(rows of nuclei_grouped_UMI_count.h5ad, i.e. the raw aggregated counts BEFORE any QC filtering):
  fixed  the cutoff actually used in preprocessing: total_counts > 80 (CustomBinning.ipynb / BinPreprocessing.ipynb).
  knee   elbow of the sorted-UMI ("barcode rank") curve, as in CustomBinning.ipynb: sort log1p(counts) descending,
         normalise rank and value to [0, 1], take the point farthest from the straight line joining the first and
         last point of the curve; its count is the cutoff.
  MAD    lower outlier bound on the log scale: expm1(median - 3 * MAD) of log1p(counts), MAD scaled to a standard
         deviation (scipy median_abs_deviation, scale="normal"), clipped at 0.
The counts are read by streaming the sparse matrix from disk (row sums), so nothing near the full matrix is held in RAM.

Outputs:
  <output-dir>/umi_cutoff_sensitivity.csv      one row per (unit, slide): n_units, median UMI, cutoffs, units kept
  <figure>                                     fraction of units kept as a function of the cutoff, one line per slide,
                                               per unit, with the fixed / knee / MAD cutoffs marked
  Tables are generated from the CSV by make_thesis_tables.py.

Usage (from 04_eval/; I/O bound, ~10-20 min):
    python -u umi_cutoff_sensitivity.py
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import median_abs_deviation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEDIR = REPO_ROOT / "data"
FILES = {
    "nuclei": BASEDIR / "combined/01_preprocessing/cell_seg_stardist/002um/custom_binning/nuclei_grouped_UMI_count.h5ad",
    "bins": BASEDIR / "combined/01_preprocessing/grid_bin/008um/custom_binning/nuclei_grouped_UMI_count.h5ad",
}
FIXED_CUTOFF = 80
SLIDES = ["CRC_P1", "CRC_P2", "CRC_P3_SMAP", "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5",
          "CRC_P6_5mm", "CRC_P11mm", "CRC_P11mm_2_cut"]
# The 11 mm frame of this slide holds three tissue samples; only the colorectal region is analysed (bin_preprocessing.py keeps
# 438,578 of the 1,290,617 native 8um bins, 283,676 after the UMI filter). This script's own raw-count loader instead falls
# back to aggregating the cropped 2um matrix (aggregate_expression_to_grid, via native_grid_is_consistent's region check in
# aggregate_grid_bin_UMI_count.py), which covers a genuinely smaller, already-cropped area and gives only 27,705 grid bins --
# NOT a stale/incomplete file: re-running the aggregation reproduces the same 27,705, since both loaders are correct for the
# region they cover. The two loaders would need to be reconciled to use the same region before this slide can be included.
INCOMPLETE = {("bins", "CRC_P11mm_2_cut"): "grid_bin nuclei_grouped_UMI_count.h5ad covers a smaller region than the trained model uses (27,705 vs. 438,578 in-scope bins)"}
DEFAULT_FIG = Path("/home/charlottewolf/Nextcloud/Arbeit UKA/MA/03_Ausarbeitung/thesis/figures/07_umi_cutoff_all_slides.png")


def row_sums(path, chunk_nnz=80_000_000):
    """Total counts per row of a CSR matrix stored in an .h5ad, streamed in chunks."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        indptr = X["indptr"][:]
        n = len(indptr) - 1
        out = np.zeros(n, dtype=np.float64)
        r0 = 0
        while r0 < n:
            r1 = int(np.searchsorted(indptr, indptr[r0] + chunk_nnz, side="right")) - 1
            r1 = min(max(r1, r0 + 1), n)
            data = X["data"][indptr[r0]:indptr[r1]].astype(np.float64)
            c = np.concatenate([[0.0], np.cumsum(data)])
            local = indptr[r0:r1 + 1] - indptr[r0]
            out[r0:r1] = c[local[1:]] - c[local[:-1]]  # cumsum difference: correct for empty rows too
            r0 = r1
    return out


def knee_cutoff(counts):
    s = np.sort(counts)[::-1]
    y = np.log1p(s)
    x = np.arange(len(s), dtype=np.float64)
    if len(s) < 3 or y.max() == y.min():
        return float(s[0]) if len(s) else np.nan
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    line = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    line /= np.linalg.norm(line)
    pts = np.column_stack([xn - xn[0], yn - yn[0]])
    dist = np.linalg.norm(pts - np.outer(pts @ line, line), axis=1)
    return float(s[int(np.argmax(dist))])


def mad_cutoff(counts, n_mads=3):
    lc = np.log1p(counts)
    return float(max(np.expm1(np.median(lc) - n_mads * median_abs_deviation(lc, scale="normal")), 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=BASEDIR / "combined/03_integration/autoencoders")
    ap.add_argument("--figure", type=Path, default=DEFAULT_FIG)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    for unit, path in FILES.items():
        print(f"{unit}: reading {path.name}", flush=True)
        slide_id = ad.read_h5ad(path, backed="r").obs["Slide_ID"].astype(str).to_numpy()
        totals = row_sums(path)
        for slide in SLIDES:
            c = totals[slide_id == slide]
            c = c[c > 0]
            if not len(c):
                continue
            if (unit, slide) in INCOMPLETE:
                print(f"  {unit}/{slide}: SKIPPED -- {INCOMPLETE[(unit, slide)]}", flush=True)
                continue
            knee, mad = knee_cutoff(c), mad_cutoff(c)
            row = {"unit": unit, "slide": slide, "n_units": len(c), "median_umi": float(np.median(c)),
                   "fixed_cutoff": FIXED_CUTOFF, "knee_cutoff": knee, "mad_cutoff": mad}
            for name, cut in (("fixed", FIXED_CUTOFF), ("knee", knee), ("mad", mad)):
                row[f"kept_{name}"] = int((c > cut).sum())
                row[f"frac_kept_{name}"] = float((c > cut).mean())
            rows.append(row)
            grid = np.unique(np.round(np.geomspace(1, max(c.max(), 2), 300)))
            s = np.sort(c)
            curves[(unit, slide)] = (grid, 1.0 - np.searchsorted(s, grid, side="right") / len(s))
            print(f"  {unit}/{slide}: n={len(c)} median={np.median(c):.0f} knee={knee:.0f} MAD={mad:.1f} "
                  f"kept fixed/knee/MAD = {row['frac_kept_fixed']:.1%}/{row['frac_kept_knee']:.1%}/{row['frac_kept_mad']:.1%}", flush=True)
        del totals
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "umi_cutoff_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, unit in zip(axes, FILES):
        for i, slide in enumerate(SLIDES):
            if (unit, slide) not in curves:
                continue
            g, frac = curves[(unit, slide)]
            ax.plot(g, frac, color=cmap(i), lw=1.2, label=slide)
            r = df[(df.unit == unit) & (df.slide == slide)].iloc[0]
            ax.plot([r.knee_cutoff], [r.frac_kept_knee], marker="^", color=cmap(i), ms=6, ls="")
            ax.plot([r.mad_cutoff if r.mad_cutoff > 0 else 1], [r.frac_kept_mad], marker="s", color=cmap(i), ms=5, ls="")
        ax.axvline(FIXED_CUTOFF, color="black", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("UMI cutoff (total counts per unit)")
        ax.set_title(f"{unit}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fraction of units kept")
    axes[0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("Fraction of units kept vs. UMI cutoff; dashed: fixed cutoff (> 80), triangles: knee, squares: MAD", fontsize=10)
    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    print("saved", args.figure, flush=True)


if __name__ == "__main__":
    main()
