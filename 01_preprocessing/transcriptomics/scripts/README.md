# Transcriptomics scripts

Script versions of `CustomBinning.ipynb` (cells) and `BinPreprocessing.ipynb` (8 µm grid bins). The steps are slow and need a lot of RAM on large slides, so use the scripts for whole samples and the notebooks for exploring cutoffs and plots. See `01_preprocessing/README.md` for the full pipeline. Paths are resolved from the repo, so the scripts can be started from any folder. Merging all samples and the global gene selection exist only in the notebooks.

Order per sample: `preprocess.py` → `get_svg.py`. `aggregate_grid_bin_UMI_count.py` is independent of both.

```bash
python preprocess.py --sample CRC_P1 --unit cell
python get_svg.py    --sample CRC_P1 --unit cell

# all samples
for s in CRC_P1 CRC_P2 CRC_P3_SMAP CRC_P4_SMAP CRC_P5 Normal_P3 Normal_P5 CRC_P6_5mm CRC_P11mm CRC_P11mm_2_cut; do
    python preprocess.py --sample $s --unit cell && python get_svg.py --sample $s --unit cell
done
```
`--unit` is `cell` (StarDist nuclei) or `bin` (8 µm grid bins). All scripts run in the `autoencoder` environment.

## preprocess.py

Loads the counts, applies QC, clusters and assigns every nucleus / bin to a histology patch (`Patch_ID`).

| Option | Default | Meaning |
|---|---|---|
| `--sample` | required | sample folder name |
| `--unit` | required | `cell` or `bin` |
| `--umi-cutoff` | 80 | keep nuclei / bins with total counts above this |
| `--nuclei-area-cutoff` | 1000 | `cell` only, drop nuclei with an area at or above this (px²) |
| `--group-size` | 4 | `bin` only, 2 µm bins per 8 µm bin side, used if there is no native 8 µm output |

- **Needs:** `output/binned_outputs/square_002um` (`cell`) or `square_008um` / `square_002um` (`bin`) of the sample. For `cell` also `01_preprocessing/cell_seg_stardist/002um/polys_FullSection.pckl` from `NucleiSegmentation.py`.
- **Patch assignment** needs `unimodel/metadata/filtered_WSI_patch_embeddings.h5ad` (histology step 3). Without it the step is skipped with a message (for `bin` all bins are kept) and the script has to be run again later. For `cell` the patch-assignment cells in `CustomBinning.ipynb` can add `Patch_ID` afterwards.
- **Writes** into `data/<sample>/01_preprocessing/`, existing files are overwritten:
  - `cell`: `cell_seg_stardist/002um/custom_binning/` with `filtered_custom_bins.h5ad`, `nuclei_grouped_UMI_count.h5ad`, `count_area_filtered_adata.h5ad`, `nuclei_geometry.parquet`
  - `bin`: `grid_bin/008um/custom_binning/count_area_filtered_adata.h5ad`

## get_svg.py

Finds spatially variable genes (Moran's I, 1000 permutations per gene). Run after `preprocess.py` of the same unit.

| Option | Default | Meaning |
|---|---|---|
| `--sample` | required | sample folder name |
| `--unit` | `cell` | `cell` or `bin` |
| `--n-jobs` | 20 | parallel workers, lower it if workers get killed for lack of memory (e.g. `CRC_P11mm` with about 1.5 M bins) |

- **Needs:** `count_area_filtered_adata.h5ad` from `preprocess.py` (for `cell` also `nuclei_geometry.parquet`).
- **Writes** into the same `count_area_filtered_adata.h5ad`: `var['moranI']`, `var['moranI_pval_fdr']`, `var['spatially_variable']` (FDR < 0.05 and I > 0.1). The merge in the notebooks reads these.

## aggregate_grid_bin_UMI_count.py

Writes the raw, unfiltered 8 µm grid counts of each sample (no QC) and merges them into the combined file. Used by `04_eval`.

```bash
python aggregate_grid_bin_UMI_count.py                              # all ten samples
python aggregate_grid_bin_UMI_count.py --samples CRC_P1 CRC_P2      # subset
```

- `--force` redoes a sample whose file already exists, otherwise finished samples are skipped, so a rerun after a crash only does the missing ones.
- **Writes** `data/<sample>/01_preprocessing/grid_bin/008um/custom_binning/nuclei_grouped_UMI_count.h5ad` and adds each sample to `data/combined/01_preprocessing/grid_bin/008um/custom_binning/nuclei_grouped_UMI_count.h5ad`.
- **RAM:** every merge loads the whole combined file, so memory grows with each sample (about 220 GB for all ten slides).
- Uses the native 8 µm output if its region matches the 2 µm data, otherwise it aggregates from 2 µm (e.g. `CRC_P11mm_2_cut`).
