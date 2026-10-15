# Histology scripts

Script versions of notebook steps that are slow or need a lot of RAM. See `01_preprocessing/README.md` for the full pipeline and the order of steps. Paths are resolved from the repo, so the scripts can be started from any folder.

## 01_extract_embeddings.py

Script version of the embedding section in `01_patch_creation_uni.ipynb`: runs the patches through UNI2-h and saves patch and token embeddings.

```bash
python 01_extract_embeddings.py                    # all samples except SKIP_SAMPLES
python 01_extract_embeddings.py --sample CRC_P1    # one sample, SKIP_SAMPLES ignored
```

| Option | Default | Meaning |
|---|---|---|
| `--sample` | all | single `Slide_ID` (sample folder name) |
| `--data_dir` | `<repo>/data` | folder with one subfolder per sample |
| `--weights_dir` | `histology/unimodel/weights` | must contain `pytorch_model.bin` (UNI2-h) |
| `--batch_size` | 16 | patches per forward pass, lower it if the GPU runs out of memory |
| `--num_workers` | 12 | CPU workers that load the PNGs |

- **Needs** (from `01_patch_creation_uni.ipynb`): `data/<sample>/01_preprocessing/unimodel/patches/*.png` and `data/combined/01_preprocessing/unimodel/metadata/WSI_patch_metadata.csv`.
- **Writes** per sample into `data/<sample>/01_preprocessing/unimodel/metadata/`: `WSI_patch_metadata.h5ad` (patch embeddings) and `WSI_patch_token_embeddings.h5ad` (token embeddings). The patch embeddings are also added to the combined `WSI_patch_metadata.h5ad`, old rows of a re-processed sample are replaced. Tokens are stored per sample only.
- **Skip list:** `SKIP_SAMPLES` at the top of the file lists samples that are already done. A sample without patches or without rows in the CSV is skipped with a message.
- A GPU is used if available, on CPU it is very slow.

## NucleiSegmentation.py

Segments nuclei on the H&E image with StarDist (`2D_versatile_he`). Provided by 10x Genomics. Run it before `preprocess.py --unit cell`. Needs the nuclei-segmentation environment (`nuclei-seg` or `stardist` in `Environments/`), not `autoencoder`.

```bash
python NucleiSegmentation.py \
    -i data/CRC_P1/input/<...>_tissue_image.btf \
    -s data/CRC_P1/output/binned_outputs/square_002um \
    -o data/CRC_P1/01_preprocessing/cell_seg_stardist/002um
```

| Option | Meaning |
|---|---|
| `-i` | full-resolution H&E image |
| `-s` | SpaceRanger folder with `filtered_feature_bc_matrix.h5` and `spatial/tissue_positions.parquet` |
| `-o` | output folder, the later steps expect `cell_seg_stardist/002um` |
| `-r1 -r2 -c1 -c2` | optional crop (row min/max, column min/max), only used for the extra cropped outputs |

- **Writes** into `-o`: `polys_FullSection.pckl` (nucleus polygons, **the only file read by later steps**), `labels_FullSection.pckl` and `Nuclei_Barcode_Map.csv`. With the crop options also `img_Stardist.tif` and `img_rois_Stardist_Subset.zip`.
- The whole image is loaded and segmented at once, so it needs a lot of RAM and time on large slides.
