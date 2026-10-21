# Genomosaic

A joint transcriptomic-histology representation for Visium HD colorectal cancer sections: a graph
encoder for spatial transcriptomics and an MLP encoder for histology (UNI2-h) tokens, aligned by a
cell/bin-level contrastive loss, validated by leave-one-slide-out (LOSO) generalisation to unseen
patients rather than only same-slide splits.

## Pipeline

Four stages, run in order, each with its own README:

| Stage | What | README |
|---|---|---|
| `01_preprocessing/` | WSI patches + UNI2-h tokens; nucleus/bin transcriptomics, QC, SVGs | [01_preprocessing/README.md](01_preprocessing/README.md) |
| `02_baselines/` | Competing methods' outputs (SpaGCN, STAGATE, SpatialFusion), each in its own environment, scored later against the same references | - |
| `03_integration/` | Trains Genomosaic: per-slide LOSO folds, and the final model on the whole cohort | [03_integration/README.md](03_integration/README.md) |
| `04_eval/` | Evaluates LOSO generalisation and a new sample's predictions | [04_eval/README.md](04_eval/README.md) |

## Data

Not part of this repository. One folder per sample under `data/<sample>/`
(raw input, then each stage's own output subfolder); `data/combined/` holds the cohort-merged files every later stage actually reads. 
The exact raw layout each sample needs, and where to get the public
Visium HD data, is in [01_preprocessing/README.md](01_preprocessing/README.md#input-data).

The data is sourced from public repositories, nameley:
- 10x Genomics
   - https://www.10xgenomics.com/datasets?sort=publishedAt+DESC
   - https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc
- SPATCH
   - https://spatch.pku-genomics.org/#/dataset/visium
   - https://spatch.pku-genomics.org/#/dataset/visiumff

The ground truth can be found on Zenodo:
- https://zenodo.org/records/11402686?utm_source=chatgpt.com

## How to proceed

1. Create the `autoencoder` conda env (`Environments/autoencoder.yml`); nuclei segmentation needs its
   own StarDist/TensorFlow env, see `01_preprocessing/README.md`.
2. Place the raw data under `data/<sample>/` as described above.
3. Run `01_preprocessing/`, then (optionally, for comparison) `02_baselines/`, then `03_integration/`,
   then `04_eval/` -- each stage's README has the exact commands and what it expects from the one
   before it.
