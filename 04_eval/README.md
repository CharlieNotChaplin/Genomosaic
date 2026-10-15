# 04_eval

Evaluates the model `03_integration` trains: how well held-out leave-one-slide-out (LOSO) folds
generalise, and what the final model says about a genuinely new sample. Three notebooks, no
model/GPU code of their own -- they load already-computed results or call the scripts in `scripts/`.

| Notebook | What |
|---|---|
| `EvaluatePreprocessing.ipynb` | QC checks on the `01_preprocessing` output (UMI/area/SVG, label transfer) |
| `EvaluateLoso.ipynb` | dashboard over the LOSO result CSVs: retrieval, clustering quality before/after both kinds of smoothing, disease prediction, baselines, embedding agreement |
| `EvaluateNewSample.ipynb` | clustering, smoothing, UMAP, CellTypist annotation and disease prediction for one sample outside the training cohort |

| Folder / file | What |
|---|---|
| `scripts/` | the actual evaluation scripts; `EvaluateLoso.ipynb` only reads what these write |
| `colors.txt` | the RWTH colour palette used throughout (`functions/plotting.py` has the same values by name) |
| `output/` | scratch figures/tables from ad hoc notebook runs, not a pipeline output |
| `logs/` | cluster job logs |

## What is done and why

**Preprocessing QC.** Sanity checks before trusting anything downstream: UMI/area distributions and
their filtering funnels, SVG burden and cross-slide reproducibility, a StarDist-vs-patch coordinate
check, label-transfer spatial maps. Independent of the model.

**LOSO evaluation (`scripts/`, read by `EvaluateLoso.ipynb`).** Each LOSO fold reports
cell-to-token retrieval directly (`loso_token_log_*.csv`, written by `run_loso.py` itself). Everything
else is scored from the saved fold embeddings afterwards:
- `evaluate_clustering_quality.py`: Leiden ARI/NMI against the label-transfer reference, raw and after
  cluster-*label* smoothing (`refine_domains`: majority vote or SpaGCN's rule).
- `evaluate_k_matched_ari.py`: the same, at a cluster-count-matched resolution (so comparing against
  baselines run at `k = 7..10` is fair), plus ARI/NMI before and after *embedding* smoothing
  (`--smooth-hops`: averaging each unit's embedding with its spatial neighbours before clustering --
  a different operation from label smoothing, scored separately on purpose), plus the baseline
  (SpaGCN/STAGATE/SpatialFusion) and embedding-agreement (z_trans vs. z_token vs. z_joint) numbers.
- `evaluate_disease_signal_weighted.py` / `evaluate_disease_signal_ground_truth.py`: tumour-vs-normal
  probes. The first trains and evaluates on the coarse slide-level label (every unit of a `CRC_` slide
  counted as diseased), with and without class re-weighting -- a diagnostic of the probe's own class
  imbalance, not a claim that the label itself is right. The second uses a real per-unit label
  throughout, training included: `CRC_P2`'s own pathologist annotation for its units, label transfer's
  per-unit prediction for the other training slides, evaluated against `CRC_P2`'s annotation on the
  held-out side. The slide-level label mislabels stroma, muscle and immune units inside a tumour slide
  as diseased (and vice versa for a unit of normal tissue picked up inside a tumour slide); this is
  the fix for that, on both the training and evaluation side.

**New-sample evaluation (`EvaluateNewSample.ipynb`).** No ground truth exists for a sample outside
the cohort, so this is prediction and visualisation, not scoring: Leiden clustering, the same
embedding-smoothing comparison as above, a UMAP, CellTypist cell-type annotation (on the sample's real
expression, not a reconstruction), and a disease-signal probe trained on the training cohort's own
saved embeddings and applied to the new sample. The disease probe's training labels follow the same
per-unit convention as `evaluate_disease_signal_ground_truth.py`: `CRC_P2`'s real annotation for its
own units, label transfer's per-unit prediction for every other training slide.

## Input data

Needs `03_integration`'s outputs:
- LOSO fold checkpoints (`run_loso.py`) for the preprocessing-QC and LOSO-evaluation notebooks/scripts
- the final model's `weights/dual_ae_latents.npz` and a new sample's own
  `03_integration/output/<sample>_<unit>_embeddings.npz` (`autoencoder.ipynb`, its last section) for
  `EvaluateNewSample.ipynb`

**Environment.** `autoencoder` conda env for everything; CellTypist additionally needs
`pip install celltypist` and its gut-atlas model cached once
(`celltypist.models.download_models(model="Cells_Intestinal_Tract.pkl")`).

## Outputs

Scripts write into `data/combined/03_integration/autoencoders/` (shared with `03_integration`, since
they score that folder's own saved embeddings): `clustering_quality.csv` (and `clustering_quality_summary.csv`),
`final_loso_kmatched_ari[_bins].csv`, `final_loso_partition_agreement[_bins].csv`,
`final_loso_baselines_ours_units.csv`, `final_loso_disease_signal_weighted.csv`,
`final_loso_disease_signal_ground_truth.csv`.

`EvaluatePreprocessing.ipynb` saves its figures to `output/`. `EvaluateNewSample.ipynb` only plots
inline (nothing saved to disk).

## How to run

| Step | What | Script |
|---|---|---|
| 1 | Preprocessing QC | `EvaluatePreprocessing.ipynb` |
| 2 | Clustering quality (raw + label-smoothed) | `scripts/evaluate_clustering_quality.py` |
| 3 | Matched-k ARI, embedding smoothing, baselines, embedding agreement | `scripts/evaluate_k_matched_ari.py [--smooth-hops 1 2 3] [--baselines]` |
| 4 | Disease-signal probes | `scripts/evaluate_disease_signal_weighted.py`, `scripts/evaluate_disease_signal_ground_truth.py` |
| 5 | LOSO results overview | `EvaluateLoso.ipynb` (after steps 2-4 have written their CSVs) |
| 6 | New-sample evaluation | `EvaluateNewSample.ipynb` (after `03_integration/autoencoder.ipynb`'s training + new-sample sections) |

Notes
- **Everything here is read-only with respect to the model.** No step here trains or fine-tunes
  anything; `evaluate_disease_signal_*` fits a small logistic-regression probe on frozen embeddings,
  nothing more.
- **Scripts are safe to re-run.** Each appends to its CSV and skips folds already present
  (`--overwrite` to redo), so a script can be run again any time more LOSO folds finish.
- **Large slides need real memory.** `CRC_P11mm` (~785k nuclei) reconstructing its held-out cells is
  the usual reason a step OOMs on a laptop; it runs fine with the cluster memory these scripts'
  (removed) sbatch wrappers used to request -- write a new one if running this on the cluster again.
- **Example commands** (from `04_eval/`):
```bash
python scripts/evaluate_clustering_quality.py --units nuclei --variants svg
python scripts/evaluate_k_matched_ari.py --smooth-hops 1 2 3 --baselines
python scripts/evaluate_disease_signal_ground_truth.py
```
