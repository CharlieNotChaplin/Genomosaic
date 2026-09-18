"""
Extract per-cell scGPT embeddings for one sample, needed as SpatialFusion's
RNA-side input (z_rna). Mirrors 01_preprocessing/histology/unimodel/notebooks/
run_uni_new.py's per-sample-h5ad-output convention on the RNA side, so the
two modalities' outputs sit next to each other in the same shape of pipeline.

Why this needs its own script rather than calling scgpt.tasks.cell_emb.embed_data
directly inline: that function does `count_matrix = adata.X ... .toarray()` on
the WHOLE adata passed in, in one shot, before ever touching the GPU -- fine
for scGPT's own typical few-thousand-cell tutorials, but at this project's
nucleus-level counts (up to ~800k cells/sample) that dense array alone would
be tens of GB (e.g. CRC_P11mm: ~785k cells x 18085 genes x 4 bytes ~= 57GB).
This script chunks each sample into --chunk-size cells (default 20_000,
~1.4GB dense per chunk at this gene count) and calls embed_data once per
chunk, writing incrementally -- same "don't materialize the whole thing at
once" principle as SpaGCN's adjacency matrix / ConGI's patch array.

Not a double-normalization risk (checked directly against scGPT's own source,
scgpt/preprocess.py's binning()): scGPT bins each cell's nonzero expression
values into 51 QUANTILE bins computed from that cell's own nonzero values --
purely rank-based. count_area_filtered_adata.h5ad's normalize_total(1e4)+
log1p (both applied upstream by cell_preprocessing.py) are a per-cell scalar
rescale + a monotonic transform, neither of which changes the within-cell
RANK ORDER of nonzero genes -- so the resulting bins are identical whether
fed raw counts or this already-normalized data. Used as-is, unlike the
un-fixable double-normalization bug found in SpaGCN/ConGI (those methods'
own math is scale-sensitive, not rank-only).

Requires the `scgpt` env described in run_scgpt_embed.sbatch (torch 2.3.0 +
numpy<2 + scgpt + ipython -- a plain `pip install scgpt` does not work;
flash-attn deliberately NOT installed, see --use-fast-transformer) and the "whole-human" pretrained checkpoint (vocab.json, args.json,
best_model.pt) at --model-dir, default weights/scGPT_human/. Fetched via:
    gdown "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
(a plain public Google Drive folder -- no login needed). This is a ~200MB
binary and should NOT be committed to git; copy it to the same relative path
on any other machine (e.g. the cluster) by hand.

Usage:
    python run_scgpt_embed.py --samples CRC_P2 CRC_P5
"""
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import torch
from scgpt.tasks.cell_emb import embed_data


def load_sample_adata(dir_custom_binning, dir_combined_adata, sample):
    """Same fallback logic as run_spagcn_svg.py's load_sample_adata: prefer the
    per-sample count_area_filtered_adata.h5ad, fall back to filtering it out
    of the combined multi-sample file (per-sample files have since been
    deleted for every sample)."""
    per_sample_path = dir_custom_binning / "count_area_filtered_adata.h5ad"
    if per_sample_path.exists():
        return sc.read_h5ad(per_sample_path)
    if not dir_combined_adata.exists():
        raise FileNotFoundError(f"Neither {per_sample_path} nor {dir_combined_adata} exist")
    print(f"{per_sample_path} not found, loading {sample} out of {dir_combined_adata} instead", flush=True)
    combined = sc.read_h5ad(dir_combined_adata, backed="r")
    mask = (combined.obs["Slide_ID"] == sample).values
    if mask.sum() == 0:
        raise ValueError(f"No rows with Slide_ID == {sample!r} in {dir_combined_adata}")
    return combined[mask].to_memory()


def process_sample(sample, args):
    basedir = Path("../../data")
    dir_custom_binning = basedir / sample / "01_preprocessing" / args.cellseg_dir / args.bin_size / "custom_binning"
    dir_combined_adata = (basedir / "combined" / "01_preprocessing" / args.cellseg_dir / args.bin_size
                           / "custom_binning" / "count_area_filtered_adata.h5ad")
    dir_out = basedir / sample / "02_baselines" / "spatialfusion" / "data"
    dir_out.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_parquet) if args.out_parquet else dir_out / "scGPT.parquet"

    if out_path.exists() and not args.force_recompute:
        print(f"{out_path} already exists, skipping (pass --force-recompute to redo)", flush=True)
        return

    print(f"\n{'=' * 70}\nSample: {sample}\n{'=' * 70}", flush=True)
    adata = load_sample_adata(dir_custom_binning, dir_combined_adata, sample)
    adata.var_names_make_unique()
    if not adata.obs["id"].is_unique:
        raise ValueError(f"Nucleus IDs in {sample}'s adata are not unique.")
    adata.obs = adata.obs.assign(id=adata.obs["id"].astype(str)).set_index("id", drop=False)
    adata.var["feature_name"] = adata.var_names
    if args.max_cells:
        adata = adata[:args.max_cells].copy()

    print(f"{sample}: {adata.n_obs} cells, {adata.n_vars} genes -- "
          f"embedding in chunks of {args.chunk_size}", flush=True)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("WARNING: --device cuda requested but CUDA is not available, using CPU instead "
              "(will be very slow at this cell count).", flush=True)

    chunks = []
    for start in range(0, adata.n_obs, args.chunk_size):
        end = min(start + args.chunk_size, adata.n_obs)
        print(f"  cells [{start}:{end}) of {adata.n_obs}", flush=True)
        chunk = adata[start:end].copy()
        chunk = embed_data(
            chunk,
            model_dir=args.model_dir,
            gene_col="feature_name",
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            use_fast_transformer=args.use_fast_transformer,
        )
        chunks.append(pd.DataFrame(chunk.obsm["X_scGPT"], index=chunk.obs_names))
        del chunk
        if device == "cuda":
            torch.cuda.empty_cache()

    embed_df = pd.concat(chunks)
    embed_df.to_parquet(out_path)
    print(f"Wrote {out_path} {embed_df.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--model-dir", default="weights/scGPT_human",
                         help="Directory with vocab.json/args.json/best_model.pt "
                              "(the 'whole-human' checkpoint)")
    parser.add_argument("--bin-size", default="002um")
    parser.add_argument("--cellseg-dir", default="cell_seg_stardist")
    parser.add_argument("--chunk-size", type=int, default=20_000,
                         help="Cells per embed_data() call -- caps peak dense-array memory "
                              "(embed_data densifies its whole input at once). At 18085 genes, "
                              "20_000 cells is ~1.4GB dense; raise/lower to trade memory for "
                              "fewer, bigger GPU batches.")
    parser.add_argument("--batch-size", type=int, default=64, help="scGPT's own inference batch size")
    parser.add_argument("--max-length", type=int, default=1200, help="scGPT's own default max_length")
    parser.add_argument("--use-fast-transformer", action="store_true",
                         help="Use flash-attn instead of plain PyTorch attention. Off by default: "
                              "flash-attn is a fragile, CUDA-version-pinned build dependency only "
                              "needed for TRAINING speed, not inference -- this script only ever "
                              "does zero-shot embedding extraction, so it's deliberately not "
                              "installed/required. Only pass this if flash-attn is confirmed "
                              "installed in this env.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None,
                         help="Smoke-test only: embed just the first N cells. Requires --out-parquet "
                              "so a partial result can never land at the real scGPT.parquet path "
                              "(run_spatialfusion_range.py would silently use it as if complete).")
    parser.add_argument("--out-parquet", default=None,
                         help="Override the output path (default: "
                              "data/{sample}/02_baselines/spatialfusion/data/scGPT.parquet)")
    args = parser.parse_args()
    if args.max_cells and not args.out_parquet:
        parser.error("--max-cells requires --out-parquet")

    for sample in args.samples:
        process_sample(sample, args)


if __name__ == "__main__":
    main()
