"""
Run ConGI (contrastive gene-expression/histology spatial-domain clustering)
for one sample, pooled onto a coarse square grid instead of per-nucleus.

Why pooled, unlike SpaGCN (which runs at full nucleus resolution): ConGI's
own code assumes Visium-scale input (a few thousand spots per slide) in three
separate places --
  1. dataset.py holds every spot's image patch in one in-memory array,
  2. training runs a full DenseNet121 forward+backward pass per spot per
     epoch (250 epochs by default),
  3. utils.refine() builds a dense n x n pandas DataFrame of pairwise
     spatial distances --
all three would replicate (2 and 3 would badly exceed) the exact O(n^2)/
compute blowups this project's SpaGCN baseline needed real memory-efficiency
surgery for, except here there is no straightforward in-place fix: (1) and
(2) are fundamentally per-spot CNN passes, not a vectorizable array op. So
rather than adapting ConGI's compute pattern, this script adapts the INPUT:
nuclei are pooled onto a coarse grid (--bin-um, default 100 -- matching
classic Visium's ~100um spot pitch) before ConGI ever sees them, via
functions/label_transfer.py's aggregate_expression_to_grid (already used
elsewhere in this project, e.g. 01_preprocessing/transcriptomics/
bin_preprocessing.py, for exactly this "coarser square grid from the base
002um bins" operation -- reused here rather than reimplemented). At a few
thousand pooled bins per sample, all three of the above are simply cheap
enough as-is; nothing needed patching.

Two more small vendored-code fixes, done the same way SpaGCN's monkeypatches
were (fixing behavior from this script rather than editing site-packages):
  - metrics.mclust_R hardcodes the original authors' own R_HOME/R_USER
    paths -- overridden here to this conda env's actual R install via
    sys.prefix, so it works wherever this env happens to live.
  - ConGI's own Dataset class only knows how to load a few named public
    datasets (SpatialLIBD/Her2st/Mouse_brain_anterior/IDC) -- this script
    defines its own Dataset subclass for our pooled-grid data instead of
    editing dataset.py, following the same "IDC" pattern upstream already
    uses for a dataset with no ground-truth labels.

One convenient property of this baseline vs. SpaGCN: mclust's clustering
(the only step that depends on --n-clusters) runs on the ALREADY-TRAINED
embedding, so a single train() covers every --n-clusters value swept --
unlike SpaGCN, which needs a full resolution-search + retrain per
n_clusters value.

Requires (already set up in the `congi` conda env on this machine):
  torch 1.7.1 (CPU-numpy-incompatible, must match ConGI's vendored code),
  a working libvips (for cropping the WSI -- same as SpaGCN), and R + the
  mclust package + rpy2 (`conda install -c conda-forge r-mclust rpy2` gets
  all three at once). The DenseNet121/ResNet50 ImageNet backbone weights are
  downloaded automatically by torchvision on first use IF this machine has
  internet access at that moment -- pre-fetch with:
      python -c "from torchvision.models import densenet121; densenet121(pretrained=True)"
  then copy ~/.cache/torch/hub/checkpoints/densenet121-*.pth to the same
  path on any machine (e.g. cluster nodes) without internet access.

Usage:
    python run_congi_grid.py --sample CRC_P2 --n-clusters-min 7 --n-clusters-max 10
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchtoolbox.transform import Cutout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `functions`
from functions.label_transfer import aggregate_expression_to_grid

# utils.py does `from skimage.feature import graycomatrix, graycoprops` (used
# only inside extract_wash_patches, which this script never calls), but the
# pinned skimage==0.18.3 in this env only has the old British-spelling names
# (greycomatrix/greycoprops -- renamed in a later skimage than ConGI's own
# requirements.txt actually pins). That's a plain import-time crash before
# any of our code runs, so it can't be monkeypatched after the fact the way
# SpaGCN's fixes were -- inject the alias into skimage.feature's own
# namespace first instead, same spirit as the scipy `.A` shim in
# run_spagcn_svg.py.
import skimage.feature as _skimage_feature
if not hasattr(_skimage_feature, "graycomatrix"):
    _skimage_feature.graycomatrix = _skimage_feature.greycomatrix
    _skimage_feature.graycoprops = _skimage_feature.greycoprops

# adata_preprocess_pca (below) calls scanpy's pp.pca with its own default
# svd_solver='arpack', passed explicitly to sklearn's PCA every call (so
# patching sklearn.decomposition.PCA's own __init__ default would be a
# no-op here -- scanpy's default has to be intercepted instead). Confirmed
# flaky on our pooled-grid data: the exact same CRC_P2 gene matrix raised
# "ArpackError: Could not build an Arnoldi factorization" in one run and
# converged fine in an otherwise-identical isolated repro, with no seed/
# import-order difference found that reliably reproduces either outcome --
# consistent with the well-documented non-reproducibility of multi-threaded
# BLAS reduction order inside ARPACK's Arnoldi iteration, which can tip a
# near-degenerate matrix (ours: a handful of very low-count, edge-of-tissue
# bins even after --umi-cutoff) from converging to not, run to run.
# svd_solver='randomized' (sklearn's randomized_svd, no Arnoldi iteration at
# all) is not subject to this and was confirmed to succeed on the exact
# matrix that broke 'arpack'.
_orig_sc_pp_pca = sc.pp.pca


def _pca_default_randomized(data, n_comps=None, zero_center=True, svd_solver="randomized", **kwargs):
    return _orig_sc_pp_pca(data, n_comps=n_comps, zero_center=zero_center, svd_solver=svd_solver, **kwargs)


sc.pp.pca = _pca_default_randomized

import metrics as congi_metrics
from utils import adata_preprocess_pca, calculate_adj_matrix, refine
from dataset import GeneTransforms
import model as congi_model


class _NoOpSummaryWriter:
    """
    TrainerSpaCLR.__init__ unconditionally creates two real
    torch.utils.tensorboard.SummaryWriter instances, whose first real write
    pulls in tensorboard's full tensorflow_stub (absl-py, grpcio, markdown,
    werkzeug, google-auth, ...) -- none of which this script's output
    (the per-n_clusters parquet files) depends on. Stubbed out rather than
    installing a pile of otherwise-unused dependencies just to satisfy an
    import chain.
    """

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass


congi_model.SummaryWriter = _NoOpSummaryWriter
from model import SpaCLR, TrainerSpaCLR


def _mclust_r_env_fixed(x, n_clusters, model="EEE", random_seed=2020):
    # Same math as the original metrics.mclust_R, but R_HOME points at
    # sys.prefix (this conda env's own R), not the original authors'
    # hardcoded '/GPUFS/sysu_ydyang_10/...' path, which doesn't exist here.
    os.environ["R_HOME"] = str(Path(sys.prefix) / "lib" / "R")

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    rmclust = robjects.r["Mclust"]

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(x), n_clusters, model)
    return np.array(res[-2]).astype(int) - 1


# metrics.eval_mclust_ari calls mclust_R as a bare name resolved from
# metrics.py's own module globals at call time (same pattern as SpaGCN's
# spg.util.search_l -> calculate_p) -- patching the module attribute, not a
# local import, is what actually reaches it. Not used directly by this
# script (mclust_R is called straight from the loop below instead), but kept
# patched in case any vendored code path still goes through eval_mclust_ari.
congi_metrics.mclust_R = _mclust_r_env_fixed


def seed_torch(seed):
    # Same as train.py's own seed_torch(1), called at the same point (first
    # thing in process_sample, before any data loading) -- kept for the same
    # reproducibility reason upstream has it. (A flaky ARPACK convergence
    # failure hit during development turned out to be caused by
    # multi-threaded BLAS non-determinism, not RNG state -- see the
    # OMP_NUM_THREADS etc. comment in run_congi_grid.sbatch for the actual
    # fix -- so this alone would not have been sufficient for that.)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Path to the standalone libvips env (see run_congi_grid.sbatch) -- override
# with the CONGI_LIBVIPS_ENV env var if it lives somewhere else (e.g. this
# machine vs. the cluster: /home/charlottewolf/miniconda3/envs/_libvips_only
# vs. /data/wv377102/miniconda3/envs/_libvips_only).
LIBVIPS_ENV = os.environ.get(
    "CONGI_LIBVIPS_ENV",
    str(Path(sys.prefix).parent / "_libvips_only"),
)


def crop_wsi_to_bins(dir_base_input, pxl_col, pxl_row, img_size):
    """
    Crop the sample's WSI to the bounding box of the pooled bins' pixel
    centers and slice out one (2*img_size, 2*img_size) patch per bin.

    Delegates the actual work to _crop_wsi_worker.py, run with the
    STANDALONE _libvips_only env's OWN python interpreter (which has its own
    self-consistent numpy + pyvips + libvips, nothing from `congi` at all) --
    confirmed necessary, not just tidy, after two narrower attempts both
    failed:
      - this env's own libvips is ABI-broken (same class of issue SpaGCN's
        pyvips hit), so `congi` can't just dlopen it directly;
      - running the worker with `congi`'s own python + LD_LIBRARY_PATH
        pointed at the standalone env's libvips looked like it should work
        (an isolated one-liner test with pyvips as the very first import
        succeeded), but starting congi's python interpreter itself already
        resident-loads its own (older) libcrypto before user code runs, so
        the subprocess variant hit the same OPENSSL_3.3.0 symbol-version
        conflict every time regardless of import order;
      - setting that LD_LIBRARY_PATH for the whole TRAINING process instead
        (rather than a subprocess at all) crashed with "free(): invalid
        pointer" partway through a real run -- it shadows
        glibc/libstdc++/libgomp process-wide with a different build than the
        one torch/CUDA already linked against, corrupting the heap once both
        are exercised together.
    Using the standalone env's own interpreter sidesteps all three: nothing
    from `congi`'s runtime is ever loaded into the process that touches vips.
    """
    # Prefer .btf when present: some sample dirs (e.g. CRC_P2) hold both the
    # full-res .btf AND a small downsampled .tif preview shipped alongside it
    # in the original 10x bundle -- .tif/.tiff is only the real WSI for
    # samples that ship no .btf at all (e.g. CRC_P3_SMAP/CRC_P4_SMAP's
    # "HE.tif"), so only fall back to it when no .btf exists.
    btf_candidates = sorted(set(dir_base_input.glob("*.btf")))
    wsi_candidates = btf_candidates or sorted(set(
        p for pattern in ("*.tif", "*.tiff") for p in dir_base_input.glob(pattern)
    ))
    if len(wsi_candidates) != 1:
        raise RuntimeError(f"Expected exactly one .btf, or exactly one .tif/.tiff if no .btf exists, "
                            f"in {dir_base_input}, found {wsi_candidates}")
    f_wsi = wsi_candidates[0]
    print("Using WSI:", f_wsi, flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pxl_npy = Path(tmpdir) / "pxl.npy"
        out_npy = Path(tmpdir) / "patches.npy"
        np.save(pxl_npy, np.column_stack([pxl_col, pxl_row]))

        worker_python = str(Path(LIBVIPS_ENV) / "bin" / "python")
        subprocess.run(
            [worker_python, str(Path(__file__).resolve().parent / "_crop_wsi_worker.py"),
             "--wsi", str(f_wsi), "--pxl-npy", str(pxl_npy),
             "--img-size", str(img_size), "--out-npy", str(out_npy)],
            check=True,
        )
        return np.load(out_npy)


class GridDataset(Dataset):
    """
    ConGI's own Dataset class only knows a handful of named public datasets
    (SpatialLIBD/Her2st/Mouse_brain_anterior/IDC); this follows the same
    shape (see its "IDC" branch, which likewise has no ground-truth labels)
    for our pooled-grid nuclei data instead of editing dataset.py.
    """

    def __init__(self, gene, image, spatial, img_size, train=True):
        self.gene = gene.astype(np.float32)
        self.image = image
        self.spatial = spatial
        self.label = np.zeros(len(gene), dtype=int)  # no ground truth, same as ConGI's IDC/Mouse_brain branches
        self.n_clusters = 1  # unused by this script (mclust n_clusters passed explicitly instead)
        self.n_pos = int(self.spatial.max()) + 1
        self.train = train

        self.img_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.img_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gene_train_transform = GeneTransforms(
            gene.shape[1],
            prob_mask=0.5, pct_mask=0.2,
            prob_noise=0.5, pct_noise=0.8, sigma_noise=0.5,
            prob_swap=0.5, pct_swap=0.1,
        )

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        spatial = torch.from_numpy(self.spatial[idx])
        y = self.label[idx]

        if self.train:
            xg = self.gene[idx]
            xg_u = self.gene_train_transform(deepcopy(xg))
            xg_v = self.gene_train_transform(deepcopy(xg))

            xg = torch.from_numpy(xg)
            xg_u = torch.from_numpy(xg_u)
            xg_v = torch.from_numpy(xg_v)

            xi_u = self.img_train_transform(self.image[idx])
            xi_v = self.img_train_transform(self.image[idx])

            return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx
        else:
            xg = torch.from_numpy(self.gene[idx])
            xi = self.img_test_transform(self.image[idx])
            return xg, xi, spatial, y, idx


def process_sample(sample, args):
    seed_torch(args.seed)
    basedir = Path("../../data")
    dir_base_input = basedir / sample / "input"
    group_size = round(args.bin_um / 2)

    print(f"\n{'=' * 70}\nSample: {sample} (grid: {group_size * 2}um bins, group_size={group_size})\n{'=' * 70}",
          flush=True)

    grid_adata = aggregate_expression_to_grid(sample, group_size=group_size, data_dir=basedir)

    # A handful of individual 002um bins carry negative pxl_row_in_fullres
    # (a Space Ranger fiducial-registration edge artifact -- confirmed
    # directly on CRC_P2: 4,809 of 8.7M in-tissue bins, down to -1905px,
    # pxl_col unaffected) -- outside the WSI's actual pixel bounds, so they
    # have no real histology to crop. Individually harmless (mean-pooled
    # away in most grid cells), but a handful of pooled cells sit almost
    # entirely within this bad region and inherit a negative mean pixel
    # position -- same fix as bin_preprocessing.py's own "drop bins with no
    # histology counterpart" step, just at the pooled-grid granularity.
    valid = (grid_adata.obsm["spatial"][:, 0] >= 0) & (grid_adata.obsm["spatial"][:, 1] >= 0)
    if (~valid).sum():
        print(f"{sample}: dropping {(~valid).sum()} pooled bin(s) with out-of-bounds pixel "
              f"coords (no real histology there)", flush=True)
        grid_adata = grid_adata[valid].copy()

    # A separate handful of pooled bins can have near-zero total counts (a
    # coarse grid cell barely clipping the tissue edge) -- confirmed on
    # CRC_P2: totals as low as 2-9 vs. a max of 215,445. These are
    # numerically degenerate for adata_preprocess_pca's truncated SVD
    # (ARPACK fails to converge -- "Could not build an Arnoldi
    # factorization" -- on rows that are effectively noise), and are exactly
    # the kind of low-quality bin 01_preprocessing/transcriptomics/
    # bin_preprocessing.py already drops via the same --umi-cutoff (default
    # 80) before clustering -- reused here rather than inventing a new
    # threshold.
    total_counts = np.asarray(grid_adata.X.sum(axis=1)).ravel()
    low_count = total_counts < args.umi_cutoff
    if low_count.sum():
        print(f"{sample}: dropping {low_count.sum()} pooled bin(s) below --umi-cutoff={args.umi_cutoff}",
              flush=True)
        grid_adata = grid_adata[~low_count].copy()
    print(f"{sample}: {grid_adata.n_obs} pooled bins, {grid_adata.n_vars} genes", flush=True)

    gene = adata_preprocess_pca(grid_adata, pca_n_comps=args.n_gene).astype(np.float32)

    pxl_col = grid_adata.obsm["spatial"][:, 0]
    pxl_row = grid_adata.obsm["spatial"][:, 1]
    image = crop_wsi_to_bins(dir_base_input, pxl_col, pxl_row, args.img_size)

    spatial = grid_adata.obsm["spatial"].astype(np.int64)
    bin_ids = grid_adata.obs_names.tolist()

    # adj_2d (for the post-training refine() step) uses array_row/array_col,
    # not raw pixel coords -- same as ConGI's own SpatialLIBD branch
    # (model.py's TrainerSpaCLR.__init__) -- and is only O(bins^2), which at
    # grid-pooled scale (thousands, not hundreds of thousands) is a few tens
    # of MB, not the multi-hundred-GB problem this matrix is at nucleus
    # resolution.
    adj_2d = calculate_adj_matrix(
        x=grid_adata.obs["array_row"].tolist(), y=grid_adata.obs["array_col"].tolist(), histology=False,
    )

    trainset = GridDataset(gene, image, spatial, args.img_size, train=True)
    testset = GridDataset(gene, image, spatial, args.img_size, train=False)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    network = SpaCLR(
        gene_dims=[args.n_gene, 2 * args.last_dim], image_dims=[args.n_gene], p_drop=args.p_drop,
        n_pos=trainset.n_pos, backbone="densenet", projection_dims=[args.last_dim, args.last_dim],
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)

    dir_congi_data = basedir / sample / "02_baselines/congi/data"
    dir_congi_data.mkdir(parents=True, exist_ok=True)
    log_dir = str(dir_congi_data / "tb_logs")

    # TrainerSpaCLR.__init__ tries to build self.adj_2d/self.sample_id itself
    # from args.dataset (SpatialLIBD/Her2st only) -- we pass a dataset name
    # it doesn't recognize on purpose (neither branch fires, so those
    # attributes are simply left unset) and use our own adj_2d/bin_ids with
    # the refine() call below instead, rather than its
    # eval_mclust_refined_ari method.
    args.dataset = "NucleiGrid"
    trainer = TrainerSpaCLR(args, n_clusters=None, network=network, optimizer=optimizer,
                             log_dir=log_dir, device=args.device)
    trainer.fit(trainloader, args.epochs)

    xg, xi, _ = trainer.valid(testloader)
    z = xg + xi * 0.1

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for n_clusters in range(args.n_clusters_min, args.n_clusters_max + 1):
        print(f"\n--- {sample}: n_clusters={n_clusters} ---", flush=True)
        raw_preds = congi_metrics.mclust_R(z, n_clusters)
        num_nbs = 24 if len(bin_ids) > 1000 else 4
        refined_preds = refine(sample_id=bin_ids, pred=raw_preds, dis=adj_2d, num_nbs=num_nbs)

        out_df = pd.DataFrame({"id": bin_ids, "sample": sample, "n_clusters": n_clusters,
                                "pred": raw_preds, "refined_pred": refined_preds})
        out_path = outputs_dir / f"{sample}_cluster_{n_clusters}.{args.format}"
        if args.format == "csv":
            out_df.to_csv(out_path, index=False)
        else:
            out_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--n-clusters-min", type=int, required=True)
    parser.add_argument("--n-clusters-max", type=int, required=True)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")

    parser.add_argument("--bin-um", type=float, default=100,
                         help="Pooled bin pitch in microns -- classic Visium's own spot pitch is "
                              "~100um, used as the default so ConGI runs at roughly the spot "
                              "density it was designed for. Rounded to the nearest multiple of "
                              "the base 2um grid (group_size = round(bin_um / 2)).")
    parser.add_argument("--n-gene", type=int, default=300, choices=(300, 1000))
    parser.add_argument("--umi-cutoff", type=float, default=80,
                         help="Drop pooled bins with fewer total counts than this before PCA -- "
                              "same default as bin_preprocessing.py's own --umi-cutoff.")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--last-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--p-drop", type=float, default=0)
    parser.add_argument("--w-g2g", type=float, default=0.1)
    parser.add_argument("--w-i2i", type=float, default=0.1)
    parser.add_argument("--w-recon", type=float, default=0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1, help="Matches train.py's own seed_torch(1) default")

    args = parser.parse_args()
    if args.n_clusters_min > args.n_clusters_max:
        raise ValueError("--n-clusters-min must be <= --n-clusters-max")

    for sample in args.samples:
        process_sample(sample, args)


if __name__ == "__main__":
    main()
