"""
Standalone worker: crops a WSI to a set of pixel centers and writes the
resulting (N, 2*img_size, 2*img_size, 3) uint8 patch array to --out-npy.

Why a separate process rather than importing pyvips directly in
run_congi_grid.py: pyvips needs libvips from a standalone conda env (see
run_congi_grid.sbatch's LD_LIBRARY_PATH comment -- congi's own env can't
resolve a working libvips.so.42 without one). Setting that LD_LIBRARY_PATH
for the whole training process is NOT safe -- confirmed directly: it also
shadows glibc/libstdc++/libgomp for the entire process with a different
build than the one torch/CUDA already linked against, and crashed with
"free(): invalid pointer" (heap corruption from two different C runtimes)
partway through a real training run. Running the crop in its own short-lived
subprocess, with that LD_LIBRARY_PATH set only in the child's environment,
keeps the main process's runtime completely untouched.

Usage (invoked by run_congi_grid.py's crop_wsi_to_bins, not run directly):
    python _crop_wsi_worker.py --wsi <path> --pxl-npy <in.npy> --img-size 112 --out-npy <out.npy>
where in.npy is an (N, 2) float array of [pxl_col, pxl_row].
"""
import argparse
import math

import numpy as np
import pyvips


def crop_wsi_to_bins(f_wsi, pxl_col, pxl_row, img_size):
    xmin = max(0, math.floor(pxl_col.min()))
    xmax = math.ceil(pxl_col.max())
    ymin = max(0, math.floor(pxl_row.min()))
    ymax = math.ceil(pxl_row.max())
    print("Cropping WSI to bounding box:", xmin, xmax, ymin, ymax, flush=True)

    image = pyvips.Image.new_from_file(str(f_wsi), access="sequential")
    cropped = image.crop(xmin, ymin, xmax - xmin, ymax - ymin)
    del image
    numpy_array = np.ndarray(
        buffer=cropped.write_to_memory(),
        dtype=np.uint8,
        shape=[cropped.height, cropped.width, cropped.bands],
    )
    img = numpy_array[:, :, :3] if cropped.bands >= 3 else numpy_array
    del numpy_array, cropped
    # Padded by img_size on every side (reflect) before slicing -- confirmed
    # necessary, not defensive-for-no-reason: a bin can sit closer than
    # img_size to the tissue bounding box's own edge (seen directly on
    # CRC_P2), which without padding silently produces an out-of-bounds,
    # smaller-than-(2*img_size,2*img_size) slice via Python's negative-index
    # wraparound rather than an error.
    img = np.pad(img, ((img_size, img_size), (img_size, img_size), (0, 0)), mode="reflect")

    patches = []
    for x, y in zip(pxl_col, pxl_row):
        cx, cy = round(x - xmin) + img_size, round(y - ymin) + img_size
        patches.append(img[cy - img_size:cy + img_size, cx - img_size:cx + img_size].copy())
    return np.array(patches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--pxl-npy", required=True)
    parser.add_argument("--img-size", type=int, required=True)
    parser.add_argument("--out-npy", required=True)
    args = parser.parse_args()

    pxl = np.load(args.pxl_npy)
    patches = crop_wsi_to_bins(args.wsi, pxl[:, 0], pxl[:, 1], args.img_size)
    np.save(args.out_npy, patches)
    print(f"Wrote {args.out_npy} {patches.shape}", flush=True)


if __name__ == "__main__":
    main()
