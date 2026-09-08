"""
Throwaway utility: recompresses one or more existing h5ad files in place with
gzip, for files that were already written without compression (see
functions/merge.py's own write call for the same fix applied at generation time
-- this script is for files that already exist and won't be regenerated).

Loads each file fully into memory (anndata.read_h5ad, not backed), so needs
enough RAM for the largest file passed in -- fine on a cluster notebook/job with
plenty of RAM, risky on a small local machine. Writes to a .tmp sibling path
first and atomically renames over the original only once the write succeeds, so
a crash or OOM mid-write never corrupts the original file.

Does NOT touch dtypes -- only adds compression. If a file's X is float64 when it
only needs to hold counts, that's a separate, per-file decision (see
functions/merge.py's X.astype(np.float32) guard for the one place this project
already does that automatically).

Usage:
    python recompress_h5ad.py /path/to/file1.h5ad /path/to/file2.h5ad ...
    python recompress_h5ad.py --glob "data/*/01_preprocessing/unimodel/metadata/WSI_patch_token_embeddings_filtered.h5ad"
"""
import argparse
import glob as globmod
import time
from pathlib import Path

import anndata as ad

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paths", nargs="*", help="h5ad file(s) to recompress in place")
parser.add_argument("--glob", action="append", default=[],
                     help="glob pattern(s) to expand into paths, repeatable. Resolved against the "
                          "current directory first; if that finds nothing and the pattern starts "
                          "with 'data/', also tried against the repo root (found by walking up from "
                          "this script), so 'data/...' patterns work regardless of your cwd.")
parser.add_argument("--dry-run", action="store_true", help="just print what would be done, don't write anything")
args = parser.parse_args()


def _resolve_glob(pattern):
    matches = sorted(globmod.glob(pattern))
    if matches or not pattern.startswith("data/"):
        return matches
    # Fallback: try relative to the repo root (parent of the first ancestor
    # directory containing a 'data/' folder), same upward-search convention
    # every other script in this project uses -- avoids the pattern only
    # working when run from one specific cwd.
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "data").is_dir():
            return sorted(globmod.glob(str(_candidate / pattern)))
    return []


paths = [Path(p) for p in args.paths]
for pattern in args.glob:
    paths.extend(Path(p) for p in _resolve_glob(pattern))
paths = sorted(set(paths))

if not paths:
    parser.error("No paths given -- pass file paths and/or --glob patterns")

for path in paths:
    if not path.exists():
        print(f"SKIP (not found): {path}")
        continue
    size_before = path.stat().st_size
    print(f"{path}  ({size_before / 1e9:.2f} GB before)")
    if args.dry_run:
        continue

    t0 = time.time()
    adata = ad.read_h5ad(path)
    tmp_path = path.with_suffix(".tmp.h5ad")
    adata.write(tmp_path, compression="gzip")
    del adata
    tmp_path.replace(path)  # atomic -- original untouched until this succeeds

    size_after = path.stat().st_size
    print(f"  -> {size_after / 1e9:.2f} GB after ({100 * (1 - size_after / size_before):.0f}% smaller), "
          f"{time.time() - t0:.0f}s")

print("\nDone." if not args.dry_run else "\nDry run -- nothing written.")
