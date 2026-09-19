#!/bin/bash
#SBATCH --job-name=stagate_range
#SBATCH --output=logs/stagate_range_%j.out
#SBATCH --error=logs/stagate_range_%j.err
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=charlotte.wolf1@rwth-aachen.de

# Runs run_stagate_range.py. CPU-ONLY -- no GPU requested, and none is needed:
# STAGATE is sparse-graph attention + dense (n_nuclei x 2000 genes) matmuls,
# linear in the number of nuclei, no O(n^2) matrix like SpaGCN. Measured on a
# 14-thread laptop CPU (real CRC_P2 nuclei, unmodified STAGATE training code):
#   ~65-70 ms per 1,000 nuclei per epoch  (1.42 s/epoch at 20k, 3.9 s at 60k)
#   peak RSS ~= 1.2 GB + 120 KB per nucleus  (3.6 GB at 20k, 8.4 GB at 60k)
# Extrapolated to 500 epochs (the default) -- cluster CPUs may differ, so treat
# as a guide and check the tqdm rate in the .err log in the first minutes:
#   sample            nuclei    train time    peak RAM   --mem to request
#   CRC_P11mm_2_cut    89k      ~0.8 h        ~12 GB     24G
#   CRC_P1            120k      ~1.1-1.2 h    ~16 GB     24G
#   CRC_P4_SMAP       151k      ~1.3-1.5 h    ~19 GB     32G
#   CRC_P2            225k      ~2.0-2.2 h    ~28 GB     40G
#   CRC_P3_SMAP       296k      ~2.6-2.9 h    ~37 GB     50G
#   CRC_P11mm         786k      ~7-8 h        ~96 GB     140G
# plus the Leiden search for each cluster count afterwards (minutes at ~200k
# nuclei, roughly an hour at 786k -- extrapolated, not measured). The
# defaults here (64G / 12h) cover everything except CRC_P11mm; for that one
# pass e.g. --mem=140G --time=1-00:00:00 on the command line.
#
# Independent samples can all run at the same time as separate jobs (each
# writes only its own data/{sample}/02_baselines/stagate/ and its own
# per-(sample, n_clusters) parquet files):
#   for s in CRC_P1 CRC_P2 ...; do
#       sbatch run_stagate_range.sbatch --samples $s --n-clusters-min 7 --n-clusters-max 10
#   done
#
# Run from this directory (the script uses paths relative to it), and create
# logs/ first if it doesn't exist -- SLURM won't create it and the job then
# fails without writing any log.
#
# One-time environment setup (modern TensorFlow via the compat.v1 API the
# vendored code already imports -- no TF 1.15 / Python 3.7 needed):
#   mamba create -p <envs-dir>/stagate python=3.10 pip -y
#   conda activate stagate
#   pip install "tensorflow-cpu==2.15.1" "numpy==1.26.4" scanpy anndata igraph pyarrow scikit-learn tqdm
#   (pin numpy<2: TF 2.15 requires it. If pip runs out of RAM unpacking wheels,
#   set TMPDIR to a disk path -- /tmp can be a RAM-backed tmpfs.)

# conda lives in a different place on each cluster -- use whichever exists
for c in /beegfs/home/cwolf/miniconda3 /data/wv377102/miniconda3 "$HOME/miniconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
conda activate stagate
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2   # hide TensorFlow's info/warning chatter, keep errors
################################################################

python3 run_stagate_range.py "$@"
