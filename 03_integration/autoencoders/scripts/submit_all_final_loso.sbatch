#!/bin/bash
# Driver for the final LOSO comparison: nuclei x {svg, colon_xenium panel,
# svg+raw-semantic-exclusion} plus bins x {svg, colon_xenium panel}.
# NOT itself an sbatch script -- run directly on the login node:
#   bash submit_all_final_loso.sh
# It submits submit_run_loso_final_nuclei.sbatch (27 tasks: 9 slides x 3
# variants, capped at 3 concurrent via that script's own `%3` -- see its own
# comment for why the 3rd variant exists and is nuclei/svg-only), then submits
# submit_run_loso_final_bins.sbatch (18 tasks: 9 slides x {svg, colon_xenium
# panel}) with a SLURM `--dependency=afterok` on the nuclei job, so bins only
# starts once nuclei fully succeeds -- the two never compete for GPUs, and you
# only run one command instead of submitting/watching/resubmitting each combo
# by hand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Submitting nuclei final LOSO (svg + colon_xenium panel + svg/semantic-excl, 27 tasks, max 3 concurrent)..."
NUCLEI_JOBID=$(sbatch --parsable submit_run_loso_final_nuclei.sbatch)
echo "  -> job $NUCLEI_JOBID"

echo "Submitting bins final LOSO (svg + colon_xenium panel, 18 tasks, max 3 concurrent)," \
     "held until job $NUCLEI_JOBID completes successfully..."
BINS_JOBID=$(sbatch --parsable --dependency=afterok:"$NUCLEI_JOBID" submit_run_loso_final_bins.sbatch)
echo "  -> job $BINS_JOBID (queued, dependency-held)"

echo
echo "Done. Track both with: squeue -u \$USER"
echo "If ANY nuclei task fails, the bins job's dependency is never satisfied and it"
echo "sits in squeue forever (reason: DependencyNeverSatisfied) rather than starting"
echo "or auto-cancelling. If that happens: fix + resubmit the failed nuclei task(s)"
echo "(sbatch --array=N submit_run_loso_final_nuclei.sbatch), then either release the"
echo "held bins job (scontrol release $BINS_JOBID) once you're satisfied, or cancel"
echo "it (scancel $BINS_JOBID) and resubmit manually."
