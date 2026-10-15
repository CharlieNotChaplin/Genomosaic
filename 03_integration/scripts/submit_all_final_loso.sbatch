#!/bin/bash
# Driver for the final LOSO comparison. Run on the login node (it is not itself an sbatch script):
#   bash submit_all_final_loso.sbatch
# Submits the nucleus-level array, then the bin-level array with an afterok dependency on it, so the two never compete for GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Submitting nuclei final LOSO (svg + colon_xenium panel + svg/semantic-excl, 27 tasks, max 3 concurrent)..."
NUCLEI_JOBID=$(sbatch --parsable submit_run_loso_final_nuclei.sbatch)
echo "  -> job $NUCLEI_JOBID"

echo "Submitting bins final LOSO (svg + colon_xenium panel + svg/semantic-excl, 27 tasks, max 3 concurrent)," \
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
