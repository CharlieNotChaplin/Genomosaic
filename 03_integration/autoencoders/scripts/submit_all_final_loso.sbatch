#!/bin/bash
# Driver for the final LOSO comparison: both nuclei AND bins x {svg,
# colon_xenium panel, svg+raw-semantic-exclusion} -- 3 variants per unit.
# NOT itself an sbatch script -- run directly on the login node:
#   bash submit_all_final_loso.sbatch
# It submits submit_run_loso_final_nuclei.sbatch (27 tasks: 9 slides x 3
# variants, capped at 3 concurrent via that script's own `%3`), then submits
# submit_run_loso_final_bins.sbatch (27 tasks: 9 slides x 3 variants, same
# `%3` cap) with a SLURM `--dependency=afterok` on the nuclei job, so bins
# only starts once nuclei fully succeeds -- the two never compete for GPUs,
# and you only run one command instead of submitting/watching/resubmitting
# each combo by hand.
#
# IMPORTANT: bins' semantic-exclusion variant (threshold=0.2, carried over
# from nuclei, UNCHECKED for bins) should be calibrated first -- run
# submit_run_loso_ablation_bins_semantic_probe.sbatch on its own BEFORE this
# driver (not at the same time -- that would put 6 GPUs in use at once) and
# check its local_frac_masked. See that script's own comment and
# submit_run_loso_final_bins.sbatch's for the full rationale.
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
