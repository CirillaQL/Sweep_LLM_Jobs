#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# E1 follow-up, version 2 (READY, NOT SUBMITTED; 2026-11-20). Times, in 5 paired blocks against ONE new physical reference
# heft_k4_fmax, the logical-model (L_cal) and executor-feedback (F_cal) selections of the frozen L40S Cholesky 16x16x4096 pool
# (Slurm 281794): 5 physical plans INCLUDING the reference (PREDECLARED.json: roles, aliases, shas, frozen Latin-square order).
# Plans are read from the ORIGINAL pool directories (plans_l40s_<family>_fit, the directories 281794 timed from), sha-checked
# against PREDECLARED.json and against the 281794 freeze on this filesystem before anything runs. Executor and runner are the
# 281794 ones (sha-checked); per-plan settings are the 281794 step-4 settings. No executor/model change.
# Expected: verify ~8 min + 25 processes x ~70 s ~= 40 min; limit 1:30.
# PREDICTIONS_ALL_PLANS.csv (all 5 plans x 4 variants; attribution only, not selection) is sha-checked and used by the analysis.
# Submit (after the author's approval): rsync this bundle to ${bundle}, then  sbatch uranus_e1_followup.sbatch
#SBATCH --job-name=e1-l40s-followup
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/data/users/chjing/cccl-cudastf-region-spike-d10a88a/research_spikes/experiment_completion_20261120/bundles/e1_l40s_followup/slurm-%j.stdout.log
#SBATCH --error=/data/users/chjing/cccl-cudastf-region-spike-d10a88a/research_spikes/experiment_completion_20261120/bundles/e1_l40s_followup/slurm-%j.stderr.log
set -uo pipefail
export QR_BINARY_SHA=392942367a1d9a7bfa0290016d07d905bb6f718831c6113caaca34447d93764e
repo_root=/data/users/chjing/cccl-cudastf-region-spike-d10a88a
spikes="${repo_root}/research_spikes"
bundle="${spikes}/experiment_completion_20261120/bundles/e1_l40s_followup"
runner="${spikes}/cholesky_l40s_chain_20261102/run_chol_minerva.sh"
declared="${bundle}/PREDECLARED.json"
DECLARED_SHA=3ce634a03b82f36c2a07b40d984896a61f268aec9692ea6629f11d26bea02f71
RUNNER_SHA_PREFIX=9cdf22113ef0e0d8
PREDICTIONS_SHA=7c0770ee12c6bc3b3364e6ae91b1eb8eca8f7e0706a5691e64379fec600f846f
top="${bundle}/results_${SLURM_JOB_ID}"
test ! -e "${top}" || { echo "Refusing to overwrite ${top}" >&2; exit 2; }
mkdir -p "${top}"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${top}/summary.txt"; }
source /data/users/chjing/miniforge3/etc/profile.d/conda.sh
conda activate cuda-env
[[ $(sha256sum "${declared}" | cut -d' ' -f1) == "${DECLARED_SHA}" ]] || { log "STOP: PREDECLARED.json changed"; exit 4; }
[[ $(sha256sum "${runner}" | cut -c1-16) == "${RUNNER_SHA_PREFIX}" ]] || { log "STOP: runner changed"; exit 4; }
[[ $(sha256sum "${bundle}/PREDICTIONS_ALL_PLANS.csv" | cut -d' ' -f1) == "${PREDICTIONS_SHA}" ]] || { log "STOP: predictions export changed"; exit 4; }
nvidia-smi -L | tee "${top}/gpus.txt" || { log "STOP: NVML unusable"; exit 3; }
processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z ${processes} ]] || { log "STOP: unexpected GPU processes ${processes}"; exit 1; }
sudo -n true || { log "STOP: no root for clocks"; exit 5; }
# role -> file mapping: every physical plan's pool file and its timing-dir copy must equal PREDECLARED and the 281794 freeze
python3 - "${declared}" "${spikes}" > "${top}/plan_dirs.txt" <<'CHECK' || { log "STOP: plan identity"; exit 4; }
import hashlib, json, sys
from pathlib import Path
d = json.load(open(sys.argv[1])); root = Path(sys.argv[2])
fr_path = root / d["source_campaign"]["frozen_predictions"]
assert hashlib.sha256(fr_path.read_bytes()).hexdigest() == d["source_campaign"]["frozen_predictions_sha256"], "freeze"
fr = json.load(open(fr_path))
assert set(d["physical_plans"]) == {d["reference"]} | {r["plan"] for r in d["roles"]}, "physical set is not reference + selections"
for p in d["physical_plans"]:
    f = d["plan_files"][p]
    for path in (root / f["pool_file"], root / f["timed_from_dir"] / f"{p}.csv"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"] == fr["plans"][p]["plan_sha256"], path
    print(p, root / f["timed_from_dir"])
CHECK
declare_dir() { awk -v p="$1" '$1==p{print $2}' "${top}/plan_dirs.txt"; }
log "identities PASS (declared ${DECLARED_SHA:0:16}); verify pass"
for p in $(awk '{print $1}' "${top}/plan_dirs.txt"); do
  PLAN_FILTER="/${p}\\.csv\$" VERIFY=1 SHUF_SEED=0 bash "${runner}" "${top}/verify/${p}" "$(declare_dir "${p}")" 2 \
    && grep -q "^status=PASS" "${top}/verify/${p}/runs/${p}/summary.txt" || { log "STOP: ${p} verification failed"; exit 6; }
done
echo "block,position,plan" > "${top}/order_actual.csv"
for b in 1 2 3 4 5; do
  log "block ${b}"
  pos=0
  for p in $(python3 -c "import json;print(' '.join(json.load(open('${declared}'))['block_orders']['${b}']))"); do
    pos=$((pos + 1))
    echo "${b},${pos},${p}" >> "${top}/order_actual.csv"
    PLAN_FILTER="/${p}\\.csv\$" SHUF_SEED=${b} bash "${runner}" "${top}/block_${b}/${pos}_${p}" "$(declare_dir "${p}")" 1 || log "PLAN_FAILED block ${b} ${p}"
  done
done
python3 "${bundle}/analyze_e1_l40s.py" "${top}" --out "${top}/analysis" > "${top}/analysis.txt" 2>&1
rc=$?
log "analysis rc=${rc}"
exit ${rc}
