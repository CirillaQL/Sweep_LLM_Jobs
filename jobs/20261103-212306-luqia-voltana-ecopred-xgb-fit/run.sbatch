#!/usr/bin/env bash
#SBATCH --job-name=voltana-xgb-fit
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261103-212306-luqia-voltana-ecopred-xgb-fit/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261103-212306-luqia-voltana-ecopred-xgb-fit/slurm-%j.err

# CPU only. Fits VoltanaLLM v3's XGBoost EcoPred (arXiv 2509.04827v3 Sec. V-D, Appx. C) on the points
# the profiling jobs kept (280101 prefill-L40S, 280104 decode-L4) and compares it on held-out rows with
# the v0.1 linear tables those jobs produced. See README.md and fit_xgb.py.
# xgboost 2.1.4 is installed once with pip --target into ${VOLTANA_ROOT}/pylib-xgboost-2.1.4 (the
# voltanallm env itself is not touched), so a later serving job can put the same copy on PYTHONPATH.
# Expected run time 30-60 min, most of it the decode fits (8.2M points, main fit + 3 leave-one-clock-out).
set -uo pipefail
START_EPOCH=$(date +%s)
JOB_NAME="20261103-212306-luqia-voltana-ecopred-xgb-fit"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
VOLTANA_ROOT="/data/users/chjing/voltana_work"
WORK="${VOLTANA_ROOT}/runs/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/voltanallm"
PY="${ENV_DIR}/bin/python"
XGB_VERSION="2.1.4"
PYLIB="${VOLTANA_ROOT}/pylib-xgboost-${XGB_VERSION}"

P_POINTS="${VOLTANA_ROOT}/runs/280101/prefill/fit/points-prefill.csv.gz"
D_POINTS="${VOLTANA_ROOT}/runs/280104/decode/fit/points-decode.csv.gz"
P_TABLE="${BROKER_ROOT}/jobs/20261026-141000-luqia-voltana-prof-prefill-l40s/results/280101/latency-prefill.csv"
D_TABLE="${BROKER_ROOT}/jobs/20261026-141030-luqia-voltana-prof-decode-l4/results/280104/latency-decode.csv"

mkdir -p "${OUT}" "${WORK}"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "job_start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-?}"
fail() { echo "RESULT=failed stage=$1"; exit 1; }
cleanup() {
  set +e
  [ ! -s "${WORK}/fit.log" ] || cp "${WORK}/fit.log" "${OUT}/"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - START_EPOCH ))"
}
trap cleanup EXIT
trap 'exit 143' TERM

export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" MPLCONFIGDIR="${WORK}/matplotlib" PYTHONPYCACHEPREFIX="${WORK}/python-pycache"
export PIP_CACHE_DIR="${WORK}/pip-cache" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${HOME}" "${TMPDIR}" "${MPLCONFIGDIR}"

# ---- inputs ----------------------------------------------------------------------------------
for f in "${P_POINTS}" "${D_POINTS}" "${P_TABLE}" "${D_TABLE}"; do
  [ -s "${f}" ] || fail "missing_input ${f}"
done
{ for f in "${P_POINTS}" "${D_POINTS}" "${P_TABLE}" "${D_TABLE}"; do
    echo "$(sha256sum "${f}" | cut -c1-16) $(stat -c %s "${f}") ${f}"
  done
} | tee "${OUT}/inputs.txt"
mkdir -p "${OUT}/tables" && cp "${P_TABLE}" "${D_TABLE}" "${OUT}/tables/"

# ---- xgboost (job-independent copy, installed once; --no-deps: numpy/scipy come from the env) ---
if ! PYTHONPATH="${PYLIB}" "${PY}" -c "import xgboost; assert xgboost.__version__ == '${XGB_VERSION}'" 2>/dev/null; then
  STAGE="${PYLIB}.tmp-${JOB_ID}"
  rm -rf "${STAGE}"
  timeout 900 "${PY}" -m pip install --no-deps --target "${STAGE}" "xgboost==${XGB_VERSION}" 2>&1 | tail -n 3
  PYTHONPATH="${STAGE}" "${PY}" -c "import xgboost" || fail "xgboost_install"
  rm -rf "${PYLIB}" && mv "${STAGE}" "${PYLIB}" || fail "xgboost_install_move"
fi
export PYTHONPATH="${PYLIB}"
"${PY}" -c "
import sys, xgboost, sklearn, pandas, numpy
print('python', sys.version.split()[0], 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__,
      'pandas', pandas.__version__, 'numpy', numpy.__version__)" | tee "${OUT}/versions.txt" || fail import_check

# ---- fit (background + wait, so the TERM before the time limit is handled at once) ------------
"${PY}" -W ignore::UserWarning "${JOB_DIR}/fit_xgb.py" \
  --prefill-points "${P_POINTS}" --prefill-table "${P_TABLE}" \
  --decode-points "${D_POINTS}" --decode-table "${D_TABLE}" \
  --out "${WORK}/fit" --threads "${OMP_NUM_THREADS}" --seed 0 > "${WORK}/fit.log" 2>&1 &
FIT_PID=$!
wait "${FIT_PID}"
FIT_RC=$?
echo "fit_rc=${FIT_RC} fit_s=$(( $(date +%s) - START_EPOCH ))"

for f in metrics.json summary.txt "${WORK}"/fit/parity-*.png; do
  f=$(basename "${f}")
  [ ! -s "${WORK}/fit/${f}" ] || cp "${WORK}/fit/${f}" "${OUT}/"
done
for f in "${WORK}"/fit/xgb-*.json; do
  [ ! -s "${f}" ] || gzip -c "${f}" > "${OUT}/$(basename "${f}").gz"
done
printf '%s\n' "${WORK}/fit" > "${OUT}/RAW_CACHE_LOCATION.txt"
[ ! -s "${OUT}/summary.txt" ] || cat "${OUT}/summary.txt"
if [ "${FIT_RC}" -eq 0 ] && [ -s "${OUT}/summary.txt" ]; then echo "RESULT=ok"; exit 0; fi
tail -n 40 "${WORK}/fit.log"
echo "RESULT=failed fit_rc=${FIT_RC}"; exit 1
