#!/usr/bin/env bash
#SBATCH --job-name=canatune-deps
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=uranus
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Install CanaTune's third-party runtime packages into the shared cuda-env
# (vLLM 0.15.1). Every package already in the env is pinned by a constraints file
# built from `pip freeze`, so nothing that vLLM depends on can be up- or
# downgraded: only missing packages are added; a version conflict stops pip.
set -uo pipefail
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-230901-luqia-canatune-deps-cuda-env"
OUT="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
WORK="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
mkdir -p "${OUT}" "${WORK}/pip-cache" "${WORK}/home" "${WORK}/tmp"
export HOME="${WORK}/home" PIP_CACHE_DIR="${WORK}/pip-cache" TMPDIR="${WORK}/tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1
exec > >(tee -a "${OUT}/job.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) python=${PY}"
"${PY}" --version

REQS=(
  "fastapi>=0.115,<1"
  "httpx>=0.27,<1"
  "nvidia-ml-py>=12,<14"
  "pyyaml>=6.0,<7"
  "uvicorn[standard]>=0.34,<1"
)
printf '%s\n' "${REQS[@]}" > "${OUT}/requirements.txt"

"${PY}" -m pip freeze --exclude-editable > "${OUT}/freeze_before.txt"
# Constraints: pin every installed distribution (skip direct URL / local installs).
grep -E '^[A-Za-z0-9_.-]+==' "${OUT}/freeze_before.txt" > "${WORK}/constraints.txt"
echo "--- installed before"
"${PY}" -m pip list 2>/dev/null | grep -iE '^(vllm|torch|fastapi|httpx|uvicorn|pyyaml|nvidia-ml-py|pynvml|starlette|pydantic|uvloop|httptools|websockets|watchfiles) ' || true

echo "--- dry run"
"${PY}" -m pip install --dry-run -c "${WORK}/constraints.txt" "${REQS[@]}" 2>&1 | tee "${OUT}/dry_run.txt"
DRY=${PIPESTATUS[0]}
if [ "${DRY}" -ne 0 ]; then
  echo "RESULT=conflict: dry run failed (an installed version does not satisfy CanaTune); nothing installed"
  exit 1
fi

echo "--- install"
"${PY}" -m pip install -c "${WORK}/constraints.txt" "${REQS[@]}" 2>&1 | tee "${OUT}/install.txt"
RC=${PIPESTATUS[0]}
"${PY}" -m pip freeze --exclude-editable > "${OUT}/freeze_after.txt"
diff "${OUT}/freeze_before.txt" "${OUT}/freeze_after.txt" > "${OUT}/freeze_diff.txt"
echo "--- freeze diff (< before, > after)"; cat "${OUT}/freeze_diff.txt"
if grep -E '^< ' "${OUT}/freeze_diff.txt" >/dev/null; then
  echo "WARNING: a previously installed package changed or disappeared (see freeze_diff.txt)"
fi

echo "--- import check"
"${PY}" - <<'PYEOF' 2>&1 | tee "${OUT}/imports.txt"
import importlib
for name in ("fastapi", "httpx", "uvicorn", "yaml", "pynvml", "vllm", "torch"):
    try:
        module = importlib.import_module(name)
        print(f"ok   {name} {getattr(module, '__version__', '?')}")
    except Exception as error:
        print(f"FAIL {name}: {error!r}")
PYEOF
"${PY}" -m pip check > "${OUT}/pip_check.txt" 2>&1; echo "pip check rc=$?"; cat "${OUT}/pip_check.txt"
FINAL=${RC}
if [ "${RC}" -eq 0 ] && ! grep -q '^FAIL' "${OUT}/imports.txt"; then
  echo "RESULT=ok"
else
  echo "RESULT=failed pip_rc=${RC}"; [ "${FINAL}" -eq 0 ] && FINAL=2
fi
echo "end=$(date -Iseconds)"
exit "${FINAL}"
