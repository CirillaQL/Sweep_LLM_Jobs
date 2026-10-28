#!/usr/bin/env bash
#SBATCH --job-name=voltana-diag
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=ganymede
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261028-233708-luqia-voltana-decode-fit-diag2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261028-233708-luqia-voltana-decode-fit-diag2/slurm-%j.err

# CPU only; reads 279535 and writes only under ${OUT}. Env-setup v5 (Slurm 279535): with the
# extra_batch_info fix both decode benchmarks succeeded (8/8, 9/9), but fit_latency.py exited
# non-zero (stage smoke_decode, decode_fit_failed) and its log stayed in WORK. This copies the
# logs, summarises the extra-info files as diag 279524 did, and reruns the decode fit into
# ${OUT}/refit with the full traceback.
set -uo pipefail
JOB_NAME="20261028-233708-luqia-voltana-decode-fit-diag2"
OUT="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}/results/${SLURM_JOB_ID:-manual}"
SRC="/data/users/chjing/voltana_work/runs/279535"
PY="/data/users/chjing/miniforge3/envs/voltanallm/bin/python"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "job_start=$(date -Iseconds) host=$(hostname) job=${SLURM_JOB_ID:-manual}"
ls -la "${SRC}" "${SRC}/smoke-decode" "${SRC}/smoke-decode/2040" "${SRC}/smoke-decode/fit" 2>&1
for f in fit-decode.log fit-prefill.log profile-decode.log; do
  [ -f "${SRC}/${f}" ] && grep -v 'FutureWarning\|import pynvml' "${SRC}/${f}" | tail -n 300 > "${OUT}/${f}"
done
"${PY}" -I - "${SRC}/smoke-decode" <<'PYEOF' > "${OUT}/decode_extra_summary.txt" 2>&1
import collections, glob, json, os, sys
import orjson
root = sys.argv[1]
for file in sorted(glob.glob(os.path.join(root, "*", "*-extra.json"))):
    print("==", file, os.path.getsize(file), "bytes")
    reqs = orjson.loads(open(file, "rb").read())
    lens = [len(v or []) for v in reqs.values()]
    print("requests", len(reqs), "infos/request min/max", min(lens, default=0), max(lens, default=0))
    modes, keys, bss, itls, types = collections.Counter(), collections.Counter(), collections.Counter(), [], collections.Counter()
    for items in reqs.values():
        for it in items or []:
            modes[repr(it.get("forward_mode"))] += 1
            keys[tuple(sorted(it))] += 1
            bss[it.get("batch_size")] += 1
            types[type(it.get("itl")).__name__] += 1
            if isinstance(it.get("itl"), (int, float)):
                itls.append(it["itl"] * 1000)
    print("forward_mode", dict(modes))
    print("batch_size", dict(bss.most_common(10)))
    print("itl types", dict(types))
    if itls:
        itls.sort()
        q = lambda p: itls[min(len(itls) - 1, int(p * len(itls)))]
        print("itl ms n=%d min=%.3f p5=%.3f p50=%.3f p95=%.3f max=%.3f in(3,300)=%d" % (
            len(itls), itls[0], q(.05), q(.5), q(.95), itls[-1], sum(3 < x < 300 for x in itls)))
    for k, n in keys.most_common(3):
        print("keys x%d" % n, k)
    first = next((v for v in reqs.values() if v), [])
    for it in first[:3] + first[-2:]:
        print(json.dumps(it)[:1500])
PYEOF
cat "${OUT}/decode_extra_summary.txt"
[ -f "${SRC}/smoke-decode/fit/points-decode.csv.gz" ] && zcat "${SRC}/smoke-decode/fit/points-decode.csv.gz" | head -n 5 > "${OUT}/points-decode-head.txt"
SCRATCH="/data/users/chjing/voltana_work/runs/${SLURM_JOB_ID:-manual}"
export MPLBACKEND=Agg MPLCONFIGDIR="${SCRATCH}/matplotlib" HOME="${SCRATCH}/home"
mkdir -p "${MPLCONFIGDIR}" "${HOME}" "${OUT}/refit"
cd "${SRC}/smoke-decode" && "${PY}" -X faulthandler /data/users/chjing/voltana_work/VoltanaLLM/voltana-llm-scripts/predictions/fit_latency.py \
  --phase decode --data-dir . --out-dir "${OUT}/refit" --tokens-min 0 > "${OUT}/refit.log" 2>&1
echo "refit_rc=$?" | tee -a "${OUT}/refit.log"
grep -v 'RuntimeWarning\|return X @ coef_' "${OUT}/refit.log" | tail -n 60
rm -f "${OUT}/refit/points-decode.csv.gz"
echo "RESULT=ok"
