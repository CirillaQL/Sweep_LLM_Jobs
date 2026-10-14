#!/usr/bin/env bash
#SBATCH --job-name=inspect-dir
# CPU only, read only: what is /data/users/chjing/cccl-cudastf-region-spike-d10a88a, and does
# it use a git broker (jobs synced through a GitHub repo, READY markers, status files)?
# Prints structure, sizes, recent changes, git metadata (credentials in URLs masked), broker
# markers, the README, and the Slurm queue. Nothing is modified; file contents other than
# README / job-description files are not printed.
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261014-140157-luqia-inspect-cccl-cudastf-dir/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261014-140157-luqia-inspect-cccl-cudastf-dir/slurm-%j.err

set -uo pipefail
JOB_NAME="20261014-140157-luqia-inspect-cccl-cudastf-dir"
OUT="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}/results/${SLURM_JOB_ID:-manual}"
D="/data/users/chjing/cccl-cudastf-region-spike-d10a88a"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/inspect.log") 2>&1
mask() { sed -E 's#(https?://)[^/@ ]+@#\1***@#g; s#(gh[pousr]_)[A-Za-z0-9]{10,}#\1***#g'; }
echo "start=$(date -Iseconds) host=$(hostname)"
[ -d "${D}" ] || { echo "RESULT=missing ${D}"; exit 0; }

echo "== 1. top level"
ls -la "${D}" | head -n 80
echo "size: $(timeout 120 du -sh "${D}" 2>/dev/null | cut -f1)"
echo "== 2. directories (depth 2)"
find "${D}" -mindepth 1 -maxdepth 2 -type d -not -path '*/.git/*' | sed "s#${D}/##" | sort | head -n 150
echo "== 3. most recently modified files (top 30, excluding .git)"
find "${D}" -type f -not -path '*/.git/*' -printf '%TY-%Tm-%Td %TH:%TM  %s  %P\n' 2>/dev/null | sort -r | head -n 30
echo "files changed in the last 7 days: $(find "${D}" -type f -not -path '*/.git/*' -newermt '-7 days' 2>/dev/null | wc -l)"

echo "== 4. git"
if git -C "${D}" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "${D}" remote -v | mask
  echo "branch: $(git -C "${D}" rev-parse --abbrev-ref HEAD)  head: $(git -C "${D}" rev-parse --short HEAD)"
  git -C "${D}" log -15 --format='%h %ad %an | %s' --date=format:'%Y-%m-%d %H:%M' | mask
  echo "-- status (first 30)"; git -C "${D}" status --short | head -n 30
  echo "-- branches"; git -C "${D}" branch -a 2>/dev/null | head -n 20
else
  echo "not a git repository (top level)"
  find "${D}" -maxdepth 3 -name .git -type d -printf 'nested git: %P\n' 2>/dev/null | head
fi

echo "== 5. broker / job markers"
for pat in READY status.json status.log run.sbatch '*.sbatch' '*.slurm' 'slurm-*.out' 'broker*' 'gsync*' 'jobs'; do
  n=$(find "${D}" -not -path '*/.git/*' -name "${pat}" 2>/dev/null | wc -l)
  [ "${n}" -gt 0 ] && echo "${pat}: ${n} (e.g. $(find "${D}" -not -path '*/.git/*' -name "${pat}" 2>/dev/null | head -n 3 | sed "s#${D}/##" | tr '\n' ' '))"
done
echo "-- mentions of broker / READY / gsync / git push in scripts (file:line, first 30)"
grep -rIn --include='*.sh' --include='*.py' --include='*.sbatch' --include='Makefile' --include='*.md' \
  -E 'broker|gsync|READY|git push|status\.json' "${D}" 2>/dev/null | grep -v '/.git/' | cut -c1-200 | mask | head -n 30
echo "-- one status.json / run.sbatch header, if any"
f=$(find "${D}" -not -path '*/.git/*' -name status.json 2>/dev/null | head -n 1); [ -n "$f" ] && head -c 1500 "$f" | mask && echo
f=$(find "${D}" -not -path '*/.git/*' -name '*.sbatch' 2>/dev/null | head -n 1); [ -n "$f" ] && { echo "($f)"; head -n 40 "$f" | mask; }

echo "== 6. README"
for r in README.md README.rst README.txt README; do
  [ -f "${D}/${r}" ] && { echo "(${r})"; head -n 80 "${D}/${r}" | mask; break; }
done

echo "== 7. Slurm queue of this user, and recent jobs"
squeue -u "$(id -un)" -o '%.10i %.12P %.30j %.8T %.10M %.20R' 2>/dev/null | head -n 30
sacct -u "$(id -un)" -S "$(date -d '-3 days' +%F)" --format=JobID%10,JobName%30,State%12,Start%19,NodeList%20 -X 2>/dev/null | grep -iv "inspect-dir" | head -n 40
echo "RESULT=ok end=$(date -Iseconds)"
