#!/usr/bin/env bash
#SBATCH --job-name=sync-log-nixl
# 只读收集：(1) 284503 (20261115-102715-luqia-nixl-xfer-l40s-neptune) 的打印记录，查慢的原因；
# (2) 正在运行的 288073 的启动脚本 / srun 信息；(3) 已完成的 281794 281793 279485 278366 的同样信息。
# Read-only: only scontrol show / squeue / sacct / sstat queries and reads (cat) of files; nothing is
# written outside this job's own results/ dir, no signal, no scontrol update, no file touched in other
# jobs' directories. 1 CPU, 1 GB, no GPU, on ganymede. Every step is optional: a step that cannot read
# its source logs the reason and the job continues; the job always exits 0.
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodelist=ganymede
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261120-sync-log-nixl-detail/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261120-sync-log-nixl-detail/slurm-%j.err

set -u
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261120-sync-log-nixl-detail"
OUT="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}" || exit 0
exec > >(tee -a "${OUT}/console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${SLURM_JOB_ID:-manual}"
MAX_BYTES=$((20 * 1024 * 1024))   # copy at most 20 MB of any one file
q() { timeout 60 nice -n 19 "$@" 2>&1; }   # one read-only query, bounded, lowest priority

# read-only copy of one file into OUT/<dest>: plain read (cat), size-capped, gz files kept as is
grab() {  # $1 = source, $2 = destination name
  local src="$1" dst="${OUT}/$2" size
  if [ ! -r "${src}" ]; then echo "unreadable: ${src}"; return 0; fi
  size=$(stat -c %s "${src}" 2>/dev/null || echo 0)
  mkdir -p "$(dirname "${dst}")"
  if [ "${size}" -le "${MAX_BYTES}" ]; then
    timeout 120 nice -n 19 cat "${src}" > "${dst}" 2>/dev/null && echo "copied ${src} (${size} B)"
  else
    { timeout 60 nice -n 19 head -c $((MAX_BYTES / 2)) "${src}"; echo; echo "... [${size} B, middle cut] ..."
      timeout 60 nice -n 19 tail -c $((MAX_BYTES / 2)) "${src}"; } > "${dst}" 2>/dev/null
    echo "copied head+tail of ${src} (${size} B)"
  fi
}

# Slurm facts of one job, all read-only queries; the batch script if Slurm still has it
slurm_info() {  # $1 = job id, $2 = output dir name
  local id="$1" d="${OUT}/$2" cmd so se wd
  mkdir -p "${d}"
  q scontrol show job -dd "${id}" > "${d}/scontrol_show_job.txt"
  q squeue -j "${id}" -o '%i %j %u %T %M %l %D %R %S' > "${d}/squeue.txt"
  q squeue -s -j "${id}" -o '%i %j %u %M %N %o' > "${d}/squeue_steps.txt"   # srun steps and their commands
  q sacct -j "${id}" -P --format=JobID,JobName%60,User,Partition,NodeList,AllocTRES%120,State,ExitCode,Submit,Start,End,Elapsed,WorkDir%300,SubmitLine%1000 > "${d}/sacct.txt"
  q sacct -j "${id}" -B > "${d}/sacct_batch_script.txt"                    # only if accounting stores scripts
  q sstat -j "${id}" --allsteps -P --format=JobID,AveCPU,MaxRSS,NTasks > "${d}/sstat.txt"   # running steps only
  # Slurm's own copy of the batch script (owner/admin only): written to OUR file, the job is not touched
  timeout 60 scontrol write batch_script "${id}" "${d}/batch_script.sh" > "${d}/write_batch_script.log" 2>&1 \
    || echo "scontrol write batch_script ${id}: not available" >> "${d}/write_batch_script.log"
  # the script / output paths Slurm reports, read if readable (never written)
  cmd=$(grep -o 'Command=[^ ]*' "${d}/scontrol_show_job.txt" | head -1 | cut -d= -f2-)
  so=$(grep -o 'StdOut=[^ ]*' "${d}/scontrol_show_job.txt" | head -1 | cut -d= -f2-)
  se=$(grep -o 'StdErr=[^ ]*' "${d}/scontrol_show_job.txt" | head -1 | cut -d= -f2-)
  wd=$(grep -o 'WorkDir=[^ ]*' "${d}/scontrol_show_job.txt" | head -1 | cut -d= -f2-)
  [ -z "${wd}" ] && wd=$(awk -F'|' 'NR==2{print $13}' "${d}/sacct.txt")
  echo "id=${id} command=${cmd:-?} stdout=${so:-?} stderr=${se:-?} workdir=${wd:-?}" | tee "${d}/paths.txt"
  [ -n "${cmd}" ] && [ -f "${cmd}" ] && grab "${cmd}" "$2/command_file.txt"
  [ -n "${so}" ] && [ -f "${so}" ] && grab "${so}" "$2/stdout.txt"
  [ -n "${se}" ] && [ -f "${se}" ] && [ "${se}" != "${so}" ] && grab "${se}" "$2/stderr.txt"
  if [ -n "${wd}" ] && [ -d "${wd}" ]; then   # listing only, then any *.sbatch / *.sh at its top level
    timeout 60 nice -n 19 ls -la "${wd}" > "${d}/workdir_ls.txt" 2>&1
    for f in "${wd}"/*.sbatch "${wd}"/*.slurm "${wd}"/run*.sh; do [ -f "$f" ] && grab "$f" "$2/workdir_$(basename "$f")"; done
  fi
  return 0
}

# ---- 1. job 284503: its print records (gitignored slurm logs) and raw work dir ----------------
echo "=== 1. 284503"
SRC_JOB="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261115-102715-luqia-nixl-xfer-l40s-neptune"
RAW="/data/users/chjing/vllm_job_work/284503"
grab "${SRC_JOB}/slurm-284503.out" "284503/slurm-284503.out"
grab "${SRC_JOB}/slurm-284503.err" "284503/slurm-284503.err"
slurm_info 284503 "284503/slurm"
if [ -d "${RAW}" ]; then
  timeout 120 nice -n 19 find "${RAW}" -maxdepth 4 -not -path '*/huggingface/*' -not -path '*/pip-cache/*' \
    -not -path '*/xdg-*' -not -path '*/CanaTune/*' -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' > "${OUT}/284503/raw_tree.txt" 2>&1
  # every text log / record of the three variants (vLLM, proxy, controller, UCX), read-only, size-capped
  while IFS= read -r f; do
    rel="${f#${RAW}/}"
    used=$(du -sm "${OUT}" 2>/dev/null | cut -f1)
    if [ "${used:-0}" -ge 150 ]; then echo "copy budget (150 MB) reached; rest listed in raw_tree.txt"; break; fi
    grab "${f}" "284503/raw/${rel}"
    case "${rel}" in *.gz) ;; *) [ -f "${OUT}/284503/raw/${rel}" ] && gzip -f "${OUT}/284503/raw/${rel}" ;; esac
  done < <(timeout 120 find "${RAW}" -maxdepth 6 -type f \( -name '*.log' -o -name '*.jsonl' -o -name '*.json' -o -name '*.txt' -o -name '*.yaml' \) \
             -not -path '*/huggingface/*' -not -path '*/pip-cache/*' -not -path '*/xdg-*' -not -path '*/CanaTune/*' -not -path '*/home/*' 2>/dev/null | head -400)
else
  echo "no raw dir ${RAW}"
fi
# compress the copies (in OUT only) so the broker push stays small
find "${OUT}/284503" -type f -size +1M ! -name '*.gz' -exec gzip -f {} \; 2>/dev/null

# ---- 2. job 288073 (running): read-only Slurm facts, never signalled or updated ----------------
echo "=== 2. 288073"
slurm_info 288073 "288073"

# ---- 3. finished jobs ---------------------------------------------------------------------------
for id in 281794 281793 279485 278366; do
  echo "=== 3. ${id}"
  slurm_info "${id}" "${id}"
done
find "${OUT}" -type f -size +1M ! -name '*.gz' ! -name console.log -exec gzip -f {} \; 2>/dev/null
du -sh "${OUT}" 2>/dev/null
echo "end=$(date -Iseconds)"
exit 0
