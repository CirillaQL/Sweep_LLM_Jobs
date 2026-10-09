#!/usr/bin/env bash
#SBATCH --job-name=nixl-bw
# NIXL 带宽测试 (NIXL transfer bandwidth without vLLM, CanaTune 20c1fb1 scripts/analysis/nixl_bw.py)
# Smoke job 274777 (vLLM 0.15.1 NixlConnector, D reads P's KV) moved KV at ~0.23 GB/s on
# ganymede; vllm-project/vllm#52607 reports UCX cuda_ipc serving WRITE at peer-copy speed
# but READ at ~0.5 GB/s. Two processes, one GPU each (as vLLM), READ and WRITE, 1 and 64
# descriptors, 2 / 16 / 64 / 256 MB, UCX_PROTO_INFO=y (the protocol UCX selects):
#   intra (ganymede GPU a <- b):  vram | vram, both GPUs visible (own first) | dram (pinned,
#                                 = kv_buffer_device cpu) | vram + UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on
#   cross (ganymede <- europa):   vram | dram | vram with UCX_TLS=tcp,cuda_copy (no RDMA) |
#                                 vram with UCX_NET_DEVICES = the 10.1.x interface
# Plus per node: RDMA devices (/sys/class/infiniband), NIC speed and driver, loaded RDMA /
# nvidia_peermem modules, nvidia-smi topo -m / -p2p r, UCX version; and a plain TCP
# throughput test ganymede <- europa (Python sockets, 4 streams). No vLLM, no clocks.
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=ganymede,europa
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --signal=B:TERM@60
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-224303-luqia-nixl-bandwidth-ganymede-europa/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-224303-luqia-nixl-bandwidth-ganymede-europa/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20261009-224303-luqia-nixl-bandwidth-ganymede-europa"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
NIXL_DIR="/data/users/chjing/pylibs/nixl-0.9.0"   # shared, installed by job 274777
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="20c1fb1b753d440bb2cc583aaba6d00baf4b8ebc"
A_NODE=ganymede   # initiator (the D side: it reads)
B_NODE=europa     # target in the cross-node runs (the P side)
mkdir -p "${OUT}/bw"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} commit=${COMMIT}"

export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" XDG_CACHE_HOME="${WORK}/xdg-cache"
export PIP_CACHE_DIR="${WORK}/pip-cache" PYTHONPYCACHEPREFIX="${WORK}/python-pycache"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_TERMINAL_PROMPT=0
export PATH="${ENV_DIR}/bin:${PATH}"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"
trap 'echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"' EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }
on() { local node="$1"; shift; srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=2G "$@"; }

# ---- 0. NVML, inventory -----------------------------------------------------------------
for node in "${A_NODE}" "${B_NODE}"; do
  on "${node}" bash -c '
    echo "== $(hostname)"; head -1 /proc/driver/nvidia/version; modinfo -F version nvidia
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
    echo "-- topo -m"; nvidia-smi topo -m; echo "-- topo -p2p r"; nvidia-smi topo -p2p r
    echo "-- ip"; ip -br -4 addr
    for i in $(ls /sys/class/net); do
      [ "$i" = lo ] && continue
      echo "net $i speed=$(cat /sys/class/net/$i/speed 2>/dev/null) driver=$(basename "$(readlink -f /sys/class/net/$i/device/driver 2>/dev/null)")"
    done
    echo "-- infiniband: $(ls /sys/class/infiniband 2>/dev/null | tr "\n" " ")"
    for d in /sys/class/infiniband/*; do [ -d "$d" ] || continue
      for p in "$d"/ports/*; do echo "$(basename $d) port $(basename $p) state=$(cat $p/state) link=$(cat $p/link_layer) rate=$(cat $p/rate)"; done
      echo "$(basename $d) netdev: $(ls $d/device/net 2>/dev/null | tr "\n" " ")"
    done
    echo "-- modules: $(lsmod | awk "{print \$1}" | grep -E "^(nvidia_peermem|nv_peer_mem|ib_core|ib_uverbs|rdma_ucm|mlx5_ib|mlx5_core|bnxt_re|irdma|rxe)$" | tr "\n" " ")"
    command -v ibv_devinfo >/dev/null && ibv_devinfo -l
    command -v iperf3 >/dev/null && echo "iperf3 present" || echo "no iperf3"
  ' 2>&1 | grep -v NVGPUFREQ | tee -a "${OUT}/inventory.txt"
  on "${node}" nvidia-smi -L >/dev/null 2>&1 || fail "nvml_${node}" 3
done

# ---- 1. code, NIXL, UCX version ------------------------------------------------------------
timeout 300 git clone --quiet "${REPO}" "${WORK}/CanaTune" || fail clone
cd "${WORK}/CanaTune" && git checkout --quiet "${COMMIT}" || fail checkout
[ -f "${NIXL_DIR}/.complete" ] || fail nixl_not_installed
export PYTHONPATH="${NIXL_DIR}"
find "${NIXL_DIR}" -name 'libucp.so*' | head -3 | tee "${OUT}/ucx_libs.txt"
UCP=$(find "${NIXL_DIR}" -name 'libucp.so*' | head -1)
"${PY}" -c "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); l.ucp_get_version_string.restype=ctypes.c_char_p; print('UCX', l.ucp_get_version_string().decode())" "${UCP}" \
  2>&1 | tee -a "${OUT}/ucx_libs.txt"

# ---- 2. GPUs and interfaces ------------------------------------------------------------------
gpus_of() {
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --gpus-per-node=2 --mem=256M bash -c \
    'echo "gpus=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"' 2>&1 | grep '^gpus=' | tail -1 | sed 's/^gpus=//'
}
iface_of() { on "$1" bash -c "ip -br -4 addr | awk '\$3 ~ /^10\\.1\\./ {print \$1; exit}'" 2>/dev/null | grep -v NVGPUFREQ | tail -1; }
ip_of() { on "$1" bash -c "ip -br -4 addr | awk '\$3 ~ /^10\\.1\\./ {split(\$3,a,\"/\"); print a[1]; exit}'" 2>/dev/null | grep -v NVGPUFREQ | tail -1; }
IFS=, read -r A0 A1 <<< "$(gpus_of "${A_NODE}")"
IFS=, read -r B0 B1 <<< "$(gpus_of "${B_NODE}")"
A_IF=$(iface_of "${A_NODE}"); B_IF=$(iface_of "${B_NODE}"); A_IP=$(ip_of "${A_NODE}"); B_IP=$(ip_of "${B_NODE}")
echo "gpus ${A_NODE}=${A0},${A1} ${B_NODE}=${B0},${B1} iface ${A_IF}/${B_IF} ip ${A_IP}/${B_IP}" | tee "${OUT}/allocation.txt"
[[ "${A0}" =~ ^[0-9]+$ && "${A1}" =~ ^[0-9]+$ && "${B0}" =~ ^[0-9]+$ ]] || fail gpu_detect 3

# ---- 3. bandwidth pairs ------------------------------------------------------------------------
# pair NAME MEM T_NODE T_VISIBLE I_NODE I_VISIBLE [ENV=VALUE ...]   (I reads/writes T's buffer)
pair() {
  local name="$1" mem="$2" tn="$3" tv="$4" in="$5" iv="$6"; shift 6
  local dir="${WORK}/bw/${name}"
  rm -rf "${dir}"; mkdir -p "${dir}"
  echo "=== ${name} mem=${mem} target=${tn}:${tv} initiator=${in}:${iv} env=$* start=$(date +%T)"
  timeout 300 srun --overlap --nodes=1 --ntasks=1 --nodelist="${tn}" --mem=8G \
    env CUDA_VISIBLE_DEVICES="${tv}" UCX_PROTO_INFO=y "$@" \
    "${PY}" scripts/analysis/nixl_bw.py --role target --dir "${dir}" --mem "${mem}" --tag "${name}" \
    > "${OUT}/bw/${name}.target.log" 2>&1 &
  local tpid=$!
  timeout 280 srun --overlap --nodes=1 --ntasks=1 --nodelist="${in}" --mem=8G \
    env CUDA_VISIBLE_DEVICES="${iv}" UCX_PROTO_INFO=y "$@" \
    "${PY}" scripts/analysis/nixl_bw.py --role initiator --dir "${dir}" --mem "${mem}" --tag "${name}" \
    --out "${OUT}/bw/${name}.json" > "${OUT}/bw/${name}.initiator.log" 2>&1
  local rc=$?
  touch "${dir}/done"; wait "${tpid}"
  grep '^{' "${OUT}/bw/${name}.initiator.log" | "${PY}" "${WORK}/summary.py"
  echo "  rc=${rc}; UCX protocol lines (initiator):"
  grep -E -i 'remote memory (read|write)|tcp/|cuda_ipc/|cuda_copy/|rc_mlx|rc_verbs|dc_mlx|ud_|sysv/|posix/|cma/' \
    "${OUT}/bw/${name}.initiator.log" 2>/dev/null | sed 's/  */ /g' | sort | uniq -c | sort -rn | head -12
}
cat > "${WORK}/summary.py" <<'PY'
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    gb = "-" if r["gb_s"] is None else "%.2f" % r["gb_s"]
    print("  %-5s pieces=%3d %4d MB %8s GB/s  %s" % (r["op"], r["pieces"], r["size_mb"], gb, r["status"]))
PY
pair intra_vram         VRAM "${A_NODE}" "${A1}"       "${A_NODE}" "${A0}"
pair intra_vram_allvis  VRAM "${A_NODE}" "${A1},${A0}" "${A_NODE}" "${A0},${A1}"
pair intra_dram         DRAM "${A_NODE}" "${A1}"       "${A_NODE}" "${A0}"
pair intra_vram_getzcopy VRAM "${A_NODE}" "${A1}"      "${A_NODE}" "${A0}" UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on
pair cross_vram         VRAM "${B_NODE}" "${B0}"       "${A_NODE}" "${A0}"
pair cross_dram         DRAM "${B_NODE}" "${B0}"       "${A_NODE}" "${A0}"
pair cross_vram_tcp     VRAM "${B_NODE}" "${B0}"       "${A_NODE}" "${A0}" UCX_TLS=tcp,cuda_copy
[ -n "${A_IF}" ] && [ -n "${B_IF}" ] && [ "${A_IF}" = "${B_IF}" ] && \
  pair cross_vram_netdev VRAM "${B_NODE}" "${B0}"      "${A_NODE}" "${A0}" UCX_NET_DEVICES="${A_IF}"

# ---- 4. plain TCP ganymede <- europa (upper bound for a TCP path) -----------------------------
TCP_PORT=$(( 20000 + ${JOB_ID//[^0-9]/} % 1000 ))
echo "=== tcp ${B_NODE}(${B_IP}) -> ${A_NODE}, 4 streams x 1 GiB, port ${TCP_PORT}"
timeout 120 srun --overlap --nodes=1 --ntasks=1 --nodelist="${A_NODE}" --mem=2G "${PY}" -c '
import socket, sys, threading, time
port = int(sys.argv[1]); srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", port)); srv.listen(8); got = [0]; lock = threading.Lock()
def serve(c):
    n = 0
    while True:
        b = c.recv(4 << 20)
        if not b: break
        n += len(b)
    with lock: got[0] += n
threads = []; t0 = None
for _ in range(4):
    c, _ = srv.accept(); t0 = t0 or time.perf_counter()
    th = threading.Thread(target=serve, args=(c,)); th.start(); threads.append(th)
for th in threads: th.join()
dt = time.perf_counter() - t0
print(f"tcp received {got[0]/1e9:.2f} GB in {dt:.2f} s = {got[0]/dt/1e9:.2f} GB/s")' "${TCP_PORT}" &
SPID=$!
sleep 8
timeout 100 srun --overlap --nodes=1 --ntasks=1 --nodelist="${B_NODE}" --mem=2G "${PY}" -c '
import socket, sys, threading
host, port = sys.argv[1], int(sys.argv[2]); buf = b"\0" * (4 << 20)
def send():
    s = socket.create_connection((host, port))
    for _ in range(256): s.sendall(buf)
    s.close()
ts = [threading.Thread(target=send) for _ in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]' "${A_IP}" "${TCP_PORT}"
wait "${SPID}"
echo "RESULT=done"
