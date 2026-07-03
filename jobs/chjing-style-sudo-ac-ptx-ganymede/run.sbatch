#!/usr/bin/env bash
#SBATCH --job-name=chjing-sudo-ac-ptx
#SBATCH --time=00:06:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --gres=nvgpufreq:1
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

SCRIPT="/data/users/chjing/vLLM_test/phase2_highrate_l4.sh"
TARGET_SM_FREQ="${TARGET_SM_FREQ:-1200}"
TARGET_MEM_FREQ="${TARGET_MEM_FREQ:-6251}"
GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"

section() {
  echo
  echo "===== $1 ====="
}

query_clocks() {
  local label="$1"
  section "$label"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,utilization.gpu \
    --format=csv || true
}

run_cmd() {
  local label="$1"
  shift
  section "$label"
  echo "command=$*"
  "$@"
  local rc=$?
  echo "${label}_rc=${rc}"
  return 0
}

reset_clocks() {
  section "cleanup_reset_like_chjing"
  timeout 20 sudo nvidia-smi -i "$GPU_ID" -rac 2>&1 || true
  timeout 20 sudo nvidia-smi -rac 2>&1 || true
  query_clocks "after_cleanup_reset"
}

run_ptx_workload() {
  python3 - <<'PY'
import ctypes
import time

PTX = rb'''
.version 7.0
.target sm_70
.address_size 64

.visible .entry burn(
    .param .u64 data,
    .param .u32 iters
)
{
    .reg .pred %p;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [data];
    ld.param.u32 %r1, [iters];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    cvt.u64.u32 %rd2, %r5;
    mul.lo.u64 %rd3, %rd2, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.u32 %r6, 0;
LOOP:
    setp.ge.u32 %p, %r6, %r1;
    @%p bra DONE;
    fma.rn.f32 %f1, %f1, 0f3f800001, 0f358637bd;
    fma.rn.f32 %f1, %f1, 0f3f7fffff, 0f35a7c5ac;
    add.u32 %r6, %r6, 1;
    bra LOOP;
DONE:
    st.global.f32 [%rd4], %f1;
    ret;
}
'''

def load_cuda():
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise RuntimeError("libcuda.so not found")

cuda = load_cuda()

def fn(*names):
    for name in names:
        try:
            return getattr(cuda, name)
        except AttributeError:
            pass
    raise AttributeError(names)

cuInit = fn("cuInit")
cuDeviceGet = fn("cuDeviceGet")
cuCtxCreate = fn("cuCtxCreate_v2", "cuCtxCreate")
cuCtxDestroy = fn("cuCtxDestroy_v2", "cuCtxDestroy")
cuMemAlloc = fn("cuMemAlloc_v2", "cuMemAlloc")
cuMemFree = fn("cuMemFree_v2", "cuMemFree")
cuMemsetD32 = fn("cuMemsetD32_v2", "cuMemsetD32")
cuModuleLoadData = fn("cuModuleLoadData")
cuModuleGetFunction = fn("cuModuleGetFunction")
cuLaunchKernel = fn("cuLaunchKernel")
cuCtxSynchronize = fn("cuCtxSynchronize")

def check(rc, label):
    if rc != 0:
        raise RuntimeError(f"{label} failed rc={rc}")

device = ctypes.c_int()
context = ctypes.c_void_p()
module = ctypes.c_void_p()
kernel = ctypes.c_void_p()
data = ctypes.c_ulonglong()

print("ptx_workload=starting", flush=True)
check(cuInit(0), "cuInit")
check(cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
check(cuCtxCreate(ctypes.byref(context), 0, device), "cuCtxCreate")
try:
    check(cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(PTX)), "cuModuleLoadData")
    check(cuModuleGetFunction(ctypes.byref(kernel), module, ctypes.c_char_p(b"burn")), "cuModuleGetFunction")
    blocks = 4096
    threads = 256
    count = blocks * threads
    check(cuMemAlloc(ctypes.byref(data), count * 4), "cuMemAlloc")
    try:
        check(cuMemsetD32(data, 0x3f800000, count), "cuMemsetD32")
        iters = ctypes.c_uint(20000)
        data_arg = ctypes.c_ulonglong(data.value)
        params = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(data_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(iters), ctypes.c_void_p),
        )
        end = time.time() + 20
        launches = 0
        while time.time() < end:
            check(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, None, params, None), "cuLaunchKernel")
            check(cuCtxSynchronize(), "cuCtxSynchronize")
            launches += 1
        print(f"ptx_kernel_launches={launches}", flush=True)
    finally:
        cuMemFree(data)
finally:
    cuCtxDestroy(context)
PY
}

trap reset_clocks EXIT

echo "test=chjing_style_sudo_nvidia_smi_ac_with_ptx"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "TARGET_MEM_FREQ=${TARGET_MEM_FREQ}"
echo "TARGET_SM_FREQ=${TARGET_SM_FREQ}"
echo "whoami=$(whoami)"
echo "id=$(id)"

section "slurm_job_record"
scontrol show job "$SLURM_JOB_ID" || true

section "actual_chjing_script_locations"
ls -l "$SCRIPT" /data/users/chjing/vLLM_test/phase2_characterization_l4.sh /data/users/chjing/vLLM_test/validate_obs2_l4.sh 2>/dev/null || true

section "actual_chjing_frequency_lines"
if [ -r "$SCRIPT" ]; then
  grep -nE 'SUPPORTED_MEM_FREQ|ALL_GPU_FREQS|reset.*clock|nvidia-smi -rac|set_gpu_freq|sudo nvidia-smi -i|sudo nvidia-smi -pm' "$SCRIPT" || true
  echo
  echo "----- phase2_highrate_l4.sh set/reset excerpt -----"
  sed -n '135,250p' "$SCRIPT" || true
  echo
  echo "----- phase2_highrate_l4.sh startup excerpt -----"
  sed -n '535,548p' "$SCRIPT" || true
else
  echo "script_not_readable=${SCRIPT}"
fi

section "nvgpufreq_status_files"
for file in /run/nvgpufreq.run /var/run/nvgpufreq.run; do
  echo "--- ${file} ---"
  if [ -e "$file" ]; then
    ls -l "$file" || true
    cat "$file" || true
    echo
  else
    echo "missing"
  fi
done

query_clocks "baseline_before_chjing_style_commands"

run_cmd "chjing_persistence_mode" timeout 20 sudo nvidia-smi -pm 1
query_clocks "after_persistence_mode"

run_cmd "chjing_set_application_clocks" timeout 20 sudo nvidia-smi -i "$GPU_ID" -ac "${TARGET_MEM_FREQ},${TARGET_SM_FREQ}"
query_clocks "after_chjing_set_before_ptx"

section "clock_sampling_during_ptx_workload"
run_ptx_workload &
workload_pid=$!
for sample in $(seq 1 35); do
  echo "--- sample=${sample} time=$(date --iso-8601=seconds) ---"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=index,clocks.gr,clocks.mem,power.draw,utilization.gpu \
    --format=csv,noheader || true
  if ! kill -0 "$workload_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
wait "$workload_pid"
workload_rc=$?
echo "workload_rc=${workload_rc}"

query_clocks "after_workload_before_cleanup"

section "interpretation_hint"
echo "This job uses chjing's sudo nvidia-smi -pm/-ac/-rac style, not bare nvidia-smi."
echo "If clocks.gr exceeds ${TARGET_SM_FREQ} MHz under 100% util, chjing-style -ac did not cap real PTX clocks on this node."
