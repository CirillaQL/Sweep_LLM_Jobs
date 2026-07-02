#!/usr/bin/env bash
#SBATCH --job-name=gpufreq-ptx-eu
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --nodelist=europa
#SBATCH --gpus=1
#SBATCH --gres=nvgpufreq:1
#SBATCH --gpu-freq=1200,verbose
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"

query_clocks() {
  local label="$1"
  echo
  echo "===== ${label} ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,utilization.gpu --format=csv || true
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

echo "target_node=europa"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "whoami=$(whoami)"
echo "id=$(id)"

echo
echo "===== slurm_job_record ====="
scontrol show job "$SLURM_JOB_ID" || true

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found."
  exit 127
fi

query_clocks "baseline_before_workload"

echo
echo "===== clock_sampling_during_ptx_workload ====="
run_ptx_workload &
workload_pid=$!
for sample in $(seq 1 35); do
  echo "--- sample=${sample} time=$(date --iso-8601=seconds) ---"
  nvidia-smi -i "$GPU_ID" --query-gpu=index,clocks.gr,clocks.mem,power.draw,utilization.gpu --format=csv,noheader || true
  if ! kill -0 "$workload_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
wait "$workload_pid"
workload_rc=$?
echo "workload_rc=${workload_rc}"

query_clocks "after_workload_before_job_end"
