# 20261027-081838-luqia-voltana-env-setup-v2

One-time setup for the VoltanaLLM EcoPred latency profiling jobs
(`20261026-1410*-luqia-voltana-prof-{prefill,decode}-{l40s,l4}`). One L4 on ganymede, up to 3 h.

1. Conda env `/data/users/chjing/miniforge3/envs/voltanallm` (python 3.12 + CUDA 12.8.1 toolkit for
   the flashinfer JIT). The shared `cuda-env` (vLLM 0.15.1) is not touched. Reused if it exists
   (`FORCE_REBUILD=1` rebuilds).
2. Upstream VoltanaLLM (`Supercomputing-System-AI-Lab/VoltanaLLM`, f7789acf = SGLang 0.4.9)
   cloned to `/data/users/chjing/voltana_work/VoltanaLLM`, reset to that commit, then
   `voltana-llm-scripts.patch` applied: env-configurable `p_server.sh`/`d_server.sh`/`run_profile.py`
   (model, port, clocks, rates, memory, NVML index, dataset), `fit_latency.py` (headless version of
   the `draw.ipynb` regressions), Mistral-7B-v0.1 as default model, and two NIXL backend fixes:
   `disaggregation/nixl/__init__.py` imported `FakeNixl*` classes that v0.1 never shipped (so even
   `--disaggregation-transfer-backend nixl` failed at import; `fakenixl` now raises a clear error),
   and `nixl/conn.py` passed `is_sorted`, which NIXL 0.9.0 removed.
   `pip install -e python[srt]` plus scikit-learn, matplotlib, pandas, orjson, nvidia-ml-py, datasets,
   and NIXL 0.9.0 (`nixl` + `nixl-cu12`, UCX bundled; `--no-deps` so torch stays 2.7.1, checked).
3. Links `/data/users/chjing/voltana_work/models/Mistral-7B-v0.1` to the existing offline snapshot
   (job 255143's HF cache); ShareGPT is read from `/data/users/chjing/datasets/`.
4. Checks on the L4: imports (incl. the `fake` and `nixl` disaggregation backends), one NIXL agent
   with the UCX backend; then a smoke test: clock-lock permission ("All done"), a 60 s prefill profile
   at the max clock (5 req/s), a decode profile with bs 8/9, and the fit on both. This also fills the
   shared flashinfer/triton JIT caches in `/data/users/chjing/voltana_work/cache/`.
5. Writes `/data/users/chjing/voltana_work/ENV_READY`, which the profiling jobs require.

Nothing on a python command line contains "vllm" (the shared P/D launcher pkills
`python.*vllm|vllm\.entrypoints`); `smoke.txt` records the pgrep evidence.

Results: `job_console.log`, `repo_state.txt`, `pip_freeze.txt`, `imports.txt`, `gpu_allocation.txt`,
`supported_clocks_L4.txt`, `torch_version.txt`, `nixl_check.txt`, `clock_permission.txt`, `smoke.txt`, `smoke-latency-*.csv`, log tails.

## v2 (replaces 20261026-140945-luqia-voltana-env-setup, Slurm job 279367)

279367 failed at `pip_sglang` after 4.6 min: `flashinfer-python==0.2.7.post1` is sdist-only on
PyPI and its `setup.py` pins torch to the torch present in the build env. pip's isolated build
pulled the newest torch, so the package required `torch==2.14.*`, conflicting with SGLang's
`torch==2.7.1` (ResolutionImpossible; nothing was installed, the conda env is reused).

Fix: install `torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1` and the build deps first, then
`pip install --no-build-isolation flashinfer-python==0.2.7.post1` (JIT-only; kernels compile at
first use with the env's nvcc 12.8), then `pip install -e python[srt]` as before.
New failure stages: `pip_torch`, `pip_build_deps`, `pip_flashinfer`.
