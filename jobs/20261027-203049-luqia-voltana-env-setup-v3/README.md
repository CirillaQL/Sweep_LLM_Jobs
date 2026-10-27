# 20261027-203049-luqia-voltana-env-setup-v3

One-time setup for the VoltanaLLM EcoPred latency profiling jobs
(`20261026-1410*-luqia-voltana-prof-{prefill,decode}-{l40s,l4}`). One L4 on ganymede, up to 3 h.

1. Conda env `/data/users/chjing/miniforge3/envs/voltanallm` (python 3.12 + CUDA 12.8.1 toolkit for
   the flashinfer JIT). The shared `cuda-env` (vLLM 0.15.1) is not touched. Reused if it exists
   (`FORCE_REBUILD=1` rebuilds).
2. Upstream VoltanaLLM (`Supercomputing-System-AI-Lab/VoltanaLLM`, 27417323 = "Release VoltanaLLM v0.1" on branch `voltanallm`, SGLang 0.4.7.post1)
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

## v3 (replaces v2 = Slurm job 279404, and 279367)

- 279367 failed at `pip_sglang`: the sdist-only `flashinfer-python` pins torch to the torch in its
  build env, and pip's isolated build pulled the newest one (`torch==2.14.*`). Since v2: torch 2.7.1
  is installed first and flashinfer is built with `--no-build-isolation`. v3 reads the flashinfer pin
  from the checked-out `pyproject.toml` (0.2.6.post1 at 27417323).
- 279404 installed everything (imports, `pip check`, NIXL UCX agent, clock lock all OK) but the
  prefill server died at import: `schedule_batch.py` imports `TokenToKVPoolAllocator` from
  `mem_cache.memory_pool`. The repo's `main` (f7789acf) merges the v0.1 release into SGLang ~0.4.9
  and the merge left many stale imports (also `io_struct` -> removed `openai_api.protocol` and
  `mm_utils`, `scheduler` -> removed `expert_distribution`, ...). The release commit 27417323
  itself has none of these (static `from sglang... import` check), and the patch applies to it
  unchanged. The packages installed by 279404 are replaced by the 0.4.7.post1 pins.
- The NIXL check segfaults at interpreter exit (rc 139) after `NIXL_UCX_AGENT_OK`; agent creation
  works, so it is judged by that line, as before.
