# K4a third retry (neptune P) P-tier Poisson gate

Goal: energy per request and SLO violation rate for P clocks 1305/1815/2520 under
identical Poisson workloads at 2/4/6 req/s through P0(neptune)→D0(ganymede), plus
park idle power.

Red lines: do not modify prior jobs; use only the GPU Slurm allocated on neptune and on ganymede (UUID-checked); do not
change model, vLLM flags or connector (PUT_ASYNC); never retune a GPU Slurm did not
allocate.

Acceptance: preflight passes; the P0→D0 smoke request completes within 60 s before
any measurement; `pd_requests.csv` and `summary.json` (with `phase_log` holding each
window's energy) are written even if the deadline truncates later windows.

Retry gate: if either vLLM server shows no output after 180 s, stop and keep
`startup_diagnostics.txt`; never run the smoke request against a server that failed to start.
GPU gate: every clock lock/reset checks the allocated GPU UUID first (exit 97 on mismatch); each vLLM server must be verified on its allocated UUID before any measurement.
