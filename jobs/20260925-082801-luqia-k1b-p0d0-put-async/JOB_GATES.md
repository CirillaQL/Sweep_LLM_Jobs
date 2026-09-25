# K1b PUT_ASYNC gate

Goal: decide whether the per-burst KV cost seen in K1 disappears with an
asynchronous producer send, and collect P/D TTFT versus in-flight state under
Poisson arrivals.

Red lines: do not modify prior jobs; use only uranus GPU0 and ganymede GPU0; do
not change model, vLLM flags or connector other than `send_type`; never retune a
GPU Slurm did not allocate.

Acceptance: preflight passes; one P0→D0 smoke request completes within 60 s
before any measurement (otherwise `smoke_failure.json` records the failure and
the job ends); `pd_requests.csv` and `summary.json` (with `send_type`) are
written even if the deadline truncates the Poisson phase.
