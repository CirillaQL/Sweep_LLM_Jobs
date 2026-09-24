# K1 burst-overhead gate

Goal: measure per-request prefill overhead under bursts and the decode-busy
first-token delay on the unchanged r3 P0/D0 substrate.

Red lines: do not modify prior jobs or results; use only neptune GPU0 and ganymede
GPU0; do not change the connector, vLLM flags or model; never retune a GPU that
Slurm did not allocate (the GPU guard exits otherwise).

Acceptance: static preflight passes; one P0→D0 smoke request completes before any
measurement; `pd_requests.csv` and `summary.json` are produced even if the
deadline truncates later phases (the summary's `phase_log` states what ran).
