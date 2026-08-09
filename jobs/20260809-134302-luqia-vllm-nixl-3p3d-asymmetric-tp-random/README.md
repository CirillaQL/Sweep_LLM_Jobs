# NIXL 3P3D asymmetric-TP random connectivity test

This job performs a focused NixlConnector-only validation of random
heterogeneous-TP Prefill/Decode routing:

- Uranus Prefill instances: `P0(TP=1)`, `P1(TP=1)`, `P2(TP=2)`;
- Ganymede Decode instances: `D0(TP=1)`, `D1(TP=1)`, `D2(TP=2)`;
- connector: `NixlConnector`;
- candidate routes: all nine P-D pairs, including TP1-to-TP2 and TP2-to-TP1.

The quick benchmark uses `mistralai/Mistral-7B-v0.1`, 96 requests, 256 input
tokens, 32 output tokens, and a configured rate of 6 requests/s. It passes only
if all requests succeed, all nine routes are observed, the Registry reports the
correct TP topology and nine candidates, and all six vLLM instances stay alive.

All writable configuration, cache, and temporary paths are contained in
`/data/users/chjing/vllm_job_work/<job_id>` and removed on exit.
