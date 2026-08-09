# NixlConnector 1P1D TP=1 connectivity test, retry 2

This job verifies the per-job HOME and NIXL config-path fix with the minimal
cross-node topology:

- Uranus: one L40S `P0(TP=1)`;
- Ganymede: one L4 `D0(TP=1)`;
- KV connector: `NixlConnector`;
- only valid route: `P0-D0`.

The default model is `mistralai/Mistral-7B-v0.1`. The benchmark sends eight
requests with 512 input tokens and 64 output tokens. Success requires both
instances to register, the registry to report `NixlConnector`, all requests to
succeed, and the proxy log to contain only the `P0-D0` route.

All writable user, configuration, cache, and temporary paths are placed under
`/data/users/chjing/vllm_job_work/<job_id>` and removed when the job exits.

The `READY` marker causes the broker worker to submit this job after the commit
is pushed.
