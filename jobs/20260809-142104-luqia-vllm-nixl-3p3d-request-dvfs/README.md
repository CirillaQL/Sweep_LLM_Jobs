# NIXL 3P3D request-level predictive DVFS test

This pending Slurm job reuses the validated NIXL asymmetric-TP topology:

- Prefill on Uranus/L40S: `P0=TP1`, `P1=TP1`, `P2=TP2`
- Decode on Ganymede/L4: `D0=TP1`, `D1=TP1`, `D2=TP2`
- Random routing across all nine NIXL P-D pairs

For every request, the proxy predicts TTFT/TPOT safety for the selected route
using that route's actual Prefill and Decode TP values. It chooses the
lowest-predicted-power safe frequency pair, requests `nvidia-smi -lgc` from
the two instance-local clock agents, verifies the observed graphics clocks,
and only then forwards the request.

Durable results are written under `results/<job_id>/pd_runtime/`:

- `request_dvfs_decisions.jsonl`: route, workload/SLO, prediction, frequency
  command/ack, queue snapshot, and proxy latency for every request.
- `gpu_telemetry_<instance>.csv`: actual clock, power, utilization,
  temperature, and memory use for every GPU.
- `request_dvfs_check.json`: end-of-job validation and aggregate counts.

All caches and runtime copies are kept under
`/data/users/chjing/vllm_job_work/<job_id>` and removed at job exit. The
launcher resets all allocated GPU clock locks with `nvidia-smi -rgc` before
removing that directory.
