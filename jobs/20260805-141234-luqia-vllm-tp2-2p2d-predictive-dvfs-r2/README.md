# TP2 2P/2D predictive DVFS validation, retry 2

This retries the failed Job 251807 after fixing runtime TCP port allocation.
Before starting the proxy or vLLM, the shared runner snapshots occupied TCP
ports on both nodes and selects one common free range covering every HTTP and
per-TP-rank KV port. The selected range is recorded in `port_allocation.json`.

Topology and policy are unchanged:

- Neptune: two Prefill instances, each TP=2 on two L40S GPUs.
- Ganymede: two Decode instances, each TP=2 on two L4 GPUs.
- Model: `mistralai/Mistral-7B-v0.1`.
- Scheduler: minimum-power frequency predicted to satisfy P99 TTFT <= 500 ms
  and P99 TPOT <= 200 ms.
- Manual clock control: `sudo nvidia-smi -lgc` on all eight allocated GPUs.

The job records power, clock telemetry, TTFT, TPOT, ITL, throughput, and the
prediction-versus-observation error report. It reuses the validated prediction
checker from the first attempt while writing every output into this new job
directory.
