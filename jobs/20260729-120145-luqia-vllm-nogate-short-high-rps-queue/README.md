# No-gate short-request high-RPS queue buildup test

This job tests a latency-only scheduler failure mode in which each request is
individually light, but sustained arrival rate exceeds service capacity and
causes the server queue and observed TTFT to grow.

The main workload is:

```text
input_len=32
output_len=32
request_rate=32 requests/s
duration=60 s
num_prompts=1920
burstiness=4.0
max_concurrency=512
```

With the current portable model and KV-aware TTFT term, the no-gate scheduler
classifies this workload as `OK` and recommends L40S/L4 frequencies of
735/2040 MHz. Its predicted P99 TTFT/TPOT are approximately 151.6/93.4 ms,
both below the 500/200 ms SLOs. The saturation model, which is intentionally
not used for admission in this job, predicts high prefill/decode saturation
probabilities of approximately 0.783/0.949.

The trace includes a low-rate baseline, the sustained overload, and a low-rate
recovery. Unlike earlier tests, client maximum concurrency is raised from 16
to 512 so excess requests can reach vLLM rather than waiting behind the load
generator's semaphore.

Artifacts include:

- vLLM detailed per-request results;
- one-second prefill/decode `/metrics` samples;
- five-second request-order TTFT/TPOT/E2E summaries;
- five-second running/waiting queue summaries;
- the existing GPU telemetry, energy, scheduler decision, and aggregate
  benchmark summaries.
