# vLLM PD request-lifecycle observability collection

This 30-minute Slurm job runs disaggregated vLLM v0.15.1 on:

- Prefill: Neptune L40S (`10.1.0.6`)
- Decode: Ganymede L4 (`10.1.0.3`)
- Model: `mistralai/Mistral-7B-v0.1`
- KV transport: `P2pNcclConnector` over the existing 100 GbE interfaces
- `max_num_seqs=64`, automatic GPU DVFS, and one GPU per node

The server explicitly enables:

```text
--kv-cache-metrics
--kv-cache-metrics-sample 1.0
--enable-logging-iteration-details
--otlp-traces-endpoint http://10.1.0.3:<job-specific-port>/v1/traces
--collect-detailed-traces all
--enable-prefix-caching
--enable-request-id-headers
--enable-log-requests
--enable-mfu-metrics
```

Every benchmark request gets a stable `X-Request-Id` and W3C `traceparent`.
The PD proxy preserves the client ID in its CSV while constructing the special
internal request ID needed by the P2P NCCL connector. It forwards the trace
context to both Prefill and Decode.

## Workload windows

The load lasts roughly three minutes, leaving ample time for model startup,
drain observation, OTLP flushing, and cleanup inside the 30-minute allocation:

| Window | Input | Output | RPS | Duration | Requests |
|---|---:|---:|---:|---:|---:|
| `baseline_short` | 32 | 32 | 1 | 20 s | 20 |
| `prefill_heavy` | 1024 | 32 | 2 | 30 s | 60 |
| `decode_heavy` | 32 | 256 | 2 | 30 s | 60 |
| `timeout_retry_probe` | 32 | 128 | 2 | 5 s | 10 |
| `queue_pressure` | 32 | 32 | 32 | 30 s | 960 |
| `recovery_short` | 32 | 32 | 1 | 20 s | 20 |

Requests in each window share a controlled prefix so that prefix-cache metrics
are observable. The timeout probe uses a 50 ms client timeout and one retry;
2% of queue-pressure requests are deliberately cancelled after 250 ms. This
produces labeled timeout, retry, and cancellation examples instead of silently
training only on successful requests. The final no-send drain phase waits for
both queues to stay at zero for three consecutive samples.

## CSV artifacts

The job writes all primary observations under `observability_results/`:

- `requests.csv`: planned/ready/send/header/first-token/completion timestamps,
  client backlog, TTFT/TPOT/E2E, token counts, request IDs, HTTP status,
  timeouts, retries, cancellations, and errors.
- `token_timestamps.csv`: every streamed token-event arrival timestamp and
  inter-token interval.
- `client_events.csv`: attempt-level send, response, retry, timeout, cancel,
  and completion events.
- `window_summary.csv` and `workload_events.csv`: realized RPS/TPS and window
  boundaries.
- `proxy_events.csv`: proxy arrival, PD routing, Prefill/Decode dispatch,
  upstream headers, first chunks, completion, status, and ID mapping.
- `vllm_metrics_long.csv`: every Prometheus metric and histogram bucket from
  both `/metrics` endpoints every 0.5 seconds.
- `vllm_metrics_snapshots.csv`: analysis-ready running/waiting queue state,
  waiting growth, KV usage, prefix-cache counters, throughput counters,
  preemptions, scheduler iteration tokens, and latency histogram totals.
- `queue_state_at_arrival.csv`: each request joined to the latest Prefill and
  Decode queue/KV snapshot at its proxy-arrival timestamp, including sample
  age and client-to-proxy delay.
- `otel_spans.csv` and `otel_batches.csv`: decoded OpenTelemetry spans,
  attributes, events, trace IDs, span IDs, and timing. Raw protobuf batches are
  also retained for lossless reprocessing.
- `vllm_observability_log_events.csv`: iteration, scheduler, engine, request,
  KV-cache, and preemption records extracted from both vLLM server logs.
- `prefill_neptune_telemetry.csv` and `decode_ganymede_telemetry.csv`: 0.5 s
  GPU clocks, utilization, power, memory, temperature, p-state, and network
  RX/TX.
- `environment_neptune.csv` and `environment_ganymede.csv`: topology, GPU,
  driver, Slurm allocation, model, PD, and observability configuration.
- `job_metadata.csv`: fixed model, placement, scheduler, KV connector, port,
  and observability settings for this allocation.
- `drain_samples.csv` and `drain_summary.csv`: queue-to-zero observations and
  measured drain time.
