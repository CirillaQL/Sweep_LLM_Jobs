# Full-observability calibration output

The calibration runner uploads one complete batch after every workload segment.
It does not write `calibration_runs`, `calibration_gpu_samples`, or
`calibration_shards`.

| Required data | Canonical ClickHouse destination |
|---|---|
| Experiment, Slurm job, hardware and configuration | `experiments`, `jobs`, `job_nodes` |
| Workload summary, RPS/TPS and client-observed drain | `workload_windows` |
| Planned/actual send, response, first token, completion, TTFT/TPOT/E2E, tokens, status, timeout/cancel/retry | `requests` |
| Individual HTTP attempts | `request_attempts` |
| Client lifecycle events with request/trace IDs | `request_events` |
| Packed per-token arrival and inter-token offsets | `request_token_series` |
| 500 ms queue, KV-cache, prefix-cache, throughput and latency counters | `engine_samples` |
| Cumulative Prometheus histogram boundaries at window start/end | `histogram_buckets` |
| Parsed iteration, continuous-batching and scheduler state | `scheduler_iterations` |
| Per-GPU clock, utilization, power, memory, temperature and host RX/TX | `gpu_samples` |
| Detailed vLLM spans correlated by trace/request ID | `otel_spans` |
| Queue state after sending stops until three consecutive zero samples | `drain_samples` |

These are homogeneous single-pool experiments. `engine_samples.role` is
`combined`, and the job topology is `single_pool`. There is no KV transfer
between Prefill and Decode nodes, so `kv_transfer_events` is correctly empty.
The request table retains its PD-compatible queue columns; their source is the
single combined endpoint and the job topology must be used when interpreting
them.

Before uploading a segment, the runner requires non-empty request, client
event, engine, GPU/network, histogram, OTLP, drain, queue-join, and parsed-log
artifacts, and verifies that the request row count equals the planned prompt
count. Successful streamed responses are also rejected by the client if their
per-token server-ID timestamp sequence is incomplete.
