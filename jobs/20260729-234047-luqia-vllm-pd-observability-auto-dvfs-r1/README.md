# vLLM PD observability, automatic DVFS repair run

This is the repaired 30-minute run for failed Slurm job `250276`.

- Prefill: one Neptune L40S (`10.1.0.6`)
- Decode: one Ganymede L4 (`10.1.0.3`)
- Model: `mistralai/Mistral-7B-v0.1`
- KV transport: `P2pNcclConnector` over the existing 100 GbE links
- `max_num_seqs=64`

## Repair

The first run failed before server startup because the shared `cuda-env`
contains vLLM 0.15.1 but not its optional OpenTelemetry packages. This Job
vendors the required pure-Python packages in one offline zip bundle and adds it
to `PYTHONPATH`; compute nodes do not install or download anything.

Preflight now fails immediately if any required import fails. The earlier
script accidentally returned success after printing an import traceback.

## Automatic DVFS

This run has no project scheduler prediction, clock-controller process,
frequency target, clock request/ack file, placement-frequency JSON, or GPU
clock-setting/reset action. It only reads current clock, p-state, utilization,
power, temperature, and memory for telemetry.

GPU frequency remains under the NVIDIA driver/hardware automatic DVFS policy.
The benchmark changes only request workload windows; it never changes
frequency or PD placement.

## vLLM observability

Both vLLM servers enable:

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

Every benchmark request has a stable `X-Request-Id` and W3C `traceparent`.
The proxy records and forwards both while preserving the special internal ID
format required by the P2P NCCL connector.

The same six workload windows and CSV outputs documented by the first run are
retained: request/token lifecycle, retries/cancellations/status, proxy events,
vLLM metrics and histograms, queue-at-arrival joins, OpenTelemetry spans,
iteration/log events, GPU/network telemetry, drain observations, environment,
and Job metadata.
