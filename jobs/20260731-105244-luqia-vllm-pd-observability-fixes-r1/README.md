# vLLM PD observability correctness fixes (r1)

This job runs one L40S prefill worker on `neptune` and one L4 decode worker on
`ganymede` with:

- `FLASH_ATTN`
- automatic hardware DVFS
- no manual frequency control
- no scheduler-prediction frequency controller
- no proxy clock-control endpoints
- KV-cache metrics
- iteration-detail logging
- detailed OTLP traces
- request IDs propagated through the client and proxy

The six benchmark windows target about 19-20 minutes of sending time. The Slurm
allocation remains capped at 30 minutes.

This directory contains a `READY` marker for broker submission.

## Correctness fixes

- Freeze the metrics collector, proxy, both vLLM servers/GPU telemetry
  monitors, and the OTLP receiver before the final ClickHouse flush.
- Verify that each live-source CSV row count equals its persisted uploader
  offset before declaring the final upload complete.
- Request vLLM 0.15.1 `return_token_ids=true`. A multi-token SSE chunk is
  expanded into one row per returned token ID with the shared network-arrival
  timestamp. Raw chunk timing is retained separately in
  `stream_chunk_events.csv`.
- Reject a successful benchmark window if output-token usage and token-ID
  timestamp counts differ.
- Build `kv_transfer_events.csv` for every routed request. Because
  `P2pNcclConnector` does not expose request-level pure NCCL timing, the
  duration is explicitly labelled as a proxy-observed Prefill+KV handoff upper
  bound and the byte count as a model-derived upper-bound estimate.
- Record exact proxy Prefill/Decode in-flight counts on every proxy event.
  Request-at-arrival rows use the proxy Prefill in-flight count when the
  500-ms vLLM scrape misses the short Prefill execution.
- Parse vLLM iteration-detail messages into real context/generation request
  counts, token counts, iteration IDs and elapsed times instead of inserting
  zero-filled scheduler rows.

## ClickHouse ingestion

`clickhouse_batch_uploader.py` uses only the Python standard library.

During the benchmark it uploads batches of:

- `engine_samples`
- `gpu_samples`

every 60 seconds. After the benchmark it uploads the completed request, event,
token-array, window, environment, OTEL, scheduler, KV handoff and drain tables.

Continuous `vllm_metrics_long.csv` output is disabled. Only the compact wide
snapshot table is collected at 500 ms intervals.

The temporary database credential is contained in
`clickhouse_credentials.env` for this disposable test only. The script never
prints the password.

The job fails if the final ClickHouse upload fails. Persistent CSV artifacts and
the upload state remain in the broker job directory for manual retry. The
temporary `/data/users/chjing/vllm_job_work/<job-id>` workspace is removed on
exit.
