# vLLM PD 20-minute live ClickHouse ingestion

This job runs one L40S prefill worker on `neptune` and one L4 decode worker on
`ganymede` with:

- `FLASH_ATTN`
- automatic hardware DVFS
- no manual frequency control
- no scheduler-prediction frequency controller
- KV-cache metrics
- iteration-detail logging
- detailed OTLP traces
- request IDs propagated through the client and proxy

The six benchmark windows target about 19-20 minutes of sending time. The Slurm
allocation remains capped at 30 minutes.

## ClickHouse ingestion

`clickhouse_batch_uploader.py` uses only the Python standard library.

During the benchmark it uploads batches of:

- `engine_samples`
- `gpu_samples`

every 60 seconds. After the benchmark it uploads the completed request, event,
token-array, window, environment, OTEL, scheduler and drain tables.

Continuous `vllm_metrics_long.csv` output is disabled. Only the compact wide
snapshot table is collected at 500 ms intervals.

The temporary database credential is contained in
`clickhouse_credentials.env` for this disposable test only. The script never
prints the password.

The job fails if the final ClickHouse upload fails. Persistent CSV artifacts and
the upload state remain in the broker job directory for manual retry. The
temporary `/data/users/chjing/vllm_job_work/<job-id>` workspace is removed on
exit.
