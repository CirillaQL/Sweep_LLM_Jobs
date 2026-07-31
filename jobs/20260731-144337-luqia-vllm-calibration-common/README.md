# Full-grid single-pool vLLM DVFS calibration

This campaign reproduces the unique hardware/workload configurations in the
Phase2 L40S and L4 master CSV files. It is intentionally **not** a PD job:
each benchmark uses one vanilla vLLM instance and one homogeneous GPU pool.

- L40S: TP 1/2/4, 509 configurations, 3 repeats, 7 balanced shards.
- L4: TP 1/2/4/8, 566 configurations, 3 repeats, 12 balanced shards.
- A logical repetition whose historical median exceeded 20 minutes is split
  into equal prompt-count segments. Segment IDs are part of the ClickHouse key,
  and their prompt counts sum exactly to the original logical repetition.
  Repetition/segment-specific deterministic seeds avoid replaying the same
  random prompt prefix in every segment.
- All active GPUs in a TP group receive the same `nvidia-smi -lgc f,f` target.
- Memory clocks are observed but not changed.
- There is no online frequency predictor or manual scheduling policy. vLLM
  schedules requests normally while the experiment controls only the GPU core
  clock with `nvidia-smi -lgc`.
- Runtime `clocks.sm` samples are checked against the target; target frequency
  alone is not treated as proof that the lock held.
- Each segment uses the streaming request client and records planned/actual
  send times, first-token and per-token arrivals, completion, TTFT/TPOT/E2E,
  retries, cancellations, HTTP status and request IDs.
- vLLM runs with KV-cache metrics, iteration-detail logging, prefix caching,
  detailed OTLP traces, request-ID response headers, MFU metrics and
  `FLASH_ATTN`.
- Every 500 ms the collector records engine queue/cache counters and per-GPU
  clocks, utilization, power, memory, temperature and host RX/TX. Histogram
  buckets are captured at both workload-window boundaries. The postprocessor
  also emits scheduler iterations, queue state at arrival and drain samples.
- After every segment, the complete batch is inserted into the canonical
  `experiments`, `jobs`, `job_nodes`, `workload_windows`, `requests`,
  `request_attempts`, `request_events`, `request_token_series`,
  `engine_samples`, `histogram_buckets`, `scheduler_iterations`, `gpu_samples`,
  `otel_spans`, and `drain_samples` tables. The calibration-only ClickHouse
  tables are not used by the new runner.
- `kv_transfer_events` is intentionally empty because these are homogeneous
  single-pool tests, not Prefill/Decode-transfer experiments. Combined-engine
  queue samples are labelled `combined_vllm_metrics`; they are never presented
  as measurements from two independent PD endpoints.
- Segment artifacts are staged only under
  `/data/users/chjing/vllm_job_work/calibration_<job>_<array-task>/results`;
  no benchmark CSV, trace, telemetry, or server log is written below the Git
  broker tree.
- Once a segment passes integrity checks and its ClickHouse batch insert
  succeeds, its complete artifact directory is deleted immediately and the
  live vLLM server log is truncated. Failed uploads remain only until the task
  exits so their error can be printed, then the task work directory is removed.
- Before a benchmark starts, the GPU/network monitor must produce a complete
  sample for every participating TP GPU. A startup or runtime telemetry error,
  an incomplete observability bundle, or a ClickHouse upload error terminates
  the shard immediately instead of retrying every configuration. Logs report
  `clickhouse_upload_status=not_attempted_integrity_failed` separately from a
  real uploader return code.
- Every new array task also removes calibration-owned work directories and
  external Slurm logs older than two days. Every exit path resets all allocated
  GPUs with `nvidia-smi -rgc` and removes its exact task work directory.

The manifests are generated from the authorized read-only source files with:

```bash
python3 build_calibration_manifests.py \
  --source-dir /Users/lukeqian/Obsidian/Obsidian/Dual_Sweep_LLM/calibration_data \
  --output-dir .
```

The accompanying array jobs use `--partition=long` and `--time=24:00:00`.
This common directory intentionally has no `READY` marker; it is shared by
future array-job directories and must not be submitted by itself.
New submission directories should copy the matching file from `templates/` as
`run.sbatch`. Those templates send Slurm stdout/stderr outside Git as well.

Run the local regression suite before creating a submission directory:

```bash
python3 -m unittest -v test_calibration_runner.py
```
