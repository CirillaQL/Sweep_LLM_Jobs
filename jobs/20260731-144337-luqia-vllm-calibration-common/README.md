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
- There is no scheduler, online frequency predictor, or automatic DVFS policy.
- Runtime `clocks.sm` samples are checked against the target; target frequency
  alone is not treated as proof that the lock held.
- Results and raw GPU telemetry are inserted into ClickHouse after every run.
  Successful keys are queried on restart, making an array task resumable.
- `calibration_logical_runs` combines completed segments into configuration ×
  repeat totals. Counts, tokens, durations, energy, throughput, and weighted
  means aggregate exactly from segment summaries. Global median/p99 cannot be
  reconstructed from summaries, so the view labels those fields explicitly as
  maximum segment p99 rather than presenting them as exact global percentiles.
- Local raw telemetry is removed only after ClickHouse acknowledges both the
  run row and sample batch. Compact JSONL summaries remain in the job output.
- Every exit path resets all allocated GPUs with `nvidia-smi -rgc` and removes
  only `/data/users/chjing/vllm_job_work/<job>_<array-task>`.

The manifests are generated from the authorized read-only source files with:

```bash
python3 build_calibration_manifests.py \
  --source-dir /Users/lukeqian/Obsidian/Obsidian/Dual_Sweep_LLM/calibration_data \
  --output-dir .
```

The accompanying array jobs use `--partition=long` and `--time=24:00:00`.
This common directory intentionally has no `READY` marker; it is shared by the
two submitted array-job directories and must not be submitted by itself.
