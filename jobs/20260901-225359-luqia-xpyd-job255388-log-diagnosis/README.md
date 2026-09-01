# Read-only diagnosis Job for Slurm 255388

This CPU-only Job inspects the failed XPYD run `255388` from inside the cluster.
It does not use SSH, request a GPU, start vLLM, change GPU clocks, or modify the
target Job/cache directories.

It reads:

- the broker-side Slurm stdout/stderr and status files;
- `/data/users/chjing/vllm_job_work/255388/feedback_frequency_table_2p2d`;
- the persisted Frequency Table and feedback event stream;
- bounded tails of proxy, vLLM, server, audit, clock, request, and summary logs.

Results are written to `results/<diagnostic_slurm_job_id>/` as
`diagnosis.json`, `diagnosis.md`, `artifact_manifest.json`, and `log_tails/`.
Each captured log is limited to its final 2 MiB, with a 20 MiB aggregate cap,
so the broker repository does not receive unbounded raw output.

The Job requests one CPU task, two CPU cores, 4 GiB memory, no GPU, and a
10-minute limit on the `long` partition.

The `READY` marker is present for broker submission.
