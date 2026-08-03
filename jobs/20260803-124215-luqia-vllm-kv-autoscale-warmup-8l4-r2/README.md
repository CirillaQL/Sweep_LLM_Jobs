# KV-aware autoscaling warmup test r2

This rerun fixes the r1 startup failure by launching allocation-wide power
monitoring as two host-specific overlapping Slurm steps: eight L4 GPUs on
Ganymede and four L40S GPUs on Neptune.

The scheduler can scale decode from one to eight L4 instances and prefill from
one to four L40S instances. Each workload waits 30 seconds after scaling,
registration, and clock acknowledgement before requests begin. Fourteen
6--8-second request windows remain under the hard 30-minute job limit.

Results are written to `kv_scheduler_autoscale_warmup_8l4_results/`.
