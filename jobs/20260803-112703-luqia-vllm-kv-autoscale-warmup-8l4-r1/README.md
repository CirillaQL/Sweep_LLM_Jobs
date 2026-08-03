# KV-aware autoscaling test with per-window warmup

This live Slurm test reserves all eight L4 GPUs on Ganymede and all four L40S
GPUs on Neptune. The scheduler can scale the decode pool from one to eight L4
instances and the prefill pool from one to four L40S instances. The existing
proxy sends new requests across every registered instance.

For every workload window, the runner first makes the scheduling decision,
performs any scale-in or scale-out, waits for proxy registration and clock
acknowledgements, then waits another 30 seconds before sending requests. The
settling period is written to `events.csv` as `warmup_start` and `warmup_end`;
the measured workload and energy window starts only afterward.

The trace retains fourteen baseline, saturation, long-context, and recovery
windows. Formal request windows are reduced to 6--8 seconds with at most 300
requests so the extra 420 seconds of warmup fit inside the hard 30-minute job
limit.

Allocation-wide power monitoring samples twelve GPU UUID streams at 0.5-second
intervals and validates the asymmetric host counts: eight on Ganymede and four
on Neptune.

Results are written to `kv_scheduler_autoscale_warmup_8l4_results/`.
