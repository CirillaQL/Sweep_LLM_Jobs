# max_num_seqs=128 queue-backlog experiment

This is one member of a four-job controlled experiment derived from Slurm Job
250242. Prefill and Decode both use `--max-num-seqs 128`.

All other experiment settings are held constant:

- `32 input / 32 output` at `32 RPS` for `60 seconds`;
- 1,920 requests with benchmark `max_concurrency=512`;
- latency-only scheduler with KV-cache transfer delay enabled;
- no artificial L4 frequency limit;
- the same baseline and recovery windows;
- the same placement, frequency search, monitoring, and analysis code.

The Slurm allocation is limited to 30 minutes, so this value is tested in an
independent job.
