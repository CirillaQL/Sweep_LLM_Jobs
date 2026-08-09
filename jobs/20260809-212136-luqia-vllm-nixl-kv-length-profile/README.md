# Tagged NIXL KV length profile

This pending job isolates one TP=1 L40S Prefill (`P0`) and one TP=1 L4
Decode (`D0`) and fixes every request to `P0-D0`. Predictive DVFS is disabled
so clock-setting overhead does not contaminate the transfer measurements.

After two 512-token warmups initialize the NIXL path, the job sends three
sequential requests at each input length: 128, 256, 512, 1024, 2048, and 4096
tokens. The Completions API receives token-ID arrays, so the input lengths are
exact. Each request carries `X-Experiment-Tag`, `X-PD-Route`, and
`X-Input-Tokens` headers.

The proxy trace records per-request Prefill HTTP time, Decode-to-first-chunk
time, client-visible TTFT, route, and tag. The benchmark also waits 15 seconds
after every length group and associates the Decode server's NIXL transfer
metrics with that group. Expected Mistral-7B KV payloads range from 16 MiB to
512 MiB.

Primary outputs:

- `nixl_length_profile.csv`: one row per input length
- `nixl_length_profile.json`: requests plus parsed NIXL metrics
- `tagged_request_events.jsonl`: client timings and tags
- `pd_runtime/request_trace.jsonl`: instrumented proxy stage timings
- `decode_metrics_len_<tokens>.log`: Decode log slice for each group

All caches and writable runtime paths use
`/data/users/chjing/vllm_job_work/<job_id>` and are removed at job exit.
