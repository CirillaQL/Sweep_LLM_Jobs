# Mistral TP4 Prefill / TP4 Decode 8192-token retry

This retries the fixed TP=4 to TP=4 latency experiment without performing a
full GPU reset.

- Neptune: one Prefill instance using four L40S GPUs with TP=4.
- Ganymede: one Decode instance using four L4 GPUs with TP=4.
- Model: `mistralai/Mistral-7B-v0.1`.
- Input length: 8192 tokens for every workload.
- Output lengths: 128 and 512 tokens.
- Request rates: 1, 2, and 4 requests/second.
- GPU frequency mode: automatic DVFS; no `nvidia-smi --gpu-reset` and no
  manual graphics-clock lock.

The shared runner uses dynamically selected ports and requires both services
to pass `/v1/models` before the smoke request and benchmarks. Results include
detailed benchmark data, power telemetry, TTFT, TPOT, ITL, throughput, and an
automated latency-metrics report.
