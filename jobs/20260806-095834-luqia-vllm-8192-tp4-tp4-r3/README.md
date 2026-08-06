# Mistral TP4 Prefill / TP4 Decode 8192-token retry 3

This reruns the fixed TP=4 to TP=4 latency experiment to check whether the
Neptune L40S power telemetry has recovered.

- Neptune: one Prefill instance using four L40S GPUs with TP=4.
- Ganymede: one Decode instance using four L4 GPUs with TP=4.
- Model: `mistralai/Mistral-7B-v0.1`.
- Input length: 8192 tokens for every workload.
- Output lengths: 128 and 512 tokens.
- Request rates: 1, 2, and 4 requests/second.
- GPU frequency mode: automatic DVFS; no GPU reset and no manual graphics-clock
  lock.

The allocation-wide power monitor is intentionally unchanged from retry 2.
Its startup check requires two valid samples from each of all four allocated
GPUs on both nodes, so the job directly verifies whether Neptune is reporting
usable board power again before vLLM starts.
