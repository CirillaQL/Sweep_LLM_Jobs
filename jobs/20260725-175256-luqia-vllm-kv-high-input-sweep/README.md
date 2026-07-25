# KV-aware high-input vLLM sweep

This job runs the complete high-input portion of `experiment/input_samples.csv`:
16 windows spanning 1024, 1536, 2048, 2560, 3072, 3584, 4032, and 4096 input
tokens. Each length has both a low-rate and a higher-rate or burst-designated
point.

It requests an exclusive L40S on Neptune for prefill and L4 on Ganymede for
decode. The vLLM PD service starts once, while the online
`latency_plus_saturation` scheduler recomputes GPU frequencies before every
window. Safe configurations are ranked by predicted cluster board power, with
the minimum-SLO-violation fallback retained for overload points.

## KV-aware scheduling

Predicted TTFT includes the prompt KV-cache transfer term:

```text
T_kv = input_tokens
     * (2 * 32 layers * 8 KV heads * 128 head_dim * 2 bytes)
     * 8 / 24.076 Gbit/s
```

The resulting transfer time grows from about 44.6 ms at 1024 tokens to about
178.4 ms at 4096 tokens, so this trace directly exercises KV-aware candidate
admission and frequency selection.

## Runtime and outputs

Per-window prompt count is capped at 24 and benchmark timeout at 120 seconds.
The Slurm allocation has a hard 28-minute limit.

The inherited runner writes scheduler candidates and active decisions, target
and observed clocks, 0.5-second GPU/network telemetry, event brackets,
benchmark outputs, integrated per-window GPU energy, and the joined live
performance/power summary under `kv_high_input_results/`.
