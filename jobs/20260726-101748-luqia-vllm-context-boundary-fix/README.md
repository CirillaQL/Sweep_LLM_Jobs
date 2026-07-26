# vLLM context-boundary reliability validation

Job 250143 configured random input length 4096 and output length 1. The random
dataset produced 4095 text tokens, then server-side tokenization added BOS,
creating 4096 input tokens. With one requested output token, that exceeded the
server's total `--max-model-len 4096` boundary.

This validation uses input length 4095. The random dataset therefore produces
4094 text tokens; BOS makes the server-side input 4095, leaving exactly one
token of output capacity.

The trace also verifies:

- direct propagation of the pinned vLLM environment's `--burstiness` option;
- request-level success/failure parsing even when `vllm bench serve` exits 0;
- energy JSON/CSV output when a window has zero successful requests;
- memory-safe monitor-based clock acknowledgements;
- final two-node `-rgc` reset verification.

The three windows run 32 prompts each and retain KV-aware online scheduling,
exclusive L40S-prefill/L4-decode allocation, 0.5-second power telemetry, and a
hard 28-minute Slurm limit.
