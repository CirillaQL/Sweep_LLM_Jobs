# vLLM HTTP clock-ack and context-boundary validation

Job 250144 showed that both GPU nodes applied their requested clocks, but the
batch parent did not observe Neptune's in-place acknowledgement through the
shared filesystem before the 90-second deadline.

This job validates the replacement acknowledgement path. Each node publishes
the result of `nvidia-smi -lgc` to the existing in-memory PD proxy; the parent
polls that HTTP endpoint and retains the shared acknowledgement file only as a
fallback and audit artifact.

The three workloads then exercise high-input requests at the 4096 total-context
boundary:

- input 4032 plus output 64 with burstiness 0.25;
- input 4095 plus output 1 with uniform arrivals;
- input 4095 plus output 1 with burstiness 0.25.

Each window sends 32 prompts with KV-aware online scheduling, L40S prefill, L4
decode, 0.5-second power telemetry, request-success parsing, and final two-node
`-rgc` verification. The Slurm wall-time remains capped at 28 minutes.
