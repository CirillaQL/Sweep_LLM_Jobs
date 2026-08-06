# Mistral Uranus TP2 Prefill / Ganymede TP2 Decode auto-DVFS test

This fixed-topology PD experiment runs one Prefill instance on two Uranus L40S
GPUs (TP=2) and one Decode instance on two Ganymede L4 GPUs (TP=2).

- Model: `mistralai/Mistral-7B-v0.1`.
- Prefill: Uranus `10.1.0.5`, two L40S GPUs, TP=2.
- Decode and proxy: Ganymede `10.1.0.3`, two L4 GPUs, TP=2.
- GPU frequency mode: default hardware DVFS; no `nvidia-smi -lgc` or `-rgc`.
- Benchmarks: four random-input workloads covering 512 through 8192 tokens.

The checker requires all requests and latency metrics, the TP2/TP2 registry,
one telemetry stream per role, and two allocation-wide GPU power streams per
node. SLO violations are reported but do not invalidate an otherwise complete
measurement.
