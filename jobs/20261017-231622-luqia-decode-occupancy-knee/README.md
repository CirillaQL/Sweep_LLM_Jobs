# 20261017-231622-luqia-decode-occupancy-knee

Decode-side occupancy knee (paper motivation, Fig. limits b). One L4 on ganymede, Mistral-7B,
closed-loop decode at 4..64 concurrent requests (512-token prompts, 256 output tokens) with the
decode clock locked at 1170 MHz (Canary working clock) and 2040 MHz (maximum); KV-cache usage,
running/waiting requests, preemptions, energy per token per 30 s window; TTFT/TPOT per request.
gpu-memory-utilization 0.7826 as in production. Code: K3b's gpu_char.py / serve.py, unchanged.
Expected run time about 23 min (short partition).
