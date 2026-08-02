# KV scheduler autoscale and multi-GPU power run

This broker task reruns the fourteen-window saturation trace with prefill and
decode scale-out/scale-in enabled. It reserves four L40S GPUs on Neptune and
four L4 GPUs on Ganymede, records allocation-wide board power, and has a hard
30-minute Slurm limit.
