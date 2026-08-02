# KV scheduler autoscale and multi-GPU power run, retry 3

This retry disconnects background `srun` and benchmark stdin from the workload
CSV loop. It also requires all fourteen benchmark windows to complete before
the runner may return success. The Slurm time limit remains 30 minutes.
