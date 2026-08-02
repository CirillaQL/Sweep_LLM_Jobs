# KV scheduler autoscale and multi-GPU power run, retry 1

Retry after clearing inherited `SLURM_TRES_PER_TASK` before nested `srun`
steps. The Slurm allocation retains the hard 30-minute limit.
