# KV scheduler autoscale and multi-GPU power run, retry 2

This retry removes inherited batch-level GPU and memory defaults before nested
`srun` steps. Each vLLM step requests one GPU and 24 GiB explicitly, and the
node launcher rejects any vLLM step that can see more than one GPU. The job has
a hard 30-minute limit.
