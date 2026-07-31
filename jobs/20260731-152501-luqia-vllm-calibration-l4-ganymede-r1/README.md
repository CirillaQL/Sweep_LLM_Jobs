# L4 full-grid calibration on Ganymede

Twelve balanced shards run strictly one at a time (`--array=0-11%1`) on the
single allowed node `ganymede`. Each shard reserves all eight L4 GPUs for at
most 24 hours in the `long` partition. A runtime hostname guard exits before
any GPU clock change if Slurm places the task anywhere except Ganymede.
The `READY` marker submits this serial array through the broker.
