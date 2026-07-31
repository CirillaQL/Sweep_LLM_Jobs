# L40S full-grid calibration on Neptune

Seven balanced shards run strictly one at a time (`--array=0-6%1`) on the
single allowed node `neptune`. Each shard reserves all four L40S GPUs for at
most 24 hours in the `long` partition. A runtime hostname guard exits before
any GPU clock change if Slurm places the task anywhere except Neptune.
The `READY` marker submits this serial array through the broker.
