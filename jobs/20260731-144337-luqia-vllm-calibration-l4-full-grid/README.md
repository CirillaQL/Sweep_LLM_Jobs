# L4 full-grid array job

This is a twelve-task Slurm array for the 566 unique L4 configurations. Each
task reserves one entire eight-L4 node for at most 24 hours on `long`; at most
four tasks run concurrently across `ganymede`, `io`, `europa`, and `callisto`.

The `READY` marker asks the broker to submit `run.sbatch` once; Slurm expands
the twelve-task array. Exit code 75 is a safe checkpoint; resubmitting the same
array task continues from ClickHouse.
