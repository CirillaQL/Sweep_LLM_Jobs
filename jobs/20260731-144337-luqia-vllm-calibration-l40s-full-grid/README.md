# L40S full-grid array job

This is a seven-task Slurm array for the 509 unique L40S configurations. Each
task reserves one entire four-L40S node for at most 24 hours on `long`; at most
two tasks run concurrently across `neptune` and `uranus`.

The `READY` marker asks the broker to submit `run.sbatch` once; Slurm expands
the seven-task array.
Exit code 75 means the task stopped at a safe checkpoint and the same array
task can be submitted again. ClickHouse success keys prevent duplicate work.
