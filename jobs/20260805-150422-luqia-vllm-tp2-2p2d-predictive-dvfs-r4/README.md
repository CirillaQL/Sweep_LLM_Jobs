# TP2 2P/2D predictive DVFS validation, retry 4

This retries Job 251809 after Neptune L40S GPU 0 entered a `GPU requires
reset` state and P2P NCCL sends failed with an unspecified CUDA launch error.

The topology and workload remain unchanged: two TP=2 Prefill instances on four
Neptune L40S GPUs and two TP=2 Decode instances on four Ganymede L4 GPUs,
using `mistralai/Mistral-7B-v0.1` and the predictive DVFS scheduler.

Before any vLLM or power-monitor process starts, this job performs a privileged
full `nvidia-smi --gpu-reset` of all four allocated Neptune GPUs and validates a
small CUDA operation on every device. It repeats the full reset after all
servers and monitors have stopped, including failure and signal paths. The
reset logs are stored as `gpu_reset_before.log` and `gpu_reset_after.log`.

Unlike retry 3, vLLM startup does not lock the GPUs to their maximum graphics
clock. It starts at the scheduler-selected frequency, while retaining the
sequential Decode-before-Prefill startup and per-instance HTTP readiness checks.
