# Clean up stuck vLLM servers on uranus

K4a jobs 267652 and 267664 left vLLM prefill servers on uranus in D state
(`nfs_wait_bit_killable`): uranus could not read `/data` over NFS.

This job runs on uranus with 1 CPU, 1 GB and no GPU, for at most 10 minutes. It
finds this account's `vllm.entrypoints.openai.api_server` processes whose cgroup
belongs to job 267652 or 267664, and sends them SIGTERM, then SIGKILL. The NFS
wait is killable, so SIGKILL should end them. Their state is recorded before and
after in `results/<id>/cleanup.log`.

It then times reads from `/data`, killing each after its timeout:

* `stat` of the env Python;
* Python start-up;
* `import torch`;
* a 64 MiB read of the model weights.

It uses no sudo and makes no clock changes, and it touches no other process or
job.
