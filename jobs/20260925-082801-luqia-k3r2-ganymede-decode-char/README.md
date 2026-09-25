# K3 retry: plain-vLLM decode characterization on one L4 (ganymede)

Retry of job 267572, which locked the clock successfully (`clock_permission.txt`)
and then exited with status 9 two seconds later; its stderr was not visible
because `slurm-*.out/err` are gitignored. Changes:

* the script's own stdout/stderr are mirrored to `results/<id>/job_console.log`;
* clock-lock success is judged by nvidia-smi's "All done" text and its exit status
  is recorded (`lock_rc=`), in the shell preflight and in the driver;
* a generated token that decodes to "" is still counted;
* job name `cantuning-ganymede` + `--dependency=singleton`, shared with K1b, so the
  two jobs never run on ganymede concurrently.

Design unchanged from the first K3: D clock 1500 → 1050 → 750 → 1275 MHz;
input 128 × concurrency 1/2/4/8/16/32 and input 1024 × concurrency 1/8/24;
256 output tokens; idle power (unlocked/1500/1050) and 1050↔1500 switch latency.
Outputs: `decode.csv`, `gpu_events.csv`, `power_trace.csv`, `summary.json`,
`gpu_allocation.txt`, `clock_permission.txt`, `job_console.log`, `vllm_server_tail.log`.
