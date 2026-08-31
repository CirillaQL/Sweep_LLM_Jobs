# XPYD Job 255348 log/audit diagnostic

This read-only diagnostic job extracts bounded error evidence and small
per-window audit/summary artifacts from:

`/data/users/chjing/vllm_job_work/255348`

It runs on Uranus, requests no GPU, does not start vLLM, and does not modify the
failed experiment cache.  The Git result contains only a file inventory,
redacted error matches, bounded log tails, and compact controller artifacts.
Raw
logs remain in the cache directory.

Expected result directory:

`results/<diagnostic_slurm_job_id>/`
