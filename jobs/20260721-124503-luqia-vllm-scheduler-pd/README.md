# Live SWEEP scheduler + vLLM PD job

This job allocates one L40S GPU on `neptune` and one L4 GPU on `ganymede`.
It runs `scheduler.py` inside the Slurm allocation before either vLLM server is
started. The scheduler enumerates both Prefill/Decode placements and every
supported clock pair at TP=1, keeps the phase-model safe configurations, and
selects the lowest predicted cluster power.

The resulting `scheduler_pd_results/placement.json` controls:

- which node starts the Prefill and Decode vLLM roles;
- which GPU clock each node applies with `nvidia-smi -lgc`;
- the role-specific P2pNcclConnector producer/consumer settings.

The benchmark and scheduling workload are the same: input length 512, output
length 64, request rate 1 request/s, TTFT SLO 500 ms, and TPOT SLO 200 ms.
The job resets GPU clocks with `-rgc` during cleanup.

`model_bundle.json` is a standard-library runtime export of the copied fitted
models. `models/` retains the original joblib artifacts for provenance, while
`export_models.py` reproduces the portable bundle in an environment that has
scikit-learn, pandas, NumPy, and joblib installed.
