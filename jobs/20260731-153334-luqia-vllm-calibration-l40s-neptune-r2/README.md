# L40S calibration retry 2

Runs the seven calibration shards sequentially on Neptune only. This retry
removes the unreliable vLLM help-text flag check; actual server startup,
benchmark completion, and successful ClickHouse insertion determine success.
