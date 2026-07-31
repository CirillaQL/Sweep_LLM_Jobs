# L4 calibration retry 2

Runs the twelve calibration shards sequentially on Ganymede only. This retry
removes the unreliable vLLM help-text flag check; actual server startup,
benchmark completion, and successful ClickHouse insertion determine success.
