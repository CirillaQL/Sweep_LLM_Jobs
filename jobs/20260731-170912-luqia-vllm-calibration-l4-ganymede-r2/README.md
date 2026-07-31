# L4 full-grid calibration on Ganymede (full observability)

This serial array runs only on `ganymede`, reserves its eight L4 GPUs, and
loads the shared full-observability calibration runner. The broker submits the
array once after seeing `READY`; at most one array element runs at a time.
