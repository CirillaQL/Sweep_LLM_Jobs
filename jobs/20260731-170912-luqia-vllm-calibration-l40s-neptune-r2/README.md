# L40S full-grid calibration on Neptune (full observability)

This serial array runs only on `neptune`, reserves its four L40S GPUs, and
loads the shared full-observability calibration runner. The broker submits the
array once after seeing `READY`; at most one array element runs at a time.
