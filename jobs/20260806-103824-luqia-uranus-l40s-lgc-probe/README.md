# Uranus L40S graphics-clock lock and restore probe

This job allocates exactly one GPU on Uranus and tests whether its L40S accepts
and enforces a 990 MHz graphics-clock lock through `sudo -n nvidia-smi -lgc`.
It validates the active SM clock under a CUDA matrix-multiplication load rather
than trusting the command return code alone.

Every exit path runs `sudo -n nvidia-smi -rgc`. A post-reset CUDA probe must
show the active clock has departed the 990 MHz target, after which a second
`-rgc` is issued as the final GPU-control operation.
