# Uranus network and sudo nvidia-smi probe

This read-only diagnostic job allocates one GPU on both Uranus and Ganymede.
It discovers the Uranus source IP and interface selected for Ganymede's known
100GbE address (`10.1.0.3`), records interface/link details, tests bidirectional
ICMP connectivity, and verifies both ordinary and passwordless-sudo
`nvidia-smi` queries.

The job does not reset GPUs, lock clocks, change persistence mode, or mutate
network configuration.
