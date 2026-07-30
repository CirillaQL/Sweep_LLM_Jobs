# Slurm compute-node external HTTPS probe

This minimal job runs on `ganymede` and checks whether a Slurm compute node can:

1. resolve `eduardoqian.com` through DNS; and
2. complete a validated HTTPS GET request to `https://eduardoqian.com/`.

The response body is discarded. The job records only connectivity metadata such
as the HTTP status, remote address, redirects, and timing. It neither downloads
artifacts nor writes data to an external service.
