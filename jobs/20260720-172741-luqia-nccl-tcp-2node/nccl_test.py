import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def log(message: str) -> None:
    rank = os.environ.get("RANK", "?")
    host = socket.gethostname()
    print(f"rank={rank} host={host} {message}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"expected world_size=2, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch NCCL backend is unavailable")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    iface = os.environ.get("NCCL_SOCKET_IFNAME", "unset")
    route = subprocess.run(
        ["ip", "route", "get", os.environ["MASTER_ADDR"]],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    log(
        "pre_init "
        f"world_size={world_size} master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} "
        f"interface={iface} CUDA_VISIBLE_DEVICES={visible} route={route!r} "
        f"gpu={torch.cuda.get_device_name(0)!r} torch={torch.__version__} "
        f"nccl={torch.cuda.nccl.version()}"
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    log("init_process_group_ok=true")

    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2
    actual = float(value.item())
    correct = actual == expected
    log(f"correctness_all_reduce expected={expected:.1f} actual={actual:.1f} ok={correct}")
    if not correct:
        raise RuntimeError(f"all_reduce mismatch: expected {expected}, got {actual}")

    cases = [
        (1, 30),
        (16, 20),
        (64, 10),
        (256, 5),
    ]
    for size_mib, iterations in cases:
        size_bytes = size_mib * 1024 * 1024
        tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=device)

        for _ in range(3):
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(iterations):
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        max_elapsed = float(elapsed_tensor.item())
        sample = float(tensor[0].item())
        if sample != expected:
            raise RuntimeError(
                f"payload validation failed for {size_mib} MiB: expected {expected}, got {sample}"
            )

        algorithm_gbps = size_bytes * iterations * 8 / max_elapsed / 1e9
        bus_factor = 2 * (world_size - 1) / world_size
        bus_gbps = algorithm_gbps * bus_factor
        if rank == 0:
            print(
                "NCCL_BW "
                f"size_mib={size_mib} iterations={iterations} elapsed_s={max_elapsed:.6f} "
                f"algbw_gbps={algorithm_gbps:.3f} busbw_gbps={bus_gbps:.3f}",
                flush=True,
            )
        del tensor

    dist.barrier()
    log("nccl_all_reduce_complete=true")
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
