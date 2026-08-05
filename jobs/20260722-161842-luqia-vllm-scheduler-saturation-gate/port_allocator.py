#!/usr/bin/env python3

import argparse
import json
import socket
from pathlib import Path


PORT_BASES = {
    "proxy_http": 30000,
    "proxy_register": 31000,
    "prefill_http": 32000,
    "decode_http": 33000,
    "kv": 34000,
}
PORT_BAND_SIZE = 1000


def occupied_tcp_ports():
    ports = set()
    for proc_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = proc_path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
            except (IndexError, ValueError):
                continue
    return ports


def required_ports(offset, max_prefill, max_decode, prefill_tp, decode_tp):
    return {
        "ganymede": {
            PORT_BASES["proxy_http"] + offset,
            PORT_BASES["proxy_register"] + offset,
            *(PORT_BASES["decode_http"] + offset + i for i in range(max_decode)),
            *(PORT_BASES["kv"] + offset + i for i in range(max_decode * decode_tp)),
        },
        "neptune": {
            *(PORT_BASES["prefill_http"] + offset + i for i in range(max_prefill)),
            *(PORT_BASES["kv"] + offset + i for i in range(max_prefill * prefill_tp)),
        },
    }


def load_snapshot(path, expected_host):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    host = str(data.get("host", "")).split(".", 1)[0]
    if host != expected_host:
        raise SystemExit(
            f"port snapshot host mismatch: expected={expected_host} actual={host or 'unset'}"
        )
    try:
        return {int(port) for port in data["occupied_tcp_ports"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid port snapshot {path}: {exc}") from exc


def select_offset(
    occupied_by_host,
    job_id,
    max_prefill,
    max_decode,
    prefill_tp,
    decode_tp,
    offset_override=None,
):
    max_span = max(
        1,
        max_prefill,
        max_decode,
        max_prefill * prefill_tp,
        max_decode * decode_tp,
    )
    candidate_count = PORT_BAND_SIZE - max_span + 1
    if candidate_count <= 0:
        raise SystemExit(
            f"requested port span does not fit: span={max_span} band={PORT_BAND_SIZE}"
        )

    if offset_override is None:
        start_offset = job_id % candidate_count
        candidates = (
            (start_offset + attempt) % candidate_count
            for attempt in range(candidate_count)
        )
    else:
        if not 0 <= offset_override < candidate_count:
            raise SystemExit(
                f"port offset override out of range: offset={offset_override} "
                f"valid=0..{candidate_count - 1}"
            )
        start_offset = offset_override
        candidates = iter((offset_override,))

    rejected = []
    for offset in candidates:
        required = required_ports(
            offset, max_prefill, max_decode, prefill_tp, decode_tp
        )
        conflicts = {
            host: sorted(ports & occupied_by_host[host])
            for host, ports in required.items()
        }
        conflicts = {host: ports for host, ports in conflicts.items() if ports}
        if not conflicts:
            return offset, start_offset, rejected, required
        rejected.append({"offset": offset, "conflicts": conflicts})

    detail = rejected[0]["conflicts"] if rejected else {}
    if offset_override is not None:
        raise SystemExit(
            f"requested port offset is occupied: offset={offset_override} conflicts={detail}"
        )
    raise SystemExit(
        f"no common free port offset: candidates={candidate_count} first_conflicts={detail}"
    )


def snapshot_command(args):
    host = socket.gethostname().split(".", 1)[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"tcp_ports_{host}.json"
    data = {
        "host": host,
        "occupied_tcp_ports": sorted(occupied_tcp_ports()),
    }
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"port_snapshot host={host} "
        f"occupied={len(data['occupied_tcp_ports'])} output={output}"
    )


def select_command(args):
    occupied_by_host = {
        "ganymede": load_snapshot(args.ganymede_snapshot, "ganymede"),
        "neptune": load_snapshot(args.neptune_snapshot, "neptune"),
    }
    offset, start_offset, rejected, required = select_offset(
        occupied_by_host=occupied_by_host,
        job_id=args.job_id,
        max_prefill=args.max_prefill_instances,
        max_decode=args.max_decode_instances,
        prefill_tp=args.prefill_tp,
        decode_tp=args.decode_tp,
        offset_override=args.offset,
    )
    data = {
        "job_id": args.job_id,
        "start_offset": start_offset,
        "selected_offset": offset,
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
        "ports": {
            "proxy_http": PORT_BASES["proxy_http"] + offset,
            "proxy_register": PORT_BASES["proxy_register"] + offset,
            "prefill_http_base": PORT_BASES["prefill_http"] + offset,
            "decode_http_base": PORT_BASES["decode_http"] + offset,
            "kv_base": PORT_BASES["kv"] + offset,
        },
        "reserved": {host: sorted(ports) for host, ports in required.items()},
    }
    output = Path(args.output)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"port_selection start_offset={start_offset} selected_offset={offset} "
        f"rejected={len(rejected)} output={output}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output-dir", required=True)
    snapshot.set_defaults(func=snapshot_command)

    select = subparsers.add_parser("select")
    select.add_argument("--ganymede-snapshot", required=True)
    select.add_argument("--neptune-snapshot", required=True)
    select.add_argument("--job-id", type=int, required=True)
    select.add_argument("--max-prefill-instances", type=int, required=True)
    select.add_argument("--max-decode-instances", type=int, required=True)
    select.add_argument("--prefill-tp", type=int, required=True)
    select.add_argument("--decode-tp", type=int, required=True)
    select.add_argument("--offset", type=int)
    select.add_argument("--output", required=True)
    select.set_defaults(func=select_command)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
