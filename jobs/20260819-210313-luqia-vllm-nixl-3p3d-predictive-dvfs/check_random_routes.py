#!/usr/bin/env python3
"""检查正式测量阶段的请求是否都使用合法的 3P3D 路由。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# 代理为每个请求输出 count 和 P-D 路由；这里只解析这类审计行。
ROUTE_LINE = re.compile(r"^route count=(\d+) policy_route=(\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-count-start", type=int, required=True)
    parser.add_argument("--prefill-tp-sizes", required=True)
    parser.add_argument("--decode-tp-sizes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


# 生成完整 3x3 路由，并检查编号连续性、重复项和非法路径。
def main() -> int:
    args = parse_args()
    prefill_tp = [int(value) for value in args.prefill_tp_sizes.split(",")]
    decode_tp = [int(value) for value in args.decode_tp_sizes.split(",")]
    supported = {
        f"P{p_index}-D{d_index}"
        for p_index in range(len(prefill_tp))
        for d_index in range(len(decode_tp))
    }

    observed: dict[int, str] = {}
    duplicates: list[int] = []
    for line in args.proxy_log.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = ROUTE_LINE.match(line)
        if match is None:
            continue
        count, route = int(match.group(1)), match.group(2)
        if count in observed:
            duplicates.append(count)
        observed[count] = route

    expected = set(
        range(
            args.expected_count_start,
            args.expected_count_start + args.expected_requests,
        )
    )
    actual = set(observed)
    invalid = [
        {"count": count, "route": route}
        for count, route in sorted(observed.items())
        if route not in supported
    ]
    passed = not duplicates and actual == expected and not invalid
    route_counts = Counter(observed.values())

    # 保存机器可读报告，任务用退出码决定本轮实验是否有效。
    report = {
        "passed": passed,
        "policy": "independent_prefill_decode_random",
        "allow_asymmetric_tp": True,
        "prefill_tp_sizes": prefill_tp,
        "decode_tp_sizes": decode_tp,
        "supported_routes": sorted(supported),
        "expected_requests": args.expected_requests,
        "observed_requests": len(observed),
        "route_counts": dict(sorted(route_counts.items())),
        "missing_counts": sorted(expected - actual),
        "unexpected_counts": sorted(actual - expected),
        "duplicate_counts": duplicates,
        "invalid_routes": invalid,
        "unobserved_routes": sorted(supported - set(observed.values())),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"random_route_check passed={str(passed).lower()} "
        f"expected={args.expected_requests} observed={len(observed)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
