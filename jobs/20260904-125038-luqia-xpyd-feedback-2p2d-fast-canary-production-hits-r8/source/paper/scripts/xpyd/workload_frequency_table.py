"""Thread-safe workload-to-frequency knowledge table for feedback control.

This module stores evidence only.  It deliberately performs no DVFS actuation
and no frequency search; a later feedback controller may read and update it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Optional, Sequence


WORKLOAD_KEYS = (
    "small_light",
    "prefill_medium",
    "prefill_heavy",
    "decode_medium",
    "decode_heavy",
    "balanced_medium",
    "both_heavy",
)
PREFILL_ENDPOINT_IDS = ("P0", "P1", "P2", "P3")
DECODE_ENDPOINT_IDS = ("D0", "D1", "D2", "D3")
TTFT_SLO_MS = 500.0
TPOT_SLO_MS = 200.0


class FrequencyTableError(ValueError):
    """A table key, value, or conditional update is invalid."""


@dataclass(frozen=True)
class FrequencyTableValue:
    """One SLO-safe, energy-ranked P/D observation for a workload class."""

    prefill_frequency_mhz: int
    decode_frequency_mhz: int
    measured_power_w: float
    measured_energy_j: float
    ttft_ms: float
    tpot_ms: float
    prefill_endpoint_id: str
    decode_endpoint_id: str
    sample_count: int
    updated_unix_s: float
    source: str
    slo_met: bool = True

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, clock=time.time,
    ) -> "FrequencyTableValue":
        result = cls(
            prefill_frequency_mhz=int(value["prefill_frequency_mhz"]),
            decode_frequency_mhz=int(value["decode_frequency_mhz"]),
            measured_power_w=float(value["measured_power_w"]),
            measured_energy_j=float(value["measured_energy_j"]),
            ttft_ms=float(value["ttft_ms"]),
            tpot_ms=float(value["tpot_ms"]),
            prefill_endpoint_id=str(value["prefill_endpoint_id"]),
            decode_endpoint_id=str(value["decode_endpoint_id"]),
            sample_count=int(value["sample_count"]),
            updated_unix_s=float(value.get("updated_unix_s", clock())),
            source=str(value.get("source", "feedback")),
            slo_met=value.get("slo_met") is True,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.prefill_frequency_mhz <= 0 or self.decode_frequency_mhz <= 0:
            raise FrequencyTableError("P and D frequencies must be positive")
        if not math.isfinite(self.measured_power_w) or self.measured_power_w <= 0:
            raise FrequencyTableError("measured power must be positive")
        if not math.isfinite(self.measured_energy_j) or self.measured_energy_j <= 0:
            raise FrequencyTableError("measured energy must be positive")
        if (
            not math.isfinite(self.ttft_ms) or not math.isfinite(self.tpot_ms)
            or self.ttft_ms < 0 or self.tpot_ms < 0 or self.sample_count <= 0
        ):
            raise FrequencyTableError("latencies must be nonnegative and samples positive")
        if not math.isfinite(self.updated_unix_s) or self.updated_unix_s <= 0:
            raise FrequencyTableError("update timestamp must be positive and finite")
        if self.prefill_endpoint_id not in PREFILL_ENDPOINT_IDS:
            raise FrequencyTableError("prefill endpoint must be one of P0..P3")
        if self.decode_endpoint_id not in DECODE_ENDPOINT_IDS:
            raise FrequencyTableError("decode endpoint must be one of D0..D3")
        if not self.source.strip():
            raise FrequencyTableError("source must be non-empty")
        if not self.slo_met:
            raise FrequencyTableError("only SLO-safe observations may enter the table")
        if self.ttft_ms >= TTFT_SLO_MS or self.tpot_ms > TPOT_SLO_MS:
            raise FrequencyTableError(
                "observation exceeds TTFT/TPOT SLO: %.3f/%.3f ms"
                % (self.ttft_ms, self.tpot_ms)
            )


@dataclass(frozen=True)
class FrequencyTableEntry:
    workload_id: str
    revision: int
    value: Optional[FrequencyTableValue]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "revision": self.revision,
            "value": asdict(self.value) if self.value is not None else None,
        }


class WorkloadFrequencyTable:
    """Linearizable in-process reads and writes over a fixed key set."""

    def __init__(
        self, workload_ids: Sequence[str] = WORKLOAD_KEYS,
        persistence_path: Optional[str] = None,
    ) -> None:
        keys = tuple(str(item).strip() for item in workload_ids)
        if not keys or any(not item for item in keys):
            raise FrequencyTableError("workload keys must be non-empty")
        if len(keys) != len(set(keys)):
            raise FrequencyTableError("workload keys must be unique")
        self._lock = threading.RLock()
        self._values: dict[str, Optional[FrequencyTableValue]] = {
            key: None for key in keys
        }
        self._revisions = {key: 0 for key in keys}
        self._persistence_path = (
            Path(os.path.expandvars(persistence_path)) if persistence_path else None
        )
        with self._lock:
            self._persist_locked()

    def _snapshot_locked(self) -> dict[str, dict[str, Any]]:
        return {
            key: FrequencyTableEntry(
                key, self._revisions[key], self._values[key]
            ).as_dict()
            for key in self._values
        }

    def _persist_locked(self) -> None:
        if self._persistence_path is None:
            return
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._persistence_path.with_suffix(
            self._persistence_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps({"schema_version": 1, "entries": self._snapshot_locked()},
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._persistence_path)

    @property
    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._values)

    def _require_key(self, workload_id: str) -> str:
        key = str(workload_id)
        if key not in self._values:
            raise KeyError("unknown workload: %s" % key)
        return key

    def read(self, workload_id: str) -> FrequencyTableEntry:
        with self._lock:
            key = self._require_key(workload_id)
            return FrequencyTableEntry(
                workload_id=key,
                revision=self._revisions[key],
                value=self._values[key],
            )

    def write(
        self,
        workload_id: str,
        value: FrequencyTableValue | Mapping[str, Any],
        *,
        expected_revision: Optional[int] = None,
    ) -> FrequencyTableEntry:
        candidate = (
            value if isinstance(value, FrequencyTableValue)
            else FrequencyTableValue.from_mapping(value)
        )
        candidate.validate()
        with self._lock:
            key = self._require_key(workload_id)
            revision = self._revisions[key]
            if expected_revision is not None and int(expected_revision) != revision:
                raise FrequencyTableError(
                    "revision conflict for %s: expected=%d actual=%d"
                    % (key, int(expected_revision), revision)
                )
            current = self._values[key]
            if current is not None and candidate.measured_energy_j >= current.measured_energy_j:
                return FrequencyTableEntry(key, revision, current)
            self._values[key] = candidate
            self._revisions[key] = revision + 1
            self._persist_locked()
            return FrequencyTableEntry(key, revision + 1, candidate)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self._snapshot_locked()
