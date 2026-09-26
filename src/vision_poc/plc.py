"""PLC/reject-station simulation for end-to-end inspection integration.

The simulator models the information flow that a production vision station must
preserve between image acquisition and a downstream reject actuator. It does
not claim to implement a vendor-specific PLC protocol.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PLCEvent:
    """One traceable result transferred from vision to the reject controller."""

    part_id: str
    sequence: int
    decision: str
    reject_output: bool
    inspection_time_ms: float
    reject_delay_ms: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event representation."""
        return asdict(self)


class PLCRejectSimulator:
    """Simulate deterministic PASS/REJECT handoff to a downstream actuator."""

    def __init__(self, conveyor_speed_mm_s: float, reject_distance_mm: float) -> None:
        """Initialize reject timing from conveyor speed and camera-to-reject distance.

        Args:
            conveyor_speed_mm_s: Conveyor surface speed in millimetres per second.
            reject_distance_mm: Distance from inspection point to reject actuator.

        Raises:
            ValueError: If either physical quantity is non-positive.
        """
        if conveyor_speed_mm_s <= 0 or reject_distance_mm <= 0:
            raise ValueError("PLC simulator dimensions must be positive")
        self.reject_delay_ms = 1000.0 * reject_distance_mm / conveyor_speed_mm_s
        self._sequence = 0

    def submit(self, part_id: str, result: dict[str, Any]) -> PLCEvent:
        """Convert one vision result into a traceable simulated PLC event.

        Args:
            part_id: Unique part identifier from the inspection dataset/line.
            result: Hybrid inspection result containing decision and latency.

        Returns:
            Ordered PLC event with reject command and nominal actuator delay.
        """
        self._sequence += 1
        decision = str(result["decision"])
        reject = decision == "REJECT"
        geometry = bool(result["geometry"]["geometry_damage"])
        scratch = bool(result["scratch_detected"])
        reasons = []
        if geometry:
            reasons.append("geometry")
        if scratch:
            reasons.append("scratch")
        return PLCEvent(
            part_id=part_id,
            sequence=self._sequence,
            decision=decision,
            reject_output=reject,
            inspection_time_ms=float(result["latency_ms"]),
            reject_delay_ms=self.reject_delay_ms,
            reason="+".join(reasons) if reasons else "none",
        )
