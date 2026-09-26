"""Tests for the Step #4 PLC/reject-station simulator."""
import pytest

from vision_poc.plc import PLCRejectSimulator


def test_plc_reject_event_and_delay():
    """A rejected part should assert reject output at the computed travel delay."""
    plc = PLCRejectSimulator(conveyor_speed_mm_s=500.0, reject_distance_mm=750.0)
    result = {
        "decision": "REJECT",
        "latency_ms": 12.5,
        "geometry": {"geometry_damage": True},
        "scratch_detected": False,
    }
    event = plc.submit("part-1", result)
    assert event.sequence == 1
    assert event.reject_output is True
    assert event.reject_delay_ms == pytest.approx(1500.0)
    assert event.reason == "geometry"


def test_plc_requires_positive_physical_values():
    """Reject invalid conveyor/reject-station geometry."""
    with pytest.raises(ValueError):
        PLCRejectSimulator(conveyor_speed_mm_s=0.0, reject_distance_mm=750.0)
