"""Production-readiness evidence tests."""
from vision_poc.production_readiness import build_acceptance_plan, build_fmea


def test_fmea_has_prioritized_risks():
    """Verify FMEA rows contain positive risk-priority numbers."""
    rows = build_fmea()
    assert len(rows) >= 8
    assert all(row["rpn"] > 0 for row in rows)


def test_acceptance_plan_covers_lifecycle():
    """Verify production qualification covers major lifecycle gates."""
    plan = build_acceptance_plan()
    assert {"FAT", "SAT", "commissioning", "monitoring", "maintenance"} <= set(plan)
