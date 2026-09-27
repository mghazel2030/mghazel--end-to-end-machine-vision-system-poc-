"""Production-readiness evidence for commissioning and lifecycle governance.

This module does not claim that synthetic validation replaces factory acceptance.
It converts known system risks and deployment obligations into traceable FMEA,
FAT/SAT, commissioning, monitoring, fail-safe, and maintenance artifacts.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-26
"""
import csv
import json
from pathlib import Path
from typing import Any


def build_fmea() -> list[dict[str, Any]]:
    """Build the baseline failure-mode-and-effects-analysis register.

    Returns:
        List of risk dictionaries containing failure mode, effect, cause,
        severity, occurrence, detection, RPN, and mitigation.

    Raises:
        This deterministic in-memory function raises no expected exceptions.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-26
    """
    # Scores use a conventional 1-10 ordinal scale. RPN is a prioritization aid,
    # not a substitute for safety analysis or customer-specific risk procedures.
    rows = [
        ("Motion blur", "Small defects missed", "Speed/exposure change", 8, 4, 4,
         "Encoder/speed monitoring; exposure lock; blur verification"),
        ("Illumination drift", "Contrast loss or false rejects", "LED aging/contamination", 7, 5, 4,
         "Reference target; intensity monitoring; preventive cleaning"),
        ("Part pose outside envelope", "Localization/rectification failure", "Guide or feed upset", 7, 3, 3,
         "Pose plausibility gate; reject/stop policy; mechanical guides"),
        ("Logo/text false positive", "Good part rejected", "Legitimate marking variability", 5, 5, 5,
         "Representative validation set; masking/augmentation; review failures"),
        ("Synthetic-to-real domain gap", "Unknown field accuracy", "Unmodeled optics/material variation", 9, 6, 6,
         "Real-data qualification; domain-gap study; controlled adaptation"),
        ("Camera/acquisition loss", "No inspection image", "Cable/power/SDK fault", 9, 3, 2,
         "Heartbeat; timeout; fail-safe reject/line stop; spare cable/camera"),
        ("PLC communication timeout", "Reject command not delivered", "Network/controller fault", 10, 3, 2,
         "Handshake/watchdog; sequence IDs; fail-safe state; alarm"),
        ("Reject actuator failure", "Defective part escapes", "Pneumatic/mechanical fault", 10, 3, 4,
         "Actuation confirmation; downstream sensor; maintenance checks"),
        ("Disk/log storage full", "Traceability loss", "Retention growth", 6, 4, 2,
         "Disk threshold alarms; rotation; retention policy; central archive"),
        ("Model/config drift", "Uncontrolled decision behavior", "Unmanaged update", 9, 3, 4,
         "Version pinning; checksum; approval; rollback; golden regression set"),
    ]
    return [
        {"failure_mode": m, "effect": e, "cause": c, "severity": s, "occurrence": o,
         "detection": d, "rpn": s * o * d, "mitigation": mitigation}
        for m, e, c, s, o, d, mitigation in rows
    ]


def build_acceptance_plan() -> dict[str, list[str]]:
    """Define FAT, SAT, commissioning, monitoring, and maintenance gates.

    Returns:
        Mapping from lifecycle phase to concise, auditable acceptance activities.

    Raises:
        This deterministic in-memory function raises no expected exceptions.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-26
    """
    return {
        "FAT": [
            "Verify camera/lens/light/trigger/IPC/PLC/reject hardware against approved BOM.",
            "Verify calibration, FOV, sampling, exposure, blur, focus, and illumination uniformity.",
            "Run golden good/defective parts and challenge boundary-size defects.",
            "Verify software version, configuration checksum, logs, alarms, and backup/restore.",
            "Verify PLC handshake, sequence tracking, reject timing, timeout, and fail-safe behavior.",
        ],
        "SAT": [
            "Repeat imaging checks at installed line speed under plant ambient conditions.",
            "Run representative production lots across shifts, materials, markings, and nuisance variation.",
            "Measure false-accept/false-reject rates on independently adjudicated real parts.",
            "Verify reject confirmation, operator HMI/alarm workflow, traceability, and recovery procedures.",
            "Obtain customer sign-off against agreed acceptance criteria.",
        ],
        "commissioning": [
            "Mechanical/optical installation and safety review.",
            "Camera/lighting setup, geometric calibration, focus/exposure lock, and trigger verification.",
            "Collect real golden/defect data; quantify synthetic-to-real domain gap.",
            "Tune only on designated real validation data; freeze thresholds/model before acceptance test.",
            "Execute FAT/SAT, train operators/maintenance, baseline KPIs, and archive release package.",
        ],
        "monitoring": [
            "Track pass/reject counts, confidence/defect area, latency, acquisition errors, and PLC timeouts.",
            "Trend illumination/reference-target statistics and pose/localization distributions.",
            "Review false rejects and escaped defects with human adjudication and root-cause labels.",
            "Trigger investigation on KPI drift; retrain only through controlled validation/release process.",
        ],
        "maintenance": [
            "Clean optics/lighting and inspect mounts/cables on preventive-maintenance schedule.",
            "Verify calibration and golden-part response after maintenance or hardware movement.",
            "Pin software/model/config versions and retain rollback artifacts.",
            "Back up logs/configuration/model; periodically test restore and spare-hardware procedure.",
        ],
    }


def save_production_readiness_evidence(cfg: dict[str, Any], output: Path) -> None:
    """Persist production-readiness artifacts beside quantitative PoC evidence.

    Args:
        cfg: Validated configuration used to capture key operational assumptions.
        output: Destination directory for FMEA and lifecycle acceptance artifacts.

    Returns:
        None. Files are written to ``output``.

    Raises:
        OSError: If the destination cannot be created or written.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-26
    """
    # Create one dedicated evidence directory so deployment governance is not
    # confused with synthetic model-performance evidence.
    output.mkdir(parents=True, exist_ok=True)
    fmea = build_fmea()
    plan = build_acceptance_plan()

    # CSV supports engineering review and spreadsheet-based FMEA extension.
    with (output / "fmea.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fmea[0]))
        writer.writeheader()
        writer.writerows(fmea)

    # JSON preserves structured lifecycle gates for automation/reporting.
    evidence = {
        "operational_assumptions": {
            "conveyor_speed_mm_s": cfg["motion"]["speed_mm_s"],
            "software_latency_target_ms": cfg["integration"]["software_latency_target_ms"],
            "reject_distance_mm": cfg["integration"]["reject_distance_mm"],
        },
        "fmea": fmea,
        "acceptance_plan": plan,
        "qualification_status": "PoC software demonstrated; real factory qualification required",
    }
    (output / "production_readiness.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    # Markdown is optimized for human review during design/commissioning meetings.
    lines = ["# Production Readiness and Commissioning Plan", "", "## Qualification boundary",
             "Synthetic-domain results demonstrate software feasibility; ",
             "they do not constitute factory acceptance.", ""]
    for phase, items in plan.items():
        lines += [f"## {phase.upper()}"] + [f"- {item}" for item in items] + [""]
    (output / "production_readiness.md").write_text("\n".join(lines), encoding="utf-8")
