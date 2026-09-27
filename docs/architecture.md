# End-to-End Machine-Vision System Architecture

```mermaid
flowchart LR
  A[Requirements & Acceptance Criteria] --> B[Hardware Engineering]
  B --> C[Synthetic Acquisition + Exact Labels]
  C --> D[Preprocessing]
  D --> E[Localization & Orientation]
  E --> F[Pose Rectification]
  F --> G[Classical Edge/Corner Geometry]
  F --> H[PyTorch U-Net Scratch Segmentation]
  G --> I[Decision Fusion]
  H --> I
  I --> J[Validation-Only Calibration]
  J --> K[Held-Out Evaluation]
  K --> L[PLC / Reject Simulation]
  L --> M[Evidence, Logging & Traceability]
  M --> N[FMEA / FAT / SAT / Commissioning / Monitoring]
```

The PoC implements the software path and a simulated PLC/reject contract. Physical camera, PLC, actuator and factory acceptance require commissioning with representative real parts.
