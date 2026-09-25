# Step 1 — High-level architecture

```mermaid
flowchart TD
  A[Requirements & acceptance criteria] --> B[Quantitative engineering]
  B --> C[Camera / optics / illumination]
  C --> D[Trigger & acquisition — Step 2 simulation]
  D --> E[Preprocessing / localization / rectification — Step 3]
  E --> F[Classical CV: edges & corners — Step 3]
  E --> G[PyTorch segmentation: scratches — Step 3]
  F --> H[Decision fusion — Step 3]
  G --> H
  H --> I[PLC handshake / reject simulation — Step 4]
  H --> J[Traceability, QA and reports — Step 4]
  I --> K[FAT / SAT / monitoring — Step 5 design]
  J --> K
```

**Step 1 implemented:** YAML config, validated mathematical design, checks, reports, visualizations, tests and CI. The acquisition/CV/AI/PLC modules shown above are planned, **not implemented** in Step 1.
