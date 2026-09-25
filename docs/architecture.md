# Architecture — cumulative through Step 2

```mermaid
flowchart TD
 A[Physical requirements] --> B[Step 1 quantitative engineering]
 B --> C[Camera / lens / lighting design]
 C --> D[Step 2 synthetic acquisition simulator]
 D --> E[Pose + matte texture + illumination + noise]
 E --> F{Defect injection}
 F --> G[Normal]
 F --> H[Scratch / cut]
 F --> I[Edge damage]
 F --> J[Corner damage]
 F --> K[Mixed]
 G --> L[Image + mask + annotation]
 H --> L
 I --> L
 J --> L
 K --> L
 L --> M[Train / Validation / Test]
 M --> N[QA montage + manifest]
 N --> O[Step 3: OpenCV + PyTorch]
```

Step 2 simulates data acquisition only. It does not claim optical photorealism,
factory validation, trained-model accuracy, PLC integration or deployment.
