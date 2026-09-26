# Step #4 Representative Synthetic-Domain Results

**Run:** default 400-image development configuration, 3 training epochs  
**Scope:** software PoC evidence only; not factory acceptance

- Validation-selected scratch fraction threshold: **0.002103**
- Held-out test samples: **60**
- Accuracy: **0.900**
- Defect recall: **0.875**
- False-accept rate: **0.125**
- False-reject rate: **0.000**
- p95 software latency: **35.5 ms**
- 200-ms PoC latency target met: **True**

These results intentionally expose remaining limitations: the default short training run does **not** meet the earlier aspirational ≥95% defect-recall / ≤5% false-reject targets. Step #5 therefore treats model/data/domain improvement and real-data validation as commissioning requirements rather than presenting the PoC as production-qualified.
