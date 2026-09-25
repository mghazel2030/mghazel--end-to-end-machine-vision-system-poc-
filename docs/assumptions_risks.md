# Assumptions, acceptance and risk register

**Proposed assumptions, not supplied by Ascension:** 200×120 mm matte gray parts; ±20° rotation; ±3 mm translation; 240×200 mm FOV; 0.5 mm scratch width; 2 mm edge/corner damage; 0.5 m/s conveyor; 450 mm pitch; 200 ms software processing budget. Date 2026-09-25.

**Acceptance goals, not achieved results:** scratch recall ≥95%, false-reject rate ≤5%, ≥5 px across minimum scratch, ≤0.2 px motion blur, no unassociated PLC result. Test against independent real production data before production claims.

| Risk | Effect | Mitigation | Verification | Status |
|---|---|---|---|---|
| Tight worst-case part/FOV margin | Cropping | Mechanical guides, verify tolerance, enlarge FOV if needed | 1000-position tolerance study | Open |
| Short strobe exposure | Low SNR | High-output diffuse strobe, aperture/illumination trial | Real exposure/SNR bench | Open |
| Lens MTF at object detail | Undetectable 0.5mm scratch | Vendor MTF and target chart | Bench measurement at actual WD | Open |
| Lens distortion and tilt | False edge damage | Calibration and geometric rectification | Metrology target | Open |
| Text/logos resemble defects | False rejects | Masking and labeled hard negatives | Stratified dataset | Open |
| Synthetic/real domain gap | False confidence | Collect representative real images | Blind site acceptance test | Open |
| Camera/PLC timing | Wrong part rejected | Unique part IDs, timestamps, ack and timeout | HIL fault injection | Open |
| No verified vendor quotation | Cost uncertainty | RFQ and alternatives | Procurement | Open |

**FAT/SAT and maintenance:** planned Step 5, not yet implemented.
