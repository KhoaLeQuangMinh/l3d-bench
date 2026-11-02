# Gate log

| Gate | Check | Date | Outcome |
| --- | --- | --- | --- |
| G0 Setup | Dummy run on a lab account: clone, read one L3D image, write to Drive, resume in a new session | | |
| G1 Data | `scripts/phase1_data_checks.py`: counts vs D2GPLand (921/122/109, 39 patients), label inventory, class-overlap share, overlays checked by eye | | |
| G2 Metric | Tests pass; TopoNet released weights scored with our metric under TopoNet's settings; gap to 65.19 / 50.56 / 28.07 explained | | |
| G3 Pipeline | M1 overfits 10 images; trains on the official split; hours per run known | | |
| G4 Switches | M2, M3 one-epoch smoke runs; config diff shows only their switch | | |
| G5 Freeze | `configs/frozen.yaml` committed, tag `study-v1` | | |
