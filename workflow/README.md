# Scan2BIM Workflow Guide

This folder contains the practical Scan2BIM execution path inside the larger research repository.

Current target workflow:

`point cloud -> semantic segmentation -> BIM candidate JSON -> Revit import`

The original training code, checkpoints, dataloaders, and datasets still live at the repository root.
These workflow scripts intentionally reuse those research assets instead of duplicating them.

## Folder Layout
```text
workflow/
  README.md
  configs/
  docs/
  outputs/
  scripts/
```

## Main Scripts
- `workflow/scripts/infer_pointcloud.py`: point-cloud semantic inference entrypoint with `single_cube` and `global_local_fusion` modes
- `workflow/scripts/export_bim_candidates.py`: semantic-to-BIM candidate exporter
- `workflow/scripts/revit_import_bim_candidates.py`: Revit `DirectShape` importer
- `workflow/scripts/revit_import_hybrid_bim_candidates.py`: native-first Revit importer with fallback
- `workflow/scripts/inspect_gt_vs_semantic.py`: interactive GT vs semantic comparison viewer with point alignment and IoU summary
- `workflow/scripts/inspect_semantic_vs_bim.py`: interactive semantic vs BIM candidate comparison viewer
- `workflow/scripts/export_pointcloud_for_recap.py`: convert raw point clouds to ReCap-friendly `.pts` or `.xyz` for Revit point-cloud display

## Supporting Docs
- `workflow/docs/scan2bim_progress_2026-03-11.md`
- `workflow/docs/context_preserving_chunked_inference_spec_2026-03-11.md`
- `workflow/revit2024/README.md`

## Required Root-Level Dependencies
These workflow scripts still depend on the research repo root:

- `model/`
- `log/`
- `scan2bim.yml`
- selected datasets under `data/`

Do not move or delete those root folders unless the workflow is refactored to package its own runtime assets.

## Environment
Tested locally with:
- Python 3.12
- PyTorch 2.6.0+cpu in the current shell
- `plyfile`
- `pyvista` for visual debugging
- optional Revit-side runtime: `pyRevit CPython 3`

Base setup:
```bash
conda env create -f scan2bim.yml
conda activate scan2bim
python -c "import torch, plyfile"
```

## Supported Input Formats
### `workflow/scripts/infer_pointcloud.py`
- `.pcd`
  - ascii
  - binary
- `.ply`
- `.txt`
- `.csv`

Expected fields:
- at least `x y z`

Notes:
- `.txt` and `.csv` currently use the first three numeric columns
- `.pcd` currently uses `x`, `y`, `z`
- RGB and multimodal features are not used yet in the current inference path

## Reproducible Workflow
Run commands from the repository root: `D:\2. Area\2. Scan2BIM\1. code`

### 1. Semantic inference
Basic usage:
```bash
python workflow/scripts/infer_pointcloud.py --input your_scan.pcd --output-dir workflow/outputs/my_scan
```

Context-preserving fusion mode:
```bash
python workflow/scripts/infer_pointcloud.py --input your_scan.pcd --mode global_local_fusion --save-fusion-debug --output-dir workflow/outputs/my_scan
```

Examples:
```bash
python workflow/scripts/infer_pointcloud.py --input test_files\UDACITY\highway.pcd --output-dir workflow/outputs/inference_smoke/highway
python workflow/scripts/infer_pointcloud.py --input test_files\sdc.pcd --output-dir workflow/outputs/inference_smoke/sdc
python workflow/scripts/infer_pointcloud.py --input data\HePIC\1_Eremitani\train\101.txt --output-dir workflow/outputs/inference_smoke/txt_101
```

Outputs:
- `<name>_predicted.ply`
- `<name>_summary.json`
- optional `<name>_predicted.npz`
- optional `<name>_fusion_debug.json` in fusion mode

### 2. Export BIM element candidates
```bash
python workflow/scripts/export_bim_candidates.py --input workflow/outputs/my_scan/your_scan_predicted.ply
```

Example:
```bash
python workflow/scripts/export_bim_candidates.py --input workflow/outputs/inference_smoke/txt_101/101_predicted.ply --cluster-voxel-size 0.5
```

Output:
- `<name>_bim_candidates.json`

Current exporter behavior:
- dominant-axis angle regularization is enabled by default
- wall clusters are decomposed into multiple Manhattan wall segments when possible
- floor / ceiling / roof profiles prefer occupancy-boundary footprints before falling back to a simple rectangle
- use `--angle-regularization none` to disable angle snapping

### 2.5. Visual debugging
Ground-truth vs semantic prediction:
```bash
python workflow/scripts/inspect_gt_vs_semantic.py --ground-truth data\HePIC\1_Eremitani\train\101.txt --prediction workflow/outputs/fusion_smoke/global_local/101_predicted.ply
```

S3DIS labeled PLY example:
```bash
python workflow/scripts/inspect_gt_vs_semantic.py --ground-truth data\S3DIS\S3DIS_labeled\Area_1_conferenceRoom_1.ply --prediction workflow/outputs/s3dis_check/conferenceRoom_1_predicted.ply --label-space s3dis
```

Semantic prediction vs BIM candidate export:
```bash
python workflow/scripts/inspect_semantic_vs_bim.py --prediction workflow/outputs/fusion_smoke/global_local/101_predicted.ply --bim-json workflow/outputs/fusion_smoke/global_local/101_predicted_bim_candidates.json
```

Useful options:
- `--summary-only`: skip the window and print or save metrics only
- `--summary-json <path>`: save a machine-readable summary
- `--screenshot <path>`: render off-screen and save a screenshot

Viewer layout:
- `inspect_gt_vs_semantic.py`
  - panel 1: ground truth
  - panel 2: semantic prediction
  - panel 3: correct vs incorrect aligned points
- `inspect_semantic_vs_bim.py`
  - panel 1: semantic prediction
  - panel 2: semantic points with BIM candidate overlays
  - panel 3: BIM candidate wireframes only

Exported categories may include:
- wall
- floor
- ceiling
- roof
- column
- beam
- door
- window
- stair

### 3. Import into Revit
#### Option A: safest import
Use `workflow/scripts/revit_import_bim_candidates.py` to create `DirectShape` geometry.

#### Option B: native-first import
Use `workflow/scripts/revit_import_hybrid_bim_candidates.py` to attempt native creation for:
- wall
- floor
- ceiling

If native creation fails, the importer falls back to `DirectShape`.

Recent import fixes:
- ceiling `DirectShape` geometry is now extruded downward from the ceiling elevation instead of upward
- this reduces the common "ceiling floating above the point cloud" issue

Revit runtime assumptions:
- `pyRevit CPython 3`

Before running inside Revit:
1. Open the importer script in pyRevit CPython 3
2. Set `JSON_PATH` at the top of the script, or choose the file interactively if supported
3. Set `INPUT_UNITS` to match the point-cloud units
4. Run the script inside an active Revit project

### Revit 2024 packaged workflow
For a cleaner Revit 2024 handoff with staged JSON and pyRevit buttons, use:

- `workflow/revit2024/README.md`
- `workflow/revit2024/prepare_revit2024_import.py`
- `workflow/revit2024/pyrevit/Scan2BIM.extension`

### 3.5. Show raw point clouds in Revit
If you want to display the original point cloud in Revit, do not use the BIM candidate importer.
Use the ReCap path instead:

```bash
python workflow/scripts/export_pointcloud_for_recap.py --input your_scan.ply --output workflow/outputs/recap/your_scan.pts --format pts
```

Then:
1. open the exported `.pts` in Autodesk ReCap
2. save or index it as `.rcp` / `.rcs`
3. link that point cloud into Revit

Example:
```bash
python workflow/scripts/export_pointcloud_for_recap.py --input workflow/outputs/s3dis_check/conferenceRoom_1_predicted.ply --output workflow/outputs/recap/conferenceRoom_1_predicted.pts --format pts
```

## Smoke-Test Notes
Validated in this workspace:

### Semantic inference
1. `test_files\UDACITY\highway.pcd`
- succeeded
- not architecturally meaningful, as expected for a driving sample

2. `test_files\sdc.pcd`
- succeeded
- also not architecturally meaningful for BIM use

3. `data\HePIC\1_Eremitani\train\101.txt`
- succeeded
- produced building-related classes suitable for BIM candidate export

4. `data\HePIC\1_Eremitani\train\101.txt` with `--mode global_local_fusion`
- succeeded
- processed `12` overlapping windows in the current auto layout
- produced fusion summary and debug JSON under `workflow/outputs/fusion_smoke/global_local`

### BIM candidate export
1. `workflow/outputs/inference_smoke/highway/highway_predicted.ply`
- exported element count: `0`

2. `workflow/outputs/inference_smoke/txt_101/101_predicted.ply`
- exported element count: `9`
- exported categories:
  - ceiling: 1
  - floor: 1
  - wall: 5
  - door: 2

3. `workflow/outputs/s3dis_check_ply/Area_1_conferenceRoom_2_predicted.ply`
- exported elements: `15`
- wall count increased from `3` to `6` after wall-segment decomposition
- angle regularization snapped `12` elements to the dominant `[0°, 90°]` axes
- this reduced the "one diagonal wall replacing several orthogonal walls" failure mode

### Visual debugging
1. `inspect_gt_vs_semantic.py` on `data\HePIC\1_Eremitani\train\101.txt` vs `workflow/outputs/fusion_smoke/global_local/101_predicted.ply`
- aligned `99,998 / 99,998` points
- point accuracy: `0.0555`
- mean IoU: `0.0225`
- this indicates the current quality issue is already present at the semantic stage on this sample

2. `inspect_semantic_vs_bim.py` on `workflow/outputs/fusion_smoke/global_local/101_predicted.ply`
- prediction points: `99,998`
- exported BIM candidates: `11`
- candidate categories:
  - ceiling: 1
  - floor: 1
  - wall: 5
  - door: 4

3. `inspect_gt_vs_semantic.py` on `data\S3DIS\S3DIS_labeled\Area_1_conferenceRoom_1.ply` vs `workflow/outputs/s3dis_check/conferenceRoom_1_predicted.ply`
- aligned `1,136,617 / 1,136,617` points
- point accuracy: `0.9184`
- mean IoU: `0.7784`
- this confirms the earlier collapse was caused by using an unlabeled raw TXT as GT, not by the S3DIS model itself

## Current Limitations
1. `single_cube` mode still compresses the full scene into one voxel cube.
2. `global_local_fusion` is now implemented, but it is still a Phase 1 version:
   - fixed sliding windows only
   - no floor-aware window proposal yet
   - global prior is still a whole-scene cube pass
3. Exported BIM objects are still coarse candidates, not final validated BIM elements.
4. GT BIM generated through `export_bim_candidates.py` is still `semantic GT -> BIM candidate`, not true authoring-grade BIM ground truth.
5. Door and window hosting is not solved yet.
6. Native Revit generation is still partial.
7. Current inference path is geometry-only, not multimodal.
8. Wall decomposition and floor / ceiling footprint extraction are still Phase 1 heuristics and will need more topology-aware refinement.

## Recommended Next Step
The highest-value next step is now:

- continue improving wall and opening topology
- upgrade floor and ceiling footprints from coarse candidates to cleaner polygon profiles with hole support
- improve native Revit placement quality for hosted doors and windows
- keep tuning `single_cube` vs `global_local_fusion` on BIM-relevant metrics

Specification:
- `workflow/docs/context_preserving_chunked_inference_spec_2026-03-11.md`
