# Scan2BIM Progress Log

Date: 2026-03-11
Workspace: D:\2. Area\2. Scan2BIM\1. code

## Goal
Final objective is an end-to-end Scan-to-BIM pipeline that takes point cloud data (`.pcd`, `.ply`, `.txt`) and produces Revit model geometry.

Current working direction:
1. Point cloud input
2. Semantic inference
3. BIM candidate extraction
4. Revit import
5. Later upgrade to native Revit elements and hosted families

## What Was Added
### 1. Semantic inference entrypoint
File: `workflow/scripts/infer_pointcloud.py`

Purpose:
- Read local point clouds from `.pcd`, `.ply`, `.txt`, `.csv`
- Apply the repo's current BIMNet voxel preprocessing
- Run semantic inference with a local checkpoint
- Export labeled point cloud as `.ply`
- Export a summary `.json`

Current output:
- `<name>_predicted.ply`
- `<name>_summary.json`
- optional `<name>_predicted.npz`

### 2. BIM candidate exporter
File: `workflow/scripts/export_bim_candidates.py`

Purpose:
- Read labeled `.ply`
- Cluster semantic points into object candidates
- Convert clusters into Revit-friendly JSON payloads
- Export candidate categories such as wall, floor, ceiling, roof, column, beam, door, window, stair

Current output:
- `<name>_bim_candidates.json`

Main exported geometry styles:
- wall: base line, length, thickness, height
- floor / ceiling / roof: rectangular profile + elevation + thickness
- column: vertical axis + width/depth/height
- beam: center line + section width/depth
- door / window / stair: profile + width/height style payload

### 3. Revit importer scaffold
File: `workflow/scripts/revit_import_bim_candidates.py`

Purpose:
- Load exported BIM candidate JSON inside Revit
- Create `DirectShape` geometry in matching Revit categories
- Provide a stable first bridge from the Python pipeline into Revit

Target runtime:
- pyRevit CPython 3

## Smoke Tests Performed
### Semantic inference tests
1. Input: `test_files\UDACITY\highway.pcd`
- Output created successfully
- Result was not architecturally meaningful, which is expected because this is a driving sample, not a building scan

2. Input: `test_files\sdc.pcd`
- Output created successfully
- Same note as above: not a building-specific scan

3. Input: `data\HePIC\1_Eremitani\train\101.txt`
- Output created successfully
- This produced building-related classes and was suitable for BIM candidate export

### BIM candidate export tests
1. Input: `workflow\outputs\inference_smoke\highway\highway_predicted.ply`
- Exported element count: 0
- Reasonable result because the source is not an architectural scan

2. Input: `workflow\outputs\inference_smoke\txt_101\101_predicted.ply`
- Exported element count: 9
- Exported categories:
  - ceiling: 1
  - floor: 1
  - wall: 5
  - door: 2

Output file:
- `workflow\outputs\inference_smoke\txt_101\101_predicted_bim_candidates.json`

## Current Pipeline Status
Working chain now exists as:

`raw point cloud -> semantic labeled PLY -> BIM candidate JSON -> Revit DirectShape import`

This is the first usable end-to-end prototype in the workspace.

## Known Limitations
### 1. Semantic model is still coarse
- Current BIMNet path voxelizes the entire point cloud into a dense cube
- Large building scans will lose detail
- Tiling / chunked inference is still needed for real production use

### 2. BIM candidates are coarse approximations
- Wall, floor, beam, and column shapes are estimated from clustered semantics
- Profiles are simplified, mostly rectangle-like
- Openings and fine topology are not fully reconstructed yet

### 3. Revit import currently uses DirectShape
- This is robust for first import
- It is not yet generating native Revit `Wall`, `Floor`, `FamilyInstance`, or hosted openings

### 4. Door / window hosting is not solved yet
- Candidates are exported
- Host wall matching and actual hosted family placement still need to be implemented

## Files Created During This Stage
- `workflow/scripts/infer_pointcloud.py`
- `workflow/scripts/export_bim_candidates.py`
- `workflow/scripts/revit_import_bim_candidates.py`
- `workflow\outputs\inference_smoke\...` test artifacts

## Example Commands
### 1. Run semantic inference
```bash
python workflow/scripts/infer_pointcloud.py --input your_scan.pcd --output-dir workflow/outputs/my_scan
```

### 2. Export BIM candidates
```bash
python workflow/scripts/export_bim_candidates.py --input workflow/outputs/my_scan/your_scan_predicted.ply --cluster-voxel-size 0.5
```

### 3. Import in Revit
- Open `workflow/scripts/revit_import_bim_candidates.py` in pyRevit CPython 3
- Set `JSON_PATH` to the exported candidate JSON file
- Set `INPUT_UNITS` to match the scan units
- Run the script inside Revit

## Recommended Next Step
Next development target should be:
1. Convert `DirectShape` walls and floors into native Revit system elements where possible
2. Match door/window candidates to nearby wall candidates
3. Add chunked or tiled inference for large real-world scans
4. Improve geometry fitting for wall thickness, floor boundaries, and beam/column sections

## Practical Interpretation
At this point the workspace is no longer just a segmentation research folder.
It now contains a prototype Scan-to-BIM bridge that can:
- infer semantic labels from a point cloud
- turn those labels into BIM-like objects
- hand those objects to Revit in importable form

That means the project has moved from "point cloud understanding only" to "early BIM generation pipeline".

## Update After Logging
A native-first Revit importer was added after this log was first written.

### 4. Hybrid Revit importer
File: `workflow/scripts/revit_import_hybrid_bim_candidates.py`

Purpose:
- Try native Revit creation for walls, floors, and ceilings
- Fall back to `DirectShape` if native creation fails
- Keep roof / column / beam / door / window / stair on `DirectShape` for now

Practical meaning:
- The project now has both:
  - a robust `DirectShape` importer
  - a native-first hybrid importer for the most important building elements

Current recommended Revit-side test order:
1. `workflow/scripts/revit_import_bim_candidates.py` for safest first import
2. `workflow/scripts/revit_import_hybrid_bim_candidates.py` when testing native wall / floor / ceiling creation

## Update After Workflow Restructure
The runnable Scan2BIM path was moved under `workflow/`.

Current execution-focused entrypoints are:
- `workflow/scripts/infer_pointcloud.py`
- `workflow/scripts/export_bim_candidates.py`
- `workflow/scripts/revit_import_bim_candidates.py`
- `workflow/scripts/revit_import_hybrid_bim_candidates.py`

## Update After Context-Preserving Inference Implementation
`workflow/scripts/infer_pointcloud.py` now supports:
- `single_cube`
- `global_local_fusion`

Current `global_local_fusion` behavior:
- runs a whole-scene global pass
- generates overlapping `core + halo` windows
- accumulates local logits per original point
- fuses local evidence with a weighted global prior
- exports `confidence` and `vote_count` in the predicted PLY
- optionally writes `<name>_fusion_debug.json`

Smoke test result:
- input: `data\HePIC\1_Eremitani\train\101.txt`
- processed windows: `12`
- downstream export from fused PLY succeeded
