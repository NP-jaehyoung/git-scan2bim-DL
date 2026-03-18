# Revit 2024 JSON Import Workflow

This folder packages the `BIM candidate JSON -> Revit 2024` part of the Scan2BIM pipeline.

Target runtime:
- Revit 2024
- pyRevit with CPython 3 enabled

The workflow is:

`candidate JSON -> validate/stage -> pyRevit button -> Revit model`

## Folder Layout
```text
workflow/revit2024/
  README.md
  prepare_revit2024_import.py
  install_pyrevit_extension.ps1
  session/
  pyrevit/
    Scan2BIM.extension/
```

## What Each Piece Does
- `prepare_revit2024_import.py`
  - validates a candidate JSON
  - records the JSON path and units into `session/active_import.json`
- `install_pyrevit_extension.ps1`
  - copies the bundled `Scan2BIM.extension` into the local pyRevit extensions folder
- `pyrevit/Scan2BIM.extension`
  - adds Revit ribbon buttons for:
    - `Import Hybrid`
    - `Import DirectShape`

## End-to-End Usage
Run from the repo root: `D:\2. Area\2. Scan2BIM\1. code`

### 1. Produce a BIM candidate JSON
```bash
python workflow/scripts/infer_pointcloud.py --input your_scan.pcd --mode global_local_fusion --output-dir workflow/outputs/my_scan
python workflow/scripts/export_bim_candidates.py --input workflow/outputs/my_scan/your_scan_predicted.ply
```

### 2. Validate and stage the JSON for Revit 2024
```bash
python workflow/revit2024/prepare_revit2024_import.py --input workflow/outputs/my_scan/your_scan_predicted_bim_candidates.json --units meters --mode hybrid --write-session
```

This writes:
- `workflow/revit2024/session/active_import.json`

The pyRevit extension reads this file first, so you do not have to edit the Revit-side scripts every time.

### 3. Install the pyRevit extension
```powershell
powershell -ExecutionPolicy Bypass -File workflow/revit2024/install_pyrevit_extension.ps1
```

Default install target:
- `%APPDATA%\pyRevit\Extensions\Scan2BIM.extension`

### 4. Open Revit 2024
Inside Revit 2024 with pyRevit loaded:
- go to the `Scan2BIM` tab
- choose one of:
  - `Import Hybrid`
  - `Import DirectShape`

If `session/active_import.json` exists, the button imports that staged file.
If no staged session exists, the button falls back to a JSON file picker.

## Import Modes
### `Import Hybrid`
- tries native Revit creation for:
  - wall
  - floor
  - ceiling
- falls back to `DirectShape` if native creation fails
- uses `DirectShape` for:
  - roof
  - column
  - beam
  - door
  - window
  - stair

### `Import DirectShape`
- imports every supported category as `DirectShape`
- best for first-pass geometry checking when native creation is unstable

## Units
Supported staged units:
- `meters`
- `millimeters`
- `feet`
- `inches`

Make sure the staged unit matches the point-cloud / JSON coordinate system.

## Notes For Revit 2024
- the importer creates levels automatically when no close existing level is found
- wall/floor/ceiling native creation is still heuristic
- door/window are not hosted family instances yet
- this is best treated as a Revit seed-model workflow, not final BIM authoring

## Recommended First Use
1. stage the JSON with `--mode directshape`
2. run `Import DirectShape` in Revit 2024
3. verify position, scale, and category mapping
4. restage with `--mode hybrid`
5. run `Import Hybrid` and compare native vs fallback behavior
