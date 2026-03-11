# Scan2BIM Prototype Workspace

This repository is a working Scan-to-BIM prototype built on top of the BIM-Net / Scan-to-BIM research codebase.

The current practical goal is:

`point cloud -> semantic segmentation -> BIM element candidates -> Revit import`

This workspace is no longer only a training repository. It now contains runnable scripts for inference, BIM candidate export, and Revit-side import.

## Current Status
The pipeline currently supports these stages:

1. Read a local point cloud from `.pcd`, `.ply`, `.txt`, or `.csv`
2. Run BIMNet semantic inference
3. Export a labeled point cloud as `.ply`
4. Convert semantic labels into coarse BIM element candidates as `.json`
5. Import those candidates into Revit as either:
   - `DirectShape` geometry
   - native-first hybrid import for `wall`, `floor`, and `ceiling` with fallback to `DirectShape`

Important: this is still an early Scan-to-BIM prototype. It is useful for reproducible experiments and pipeline validation, but it is not yet a production-grade BIM authoring system.

## Main Files
### Core scripts
- `infer_pointcloud.py`: point cloud semantic inference entrypoint
- `export_bim_candidates.py`: semantic-to-BIM candidate exporter
- `revit_import_bim_candidates.py`: Revit `DirectShape` importer
- `revit_import_hybrid_bim_candidates.py`: native-first Revit importer with fallback

### Supporting docs
- `docs/scan2bim_progress_2026-03-11.md`: progress log for the current prototype stage
- `docs/context_preserving_chunked_inference_spec_2026-03-11.md`: specification for the planned next-generation inference pipeline

### Original research / training code
- `train_pcs.py`, `train_s3dis.py`, `train_randlanet.py`, `train_pvcnn.py`, `train_segcloud.py`
- `dataloaders/`
- `model/`
- `validation.py`

## Environment
This workspace was tested locally with:
- Python 3.12 (Anaconda environment)
- PyTorch 2.6.0+cpu in the current local shell
- `plyfile`
- optional Revit-side runtime: `pyRevit CPython 3`

The repository already contains `scan2bim.yml` for the original research environment.

### Base setup
```bash
conda env create -f scan2bim.yml
conda activate scan2bim
```

If you are using the local Anaconda Python already configured in this workspace, make sure these packages are available:
```bash
python -c "import torch, plyfile"
```

## Input Data Formats
### Supported by `infer_pointcloud.py`
- `.pcd`
  - ascii
  - binary
- `.ply`
- `.txt`
- `.csv`

### Expected coordinate fields
The current inference pipeline expects at least `x y z` coordinates.

Notes:
- `.txt` / `.csv` inputs currently read only the first three numeric columns as xyz
- `.pcd` inputs currently use the `x`, `y`, `z` fields
- RGB and other modalities are not used yet in the current inference path

## End-to-End Reproducible Workflow
### 1. Run semantic inference
Basic usage:
```bash
python infer_pointcloud.py --input your_scan.pcd --output-dir outputs\my_scan
```

Examples:
```bash
python infer_pointcloud.py --input test_files\UDACITY\highway.pcd --output-dir outputs\inference_smoke\highway
python infer_pointcloud.py --input test_files\sdc.pcd --output-dir outputs\inference_smoke\sdc
python infer_pointcloud.py --input data\HePIC\1_Eremitani\train\101.txt --output-dir outputs\inference_smoke\txt_101
```

Outputs:
- `<name>_predicted.ply`
- `<name>_summary.json`
- optional `<name>_predicted.npz`

### 2. Export BIM element candidates
Convert the labeled `.ply` into a Revit-friendly candidate JSON:
```bash
python export_bim_candidates.py --input outputs\my_scan\your_scan_predicted.ply
```

Example:
```bash
python export_bim_candidates.py --input outputs\inference_smoke\txt_101\101_predicted.ply --cluster-voxel-size 0.5
```

Output:
- `<name>_bim_candidates.json`

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

Current exported geometry is coarse and parametric, for example:
- wall: base line, thickness, height
- floor / ceiling / roof: rectangular profile, elevation, thickness
- column: axis, width, depth, height
- beam: center line, width, depth

### 3. Import into Revit
#### Option A: safest import
Use `revit_import_bim_candidates.py` to create `DirectShape` geometry for all exported candidates.

#### Option B: native-first import
Use `revit_import_hybrid_bim_candidates.py` to attempt native Revit creation for:
- wall
- floor
- ceiling

If native creation fails, the script falls back to `DirectShape`.

### Revit runtime assumptions
Both Revit import scripts are intended for:
- `pyRevit CPython 3`

Before running inside Revit:
1. Open the importer script in pyRevit CPython 3
2. Set `JSON_PATH` at the top of the script, or choose the file interactively if supported
3. Set `INPUT_UNITS` to match the point-cloud units
4. Run the script inside an active Revit project

## Smoke-Test Results
The following tests were run in this workspace.

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

### BIM candidate export
1. `outputs\inference_smoke\highway\highway_predicted.ply`
- exported element count: `0`
- expected because the source is not an architectural scan

2. `outputs\inference_smoke\txt_101\101_predicted.ply`
- exported element count: `9`
- exported categories:
  - ceiling: 1
  - floor: 1
  - wall: 5
  - door: 2

## Current Limitations
### 1. Single-cube semantic inference
Current inference still compresses the full scene into one voxel cube.
This is acceptable for small experiments, but not for large building scans.

### 2. Semantic output is not the final BIM output
The exporter currently produces BIM candidates, not final validated BIM objects.
Geometry fitting is still coarse.

### 3. Door / window hosting is not solved yet
Door and window candidates do not yet include reliable host-wall matching.

### 4. Revit native generation is partial
Only the hybrid importer attempts native creation, and only for:
- wall
- floor
- ceiling

Other categories are still imported as `DirectShape`.

### 5. Current inference is geometry-only
The pipeline currently uses xyz-based semantic inference.
It does not yet use multimodal fusion such as:
- image + point cloud
- RGB + point cloud feature fusion
- floor-plan priors

## Recommended Next Step
The highest-value next step is to replace the current single-cube inference mode with a context-preserving inference pipeline:

`global coarse pass + overlapping local windows + logit fusion`

Specification is documented in:
- `docs/context_preserving_chunked_inference_spec_2026-03-11.md`

This is expected to improve:
- wall continuity
- floor completeness
- opening stability
- downstream BIM candidate quality
- Revit import usability

## Reproducibility Notes
To reproduce the current prototype state:
1. create the environment
2. run `infer_pointcloud.py` on a sample point cloud
3. run `export_bim_candidates.py` on the resulting labeled `.ply`
4. import the generated JSON into Revit using one of the importer scripts
5. compare the resulting semantic outputs and BIM candidates with the example smoke-test outputs

Recommended reference documents:
- `docs/scan2bim_progress_2026-03-11.md`
- `docs/context_preserving_chunked_inference_spec_2026-03-11.md`

## Upstream Reference
This workspace is derived from the research codebase for:

Devid Campagnolo, Elena Camuffo, Umberto Michieli, Paolo Borin, Simone Milani, Andrea Giordano,
"Fully Automated Scan-to-BIM Via Point Cloud Instance Segmentation,"
IEEE ICIP 2023.

Original paper / pretrained weights / dataset links are preserved in the project history and can also be referenced from earlier revisions of this repository.
