# Scan2BIM Research Workspace

This repository is now split into two layers:

1. `research assets` at the repository root
2. `workflow assets` under `workflow/`

The root still contains the original BIM-Net / Scan-to-BIM research code, datasets, logs, and experiments.
The new `workflow/` subtree contains the runnable prototype path for:

`point cloud -> semantic inference -> BIM candidate export -> Revit import`

## Repository Layout
```text
1. code/
  data/                 # research datasets and local scan assets
  dataloaders/          # original research dataloaders
  log/                  # checkpoints / experiment logs
  model/                # original BIMNet and related models
  train_*.py            # research training scripts
  workflow/
    README.md           # reproducible workflow guide
    configs/            # future runtime config files
    docs/               # workflow-specific records and specs
    outputs/            # ignored runtime outputs
    scripts/            # runnable workflow entrypoints
```

## What Lives Where
- Research and training stay at the root: `model/`, `dataloaders/`, `train_*.py`, `data/`, `log/`
- Practical Scan2BIM execution lives under `workflow/`
- The workflow still depends on root-level research assets such as `model/` and `log/`

## Workflow Entry Points
- `workflow/scripts/infer_pointcloud.py`
- `workflow/scripts/export_bim_candidates.py`
- `workflow/scripts/revit_import_bim_candidates.py`
- `workflow/scripts/revit_import_hybrid_bim_candidates.py`
- `workflow/revit2024/README.md`

## Start Here
Use `workflow/README.md` for the reproducible operational guide:

- environment assumptions
- supported input formats
- end-to-end commands
- smoke-test notes
- current limitations
- next architectural step

## Workflow Docs
- `workflow/docs/scan2bim_progress_2026-03-11.md`
- `workflow/docs/context_preserving_chunked_inference_spec_2026-03-11.md`
- `workflow/revit2024/README.md`

## Current Intent
This repository is still research-first, but the executable Scan2BIM path is now isolated enough that we can evolve it without constantly mixing:

- training code
- archived experiments
- runtime outputs
- Revit integration scripts

That separation is the basis for the next step: turning the prototype into a cleaner production-style Scan2BIM pipeline.
