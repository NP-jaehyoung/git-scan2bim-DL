# Context-Preserving Chunked Inference Specification

Date: 2026-03-11
Workspace: D:\2. Area\2. Scan2BIM\1. code
Related files:
- infer_pointcloud.py
- export_bim_candidates.py
- docs\scan2bim_progress_2026-03-11.md

## 1. Purpose
This document specifies the next-generation inference pipeline for large architectural point clouds.

Target architecture:
`global coarse pass + overlapping local windows + logit fusion`

The goal is to replace the current single-cube inference path with a context-preserving strategy that:
- handles large scans without collapsing all detail into one voxel cube
- reduces seam artifacts at block boundaries
- preserves global semantic context while still recovering local geometric detail
- improves downstream Scan-to-BIM quality for Revit modeling

This specification is focused on semantic inference behavior, not on training changes.

## 2. Problem Statement
Current inference in `infer_pointcloud.py` normalizes the entire input point cloud into one fixed voxel cube and performs a single forward pass.

Current behavior:
1. Load full point cloud
2. Center and scale full cloud to `[-1, 1]`
3. Quantize whole scene into one dense occupancy cube
4. Run BIMNet once
5. Read per-voxel hard labels
6. Map labels back to original points

This is inadequate for large Scan-to-BIM scenes because:
- large buildings lose local detail when compressed into one cube
- wall thickness, openings, columns, and beam sections blur out
- block-scale semantics dominate over room-scale geometry
- future Revit modeling needs stable continuity, not only per-voxel labels

## 3. Scope
### In scope
- redesign inference-time spatial partitioning
- preserve semantic context through multi-scale inference
- aggregate predictions in logit space instead of hard labels
- output per-point semantic labels and confidence
- support existing input formats: `.pcd`, `.ply`, `.txt`, `.csv`
- remain compatible with current BIM candidate export stage

### Out of scope
- retraining BIMNet
- changing the Revit importer in this phase
- native instance segmentation
- full geometry fitting redesign
- cloud deployment or distributed inference

## 4. High-Level Design
### New inference stages
1. Global coarse pass
2. Spatial window generation
3. Overlapping local inference passes
4. Logit fusion on original points or canonical voxels
5. Final semantic label assignment
6. Confidence and provenance export

### Core design principle
Global pass provides scene-level semantic prior.
Local passes provide geometric detail.
Fusion resolves conflicts using weighted logits instead of winner-take-all labels.

## 5. Pipeline Overview
| Stage | Name | Input | Output | Main role |
|---|---|---|---|---|
| 1 | Load | raw point cloud | raw xyz points | ingest source data |
| 2 | Global coarse pass | full cloud | coarse global logits / priors | preserve scene context |
| 3 | Window proposal | full cloud + optional global prior | overlapping windows | create local inference units |
| 4 | Local inference | each window | local logits | recover local detail |
| 5 | Fusion | all local logits + optional global logits | fused logits | remove seams and merge context |
| 6 | Final labeling | fused logits | final class id + confidence | produce downstream semantic output |
| 7 | Export | fused result | labeled PLY / NPZ / JSON | support BIM candidate export |

## 6. Data Model
### Input point model
| Field | Type | Required | Notes |
|---|---|---|---|
| x | float32 | yes | world coordinate |
| y | float32 | yes | world coordinate |
| z | float32 | yes | world coordinate |
| rgb | optional | no | ignored for first implementation unless model path supports it |
| source_index | int32 | yes | stable index of original point |

### Internal fused prediction model
| Field | Type | Required | Notes |
|---|---|---|---|
| source_index | int32 | yes | original point index |
| fused_logits | float32[num_classes] | yes | accumulated logits after fusion |
| fused_weight | float32 | yes | total accumulation weight |
| final_label_id | int16 | yes | argmax over fused logits |
| final_confidence | float32 | yes | softmax confidence of final label |
| vote_count | int16 | yes | number of contributing passes |
| global_logit | optional | no | saved if global prior is used |

### Window model
| Field | Type | Required | Notes |
|---|---|---|---|
| window_id | string | yes | stable identifier |
| bbox_min | float[3] | yes | world coordinates |
| bbox_max | float[3] | yes | world coordinates |
| core_bbox_min | float[3] | yes | central scoring area |
| core_bbox_max | float[3] | yes | central scoring area |
| halo_margin | float[3] | yes | context margin added around core |
| point_indices | int32[] | yes | points used for inference |
| core_point_mask | bool[] | yes | scoring mask for fusion |

## 7. Global Coarse Pass Specification
### Objective
Obtain scene-level semantic prior before local inference.

### Strategy
- downsample full point cloud aggressively
- normalize using full-scene coordinates
- run low-resolution BIMNet pass
- assign coarse logits back to original points through nearest coarse voxel lookup or nearest-point propagation

### Recommended first implementation
| Item | Proposed value |
|---|---|
| downsample mode | voxel downsample |
| global voxel size | larger than local window density |
| global cube edge | 96 or 128 |
| output | per-point coarse logits or coarse class prior |

### Notes
- global pass should not overwrite local detail
- it acts as a prior, not as the final label source
- fusion weight for global logits should be lower than confident local logits

## 8. Local Window Generation Specification
### Objective
Split large scenes into context-aware overlapping windows.

### Requirements
- windows must overlap
- each window must have `core region + halo context`
- only core-region predictions should contribute at full weight
- halo region predictions provide context but should be down-weighted or ignored in final scoring

### Window geometry
| Term | Meaning |
|---|---|
| core | central spatial region where predictions are trusted most |
| halo | expanded margin around core that gives context to the model |
| overlap | neighboring windows share halo/core area |

### Recommended first implementation
| Parameter | Description | Initial target |
|---|---|---|
| core_size_xy | horizontal core size | tune from real building scale |
| core_size_z | vertical core size | typically floor-height aware |
| halo_xy | horizontal context margin | 20-50% of core width |
| halo_z | vertical context margin | enough to keep floor/ceiling context |
| stride_xy | core stepping distance | less than core size to ensure overlap |
| stride_z | vertical stepping distance | usually equal to or smaller than core_size_z |

### Windowing modes
| Mode | Use case | Priority |
|---|---|---|
| fixed sliding grid | easiest initial implementation | P1 |
| floor-aware slicing | strong for buildings with level structure | P2 |
| connected-component guided windows | useful after coarse semantic prior | P3 |
| plane-aware adaptive windows | best long-term for walls/floors | P4 |

### Important rule
A simple non-overlapping block partition is prohibited for final production mode.
It may be allowed only as an internal debug mode.

## 9. Local Inference Specification
### For each window
1. collect points inside halo bbox
2. re-center and normalize with respect to that window only
3. voxelize into dense cube for BIMNet
4. run forward pass
5. recover per-point logits from predicted voxel logits
6. apply scoring weights based on distance to the core center / core boundary

### Scoring region rule
- points inside core: full contribution
- points in halo only: reduced contribution or zero contribution, depending on fusion policy

### Recommended first policy
| Region | Fusion weight |
|---|---|
| core interior | 1.0 |
| near core boundary | 0.5 to 0.8 |
| halo only | 0.1 to 0.3 or ignored |

## 10. Logit Fusion Specification
### Why logit fusion
Hard-label voting throws away uncertainty.
Logit fusion preserves class competition and allows context-aware smoothing.

### Fusion formula
For each original point `p`:

`fused_logits[p] = global_weight * global_logits[p] + sum_i(local_weight_i[p] * local_logits_i[p])`

Final label:

`final_label[p] = argmax(fused_logits[p])`

Confidence:

`final_confidence[p] = softmax(fused_logits[p])[final_label[p]]`

### Weighting rules
| Signal | Purpose | Relative importance |
|---|---|---|
| global logits | scene prior | low to medium |
| local core logits | main local evidence | high |
| local halo logits | context only | low |
| boundary penalty | reduce seam dominance | medium |

### Optional weighting factors
- distance to window center
- distance to nearest core boundary
- local point density
- class-specific reliability weight
- entropy-based confidence from local softmax

### Conflict resolution
If different windows disagree strongly:
- prefer higher-confidence local core evidence
- fall back to global prior when all local votes are weak
- record disagreement count in debug output

## 11. Output Specification
### Main outputs
| File | Description |
|---|---|
| `<name>_predicted.ply` | final fused semantic result |
| `<name>_summary.json` | inference metadata and histogram |
| `<name>_predicted.npz` | optional dense debug payload |
| `<name>_fusion_debug.json` | optional per-window / per-class diagnostics |

### Required new summary fields
| Field | Description |
|---|---|
| inference_mode | `single_cube` or `global_local_fusion` |
| num_windows | number of local windows processed |
| avg_votes_per_point | mean fusion contributions |
| min_votes_per_point | minimum contributions |
| max_votes_per_point | maximum contributions |
| global_weight | scalar weight used in fusion |
| local_weight_policy | description of weight rule |
| uncovered_points | points with no local window contribution |
| seam_risk_points | high-disagreement points |

### Optional debug outputs
- per-window bbox list
- per-window point counts
- per-point vote count
- per-point entropy
- class confusion hot spots

## 12. CLI Specification
### New CLI options proposed for `infer_pointcloud.py`
| Option | Type | Purpose |
|---|---|---|
| `--mode` | enum | `single_cube`, `global_local_fusion` |
| `--global-cube-edge` | int | coarse pass cube size |
| `--global-voxel-downsample` | float | coarse pass downsampling |
| `--core-size-xy` | float | local core width/depth |
| `--core-size-z` | float | local core height |
| `--halo-xy` | float | horizontal halo |
| `--halo-z` | float | vertical halo |
| `--stride-xy` | float | window stride in xy |
| `--stride-z` | float | window stride in z |
| `--global-weight` | float | fusion weight for coarse prior |
| `--halo-weight` | float | default halo contribution |
| `--save-fusion-debug` | flag | write window and vote diagnostics |
| `--min-points-per-window` | int | drop sparse windows |
| `--max-windows` | int | debug limiter |

## 13. Implementation Phases
### Phase 1
- keep current BIMNet model unchanged
- add `mode=global_local_fusion`
- implement overlapping windows
- implement logit accumulation
- export vote count and confidence

### Phase 2
- add global coarse pass prior
- tune global/local fusion weights
- add debug exports
- compare against current single-cube inference

### Phase 3
- add floor-aware or level-aware window proposal
- add class-aware fusion weighting
- add seam disagreement diagnostics

### Phase 4
- integrate fused confidence into BIM candidate export
- allow exporter to ignore low-confidence or seam-risk regions

## 14. Success Criteria
### Semantic criteria
| Criterion | Target direction |
|---|---|
| wall continuity | improve |
| floor/ceiling completeness | improve |
| door/window boundary sharpness | improve |
| confidence calibration | improve |
| seam artifacts | reduce |

### Scan-to-BIM criteria
| Criterion | Target direction |
|---|---|
| wall candidate fragmentation | reduce |
| false wall splits | reduce |
| floor profile stability | improve |
| door/window host matching readiness | improve |
| usable Revit import rate | improve |

### Engineering criteria
| Criterion | Requirement |
|---|---|
| backward compatibility | preserve single-cube mode |
| reproducibility | deterministic window generation when seed fixed |
| debuggability | save enough metadata to explain label disagreements |
| runtime control | support max window cap and sampling |

## 15. Risks
| Risk | Description | Mitigation |
|---|---|---|
| seam artifacts remain | overlap alone may not solve conflicts | use logit fusion + halo weighting |
| runtime explosion | too many windows on dense scans | add downsampling, sparse window skip, max window cap |
| duplicated predictions | points appear in many windows | track vote_count and weighted accumulation |
| weak global prior | coarse pass may be noisy | keep global weight low initially |
| geometry drift | local normalization may distort scale interpretation | retain original coordinates for fusion/output |

## 16. Non-Negotiable Design Rules
1. Final production inference must not use non-overlapping hard block partitioning.
2. Fusion must operate on logits or probabilities, not only hard labels.
3. Original point indices must be preserved through the full pipeline.
4. Window generation must be deterministic when seed and parameters are fixed.
5. Output must include enough metadata to diagnose disagreement and seam failures.

## 17. Immediate Recommended Build Order
1. Extend `infer_pointcloud.py` with `--mode single_cube|global_local_fusion`
2. Implement overlapping sliding windows without global prior first
3. Add per-point logit accumulation and vote count
4. Export fused confidence and debug metadata
5. Add global coarse pass
6. Tune fusion weights on real architectural scans

## 18. Expected Practical Impact
If implemented correctly, this architecture should:
- preserve more local geometry from large scans
- reduce semantic discontinuity at window boundaries
- produce better wall/floor/column candidates for `export_bim_candidates.py`
- improve the quality of Revit-native wall/floor generation downstream

In short:
this is not only a semantic-accuracy upgrade.
It is a downstream BIM-quality upgrade.
