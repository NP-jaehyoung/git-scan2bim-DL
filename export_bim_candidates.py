"""Export Revit-friendly BIM element candidates from a labeled point cloud.

Input:
- labeled PLY with x/y/z and either `class_id` or `class`

Output:
- JSON containing clustered semantic elements with coarse geometric parameters

This is an intermediate Scan-to-BIM bridge. It is intentionally conservative:
it produces candidate walls, floors, ceilings, roofs, beams, columns,
doors, windows, and stairs, but does not attempt direct Revit API calls yet.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData


LABEL_SPACES = {
    "s3dis": {
        "class_id_to_name": {
            1: "ceiling",
            2: "floor",
            3: "wall",
            4: "beam",
            5: "column",
            6: "window",
            7: "door",
            8: "table",
            9: "chair",
            10: "sofa",
            11: "bookcase",
            12: "board",
            13: "clutter",
        },
        "export_categories": {"ceiling", "floor", "wall", "beam", "column", "window", "door"},
    },
    "pcs": {
        "class_id_to_name": {
            1: "beam",
            2: "column",
            3: "door",
            4: "floor",
            5: "roof",
            6: "stair",
            7: "wall",
            8: "window",
        },
        "export_categories": {"beam", "column", "door", "floor", "roof", "stair", "wall", "window"},
    },
}

REVIT_FAMILY_CATEGORIES = {
    "wall": "Walls",
    "floor": "Floors",
    "ceiling": "Ceilings",
    "roof": "Roofs",
    "column": "StructuralColumns",
    "beam": "StructuralFraming",
    "door": "Doors",
    "window": "Windows",
    "stair": "Stairs",
}

DEFAULT_MIN_POINTS = {
    "wall": 500,
    "floor": 500,
    "ceiling": 500,
    "roof": 500,
    "column": 200,
    "beam": 200,
    "door": 100,
    "window": 100,
    "stair": 200,
}

NEIGHBOR_OFFSETS = {
    "6": np.array(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=np.int32,
    ),
    "26": np.array(
        [
            [dx, dy, dz]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ],
        dtype=np.int32,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster a labeled PLY into Revit-friendly BIM element candidates."
    )
    parser.add_argument("--input", required=True, help="Input labeled PLY path.")
    parser.add_argument(
        "--label-space",
        choices=["auto", "s3dis", "pcs"],
        default="auto",
        help="Label set used by the input file.",
    )
    parser.add_argument(
        "--cluster-voxel-size",
        type=float,
        default=0.25,
        help="Sparse clustering voxel size in input units.",
    )
    parser.add_argument(
        "--neighbor-mode",
        choices=["6", "26"],
        default="26",
        help="Voxel connectivity used during clustering.",
    )
    parser.add_argument(
        "--min-cluster-points",
        type=int,
        default=0,
        help="Override per-category minimum cluster size. 0 keeps the defaults.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <input_stem>_bim_candidates.json next to the input.",
    )
    return parser.parse_args()


def as_float_list(values: np.ndarray | list[float], ndigits: int = 6) -> list[float]:
    return [round(float(v), ndigits) for v in values]



def robust_min_max(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    values = np.asarray(values)
    if values.size == 0:
        return 0.0, 0.0
    if values.size < 20:
        return float(values.min()), float(values.max())
    lo, hi = np.percentile(values, [lower, upper])
    return float(lo), float(hi)



def load_labeled_ply(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    data = PlyData.read(path)
    vertex = data["vertex"]
    prop_names = {prop.name for prop in vertex.properties}

    if "class_id" in prop_names:
        class_property = "class_id"
    elif "class" in prop_names:
        class_property = "class"
    else:
        raise ValueError(
            "The input PLY is missing a semantic label property. Expected 'class_id' or 'class'."
        )

    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    labels = np.asarray(vertex[class_property]).astype(np.int32, copy=False)
    return points, labels, class_property



def detect_label_space(labels: np.ndarray) -> str:
    positive = labels[labels > 0]
    if positive.size == 0:
        raise ValueError("No positive class ids were found in the input file.")

    max_label = int(positive.max())
    if max_label <= 8:
        return "pcs"
    if max_label <= 13:
        return "s3dis"
    raise ValueError(
        f"Could not infer label space from max class id {max_label}. Please pass --label-space explicitly."
    )



def quantized_clusters(
    points: np.ndarray,
    voxel_size: float,
    neighbor_offsets: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    if len(points) == 0:
        return [], []

    quantized = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(quantized, axis=0, return_inverse=True)

    point_indices_by_voxel: list[list[int]] = [[] for _ in range(len(unique_voxels))]
    for point_index, voxel_index in enumerate(inverse):
        point_indices_by_voxel[int(voxel_index)].append(int(point_index))

    voxel_lookup = {tuple(voxel.tolist()): idx for idx, voxel in enumerate(unique_voxels)}
    visited = np.zeros(len(unique_voxels), dtype=bool)

    clusters: list[np.ndarray] = []
    cluster_voxel_counts: list[int] = []

    for start_idx in range(len(unique_voxels)):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        visited[start_idx] = True
        cluster_points: list[int] = []
        voxel_count = 0

        while stack:
            voxel_idx = stack.pop()
            voxel_count += 1
            cluster_points.extend(point_indices_by_voxel[voxel_idx])
            base = unique_voxels[voxel_idx]

            for offset in neighbor_offsets:
                neighbor_key = (int(base[0] + offset[0]), int(base[1] + offset[1]), int(base[2] + offset[2]))
                neighbor_idx = voxel_lookup.get(neighbor_key)
                if neighbor_idx is None or visited[neighbor_idx]:
                    continue
                visited[neighbor_idx] = True
                stack.append(neighbor_idx)

        clusters.append(np.array(cluster_points, dtype=np.int32))
        cluster_voxel_counts.append(voxel_count)

    return clusters, cluster_voxel_counts



def planar_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy = points[:, :2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid

    if len(points) < 3 or np.allclose(centered, 0):
        major = np.array([1.0, 0.0], dtype=np.float32)
        minor = np.array([0.0, 1.0], dtype=np.float32)
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        major = vh[0].astype(np.float32)
        if vh.shape[0] > 1:
            minor = vh[1].astype(np.float32)
        else:
            minor = np.array([-major[1], major[0]], dtype=np.float32)

    major = major / max(np.linalg.norm(major), 1e-8)
    minor = minor / max(np.linalg.norm(minor), 1e-8)

    proj_major = centered @ major
    proj_minor = centered @ minor
    return centroid, major, minor, proj_major, proj_minor



def oriented_rectangle(
    center_xy: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    major_lo: float,
    major_hi: float,
    minor_lo: float,
    minor_hi: float,
    elevation: float,
) -> list[list[float]]:
    corners_xy = [
        center_xy + major * major_lo + minor * minor_lo,
        center_xy + major * major_hi + minor * minor_lo,
        center_xy + major * major_hi + minor * minor_hi,
        center_xy + major * major_lo + minor * minor_hi,
    ]
    profile = [as_float_list([corner[0], corner[1], elevation]) for corner in corners_xy]
    profile.append(profile[0])
    return profile



def common_element_payload(
    element_id: str,
    category: str,
    source_label_id: int,
    source_label_name: str,
    points: np.ndarray,
    cluster_voxel_count: int,
    cluster_voxel_size: float,
) -> dict:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    dimensions = bbox_max - bbox_min
    centroid = points.mean(axis=0)
    return {
        "id": element_id,
        "category": category,
        "family_category": REVIT_FAMILY_CATEGORIES[category],
        "source_class": {
            "id": int(source_label_id),
            "name": source_label_name,
        },
        "point_count": int(len(points)),
        "bbox": {
            "min": as_float_list(bbox_min),
            "max": as_float_list(bbox_max),
        },
        "centroid": as_float_list(centroid),
        "bbox_dimensions": {
            "x": round(float(dimensions[0]), 6),
            "y": round(float(dimensions[1]), 6),
            "z": round(float(dimensions[2]), 6),
        },
        "clustering": {
            "voxel_size": round(float(cluster_voxel_size), 6),
            "voxel_count": int(cluster_voxel_count),
        },
    }



def build_element(
    category: str,
    element_id: str,
    source_label_id: int,
    source_label_name: str,
    points: np.ndarray,
    cluster_voxel_count: int,
    cluster_voxel_size: float,
) -> dict:
    payload = common_element_payload(
        element_id,
        category,
        source_label_id,
        source_label_name,
        points,
        cluster_voxel_count,
        cluster_voxel_size,
    )

    center_xy, major, minor, proj_major, proj_minor = planar_axes(points)
    major_lo, major_hi = robust_min_max(proj_major)
    minor_lo, minor_hi = robust_min_max(proj_minor)
    z_lo, z_hi = robust_min_max(points[:, 2])

    length = max(major_hi - major_lo, 0.0)
    width = max(minor_hi - minor_lo, 0.0)
    height = max(z_hi - z_lo, 0.0)
    angle_deg = math.degrees(math.atan2(float(major[1]), float(major[0])))

    if category == "wall":
        start_xy = center_xy + major * major_lo
        end_xy = center_xy + major * major_hi
        payload["dimensions"] = {
            "length": round(float(length), 6),
            "thickness": round(float(width), 6),
            "height": round(float(height), 6),
        }
        payload["geometry"] = {
            "base_line": {
                "start": as_float_list([start_xy[0], start_xy[1], z_lo]),
                "end": as_float_list([end_xy[0], end_xy[1], z_lo]),
            },
            "orientation_xy": as_float_list(major),
            "rotation_degrees": round(float(angle_deg), 6),
        }
        payload["revit_params"] = {
            "base_level_elevation": round(float(z_lo), 6),
            "top_level_elevation": round(float(z_hi), 6),
            "unconnected_height": round(float(height), 6),
            "width": round(float(width), 6),
        }
        return payload

    if category in {"floor", "ceiling", "roof"}:
        elevation = z_lo if category == "floor" else z_hi
        payload["dimensions"] = {
            "length": round(float(length), 6),
            "width": round(float(width), 6),
            "thickness": round(float(height), 6),
        }
        payload["geometry"] = {
            "profile": oriented_rectangle(center_xy, major, minor, major_lo, major_hi, minor_lo, minor_hi, elevation),
            "orientation_xy": as_float_list(major),
            "rotation_degrees": round(float(angle_deg), 6),
            "elevation": round(float(elevation), 6),
        }
        payload["revit_params"] = {
            "level_elevation": round(float(elevation), 6),
            "thickness": round(float(height), 6),
        }
        return payload

    if category == "column":
        center_point = center_xy.mean()  # placeholder to keep lint calm
        _ = center_point
        payload["dimensions"] = {
            "width": round(float(length), 6),
            "depth": round(float(width), 6),
            "height": round(float(height), 6),
        }
        payload["geometry"] = {
            "axis": {
                "start": as_float_list([points[:, 0].mean(), points[:, 1].mean(), z_lo]),
                "end": as_float_list([points[:, 0].mean(), points[:, 1].mean(), z_hi]),
            },
            "orientation_xy": as_float_list(major),
            "rotation_degrees": round(float(angle_deg), 6),
        }
        payload["revit_params"] = {
            "base_level_elevation": round(float(z_lo), 6),
            "top_level_elevation": round(float(z_hi), 6),
            "width": round(float(length), 6),
            "depth": round(float(width), 6),
        }
        return payload

    if category == "beam":
        start_xy = center_xy + major * major_lo
        end_xy = center_xy + major * major_hi
        z_mid = float(np.median(points[:, 2]))
        payload["dimensions"] = {
            "length": round(float(length), 6),
            "width": round(float(width), 6),
            "depth": round(float(height), 6),
        }
        payload["geometry"] = {
            "center_line": {
                "start": as_float_list([start_xy[0], start_xy[1], z_mid]),
                "end": as_float_list([end_xy[0], end_xy[1], z_mid]),
            },
            "orientation_xy": as_float_list(major),
            "rotation_degrees": round(float(angle_deg), 6),
            "z_center": round(float(z_mid), 6),
        }
        payload["revit_params"] = {
            "reference_level_elevation": round(float(z_mid), 6),
            "cut_length": round(float(length), 6),
            "section_width": round(float(width), 6),
            "section_depth": round(float(height), 6),
        }
        return payload

    if category in {"door", "window", "stair"}:
        payload["dimensions"] = {
            "width": round(float(length), 6),
            "depth": round(float(width), 6),
            "height": round(float(height), 6),
        }
        payload["geometry"] = {
            "profile": oriented_rectangle(center_xy, major, minor, major_lo, major_hi, minor_lo, minor_hi, z_lo),
            "rotation_degrees": round(float(angle_deg), 6),
            "host_wall_hint": None,
        }
        payload["revit_params"] = {
            "base_elevation": round(float(z_lo), 6),
            "top_elevation": round(float(z_hi), 6),
            "width": round(float(length), 6),
            "height": round(float(height), 6),
        }
        return payload

    raise ValueError(f"Unsupported category: {category}")



def label_histogram(labels: np.ndarray, class_id_to_name: dict[int, str]) -> list[dict]:
    result = []
    for label_id in sorted(np.unique(labels)):
        if int(label_id) <= 0:
            continue
        result.append(
            {
                "class_id": int(label_id),
                "class_name": class_id_to_name.get(int(label_id), f"class_{int(label_id)}"),
                "count": int(np.count_nonzero(labels == label_id)),
            }
        )
    return result



def export_candidates(
    points: np.ndarray,
    labels: np.ndarray,
    label_space: str,
    cluster_voxel_size: float,
    neighbor_mode: str,
    min_cluster_points_override: int,
) -> tuple[list[dict], dict]:
    spec = LABEL_SPACES[label_space]
    class_id_to_name = spec["class_id_to_name"]
    export_categories = spec["export_categories"]
    offsets = NEIGHBOR_OFFSETS[neighbor_mode]

    elements: list[dict] = []
    element_counts: dict[str, int] = {}
    skipped_small_clusters = 0
    skipped_non_bim_labels: list[dict] = []

    for label_id in sorted(np.unique(labels)):
        label_id = int(label_id)
        if label_id <= 0:
            continue

        source_name = class_id_to_name.get(label_id, f"class_{label_id}")
        category = source_name.lower()
        if category not in export_categories:
            skipped_non_bim_labels.append({
                "class_id": label_id,
                "class_name": source_name,
                "point_count": int(np.count_nonzero(labels == label_id)),
            })
            continue

        class_points = points[labels == label_id]
        clusters, cluster_voxel_counts = quantized_clusters(class_points, cluster_voxel_size, offsets)

        for cluster_points_idx, cluster_voxel_count in zip(clusters, cluster_voxel_counts):
            candidate_points = class_points[cluster_points_idx]
            min_points = min_cluster_points_override or DEFAULT_MIN_POINTS[category]
            if len(candidate_points) < min_points:
                skipped_small_clusters += 1
                continue

            next_index = element_counts.get(category, 0) + 1
            element_counts[category] = next_index
            element_id = f"{category}_{next_index:04d}"
            elements.append(
                build_element(
                    category,
                    element_id,
                    label_id,
                    source_name,
                    candidate_points,
                    cluster_voxel_count,
                    cluster_voxel_size,
                )
            )

    summary = {
        "element_counts": element_counts,
        "skipped_small_clusters": skipped_small_clusters,
        "skipped_non_bim_labels": skipped_non_bim_labels,
        "label_histogram": label_histogram(labels, class_id_to_name),
    }
    return elements, summary



def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = Path(args.output).resolve() if args.output else input_path.with_name(f"{input_path.stem}_bim_candidates.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    points, labels, label_property = load_labeled_ply(input_path)
    load_seconds = time.perf_counter() - t0

    label_space = args.label_space if args.label_space != "auto" else detect_label_space(labels)

    t1 = time.perf_counter()
    elements, export_summary = export_candidates(
        points,
        labels,
        label_space,
        args.cluster_voxel_size,
        args.neighbor_mode,
        args.min_cluster_points,
    )
    export_seconds = time.perf_counter() - t1

    payload = {
        "schema_version": 1,
        "source": {
            "input_path": str(input_path),
            "label_property": label_property,
            "label_space": label_space,
            "point_count": int(len(points)),
            "cluster_voxel_size": round(float(args.cluster_voxel_size), 6),
            "neighbor_mode": args.neighbor_mode,
        },
        "assumptions": {
            "up_axis": "z",
            "units": "same as input point cloud",
            "note": "These are BIM element candidates derived from semantic clusters. They are intended as seed geometry for Revit automation, not final BIM authoring output.",
        },
        "summary": {
            **export_summary,
            "exported_element_count": int(len(elements)),
            "timings_seconds": {
                "load_points": round(float(load_seconds), 6),
                "export_candidates": round(float(export_seconds), 6),
            },
        },
        "elements": elements,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Label space: {label_space}")
    print(f"Exported elements: {len(elements)}")
    for category, count in sorted(export_summary["element_counts"].items()):
        print(f"  - {category}: {count}")
    print(f"Skipped small clusters: {export_summary['skipped_small_clusters']}")


if __name__ == "__main__":
    main()
