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
from collections import defaultdict
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
    parser.add_argument(
        "--angle-regularization",
        choices=["none", "dominant_orthogonal"],
        default="dominant_orthogonal",
        help="Regularize element rotations to dominant building axes for more stable Revit import.",
    )
    parser.add_argument(
        "--snap-angle-threshold-deg",
        type=float,
        default=8.0,
        help="Maximum angular deviation allowed when snapping elements to dominant axes.",
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


def simplify_closed_polyline(profile_xy: list[np.ndarray]) -> list[np.ndarray]:
    if len(profile_xy) < 4:
        return profile_xy
    simplified = profile_xy[:]
    changed = True
    while changed and len(simplified) >= 4:
        changed = False
        next_profile: list[np.ndarray] = []
        count = len(simplified)
        for idx in range(count):
            prev_pt = simplified[idx - 1]
            curr_pt = simplified[idx]
            next_pt = simplified[(idx + 1) % count]
            v1 = curr_pt - prev_pt
            v2 = next_pt - curr_pt
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            if abs(cross) <= 1e-6:
                changed = True
                continue
            next_profile.append(curr_pt)
        simplified = next_profile
    return simplified


def polygon_area_xy(vertices: list[np.ndarray]) -> float:
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(vertices)):
        x1, y1 = vertices[idx]
        x2, y2 = vertices[(idx + 1) % len(vertices)]
        area += float(x1 * y2 - x2 * y1)
    return 0.5 * area


def trace_boundary_loops(cells: np.ndarray) -> list[list[np.ndarray]]:
    if len(cells) == 0:
        return []

    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = defaultdict(int)

    def add_edge(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        edge = (p1, p2) if p1 <= p2 else (p2, p1)
        edge_counts[edge] += 1

    for cell_x, cell_y in cells.astype(np.int32):
        left = 2 * int(cell_x) - 1
        right = 2 * int(cell_x) + 1
        bottom = 2 * int(cell_y) - 1
        top = 2 * int(cell_y) + 1
        a = (left, bottom)
        b = (right, bottom)
        c = (right, top)
        d = (left, top)
        add_edge(a, b)
        add_edge(b, c)
        add_edge(c, d)
        add_edge(d, a)

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for p1, p2 in boundary_edges:
        adjacency[p1].append(p2)
        adjacency[p2].append(p1)

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    loops: list[list[np.ndarray]] = []

    for start_edge in boundary_edges:
        canonical_start = start_edge if start_edge[0] <= start_edge[1] else (start_edge[1], start_edge[0])
        if canonical_start in visited_edges:
            continue

        start = start_edge[0]
        current = start_edge[1]
        previous = start
        loop = [np.array(start, dtype=np.float32), np.array(current, dtype=np.float32)]
        visited_edges.add(canonical_start)

        while current != start:
            neighbors = adjacency[current]
            next_candidates = [neighbor for neighbor in neighbors if neighbor != previous]
            if not next_candidates:
                loop = []
                break
            next_vertex = next_candidates[0]
            canonical = (current, next_vertex) if current <= next_vertex else (next_vertex, current)
            if canonical in visited_edges and next_vertex != start:
                alternatives = [neighbor for neighbor in next_candidates if ((current, neighbor) if current <= neighbor else (neighbor, current)) not in visited_edges]
                if not alternatives:
                    loop = []
                    break
                next_vertex = alternatives[0]
                canonical = (current, next_vertex) if current <= next_vertex else (next_vertex, current)
            visited_edges.add(canonical)
            previous, current = current, next_vertex
            if current != start:
                loop.append(np.array(current, dtype=np.float32))

        if len(loop) >= 4:
            vertices = [vertex / 2.0 for vertex in loop[:-1]]
            vertices = simplify_closed_polyline(vertices)
            if len(vertices) >= 4:
                loops.append(vertices)

    return loops


def occupancy_boundary_profile(
    points: np.ndarray,
    angle_deg: float,
    voxel_size: float,
    elevation: float,
) -> list[list[float]] | None:
    xy = points[:, :2]
    centroid = xy.mean(axis=0, keepdims=True)
    rotated = rotate_xy(xy - centroid, -angle_deg)
    cells = np.round(rotated / max(voxel_size, 1e-6)).astype(np.int32)
    unique_cells = np.unique(cells, axis=0)
    loops = trace_boundary_loops(unique_cells)
    if not loops:
        return None

    largest = max(loops, key=lambda vertices: abs(polygon_area_xy(vertices)))
    if polygon_area_xy(largest) < 0:
        largest = list(reversed(largest))

    world_rotation = rotation_matrix(angle_deg)
    world_vertices = [vertex * voxel_size @ world_rotation.T + centroid.squeeze(0) for vertex in largest]
    profile = [as_float_list([vertex[0], vertex[1], elevation]) for vertex in world_vertices]
    if profile and profile[0] != profile[-1]:
        profile.append(profile[0])
    return profile


def rotation_matrix(angle_deg: float) -> np.ndarray:
    radians = math.radians(float(angle_deg))
    return np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float32,
    )


def rotate_xy(points_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    return points_xy @ rotation_matrix(angle_deg).T


def contiguous_ranges(values: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    sorted_values = np.sort(np.unique(values.astype(np.int32)))
    ranges: list[tuple[int, int]] = []
    start = int(sorted_values[0])
    prev = start
    for value in sorted_values[1:]:
        value = int(value)
        if value - prev <= max_gap:
            prev = value
            continue
        ranges.append((start, prev))
        start = value
        prev = value
    ranges.append((start, prev))
    return ranges


def manhattan_alignment_angle(points: np.ndarray, voxel_size: float) -> float:
    xy = points[:, :2]
    centroid = xy.mean(axis=0, keepdims=True)
    centered = xy - centroid
    if len(points) < 16:
        return 0.0

    best_angle = 0.0
    best_score = -1.0
    step_deg = 2.0
    for angle_deg in np.arange(0.0, 90.0, step_deg):
        rotated = rotate_xy(centered, -float(angle_deg))
        quantized = np.round(rotated / max(voxel_size, 1e-6)).astype(np.int32)
        unique_xy = np.unique(quantized, axis=0)
        if len(unique_xy) == 0:
            continue
        x_counts = np.bincount(unique_xy[:, 0] - unique_xy[:, 0].min())
        y_counts = np.bincount(unique_xy[:, 1] - unique_xy[:, 1].min())
        score = float(np.sum(x_counts.astype(np.float64) ** 2) + np.sum(y_counts.astype(np.float64) ** 2))
        if score > best_score:
            best_score = score
            best_angle = float(angle_deg)
    return best_angle


def build_wall_element_from_segment(
    element_id: str,
    source_label_id: int,
    source_label_name: str,
    segment_points: np.ndarray,
    cluster_voxel_count: int,
    cluster_voxel_size: float,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    thickness: float,
) -> dict:
    payload = common_element_payload(
        element_id,
        "wall",
        source_label_id,
        source_label_name,
        segment_points,
        cluster_voxel_count,
        cluster_voxel_size,
    )
    z_lo, z_hi = robust_min_max(segment_points[:, 2])
    height = max(z_hi - z_lo, cluster_voxel_size)
    length = float(np.linalg.norm(end_xy - start_xy))
    direction = end_xy - start_xy
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    angle_deg = math.degrees(math.atan2(float(direction[1]), float(direction[0])))

    payload["dimensions"] = {
        "length": round(float(length), 6),
        "thickness": round(float(thickness), 6),
        "height": round(float(height), 6),
    }
    payload["geometry"] = {
        "base_line": {
            "start": as_float_list([start_xy[0], start_xy[1], z_lo]),
            "end": as_float_list([end_xy[0], end_xy[1], z_lo]),
        },
        "orientation_xy": as_float_list(direction),
        "rotation_degrees": round(float(angle_deg), 6),
    }
    payload["revit_params"] = {
        "base_level_elevation": round(float(z_lo), 6),
        "top_level_elevation": round(float(z_hi), 6),
        "unconnected_height": round(float(height), 6),
        "width": round(float(thickness), 6),
    }
    payload["segmentation"] = {
        "strategy": "manhattan_segment_decomposition",
    }
    return payload


def decompose_wall_cluster(
    points: np.ndarray,
    element_id_prefix: str,
    source_label_id: int,
    source_label_name: str,
    cluster_voxel_count: int,
    cluster_voxel_size: float,
    min_cluster_points: int,
) -> list[dict]:
    if len(points) < max(min_cluster_points * 2, 1500):
        return []

    angle_deg = manhattan_alignment_angle(points, cluster_voxel_size)
    centroid_xy = points[:, :2].mean(axis=0, keepdims=True)
    rotated_xy = rotate_xy(points[:, :2] - centroid_xy, -angle_deg)
    qx = np.round(rotated_xy[:, 0] / max(cluster_voxel_size, 1e-6)).astype(np.int32)
    qy = np.round(rotated_xy[:, 1] / max(cluster_voxel_size, 1e-6)).astype(np.int32)
    unique_xy = np.unique(np.column_stack([qx, qy]), axis=0)
    if len(unique_xy) == 0:
        return []

    x_values = unique_xy[:, 0]
    y_values = unique_xy[:, 1]
    span_x = int(x_values.max() - x_values.min() + 1)
    span_y = int(y_values.max() - y_values.min() + 1)
    if span_x < 4 or span_y < 4:
        return []

    x_hist = np.bincount(x_values - x_values.min())
    y_hist = np.bincount(y_values - y_values.min())
    vertical_support = max(4, int(round(span_y * 0.18)))
    horizontal_support = max(4, int(round(span_x * 0.18)))

    x_band_indices = np.where(x_hist >= vertical_support)[0] + x_values.min()
    y_band_indices = np.where(y_hist >= horizontal_support)[0] + y_values.min()
    x_bands = contiguous_ranges(x_band_indices)
    y_bands = contiguous_ranges(y_band_indices)

    if len(x_bands) + len(y_bands) <= 1:
        return []

    min_segment_bins = max(4, int(round(min(span_x, span_y) * 0.15)))
    elements: list[dict] = []
    next_index = 1
    world_rotation = rotation_matrix(angle_deg)

    def local_to_world(local_xy: np.ndarray) -> np.ndarray:
        return local_xy @ world_rotation.T + centroid_xy.squeeze(0)

    for x_lo, x_hi in x_bands:
        in_band = (qx >= x_lo) & (qx <= x_hi)
        runs = contiguous_ranges(qy[in_band])
        for y_lo, y_hi in runs:
            if y_hi - y_lo + 1 < min_segment_bins:
                continue
            mask = in_band & (qy >= y_lo) & (qy <= y_hi)
            segment_points = points[mask]
            if len(segment_points) < min_cluster_points:
                continue

            band_width = max((x_hi - x_lo + 1) * cluster_voxel_size, cluster_voxel_size)
            segment_length = (y_hi - y_lo + 1) * cluster_voxel_size
            if segment_length <= band_width * 1.5:
                continue

            x_center = ((x_lo + x_hi) * 0.5) * cluster_voxel_size
            y_start = y_lo * cluster_voxel_size
            y_end = y_hi * cluster_voxel_size
            start_xy = local_to_world(np.array([[x_center, y_start]], dtype=np.float32))[0]
            end_xy = local_to_world(np.array([[x_center, y_end]], dtype=np.float32))[0]
            element_id = f"{element_id_prefix}_{next_index:04d}"
            elements.append(
                build_wall_element_from_segment(
                    element_id,
                    source_label_id,
                    source_label_name,
                    segment_points,
                    cluster_voxel_count,
                    cluster_voxel_size,
                    start_xy,
                    end_xy,
                    band_width,
                )
            )
            next_index += 1

    for y_lo, y_hi in y_bands:
        in_band = (qy >= y_lo) & (qy <= y_hi)
        runs = contiguous_ranges(qx[in_band])
        for x_lo, x_hi in runs:
            if x_hi - x_lo + 1 < min_segment_bins:
                continue
            mask = in_band & (qx >= x_lo) & (qx <= x_hi)
            segment_points = points[mask]
            if len(segment_points) < min_cluster_points:
                continue

            band_width = max((y_hi - y_lo + 1) * cluster_voxel_size, cluster_voxel_size)
            segment_length = (x_hi - x_lo + 1) * cluster_voxel_size
            if segment_length <= band_width * 1.5:
                continue

            y_center = ((y_lo + y_hi) * 0.5) * cluster_voxel_size
            x_start = x_lo * cluster_voxel_size
            x_end = x_hi * cluster_voxel_size
            start_xy = local_to_world(np.array([[x_start, y_center]], dtype=np.float32))[0]
            end_xy = local_to_world(np.array([[x_end, y_center]], dtype=np.float32))[0]
            element_id = f"{element_id_prefix}_{next_index:04d}"
            elements.append(
                build_wall_element_from_segment(
                    element_id,
                    source_label_id,
                    source_label_name,
                    segment_points,
                    cluster_voxel_count,
                    cluster_voxel_size,
                    start_xy,
                    end_xy,
                    band_width,
                )
            )
            next_index += 1

    if len(elements) <= 1:
        return []
    return elements


def angle_mod_180(angle_deg: float) -> float:
    angle = float(angle_deg) % 180.0
    return angle + 180.0 if angle < 0.0 else angle


def angular_difference_180(a_deg: float, b_deg: float) -> float:
    delta = abs(angle_mod_180(a_deg) - angle_mod_180(b_deg))
    return min(delta, 180.0 - delta)


def axis_from_angle_deg(angle_deg: float) -> np.ndarray:
    radians = math.radians(float(angle_deg))
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)


def choose_angle_targets(elements: list[dict]) -> list[float]:
    evidence: list[tuple[float, float]] = []
    for element in elements:
        category = str(element.get("category", "")).lower()
        rotation = element.get("geometry", {}).get("rotation_degrees")
        if rotation is None:
            continue

        if category in {"wall", "beam"}:
            length = float(element.get("dimensions", {}).get("length", 0.0))
            weight = max(length, 0.0) * max(float(element.get("point_count", 0)), 1.0)
            if weight > 0.0:
                evidence.append((angle_mod_180(rotation), weight))
            continue

        if category in {"floor", "ceiling", "roof"}:
            length = float(element.get("dimensions", {}).get("length", 0.0))
            width = float(element.get("dimensions", {}).get("width", 0.0))
            aspect = max(length, width) / max(min(length, width), 1e-6)
            if aspect >= 1.15:
                weight = max(length, width) * max(float(element.get("point_count", 0)), 1.0) * 0.35
                evidence.append((angle_mod_180(rotation), weight))

    if not evidence:
        return []

    anchor_angle, _ = max(evidence, key=lambda item: item[1])
    cluster_members = [item for item in evidence if angular_difference_180(item[0], anchor_angle) <= 15.0]
    total_weight = sum(weight for _, weight in cluster_members)
    if total_weight <= 0.0:
        dominant = anchor_angle
    else:
        vec = np.zeros(2, dtype=np.float64)
        for angle_deg, weight in cluster_members:
            radians = math.radians(angle_deg * 2.0)
            vec += weight * np.array([math.cos(radians), math.sin(radians)], dtype=np.float64)
        dominant = angle_mod_180(0.5 * math.degrees(math.atan2(vec[1], vec[0])))

    orthogonal = angle_mod_180(dominant + 90.0)
    return [dominant, orthogonal]


def rebuild_profile(element: dict, angle_deg: float) -> None:
    center = np.asarray(element["centroid"][:2], dtype=np.float32)
    current_angle = float(element["geometry"].get("rotation_degrees", angle_deg))
    delta_deg = float(angle_deg) - current_angle
    profile = element["geometry"].get("profile")
    if profile and len(profile) >= 4:
        rot = rotation_matrix(delta_deg)
        rotated_profile = []
        for point in profile[:-1]:
            xy = np.asarray(point[:2], dtype=np.float32)
            z = float(point[2])
            rotated_xy = (xy - center) @ rot.T + center
            rotated_profile.append(as_float_list([rotated_xy[0], rotated_xy[1], z]))
        if rotated_profile and rotated_profile[0] != rotated_profile[-1]:
            rotated_profile.append(rotated_profile[0])
        element["geometry"]["profile"] = rotated_profile
    else:
        elevation = float(element["geometry"].get("elevation", 0.0))
        length = float(element["dimensions"].get("length", 0.0))
        width = float(element["dimensions"].get("width", element["dimensions"].get("depth", 0.0)))
        major = axis_from_angle_deg(angle_deg)
        minor = np.array([-major[1], major[0]], dtype=np.float32)
        major_lo = -0.5 * length
        major_hi = 0.5 * length
        minor_lo = -0.5 * width
        minor_hi = 0.5 * width
        element["geometry"]["profile"] = oriented_rectangle(center, major, minor, major_lo, major_hi, minor_lo, minor_hi, elevation)
    major = axis_from_angle_deg(angle_deg)
    element["geometry"]["orientation_xy"] = as_float_list(major)
    element["geometry"]["rotation_degrees"] = round(float(angle_deg), 6)


def rebuild_linear_geometry(element: dict, angle_deg: float) -> None:
    major = axis_from_angle_deg(angle_deg)
    center = np.asarray(element["centroid"][:2], dtype=np.float32)
    length = float(element["dimensions"].get("length", element["dimensions"].get("cut_length", 0.0)))
    half = 0.5 * length
    start_xy = center - major * half
    end_xy = center + major * half

    if element["category"] == "wall":
        z = float(element["revit_params"]["base_level_elevation"])
        element["geometry"]["base_line"] = {
            "start": as_float_list([start_xy[0], start_xy[1], z]),
            "end": as_float_list([end_xy[0], end_xy[1], z]),
        }
    elif element["category"] == "beam":
        z = float(element["geometry"]["z_center"])
        element["geometry"]["center_line"] = {
            "start": as_float_list([start_xy[0], start_xy[1], z]),
            "end": as_float_list([end_xy[0], end_xy[1], z]),
        }

    element["geometry"]["orientation_xy"] = as_float_list(major)
    element["geometry"]["rotation_degrees"] = round(float(angle_deg), 6)


def regularize_element_angles(
    elements: list[dict],
    strategy: str,
    threshold_deg: float,
) -> tuple[list[dict], dict]:
    if strategy == "none" or not elements:
        return elements, {"strategy": strategy, "applied": False, "target_angles_deg": [], "snapped_count": 0}

    targets = choose_angle_targets(elements)
    if not targets:
        return elements, {"strategy": strategy, "applied": False, "target_angles_deg": [], "snapped_count": 0}

    snapped_count = 0
    for element in elements:
        category = str(element.get("category", "")).lower()
        if category not in {"wall", "beam", "floor", "ceiling", "roof", "door", "window", "stair"}:
            continue

        current_angle = element.get("geometry", {}).get("rotation_degrees")
        if current_angle is None:
            continue

        target = min(targets, key=lambda candidate: angular_difference_180(current_angle, candidate))
        delta = angular_difference_180(current_angle, target)
        if delta > threshold_deg:
            continue

        if category in {"wall", "beam"}:
            rebuild_linear_geometry(element, target)
        else:
            rebuild_profile(element, target)

        element.setdefault("postprocess", {})
        element["postprocess"]["angle_regularization"] = {
            "from_degrees": round(float(current_angle), 6),
            "to_degrees": round(float(target), 6),
            "delta_degrees": round(float(delta), 6),
        }
        snapped_count += 1

    return elements, {
        "strategy": strategy,
        "applied": snapped_count > 0,
        "target_angles_deg": [round(float(angle), 6) for angle in targets],
        "snapped_count": int(snapped_count),
        "threshold_deg": round(float(threshold_deg), 6),
    }



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
        profile = occupancy_boundary_profile(points, angle_deg, cluster_voxel_size, elevation)
        profile_type = "occupancy_boundary"
        if profile is None:
            profile = oriented_rectangle(center_xy, major, minor, major_lo, major_hi, minor_lo, minor_hi, elevation)
            profile_type = "oriented_rectangle"
        payload["dimensions"] = {
            "length": round(float(length), 6),
            "width": round(float(width), 6),
            "thickness": round(float(height), 6),
        }
        payload["geometry"] = {
            "profile": profile,
            "orientation_xy": as_float_list(major),
            "rotation_degrees": round(float(angle_deg), 6),
            "elevation": round(float(elevation), 6),
            "profile_type": profile_type,
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

            if category == "wall":
                current_count = element_counts.get(category, 0)
                decomposed_elements = decompose_wall_cluster(
                    candidate_points,
                    category,
                    label_id,
                    source_name,
                    cluster_voxel_count,
                    cluster_voxel_size,
                    min_points,
                )
                if decomposed_elements:
                    for decomposed in decomposed_elements:
                        current_count += 1
                        decomposed["id"] = f"{category}_{current_count:04d}"
                        elements.append(decomposed)
                    element_counts[category] = current_count
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
    elements, angle_regularization_summary = regularize_element_angles(
        elements,
        args.angle_regularization,
        args.snap_angle_threshold_deg,
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
            "angle_regularization": angle_regularization_summary,
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
    if angle_regularization_summary["applied"]:
        print(
            "Angle regularization: {} elements snapped to {}".format(
                angle_regularization_summary["snapped_count"],
                angle_regularization_summary["target_angles_deg"],
            )
        )


if __name__ == "__main__":
    main()
