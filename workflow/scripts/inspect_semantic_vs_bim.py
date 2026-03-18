"""Interactive viewer for semantic prediction vs BIM candidate debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scan2bim_debug_viewer_common import CANONICAL_COLORS, colorize_labels, load_bim_candidates, load_predicted_ply, maybe_downsample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare semantic prediction points with exported BIM candidates.")
    parser.add_argument("--prediction", required=True, help="Predicted labeled PLY path.")
    parser.add_argument("--bim-json", required=True, help="Exported BIM candidate JSON path.")
    parser.add_argument("--label-space", choices=["auto", "s3dis", "pcs"], default="auto")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--summary-json", help="Optional output path for comparison summary.")
    parser.add_argument("--summary-only", action="store_true", help="Compute summary without opening a window.")
    parser.add_argument("--screenshot", help="Optional screenshot path. Implies off-screen rendering.")
    return parser.parse_args()


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista is required for the visual debug viewer.") from exc
    return pv


def line_segments_polydata(pv, segments: list[tuple[np.ndarray, np.ndarray]]):
    if not segments:
        return None
    points = []
    lines = []
    cursor = 0
    for start, end in segments:
        points.append(np.asarray(start, dtype=float))
        points.append(np.asarray(end, dtype=float))
        lines.extend([2, cursor, cursor + 1])
        cursor += 2
    return pv.PolyData(np.asarray(points), lines=np.asarray(lines))


def closed_profile_segments(profile: list[list[float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(profile) < 2:
        return []
    segments = []
    for idx in range(len(profile) - 1):
        segments.append((np.asarray(profile[idx], dtype=float), np.asarray(profile[idx + 1], dtype=float)))
    return segments


def bbox_segments(bbox_min: list[float], bbox_max: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    corners = [
        np.array([x0, y0, z0], dtype=float),
        np.array([x1, y0, z0], dtype=float),
        np.array([x1, y1, z0], dtype=float),
        np.array([x0, y1, z0], dtype=float),
        np.array([x0, y0, z1], dtype=float),
        np.array([x1, y0, z1], dtype=float),
        np.array([x1, y1, z1], dtype=float),
        np.array([x0, y1, z1], dtype=float),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in edges]


def wall_segments(element: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    line = element.get("geometry", {}).get("base_line")
    width = float(element.get("revit_params", {}).get("width", element.get("dimensions", {}).get("thickness", 0.0)))
    if not line or width <= 0:
        return bbox_segments(element["bbox"]["min"], element["bbox"]["max"])

    start = np.asarray(line["start"], dtype=float)
    end = np.asarray(line["end"], dtype=float)
    direction = end[:2] - start[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return bbox_segments(element["bbox"]["min"], element["bbox"]["max"])
    direction = direction / norm
    normal = np.array([-direction[1], direction[0]], dtype=float)
    half = width * 0.5

    bottom_z = float(start[2])
    top_z = float(element.get("revit_params", {}).get("top_level_elevation", element["bbox"]["max"][2]))
    s0 = np.array([start[0] + normal[0] * half, start[1] + normal[1] * half, bottom_z], dtype=float)
    s1 = np.array([start[0] - normal[0] * half, start[1] - normal[1] * half, bottom_z], dtype=float)
    e0 = np.array([end[0] + normal[0] * half, end[1] + normal[1] * half, bottom_z], dtype=float)
    e1 = np.array([end[0] - normal[0] * half, end[1] - normal[1] * half, bottom_z], dtype=float)
    s0_top = s0.copy()
    s1_top = s1.copy()
    e0_top = e0.copy()
    e1_top = e1.copy()
    s0_top[2] = top_z
    s1_top[2] = top_z
    e0_top[2] = top_z
    e1_top[2] = top_z

    corners = [s0, e0, e1, s1, s0_top, e0_top, e1_top, s1_top]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in edges]


def element_segments(element: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    category = element.get("category", "").lower()
    geometry = element.get("geometry", {})

    if category == "wall":
        return wall_segments(element)
    if category in {"floor", "ceiling", "roof", "door", "window", "stair"} and "profile" in geometry:
        return closed_profile_segments(geometry["profile"])
    if category == "beam" and "center_line" in geometry:
        line = geometry["center_line"]
        return [(np.asarray(line["start"], dtype=float), np.asarray(line["end"], dtype=float))]
    if category == "column" and "axis" in geometry:
        line = geometry["axis"]
        return [(np.asarray(line["start"], dtype=float), np.asarray(line["end"], dtype=float))]
    if "bbox" in element:
        return bbox_segments(element["bbox"]["min"], element["bbox"]["max"])
    return []


def build_category_meshes(pv, elements: list[dict]) -> dict[str, object]:
    grouped_segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for element in elements:
        category = str(element.get("category", "unknown")).lower()
        grouped_segments.setdefault(category, []).extend(element_segments(element))

    meshes = {}
    for category, segments in grouped_segments.items():
        mesh = line_segments_polydata(pv, segments)
        if mesh is not None:
            meshes[category] = mesh
    return meshes


def add_semantic_points(plotter, pv, points: np.ndarray, labels: np.ndarray, point_size: float, monochrome: bool = False) -> None:
    cloud = pv.PolyData(points)
    if monochrome:
        gray = np.tile(np.asarray([150, 150, 150], dtype=np.uint8), (len(points), 1))
        cloud["rgb"] = gray
    else:
        cloud["rgb"] = colorize_labels(labels)
    plotter.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        point_size=point_size,
        render_points_as_spheres=True,
    )


def summarize(payload: dict, prediction_count: int) -> dict:
    summary = payload.get("summary", {})
    return {
        "prediction_point_count": int(prediction_count),
        "candidate_count": int(len(payload.get("elements", []))),
        "element_counts": summary.get("element_counts", {}),
        "skipped_small_clusters": summary.get("skipped_small_clusters", 0),
        "skipped_non_bim_labels": summary.get("skipped_non_bim_labels", []),
        "source": payload.get("source", {}),
    }


def main() -> None:
    args = parse_args()
    pred_path = Path(args.prediction).resolve()
    bim_path = Path(args.bim_json).resolve()
    if not pred_path.exists():
        raise FileNotFoundError("Prediction file not found: {}".format(pred_path))
    if not bim_path.exists():
        raise FileNotFoundError("BIM candidate JSON not found: {}".format(bim_path))

    prediction = load_predicted_ply(pred_path, label_space=args.label_space)
    payload = load_bim_candidates(bim_path)
    elements = payload.get("elements", [])
    summary = summarize(payload, len(prediction["points"]))

    print("Prediction: {}".format(pred_path))
    print("BIM JSON:    {}".format(bim_path))
    print("Prediction points: {}".format(summary["prediction_point_count"]))
    print("Exported candidates: {}".format(summary["candidate_count"]))
    for category, count in sorted(summary["element_counts"].items()):
        print("  - {}: {}".format(category, count))
    print("Skipped small clusters: {}".format(summary["skipped_small_clusters"]))

    if args.summary_json:
        output_path = Path(args.summary_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Summary JSON: {}".format(output_path))

    if args.summary_only:
        return

    pv = require_pyvista()
    meshes = build_category_meshes(pv, elements)
    off_screen = bool(args.screenshot)
    plotter = pv.Plotter(shape=(1, 3), window_size=(2100, 700), off_screen=off_screen)
    sample = maybe_downsample_indices(len(prediction["points"]), args.max_points, args.seed)
    sampled_points = prediction["points"][sample]
    sampled_labels = prediction["labels"][sample]

    plotter.subplot(0, 0)
    add_semantic_points(plotter, pv, sampled_points, sampled_labels, args.point_size, monochrome=False)
    plotter.add_text("Semantic Prediction", font_size=12)
    plotter.show_grid(color="gray")

    plotter.subplot(0, 1)
    add_semantic_points(plotter, pv, sampled_points, sampled_labels, max(args.point_size - 1.0, 2.0), monochrome=True)
    for category, mesh in meshes.items():
        color = CANONICAL_COLORS.get(category, CANONICAL_COLORS["unknown"])
        plotter.add_mesh(mesh, color=color, line_width=3.0)
    overlay_text = "Semantic + BIM overlay\ncandidates: {}".format(summary["candidate_count"])
    plotter.add_text(overlay_text, font_size=12)
    plotter.show_grid(color="gray")

    plotter.subplot(0, 2)
    for category, mesh in meshes.items():
        color = CANONICAL_COLORS.get(category, CANONICAL_COLORS["unknown"])
        plotter.add_mesh(mesh, color=color, line_width=4.0)
    counts_text = "BIM candidates"
    for category, count in sorted(summary["element_counts"].items()):
        counts_text += "\n{}: {}".format(category, count)
    plotter.add_text(counts_text, font_size=11)
    plotter.add_legend(
        [[category, CANONICAL_COLORS.get(category, CANONICAL_COLORS["unknown"])] for category in sorted(meshes)],
        bcolor=(20, 20, 20),
        border=True,
        size=(0.18, 0.28),
    )
    plotter.show_grid(color="gray")

    plotter.link_views()
    plotter.view_isometric()
    if args.screenshot:
        screenshot_path = Path(args.screenshot).resolve()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path))
        print("Screenshot: {}".format(screenshot_path))
    else:
        plotter.show()


if __name__ == "__main__":
    main()
