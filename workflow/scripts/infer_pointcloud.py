"""Run Scan2BIM semantic inference on a local point cloud."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.bimnet import BIMNet

EPS = 1e-6

LABEL_SPACES = {
    "s3dis": {
        "num_model_classes": 13,
        "class_names": [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",
        ],
        "display_colors": np.array(
            [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [0, 0, 142],
            ],
            dtype=np.uint8,
        ),
        "default_weights": Path("log/train_s3distest/val_best.pth"),
    },
    "pcs": {
        "num_model_classes": 8,
        "class_names": ["Beams", "Columns", "Doors", "Floors", "Roofs", "Stairs", "Walls", "Windows"],
        "display_colors": np.array(
            [
                [154, 205, 50],
                [169, 169, 169],
                [143, 48, 223],
                [255, 215, 0],
                [255, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [0, 191, 255],
            ],
            dtype=np.uint8,
        ),
        "default_weights": Path("log/train_pcs_test/latest.pth"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer semantic labels for a local point cloud with BIMNet.")
    parser.add_argument("--input", required=True, help="Input point cloud path (.pcd, .ply, .txt, .csv).")
    parser.add_argument("--label-space", choices=sorted(LABEL_SPACES), default="s3dis")
    parser.add_argument("--weights", help="Checkpoint path.")
    parser.add_argument("--mode", choices=["single_cube", "global_local_fusion"], default="single_cube")
    parser.add_argument("--cube-edge", type=int, default=96, help="Voxel resolution for local windows.")
    parser.add_argument("--global-cube-edge", type=int, default=0, help="Voxel resolution for the global pass.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=str(Path("workflow") / "outputs" / "inference"))
    parser.add_argument("--max-points", type=int, default=0, help="Optional random downsampling limit.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--save-fusion-debug", action="store_true")
    parser.add_argument("--core-size-xy", type=float, default=0.0)
    parser.add_argument("--core-size-z", type=float, default=0.0)
    parser.add_argument("--halo-xy", type=float, default=0.0)
    parser.add_argument("--halo-z", type=float, default=0.0)
    parser.add_argument("--stride-xy", type=float, default=0.0)
    parser.add_argument("--stride-z", type=float, default=0.0)
    parser.add_argument("--global-weight", type=float, default=0.35)
    parser.add_argument("--halo-weight", type=float, default=0.25)
    parser.add_argument("--min-points-per-window", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device_arg


def resolve_weights(label_space: str, weights_arg: str | None) -> Path:
    weights_path = Path(weights_arg) if weights_arg else REPO_ROOT / LABEL_SPACES[label_space]["default_weights"]
    if not weights_path.is_absolute():
        weights_path = (REPO_ROOT / weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}\nProvide --weights explicitly or add the checkpoint."
        )
    return weights_path


def infer_num_model_classes(state_dict: dict[str, torch.Tensor]) -> int:
    key = "out.cx.weight"
    if key not in state_dict:
        raise KeyError(f"Missing checkpoint tensor: {key}")
    return int(state_dict[key].shape[0])


def load_model(weights_path: Path, label_space: str, device: str) -> BIMNet:
    state_dict = torch.load(weights_path, map_location=device)
    num_classes = infer_num_model_classes(state_dict)
    expected = LABEL_SPACES[label_space]["num_model_classes"]
    if num_classes != expected:
        raise ValueError(f"Checkpoint class count {num_classes} does not match label space {expected}.")
    model = BIMNet(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def load_text_points(path: Path) -> np.ndarray:
    first_line = first_data_line(path)
    if not first_line:
        raise ValueError(f"No readable point rows were found in {path}.")
    delimiter = "," if "," in first_line else None
    points = np.loadtxt(path, delimiter=delimiter, usecols=(0, 1, 2), ndmin=2)
    return np.asarray(points, dtype=np.float32)


def load_ply_points(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"]
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)


def parse_pcd_header(handle: BinaryIO) -> dict[str, object]:
    header: dict[str, object] = {}
    while True:
        raw_line = handle.readline()
        if raw_line == b"":
            raise ValueError("Unexpected end of file while reading the PCD header.")
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        values = parts[1:]
        if key in {"FIELDS", "TYPE"}:
            header[key] = values
        elif key in {"SIZE", "COUNT"}:
            header[key] = [int(value) for value in values]
        elif key in {"WIDTH", "HEIGHT", "POINTS"}:
            header[key] = int(values[0])
        elif key == "DATA":
            header[key] = values[0].lower()
            break
    header.setdefault("COUNT", [1] * len(header["FIELDS"]))
    header.setdefault("POINTS", int(header.get("WIDTH", 0)) * int(header.get("HEIGHT", 1)))
    return header


def pcd_numpy_dtype(header: dict[str, object]) -> np.dtype:
    dtype_fields = []
    for field_name, size, type_code, count in zip(
        header["FIELDS"], header["SIZE"], header["TYPE"], header["COUNT"]
    ):
        if type_code == "F":
            base = {4: np.float32, 8: np.float64}.get(size)
        elif type_code == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(size)
        elif type_code == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
        else:
            base = None
        if base is None:
            raise ValueError(f"Unsupported PCD field: {field_name} {type_code}{size}")
        dtype_fields.append((field_name, base) if count == 1 else (field_name, base, (count,)))
    return np.dtype(dtype_fields)


def load_pcd_points(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = parse_pcd_header(handle)
        if header["DATA"] == "ascii":
            rows = np.loadtxt(handle, ndmin=2)
            fields = header["FIELDS"]
            xyz = rows[:, [fields.index("x"), fields.index("y"), fields.index("z")]]
            return np.asarray(xyz, dtype=np.float32)
        if header["DATA"] == "binary":
            points = np.frombuffer(handle.read(), dtype=pcd_numpy_dtype(header), count=header["POINTS"])
            return np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32, copy=False)
        raise NotImplementedError(f"Unsupported PCD DATA kind: {header['DATA']}")


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return load_ply_points(path)
    if suffix == ".pcd":
        return load_pcd_points(path)
    if suffix in {".txt", ".csv"}:
        return load_text_points(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def maybe_downsample(points: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, np.arange(len(points), dtype=np.int32)
    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(len(points), size=max_points, replace=False)).astype(np.int32)
    return points[keep_idx], keep_idx


def normalize_to_voxel_grid(points: np.ndarray, cube_edge: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    scale = float(np.abs(centered).max())
    if scale == 0:
        raise ValueError("The point cloud has zero extent and cannot be normalized.")
    voxel_coords = np.round((centered / scale + 1.0) * (cube_edge // 2)).astype(np.int32)
    valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < cube_edge), axis=1)
    return centroid.squeeze(0), voxel_coords, valid_mask, scale


def build_occupancy_grid(voxel_coords: np.ndarray, valid_mask: np.ndarray, cube_edge: int) -> np.ndarray:
    geom = np.zeros((cube_edge, cube_edge, cube_edge), dtype=np.float32)
    valid_coords = voxel_coords[valid_mask]
    if len(valid_coords) > 0:
        geom[tuple(valid_coords.T)] = 1.0
    return geom


def run_model_logits(model: BIMNet, geom: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(geom).unsqueeze(0).unsqueeze(0).to(device)
        return model(x).squeeze(0).cpu().numpy().astype(np.float32)


def extract_point_logits(logits_grid: np.ndarray, voxel_coords: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    point_logits = np.zeros((len(voxel_coords), logits_grid.shape[0]), dtype=np.float32)
    if np.any(valid_mask):
        valid_coords = voxel_coords[valid_mask]
        point_logits[valid_mask] = logits_grid[
            :,
            valid_coords[:, 0],
            valid_coords[:, 1],
            valid_coords[:, 2],
        ].T
    return point_logits


def logits_to_predictions(point_logits: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_channel_ids = np.full(len(point_logits), -1, dtype=np.int16)
    point_label_ids = np.zeros(len(point_logits), dtype=np.int16)
    confidence = np.zeros(len(point_logits), dtype=np.float32)
    if not np.any(valid_mask):
        return point_channel_ids, point_label_ids, confidence
    valid_logits = point_logits[valid_mask]
    channel_ids = np.argmax(valid_logits, axis=1).astype(np.int16)
    shifted = valid_logits - valid_logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), EPS, None)
    point_channel_ids[valid_mask] = channel_ids
    point_label_ids[valid_mask] = channel_ids + 1
    confidence[valid_mask] = probs[np.arange(len(channel_ids)), channel_ids].astype(np.float32)
    return point_channel_ids, point_label_ids, confidence


def class_histogram(label_ids: np.ndarray, label_space: str) -> list[dict[str, int | str]]:
    histogram = []
    for class_id, class_name in enumerate(LABEL_SPACES[label_space]["class_names"], start=1):
        count = int(np.count_nonzero(label_ids == class_id))
        if count:
            histogram.append({"class_id": class_id, "class_name": class_name, "count": count})
    histogram.sort(key=lambda item: item["count"], reverse=True)
    return histogram


def write_prediction_ply(
    out_path: Path,
    points: np.ndarray,
    point_channel_ids: np.ndarray,
    point_label_ids: np.ndarray,
    label_space: str,
    confidence: np.ndarray,
    vote_count: np.ndarray,
) -> None:
    palette = LABEL_SPACES[label_space]["display_colors"]
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    valid_mask = point_channel_ids >= 0
    if np.any(valid_mask):
        colors[valid_mask] = palette[point_channel_ids[valid_mask]]
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class_id", "i4"),
            ("confidence", "f4"),
            ("vote_count", "i4"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    vertices["class_id"] = point_label_ids.astype(np.int32)
    vertices["confidence"] = confidence.astype(np.float32)
    vertices["vote_count"] = vote_count.astype(np.int32)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(out_path)


def scene_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return bbox_min, bbox_max, bbox_max - bbox_min


def count_occupied_voxels(voxel_coords: np.ndarray, valid_mask: np.ndarray) -> int:
    return 0 if not np.any(valid_mask) else int(np.unique(voxel_coords[valid_mask], axis=0).shape[0])


def run_scene_pass(model: BIMNet, points: np.ndarray, cube_edge: int, device: str) -> dict[str, object]:
    centroid, voxel_coords, valid_mask, scale = normalize_to_voxel_grid(points, cube_edge)
    logits_grid = run_model_logits(model, build_occupancy_grid(voxel_coords, valid_mask, cube_edge), device)
    point_logits = extract_point_logits(logits_grid, voxel_coords, valid_mask)
    channel_ids, label_ids, confidence = logits_to_predictions(point_logits, valid_mask)
    return {
        "cube_edge": int(cube_edge),
        "centroid": centroid,
        "voxel_coords": voxel_coords,
        "valid_mask": valid_mask,
        "scale": float(scale),
        "occupied_voxels": count_occupied_voxels(voxel_coords, valid_mask),
        "point_logits": point_logits,
        "point_channel_ids": channel_ids,
        "point_label_ids": label_ids,
        "confidence": confidence,
    }


def resolve_window_params(points: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    _, _, extents = scene_bounds(points)
    scene_xy_extent = max(float(extents[0]), float(extents[1]), EPS)
    scene_z_extent = max(float(extents[2]), EPS)
    core_xy = args.core_size_xy if args.core_size_xy > 0 else max(scene_xy_extent * 0.6, 1.0)
    core_z = args.core_size_z if args.core_size_z > 0 else max(scene_z_extent * 0.8, 1.0)
    halo_xy = args.halo_xy if args.halo_xy > 0 else max(core_xy * 0.25, 0.25)
    halo_z = args.halo_z if args.halo_z > 0 else max(core_z * 0.25, 0.25)
    stride_xy = args.stride_xy if args.stride_xy > 0 else max(core_xy * 0.5, 0.25)
    stride_z = args.stride_z if args.stride_z > 0 else max(core_z * 0.5, 0.25)
    return {
        "core_size_xy": float(core_xy),
        "core_size_z": float(core_z),
        "halo_xy": float(halo_xy),
        "halo_z": float(halo_z),
        "stride_xy": float(stride_xy),
        "stride_z": float(stride_z),
    }


def axis_starts(axis_min: float, axis_max: float, size: float, stride: float) -> list[float]:
    if axis_max - axis_min <= size + EPS:
        return [float(axis_min)]
    last_start = float(axis_max - size)
    starts = [float(v) for v in np.arange(axis_min, last_start + EPS, stride)]
    if starts[-1] < last_start - EPS:
        starts.append(last_start)
    return starts


def mask_in_bbox(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.all((points >= bbox_min) & (points <= bbox_max), axis=1)


def generate_windows(points: np.ndarray, params: dict[str, float], args: argparse.Namespace) -> tuple[list[dict[str, object]], bool]:
    bbox_min, bbox_max, _ = scene_bounds(points)
    core_size = np.array([params["core_size_xy"], params["core_size_xy"], params["core_size_z"]], dtype=np.float32)
    halo_margin = np.array([params["halo_xy"], params["halo_xy"], params["halo_z"]], dtype=np.float32)
    windows: list[dict[str, object]] = []
    truncated = False
    for z_start in axis_starts(float(bbox_min[2]), float(bbox_max[2]), params["core_size_z"], params["stride_z"]):
        for y_start in axis_starts(float(bbox_min[1]), float(bbox_max[1]), params["core_size_xy"], params["stride_xy"]):
            for x_start in axis_starts(float(bbox_min[0]), float(bbox_max[0]), params["core_size_xy"], params["stride_xy"]):
                core_min = np.array([x_start, y_start, z_start], dtype=np.float32)
                core_max = core_min + core_size
                halo_min = core_min - halo_margin
                halo_max = core_max + halo_margin
                halo_indices = np.flatnonzero(mask_in_bbox(points, halo_min, halo_max)).astype(np.int32)
                if len(halo_indices) < args.min_points_per_window:
                    continue
                core_mask = mask_in_bbox(points[halo_indices], core_min, core_max)
                if not np.any(core_mask):
                    continue
                windows.append(
                    {
                        "window_id": f"w{len(windows):04d}",
                        "bbox_min": halo_min,
                        "bbox_max": halo_max,
                        "core_bbox_min": core_min,
                        "core_bbox_max": core_max,
                        "halo_margin": halo_margin.copy(),
                        "point_indices": halo_indices,
                        "core_point_mask": core_mask,
                    }
                )
                if args.max_windows > 0 and len(windows) >= args.max_windows:
                    truncated = True
                    return windows, truncated
    return windows, truncated


def fuse_global_local(model: BIMNet, points: np.ndarray, args: argparse.Namespace, device: str) -> dict[str, object]:
    global_pass = run_scene_pass(model, points, args.global_cube_edge or args.cube_edge, device)
    num_classes = global_pass["point_logits"].shape[1]
    local_logit_sum = np.zeros((len(points), num_classes), dtype=np.float32)
    local_weight_sum = np.zeros(len(points), dtype=np.float32)
    local_vote_count = np.zeros(len(points), dtype=np.int16)
    local_top_label_weight_sum = np.zeros((len(points), num_classes), dtype=np.float32)
    window_params = resolve_window_params(points, args)
    windows, truncated = generate_windows(points, window_params, args)
    window_records = []

    for order, window in enumerate(windows, start=1):
        point_indices = window["point_indices"]
        window_points = points[point_indices]
        weights = np.full(len(window_points), float(args.halo_weight), dtype=np.float32)
        weights[window["core_point_mask"]] = 1.0
        centroid, voxel_coords, valid_mask, scale = normalize_to_voxel_grid(window_points, args.cube_edge)
        logits_grid = run_model_logits(model, build_occupancy_grid(voxel_coords, valid_mask, args.cube_edge), device)
        point_logits = extract_point_logits(logits_grid, voxel_coords, valid_mask)
        contribute_mask = valid_mask & (weights > 0)
        contributed = int(np.count_nonzero(contribute_mask))
        if contributed:
            contrib_indices = point_indices[contribute_mask]
            contrib_weights = weights[contribute_mask]
            contrib_logits = point_logits[contribute_mask]
            local_logit_sum[contrib_indices] += contrib_logits * contrib_weights[:, None]
            local_weight_sum[contrib_indices] += contrib_weights
            local_vote_count[contrib_indices] += 1
            top_channels = np.argmax(contrib_logits, axis=1)
            local_top_label_weight_sum[contrib_indices, top_channels] += contrib_weights
        window_records.append(
            {
                "window_id": window["window_id"],
                "order": order,
                "bbox_min": window["bbox_min"].round(6).tolist(),
                "bbox_max": window["bbox_max"].round(6).tolist(),
                "core_bbox_min": window["core_bbox_min"].round(6).tolist(),
                "core_bbox_max": window["core_bbox_max"].round(6).tolist(),
                "points_in_window": int(len(point_indices)),
                "core_points": int(np.count_nonzero(window["core_point_mask"])),
                "contributed_points": contributed,
                "valid_points_after_voxel_bounds": int(np.count_nonzero(valid_mask)),
                "occupied_voxels": count_occupied_voxels(voxel_coords, valid_mask),
                "scale_abs_max": float(scale),
                "center_xyz": centroid.round(6).tolist(),
            }
        )

    weight_sum_total = local_weight_sum.copy()
    logit_sum_total = local_logit_sum.copy()
    global_valid_mask = global_pass["valid_mask"]
    if args.global_weight > 0 and np.any(global_valid_mask):
        logit_sum_total[global_valid_mask] += global_pass["point_logits"][global_valid_mask] * float(args.global_weight)
        weight_sum_total[global_valid_mask] += float(args.global_weight)
    avg_logits = np.zeros_like(logit_sum_total, dtype=np.float32)
    final_valid_mask = weight_sum_total > 0
    avg_logits[final_valid_mask] = logit_sum_total[final_valid_mask] / weight_sum_total[final_valid_mask, None]
    channel_ids, label_ids, confidence = logits_to_predictions(avg_logits, final_valid_mask)
    sorted_weights = np.sort(local_top_label_weight_sum, axis=1)
    secondary = sorted_weights[:, -2] if num_classes > 1 else np.zeros(len(points), dtype=np.float32)
    primary = sorted_weights[:, -1]
    seam_risk_mask = (
        (local_vote_count > 1)
        & (np.count_nonzero(local_top_label_weight_sum > 0, axis=1) >= 2)
        & (secondary >= np.maximum(primary * 0.5, 0.5))
    )
    return {
        "mode": "global_local_fusion",
        "cube_edge": int(args.cube_edge),
        "global_cube_edge": int(args.global_cube_edge or args.cube_edge),
        "global_pass": global_pass,
        "point_logits": avg_logits,
        "point_channel_ids": channel_ids,
        "point_label_ids": label_ids,
        "confidence": confidence,
        "vote_count_total": (local_vote_count + global_valid_mask.astype(np.int16)).astype(np.int16),
        "local_vote_count": local_vote_count,
        "weight_sum_total": weight_sum_total,
        "local_weight_sum": local_weight_sum,
        "final_valid_mask": final_valid_mask,
        "seam_risk_mask": seam_risk_mask,
        "window_params": window_params,
        "window_records": window_records,
        "truncated_windows": truncated,
        "global_prior_weight": float(args.global_weight),
        "halo_weight": float(args.halo_weight),
        "local_top_label_weight_sum": local_top_label_weight_sum,
    }


def build_summary(
    input_path: Path,
    weights_path: Path,
    output_ply_path: Path,
    output_npz_path: Path | None,
    output_debug_path: Path | None,
    label_space: str,
    device: str,
    total_input_points: int,
    processed_points: int,
    kept_index_count: int,
    points: np.ndarray,
    result: dict[str, object],
    timings: dict[str, float],
) -> dict[str, object]:
    final_valid_mask = result["final_valid_mask"]
    summary: dict[str, object] = {
        "input_path": str(input_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "label_space": label_space,
        "inference_mode": result["mode"],
        "cube_edge": int(result["cube_edge"]),
        "device": device,
        "total_input_points": int(total_input_points),
        "processed_points": int(processed_points),
        "kept_index_count": int(kept_index_count),
        "valid_points_after_fusion": int(np.count_nonzero(final_valid_mask)),
        "invalid_points_after_fusion": int(np.count_nonzero(~final_valid_mask)),
        "outputs": {
            "prediction_ply": str(output_ply_path.resolve()),
            "prediction_npz": str(output_npz_path.resolve()) if output_npz_path else None,
            "fusion_debug_json": str(output_debug_path.resolve()) if output_debug_path else None,
        },
        "input_bounds": {
            "bbox_min_xyz": points.min(axis=0).round(6).tolist(),
            "bbox_max_xyz": points.max(axis=0).round(6).tolist(),
            "centroid_xyz": points.mean(axis=0).round(6).tolist(),
        },
        "class_histogram": class_histogram(result["point_label_ids"], label_space),
        "confidence": {
            "mean": float(result["confidence"][final_valid_mask].mean()) if np.any(final_valid_mask) else 0.0,
            "min": float(result["confidence"][final_valid_mask].min()) if np.any(final_valid_mask) else 0.0,
            "max": float(result["confidence"][final_valid_mask].max()) if np.any(final_valid_mask) else 0.0,
        },
        "fusion": {
            "num_windows": int(len(result.get("window_records", []))),
            "avg_votes_per_point": float(result["vote_count_total"].mean()) if len(points) else 0.0,
            "min_votes_per_point": int(result["vote_count_total"].min()) if len(points) else 0,
            "max_votes_per_point": int(result["vote_count_total"].max()) if len(points) else 0,
            "avg_local_votes_per_point": float(result["local_vote_count"].mean()) if len(points) else 0.0,
            "uncovered_points_local_only": int(np.count_nonzero(result["local_vote_count"] == 0)),
            "seam_risk_points": int(np.count_nonzero(result["seam_risk_mask"])),
            "global_weight": float(result.get("global_prior_weight", 0.0)),
            "halo_weight": float(result.get("halo_weight", 0.0)),
        },
        "timings_seconds": timings,
    }
    if result["mode"] == "single_cube":
        summary["global_pass"] = {
            "cube_edge": int(result["cube_edge"]),
            "valid_points_after_voxel_bounds": int(np.count_nonzero(result["valid_mask"])),
            "invalid_points_after_voxel_bounds": int(np.count_nonzero(~result["valid_mask"])),
            "occupied_voxels": int(result["occupied_voxels"]),
            "scale_abs_max": float(result["scale"]),
        }
    else:
        global_pass = result["global_pass"]
        summary["global_pass"] = {
            "cube_edge": int(result["global_cube_edge"]),
            "valid_points_after_voxel_bounds": int(np.count_nonzero(global_pass["valid_mask"])),
            "invalid_points_after_voxel_bounds": int(np.count_nonzero(~global_pass["valid_mask"])),
            "occupied_voxels": int(global_pass["occupied_voxels"]),
            "scale_abs_max": float(global_pass["scale"]),
        }
        summary["windowing"] = {**result["window_params"], "truncated_windows": bool(result["truncated_windows"])}
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print(f"Mode: {summary['inference_mode']}")
    print(f"Valid points: {summary['valid_points_after_fusion']:,}")
    print(f"Invalid points: {summary['invalid_points_after_fusion']:,}")
    if summary["inference_mode"] == "global_local_fusion":
        print(f"Windows processed: {summary['fusion']['num_windows']:,}")
        print(f"Average votes per point: {summary['fusion']['avg_votes_per_point']:.2f}")
        print(f"Seam-risk points: {summary['fusion']['seam_risk_points']:,}")
    print("Predicted class counts:")
    for item in summary["class_histogram"]:
        print(f"  - {item['class_name']}: {item['count']:,}")
    print("Timings:")
    for key, value in summary["timings_seconds"].items():
        print(f"  - {key}: {value:.2f}s")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    device = resolve_device(args.device)
    weights_path = resolve_weights(args.label_space, args.weights)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_ply_path = output_dir / f"{stem}_predicted.ply"
    output_json_path = output_dir / f"{stem}_summary.json"
    output_npz_path = output_dir / f"{stem}_predicted.npz" if args.save_npz else None
    output_debug_path = output_dir / f"{stem}_fusion_debug.json" if args.mode == "global_local_fusion" and args.save_fusion_debug else None

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    raw_points = load_points(input_path)
    timings["load_points"] = time.perf_counter() - t0
    points, keep_idx = maybe_downsample(raw_points, args.max_points, args.seed)

    t1 = time.perf_counter()
    model = load_model(weights_path, args.label_space, device)
    timings["load_model"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    if args.mode == "single_cube":
        result = run_scene_pass(model, points, args.cube_edge, device)
        result.update(
            {
                "mode": "single_cube",
                "vote_count_total": result["valid_mask"].astype(np.int16),
                "local_vote_count": np.zeros(len(points), dtype=np.int16),
                "weight_sum_total": result["valid_mask"].astype(np.float32),
                "local_weight_sum": np.zeros(len(points), dtype=np.float32),
                "final_valid_mask": result["valid_mask"],
                "seam_risk_mask": np.zeros(len(points), dtype=bool),
                "window_records": [],
                "global_prior_weight": 0.0,
                "halo_weight": 0.0,
                "local_top_label_weight_sum": np.zeros((len(points), result["point_logits"].shape[1]), dtype=np.float32),
            }
        )
    else:
        result = fuse_global_local(model, points, args, device)
    timings["model_inference"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    write_prediction_ply(
        output_ply_path,
        points,
        result["point_channel_ids"],
        result["point_label_ids"],
        args.label_space,
        result["confidence"],
        result["vote_count_total"],
    )
    if output_npz_path is not None:
        np.savez_compressed(
            output_npz_path,
            points=points,
            source_indices=keep_idx,
            kept_indices=keep_idx,
            point_channel_ids=result["point_channel_ids"],
            point_label_ids=result["point_label_ids"],
            confidence=result["confidence"],
            vote_count_total=result["vote_count_total"],
            local_vote_count=result["local_vote_count"],
            weight_sum_total=result["weight_sum_total"],
            local_weight_sum=result["local_weight_sum"],
            final_valid_mask=result["final_valid_mask"],
            seam_risk_mask=result["seam_risk_mask"],
            point_logits=result["point_logits"],
        )
    timings["write_outputs"] = time.perf_counter() - t3

    summary = build_summary(
        input_path,
        weights_path,
        output_ply_path,
        output_npz_path,
        output_debug_path,
        args.label_space,
        device,
        len(raw_points),
        len(points),
        len(keep_idx),
        points,
        result,
        timings,
    )
    output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if output_debug_path is not None:
        debug_payload = {
            "input_path": str(input_path.resolve()),
            "summary_snapshot": summary,
            "window_records": result["window_records"],
        }
        output_debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Weights: {weights_path}")
    print(f"Output PLY: {output_ply_path}")
    print(f"Summary JSON: {output_json_path}")
    if output_npz_path is not None:
        print(f"Output NPZ: {output_npz_path}")
    if output_debug_path is not None:
        print(f"Fusion Debug JSON: {output_debug_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
