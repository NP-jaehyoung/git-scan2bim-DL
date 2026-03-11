"""Run BIMNet inference on a local point cloud and export labeled results.

This script mirrors the repo's current training-time preprocessing:
- center the full cloud
- scale it into [-1, 1]
- voxelize it into a dense occupancy cube
- run BIMNet
- map predicted voxel classes back onto the original input points

It is the first practical bridge from raw point clouds toward Scan-to-BIM.
The current output is still semantic labeling, not Revit geometry.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from model.bimnet import BIMNet


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
        # Match the repo's current S3DIS visualization palette.
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
        "class_names": [
            "Beams",
            "Columns",
            "Doors",
            "Floors",
            "Roofs",
            "Stairs",
            "Walls",
            "Windows",
        ],
        # Match the repo's current PCS visualization palette.
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
    parser = argparse.ArgumentParser(
        description="Infer semantic labels for a local point cloud with BIMNet."
    )
    parser.add_argument("--input", required=True, help="Input point cloud path (.pcd, .ply, .txt, .csv).")
    parser.add_argument(
        "--label-space",
        choices=sorted(LABEL_SPACES.keys()),
        default="s3dis",
        help="Class set and default checkpoint to use.",
    )
    parser.add_argument(
        "--weights",
        help="Checkpoint path. Defaults to a workspace checkpoint for the selected label space.",
    )
    parser.add_argument(
        "--cube-edge",
        type=int,
        default=96,
        help="Voxel cube resolution used for occupancy inference.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory where prediction artifacts will be written.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional random downsampling limit before voxelization. 0 keeps all points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for optional point downsampling.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save compressed per-point arrays for downstream processing.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment.")
    return device_arg


def resolve_weights(script_dir: Path, label_space: str, weights_arg: str | None) -> Path:
    if weights_arg:
        weights_path = Path(weights_arg)
    else:
        weights_path = script_dir / LABEL_SPACES[label_space]["default_weights"]

    if not weights_path.is_absolute():
        weights_path = (script_dir / weights_path).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}\n"
            f"Provide --weights explicitly or add the checkpoint for '{label_space}'."
        )
    return weights_path


def infer_num_model_classes(state_dict: dict[str, torch.Tensor]) -> int:
    key = "out.cx.weight"
    if key not in state_dict:
        raise KeyError(f"Could not infer class count because '{key}' is missing in the checkpoint.")
    return int(state_dict[key].shape[0])


def load_model(weights_path: Path, label_space: str, device: str) -> BIMNet:
    state_dict = torch.load(weights_path, map_location=device)
    num_model_classes = infer_num_model_classes(state_dict)
    expected = LABEL_SPACES[label_space]["num_model_classes"]
    if num_model_classes != expected:
        raise ValueError(
            f"Checkpoint class count ({num_model_classes}) does not match "
            f"label space '{label_space}' ({expected})."
        )

    model = BIMNet(num_classes=num_model_classes).to(device)
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
    data = PlyData.read(path)
    vertex = data["vertex"]
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    return xyz


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
    fields = header["FIELDS"]
    sizes = header["SIZE"]
    types = header["TYPE"]
    counts = header["COUNT"]

    dtype_fields = []
    for field_name, size, type_code, count in zip(fields, sizes, types, counts):
        if type_code == "F":
            base = {4: np.float32, 8: np.float64}.get(size)
        elif type_code == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(size)
        elif type_code == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
        else:
            base = None

        if base is None:
            raise ValueError(
                f"Unsupported PCD field type/size combination: field={field_name}, type={type_code}, size={size}"
            )

        if count == 1:
            dtype_fields.append((field_name, base))
        else:
            dtype_fields.append((field_name, base, (count,)))

    return np.dtype(dtype_fields)


def load_pcd_points(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = parse_pcd_header(handle)
        data_kind = header["DATA"]

        if data_kind == "ascii":
            rows = np.loadtxt(handle, ndmin=2)
            fields = header["FIELDS"]
            x_idx = fields.index("x")
            y_idx = fields.index("y")
            z_idx = fields.index("z")
            xyz = rows[:, [x_idx, y_idx, z_idx]]
            return np.asarray(xyz, dtype=np.float32)

        if data_kind == "binary":
            dtype = pcd_numpy_dtype(header)
            payload = handle.read()
            points = np.frombuffer(payload, dtype=dtype, count=header["POINTS"])
            xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32, copy=False)
            return xyz

        raise NotImplementedError(
            f"PCD DATA '{data_kind}' is not supported yet. "
            "This script currently supports ascii and binary PCD files."
        )


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
        return points, np.arange(len(points))

    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[keep_idx], keep_idx


def normalize_to_voxel_grid(points: np.ndarray, cube_edge: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.abs(centered).max())
    if scale == 0:
        raise ValueError("The point cloud has zero extent and cannot be normalized.")

    voxel_coords = np.round((centered / scale + 1.0) * (cube_edge // 2)).astype(np.int32)
    valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < cube_edge), axis=1)
    return centered, voxel_coords, valid_mask, scale


def build_occupancy_grid(voxel_coords: np.ndarray, valid_mask: np.ndarray, cube_edge: int) -> np.ndarray:
    geom = np.zeros((cube_edge, cube_edge, cube_edge), dtype=np.float32)
    valid_coords = voxel_coords[valid_mask]
    if len(valid_coords) > 0:
        geom[tuple(valid_coords.T)] = 1.0
    return geom


def run_inference(model: BIMNet, geom: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(geom).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    return pred


def map_predictions_to_points(
    pred_channels: np.ndarray,
    voxel_coords: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_channel_ids = np.full(len(voxel_coords), -1, dtype=np.int16)
    point_label_ids = np.zeros(len(voxel_coords), dtype=np.int16)

    if np.any(valid_mask):
        valid_channels = pred_channels[tuple(voxel_coords[valid_mask].T)]
        point_channel_ids[valid_mask] = valid_channels
        point_label_ids[valid_mask] = valid_channels + 1

    return point_channel_ids, point_label_ids


def class_histogram(label_ids: np.ndarray, label_space: str) -> list[dict[str, int | str]]:
    class_names = LABEL_SPACES[label_space]["class_names"]
    histogram = []
    for class_id, class_name in enumerate(class_names, start=1):
        count = int(np.count_nonzero(label_ids == class_id))
        if count:
            histogram.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "count": count,
                }
            )
    histogram.sort(key=lambda item: item["count"], reverse=True)
    return histogram


def write_prediction_ply(
    out_path: Path,
    points: np.ndarray,
    point_channel_ids: np.ndarray,
    point_label_ids: np.ndarray,
    label_space: str,
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
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    vertices["class_id"] = point_label_ids.astype(np.int32)

    ply = PlyData([PlyElement.describe(vertices, "vertex")], text=False)
    ply.write(out_path)


def write_summary_json(
    out_path: Path,
    *,
    input_path: Path,
    weights_path: Path,
    output_ply_path: Path,
    output_npz_path: Path | None,
    label_space: str,
    cube_edge: int,
    device: str,
    total_input_points: int,
    processed_points: int,
    sampled_points: int,
    kept_index_count: int,
    valid_mask: np.ndarray,
    voxel_coords: np.ndarray,
    scale: float,
    point_label_ids: np.ndarray,
    input_points: np.ndarray,
    timings: dict[str, float],
) -> None:
    occupied_voxels = 0
    if np.any(valid_mask):
        occupied_voxels = int(np.unique(voxel_coords[valid_mask], axis=0).shape[0])

    summary = {
        "input_path": str(input_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "label_space": label_space,
        "cube_edge": cube_edge,
        "device": device,
        "total_input_points": int(total_input_points),
        "processed_points": int(processed_points),
        "sampled_points": int(sampled_points),
        "kept_index_count": int(kept_index_count),
        "valid_points_after_voxel_bounds": int(np.count_nonzero(valid_mask)),
        "invalid_points_after_voxel_bounds": int(np.count_nonzero(~valid_mask)),
        "occupied_voxels": occupied_voxels,
        "normalization": {
            "input_centroid_xyz": input_points.mean(axis=0).tolist(),
            "scale_abs_max": scale,
        },
        "outputs": {
            "prediction_ply": str(output_ply_path.resolve()),
            "prediction_npz": str(output_npz_path.resolve()) if output_npz_path else None,
        },
        "class_histogram": class_histogram(point_label_ids, label_space),
        "timings_seconds": timings,
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(point_label_ids: np.ndarray, label_space: str, valid_mask: np.ndarray, timings: dict[str, float]) -> None:
    print(f"Valid points: {int(np.count_nonzero(valid_mask)):,}")
    print(f"Invalid points: {int(np.count_nonzero(~valid_mask)):,}")
    print("Predicted class counts:")
    for item in class_histogram(point_label_ids, label_space):
        print(f"  - {item['class_name']}: {item['count']:,}")
    print("Timings:")
    for key, value in timings.items():
        print(f"  - {key}: {value:.2f}s")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    device = resolve_device(args.device)
    weights_path = resolve_weights(script_dir, args.label_space, args.weights)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    output_ply_path = output_dir / f"{stem}_predicted.ply"
    output_json_path = output_dir / f"{stem}_summary.json"
    output_npz_path = output_dir / f"{stem}_predicted.npz" if args.save_npz else None

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    raw_points = load_points(input_path)
    timings["load_points"] = time.perf_counter() - t0

    total_input_points = len(raw_points)
    points, keep_idx = maybe_downsample(raw_points, args.max_points, args.seed)
    sampled_points = len(points)

    t1 = time.perf_counter()
    _, voxel_coords, valid_mask, scale = normalize_to_voxel_grid(points, args.cube_edge)
    geom = build_occupancy_grid(voxel_coords, valid_mask, args.cube_edge)
    timings["preprocess"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    model = load_model(weights_path, args.label_space, device)
    pred_channels = run_inference(model, geom, device)
    point_channel_ids, point_label_ids = map_predictions_to_points(pred_channels, voxel_coords, valid_mask)
    timings["model_inference"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    write_prediction_ply(output_ply_path, points, point_channel_ids, point_label_ids, args.label_space)
    if output_npz_path is not None:
        np.savez_compressed(
            output_npz_path,
            points=points,
            kept_indices=keep_idx,
            voxel_coords=voxel_coords,
            valid_mask=valid_mask,
            point_channel_ids=point_channel_ids,
            point_label_ids=point_label_ids,
        )
    timings["write_outputs"] = time.perf_counter() - t3

    write_summary_json(
        output_json_path,
        input_path=input_path,
        weights_path=weights_path,
        output_ply_path=output_ply_path,
        output_npz_path=output_npz_path,
        label_space=args.label_space,
        cube_edge=args.cube_edge,
        device=device,
        total_input_points=total_input_points,
        processed_points=len(points),
        sampled_points=sampled_points,
        kept_index_count=len(keep_idx),
        valid_mask=valid_mask,
        voxel_coords=voxel_coords,
        scale=scale,
        point_label_ids=point_label_ids,
        input_points=points,
        timings=timings,
    )

    print(f"Input: {input_path}")
    print(f"Weights: {weights_path}")
    print(f"Output PLY: {output_ply_path}")
    print(f"Summary JSON: {output_json_path}")
    if output_npz_path is not None:
        print(f"Output NPZ: {output_npz_path}")
    print_summary(point_label_ids, args.label_space, valid_mask, timings)


if __name__ == "__main__":
    main()
