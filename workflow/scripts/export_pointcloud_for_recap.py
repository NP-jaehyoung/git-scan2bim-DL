"""Convert local point clouds into ReCap-friendly interchange formats.

Current focus:
- write `.pts` for Autodesk ReCap / Revit point-cloud linking

Supported inputs:
- .pcd
- .ply
- .txt
- .csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import BinaryIO

import numpy as np
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a point cloud to a ReCap-friendly format such as .pts.")
    parser.add_argument("--input", required=True, help="Input point cloud path (.pcd, .ply, .txt, .csv).")
    parser.add_argument("--output", required=True, help="Output path, typically ending with .pts.")
    parser.add_argument("--format", choices=["pts", "xyz"], default="pts")
    parser.add_argument("--max-points", type=int, default=0, help="Optional random downsampling limit.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--uniform-color", nargs=3, type=int, metavar=("R", "G", "B"), help="Override all point colors.")
    parser.add_argument("--intensity", type=float, default=0.0, help="Intensity value used for .pts export.")
    return parser.parse_args()


def first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def detect_text_delimiter(path: Path) -> str | None:
    return "," if "," in first_data_line(path) else None


def maybe_downsample(points: np.ndarray, colors: np.ndarray | None, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(points), size=max_points, replace=False))
    sampled_points = points[keep]
    sampled_colors = colors[keep] if colors is not None else None
    return sampled_points, sampled_colors


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
            raise ValueError("Unsupported PCD field: {} {}{}".format(field_name, type_code, size))
        dtype_fields.append((field_name, base) if count == 1 else (field_name, base, (count,)))
    return np.dtype(dtype_fields)


def unpack_packed_rgb(values: np.ndarray) -> np.ndarray:
    packed = np.asarray(values)
    if packed.dtype.kind == "f":
        packed = packed.view(np.uint32)
    else:
        packed = packed.astype(np.uint32, copy=False)
    rgb = np.empty((len(packed), 3), dtype=np.uint8)
    rgb[:, 0] = ((packed >> 16) & 255).astype(np.uint8)
    rgb[:, 1] = ((packed >> 8) & 255).astype(np.uint8)
    rgb[:, 2] = (packed & 255).astype(np.uint8)
    return rgb


def extract_rgb_from_structured(points: np.ndarray, field_names: list[str]) -> np.ndarray | None:
    lower = [name.lower() for name in field_names]
    if all(channel in lower for channel in ("r", "g", "b")):
        return np.column_stack([points["r"], points["g"], points["b"]]).astype(np.uint8, copy=False)
    if "rgb" in lower:
        return unpack_packed_rgb(points["rgb"])
    if "rgba" in lower:
        return unpack_packed_rgb(points["rgba"])
    return None


def load_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as handle:
        header = parse_pcd_header(handle)
        fields = [str(name) for name in header["FIELDS"]]
        if header["DATA"] == "ascii":
            rows = np.loadtxt(handle, ndmin=2)
            xyz = rows[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(np.float32, copy=False)
            rgb = None
            if all(channel in fields for channel in ("r", "g", "b")):
                rgb = rows[:, [fields.index("r"), fields.index("g"), fields.index("b")]].astype(np.uint8)
            elif "rgb" in fields:
                rgb = unpack_packed_rgb(rows[:, fields.index("rgb")])
            elif "rgba" in fields:
                rgb = unpack_packed_rgb(rows[:, fields.index("rgba")])
            return xyz, rgb

        if header["DATA"] == "binary":
            structured = np.frombuffer(handle.read(), dtype=pcd_numpy_dtype(header), count=header["POINTS"])
            xyz = np.column_stack([structured["x"], structured["y"], structured["z"]]).astype(np.float32, copy=False)
            rgb = extract_rgb_from_structured(structured, fields)
            return xyz, rgb

    raise NotImplementedError("Unsupported PCD DATA kind.")


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    vertex = PlyData.read(path)["vertex"]
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    property_names = {prop.name for prop in vertex.properties}
    if {"red", "green", "blue"} <= property_names:
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8, copy=False)
        return points, rgb
    if {"r", "g", "b"} <= property_names:
        rgb = np.column_stack([vertex["r"], vertex["g"], vertex["b"]]).astype(np.uint8, copy=False)
        return points, rgb
    return points, None


def load_text(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    delimiter = detect_text_delimiter(path)
    rows = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    if rows.shape[1] < 3:
        raise ValueError("Text point cloud must contain at least x y z columns.")
    points = rows[:, :3].astype(np.float32, copy=False)
    rgb = None
    if rows.shape[1] >= 6:
        candidate = rows[:, 3:6]
        if np.all((candidate >= 0) & (candidate <= 255)):
            rgb = candidate.astype(np.uint8)
    return points, rgb


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".pcd":
        return load_pcd(path)
    if suffix == ".ply":
        return load_ply(path)
    if suffix in {".txt", ".csv"}:
        return load_text(path)
    raise ValueError("Unsupported input format: {}".format(path.suffix))


def resolve_colors(colors: np.ndarray | None, count: int, uniform_color: tuple[int, int, int] | None) -> np.ndarray:
    if uniform_color is not None:
        rgb = np.empty((count, 3), dtype=np.uint8)
        rgb[:] = np.asarray(uniform_color, dtype=np.uint8)
        return rgb
    if colors is not None:
        return colors.astype(np.uint8, copy=False)
    rgb = np.empty((count, 3), dtype=np.uint8)
    rgb[:] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def write_xyz(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for point, color in zip(points, colors):
            handle.write(
                "{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )


def write_pts(path: Path, points: np.ndarray, colors: np.ndarray, intensity: float) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n".format(len(points)))
        for point, color in zip(points, colors):
            handle.write(
                "{:.6f} {:.6f} {:.6f} {:.3f} {} {} {}\n".format(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(intensity),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError("Input not found: {}".format(input_path))

    points, colors = load_point_cloud(input_path)
    points, colors = maybe_downsample(points, colors, args.max_points, args.seed)
    uniform_color = tuple(args.uniform_color) if args.uniform_color else None
    colors = resolve_colors(colors, len(points), uniform_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "xyz":
        write_xyz(output_path, points, colors)
    else:
        write_pts(output_path, points, colors, args.intensity)

    print("Input: {}".format(input_path))
    print("Output: {}".format(output_path))
    print("Format: {}".format(args.format))
    print("Point count: {:,}".format(len(points)))
    print("Has RGB: {}".format("yes" if colors is not None else "no"))
    print("Next step: import this file into Autodesk ReCap, create .rcp/.rcs, then link the point cloud in Revit.")


if __name__ == "__main__":
    main()
