"""Shared utilities for Scan2BIM visual debugging tools."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData

LABEL_SPACE_SPECS = {
    "s3dis": {
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
    "pcs": {
        1: "beam",
        2: "column",
        3: "door",
        4: "floor",
        5: "roof",
        6: "stair",
        7: "wall",
        8: "window",
    },
}

CANONICAL_COLORS = {
    "ceiling": (128, 64, 128),
    "floor": (244, 35, 232),
    "wall": (220, 20, 60),
    "beam": (102, 102, 156),
    "column": (190, 153, 153),
    "window": (0, 191, 255),
    "door": (250, 170, 30),
    "table": (220, 220, 0),
    "chair": (107, 142, 35),
    "sofa": (152, 251, 152),
    "bookcase": (70, 130, 180),
    "board": (255, 255, 255),
    "clutter": (110, 110, 110),
    "roof": (255, 255, 0),
    "stair": (0, 0, 255),
    "unknown": (180, 180, 180),
    "correct": (170, 170, 170),
    "incorrect": (255, 70, 70),
}

LABEL_ALIASES = {
    "beams": "beam",
    "beam": "beam",
    "columns": "column",
    "column": "column",
    "doors": "door",
    "door": "door",
    "floors": "floor",
    "floor": "floor",
    "roofs": "roof",
    "roof": "roof",
    "stairs": "stair",
    "stair": "stair",
    "walls": "wall",
    "wall": "wall",
    "windows": "window",
    "window": "window",
    "ceilings": "ceiling",
    "ceiling": "ceiling",
    "tables": "table",
    "table": "table",
    "chairs": "chair",
    "chair": "chair",
    "sofas": "sofa",
    "sofa": "sofa",
    "bookcases": "bookcase",
    "bookcase": "bookcase",
    "boards": "board",
    "board": "board",
    "clutter": "clutter",
}


def canonicalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ")
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    if normalized.endswith("s") and normalized[:-1] in LABEL_ALIASES:
        return LABEL_ALIASES[normalized[:-1]]
    return normalized


def infer_label_space_from_ids(class_ids: np.ndarray) -> str:
    positive = class_ids[class_ids > 0]
    if positive.size == 0:
        raise ValueError("No positive class ids were found.")
    max_label = int(positive.max())
    if max_label <= 8:
        return "pcs"
    if max_label <= 13:
        return "s3dis"
    raise ValueError("Could not infer label space from class ids up to {}.".format(max_label))


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((len(labels), 3), dtype=np.uint8)
    for idx, label in enumerate(labels):
        rgb[idx] = np.asarray(CANONICAL_COLORS.get(str(label), CANONICAL_COLORS["unknown"]), dtype=np.uint8)
    return rgb


def first_non_comment_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def detect_text_delimiter(path: Path) -> str | None:
    first_line = first_non_comment_line(path)
    if "," in first_line:
        return ","
    return None


def load_ground_truth_txt(path: Path) -> dict[str, np.ndarray | list[str] | str]:
    delimiter = detect_text_delimiter(path)
    points: list[list[float]] = []
    raw_labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        if delimiter == ",":
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                try:
                    xyz = [float(row[0]), float(row[1]), float(row[2])]
                except ValueError:
                    continue
                points.append(xyz)
                raw_labels.append(row[3].strip())
        else:
            for line in handle:
                row = line.strip().split()
                if len(row) < 4:
                    continue
                try:
                    xyz = [float(row[0]), float(row[1]), float(row[2])]
                except ValueError:
                    continue
                points.append(xyz)
                raw_labels.append(row[3].strip())
    if not points:
        raise ValueError("No GT point rows were parsed from {}.".format(path))
    canonical = np.asarray([canonicalize_label(label) for label in raw_labels], dtype=object)
    return {
        "points": np.asarray(points, dtype=np.float32),
        "raw_labels": raw_labels,
        "labels": canonical,
        "label_space": "pcs",
    }


def load_ground_truth_ply(path: Path, label_space: str = "auto") -> dict[str, np.ndarray | list[str] | str]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    property_names = {prop.name for prop in vertex.properties}
    if "class_id" not in property_names and "class" not in property_names:
        raise ValueError("Ground-truth PLY is missing 'class_id' or 'class'.")

    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    if "class_id" in property_names:
        class_ids = np.asarray(vertex["class_id"]).astype(np.int32, copy=False)
    else:
        class_ids = np.asarray(vertex["class"]).astype(np.int32, copy=False)

    resolved_space = infer_label_space_from_ids(class_ids) if label_space == "auto" else label_space
    id_to_name = LABEL_SPACE_SPECS[resolved_space]
    raw_labels = [id_to_name.get(int(class_id), "unknown") for class_id in class_ids]
    labels = np.asarray([canonicalize_label(label) for label in raw_labels], dtype=object)
    return {
        "points": points,
        "raw_labels": raw_labels,
        "labels": labels,
        "label_space": resolved_space,
    }


def load_ground_truth(path: Path, label_space: str = "auto") -> dict[str, np.ndarray | list[str] | str]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return load_ground_truth_ply(path, label_space=label_space)
    if suffix in {".txt", ".csv"}:
        return load_ground_truth_txt(path)
    raise ValueError("Unsupported ground-truth format: {}".format(path.suffix))


def load_predicted_ply(path: Path, label_space: str = "auto") -> dict[str, np.ndarray | str]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    property_names = {prop.name for prop in vertex.properties}
    if "class_id" not in property_names and "class" not in property_names:
        raise ValueError("Prediction PLY is missing 'class_id' or 'class'.")

    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    if "class_id" in property_names:
        class_ids = np.asarray(vertex["class_id"]).astype(np.int32, copy=False)
        resolved_space = infer_label_space_from_ids(class_ids) if label_space == "auto" else label_space
        id_to_name = LABEL_SPACE_SPECS[resolved_space]
        labels = np.asarray([canonicalize_label(id_to_name.get(int(class_id), "unknown")) for class_id in class_ids], dtype=object)
    else:
        resolved_space = label_space if label_space != "auto" else "unknown"
        labels = np.asarray([canonicalize_label(str(value)) for value in vertex["class"]], dtype=object)
        class_ids = np.zeros(len(labels), dtype=np.int32)

    confidence = np.asarray(vertex["confidence"]).astype(np.float32, copy=False) if "confidence" in property_names else None
    vote_count = np.asarray(vertex["vote_count"]).astype(np.int32, copy=False) if "vote_count" in property_names else None

    return {
        "points": points,
        "class_ids": class_ids,
        "labels": labels,
        "label_space": resolved_space,
        "confidence": confidence,
        "vote_count": vote_count,
    }


def load_bim_candidates(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        raise ValueError("BIM candidate JSON is missing a valid 'elements' list.")
    return payload


def build_coordinate_lookup(points: np.ndarray, decimals: int) -> dict[tuple[float, float, float], list[int]]:
    rounded = np.round(points, decimals=decimals)
    lookup: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    for idx, coord in enumerate(rounded):
        lookup[(float(coord[0]), float(coord[1]), float(coord[2]))].append(idx)
    return lookup


def align_points_by_coordinate(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    decimals: int = 5,
) -> dict[str, np.ndarray | int | bool]:
    if reference_points.shape == candidate_points.shape and np.allclose(reference_points, candidate_points, atol=10 ** (-decimals)):
        indices = np.arange(len(reference_points), dtype=np.int32)
        return {
            "matched_reference": indices,
            "matched_candidate": indices,
            "matched_count": int(len(indices)),
            "unmatched_reference_count": 0,
            "unmatched_candidate_count": 0,
            "used_direct_order": True,
        }

    lookup = build_coordinate_lookup(candidate_points, decimals)
    matched_reference: list[int] = []
    matched_candidate: list[int] = []
    rounded_reference = np.round(reference_points, decimals=decimals)

    for ref_idx, coord in enumerate(rounded_reference):
        key = (float(coord[0]), float(coord[1]), float(coord[2]))
        bucket = lookup.get(key)
        if not bucket:
            continue
        matched_reference.append(ref_idx)
        matched_candidate.append(bucket.pop())

    return {
        "matched_reference": np.asarray(matched_reference, dtype=np.int32),
        "matched_candidate": np.asarray(matched_candidate, dtype=np.int32),
        "matched_count": int(len(matched_reference)),
        "unmatched_reference_count": int(len(reference_points) - len(matched_reference)),
        "unmatched_candidate_count": int(len(candidate_points) - len(matched_candidate)),
        "used_direct_order": False,
    }


def compute_semantic_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    if len(gt_labels) != len(pred_labels):
        raise ValueError("GT and prediction label arrays must have the same length.")

    classes = sorted(set(gt_labels.tolist()) | set(pred_labels.tolist()))
    accuracy = float(np.mean(gt_labels == pred_labels)) if len(gt_labels) else 0.0
    per_class = []

    for class_name in classes:
        gt_mask = gt_labels == class_name
        pred_mask = pred_labels == class_name
        intersection = int(np.count_nonzero(gt_mask & pred_mask))
        union = int(np.count_nonzero(gt_mask | pred_mask))
        gt_count = int(np.count_nonzero(gt_mask))
        pred_count = int(np.count_nonzero(pred_mask))
        iou = float(intersection / union) if union else 0.0
        precision = float(intersection / pred_count) if pred_count else 0.0
        recall = float(intersection / gt_count) if gt_count else 0.0
        per_class.append(
            {
                "class_name": class_name,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "intersection": intersection,
                "union": union,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            }
        )

    per_class.sort(key=lambda item: (-item["iou"], item["class_name"]))
    mean_iou = float(np.mean([item["iou"] for item in per_class])) if per_class else 0.0
    return {
        "point_accuracy": accuracy,
        "mean_iou": mean_iou,
        "class_count": len(classes),
        "per_class": per_class,
    }


def maybe_downsample_indices(count: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_points, replace=False).astype(np.int32))


def format_metrics_text(metrics: dict | None, alignment: dict | None) -> str:
    if metrics is None or alignment is None:
        return "alignment unavailable"
    return (
        "matched: {matched}\n"
        "accuracy: {accuracy:.3f}\n"
        "mIoU: {miou:.3f}\n"
        "unmatched gt: {ugt}\n"
        "unmatched pred: {upred}"
    ).format(
        matched=alignment["matched_count"],
        accuracy=metrics["point_accuracy"],
        miou=metrics["mean_iou"],
        ugt=alignment["unmatched_reference_count"],
        upred=alignment["unmatched_candidate_count"],
    )
