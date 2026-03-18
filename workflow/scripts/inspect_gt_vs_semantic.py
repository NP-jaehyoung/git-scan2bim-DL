"""Interactive viewer for ground-truth vs semantic prediction debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scan2bim_debug_viewer_common import (
    CANONICAL_COLORS,
    align_points_by_coordinate,
    colorize_labels,
    compute_semantic_metrics,
    format_metrics_text,
    load_ground_truth,
    load_predicted_ply,
    maybe_downsample_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GT labels with semantic prediction in an interactive viewer.")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth path (.txt/.csv with labels or labeled .ply).")
    parser.add_argument("--prediction", required=True, help="Predicted labeled PLY path.")
    parser.add_argument("--label-space", choices=["auto", "s3dis", "pcs"], default="auto")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--alignment-decimals", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--summary-json", help="Optional output path for comparison metrics.")
    parser.add_argument("--summary-only", action="store_true", help="Compute metrics without opening a window.")
    parser.add_argument("--screenshot", help="Optional screenshot path. Implies off-screen rendering.")
    return parser.parse_args()


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista is required for the visual debug viewer.") from exc
    return pv


def add_colored_points(plotter, pv, points: np.ndarray, labels: np.ndarray, point_size: float) -> None:
    cloud = pv.PolyData(points)
    cloud["rgb"] = colorize_labels(labels)
    plotter.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        point_size=point_size,
        render_points_as_spheres=True,
    )


def add_label_legend(plotter, labels: np.ndarray) -> None:
    present = sorted(set(labels.tolist()))
    legend_entries = []
    for label in present:
        legend_entries.append([str(label), CANONICAL_COLORS.get(str(label), CANONICAL_COLORS["unknown"])])
    if legend_entries:
        plotter.add_legend(legend_entries, bcolor=(20, 20, 20), border=True, size=(0.16, 0.28))


def summarize_metrics(metrics: dict) -> str:
    lines = [
        "point accuracy: {:.4f}".format(metrics["point_accuracy"]),
        "mean IoU: {:.4f}".format(metrics["mean_iou"]),
        "per-class IoU:",
    ]
    ranked = sorted(metrics["per_class"], key=lambda item: (item["iou"], item["gt_count"]), reverse=True)
    for item in ranked[:12]:
        lines.append(
            "  {} | IoU {:.3f} | gt {} | pred {} | recall {:.3f}".format(
                item["class_name"],
                item["iou"],
                item["gt_count"],
                item["pred_count"],
                item["recall"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth).resolve()
    pred_path = Path(args.prediction).resolve()
    if not gt_path.exists():
        raise FileNotFoundError("Ground-truth file not found: {}".format(gt_path))
    if not pred_path.exists():
        raise FileNotFoundError("Prediction file not found: {}".format(pred_path))

    gt = load_ground_truth(gt_path, label_space=args.label_space)
    pred = load_predicted_ply(pred_path, label_space=args.label_space)
    alignment = align_points_by_coordinate(gt["points"], pred["points"], decimals=args.alignment_decimals)

    matched_reference = alignment["matched_reference"]
    matched_candidate = alignment["matched_candidate"]
    matched_gt_labels = gt["labels"][matched_reference]
    matched_pred_labels = pred["labels"][matched_candidate]
    metrics = compute_semantic_metrics(matched_gt_labels, matched_pred_labels) if alignment["matched_count"] else None

    summary = {
        "ground_truth_path": str(gt_path),
        "prediction_path": str(pred_path),
        "ground_truth_point_count": int(len(gt["points"])),
        "prediction_point_count": int(len(pred["points"])),
        "alignment": {
            "matched_count": int(alignment["matched_count"]),
            "unmatched_ground_truth_count": int(alignment["unmatched_reference_count"]),
            "unmatched_prediction_count": int(alignment["unmatched_candidate_count"]),
            "used_direct_order": bool(alignment["used_direct_order"]),
        },
        "metrics": metrics,
    }

    print("Ground truth: {}".format(gt_path))
    print("Prediction:   {}".format(pred_path))
    print("Matched points: {}".format(alignment["matched_count"]))
    print("Unmatched GT: {}".format(alignment["unmatched_reference_count"]))
    print("Unmatched prediction: {}".format(alignment["unmatched_candidate_count"]))
    if metrics:
        print(summarize_metrics(metrics))
    else:
        print("No aligned points were found, so semantic metrics were skipped.")

    if args.summary_json:
        output_path = Path(args.summary_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Summary JSON: {}".format(output_path))

    if args.summary_only:
        return

    pv = require_pyvista()
    off_screen = bool(args.screenshot)
    plotter = pv.Plotter(shape=(1, 3), window_size=(2100, 700), off_screen=off_screen)

    gt_sample = maybe_downsample_indices(len(gt["points"]), args.max_points, args.seed)
    pred_sample = maybe_downsample_indices(len(pred["points"]), args.max_points, args.seed)

    plotter.subplot(0, 0)
    add_colored_points(plotter, pv, gt["points"][gt_sample], gt["labels"][gt_sample], args.point_size)
    add_label_legend(plotter, gt["labels"][gt_sample])
    plotter.add_text("Ground Truth", font_size=12)
    plotter.show_grid(color="gray")

    plotter.subplot(0, 1)
    add_colored_points(plotter, pv, pred["points"][pred_sample], pred["labels"][pred_sample], args.point_size)
    plotter.add_text("Semantic Prediction", font_size=12)
    plotter.show_grid(color="gray")

    plotter.subplot(0, 2)
    if alignment["matched_count"]:
        error_mask = matched_gt_labels != matched_pred_labels
        correct_mask = ~error_mask
        matched_points = gt["points"][matched_reference]
        error_sample = maybe_downsample_indices(len(matched_points), args.max_points, args.seed)
        sampled_points = matched_points[error_sample]
        sampled_errors = error_mask[error_sample]
        colors = np.tile(np.asarray(CANONICAL_COLORS["correct"], dtype=np.uint8), (len(sampled_points), 1))
        colors[sampled_errors] = np.asarray(CANONICAL_COLORS["incorrect"], dtype=np.uint8)
        cloud = pv.PolyData(sampled_points)
        cloud["rgb"] = colors
        plotter.add_points(
            cloud,
            scalars="rgb",
            rgb=True,
            point_size=args.point_size,
            render_points_as_spheres=True,
        )
        panel_text = "Agreement / Error\n" + format_metrics_text(metrics, alignment)
        panel_text += "\nerrors: {}".format(int(np.count_nonzero(error_mask)))
        panel_text += "\ncorrect: {}".format(int(np.count_nonzero(correct_mask)))
        plotter.add_text(panel_text, font_size=11)
        plotter.add_legend(
            [
                ["correct", CANONICAL_COLORS["correct"]],
                ["incorrect", CANONICAL_COLORS["incorrect"]],
            ],
            bcolor=(20, 20, 20),
            border=True,
            size=(0.16, 0.2),
        )
    else:
        plotter.add_text("Agreement / Error\nalignment unavailable", font_size=11)
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
