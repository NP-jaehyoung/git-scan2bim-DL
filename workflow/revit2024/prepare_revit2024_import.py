"""Validate a Scan2BIM JSON file and stage it for Revit 2024 import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORTED_CATEGORIES = {"wall", "floor", "ceiling", "roof", "column", "beam", "door", "window", "stair"}
SUPPORTED_UNITS = {"meters", "millimeters", "feet", "inches"}

SESSION_PATH = Path(__file__).resolve().parent / "session" / "active_import.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and stage a BIM candidate JSON for Revit 2024.")
    parser.add_argument("--input", required=True, help="Input BIM candidate JSON path.")
    parser.add_argument("--units", choices=sorted(SUPPORTED_UNITS), default="meters", help="Units used by the JSON geometry.")
    parser.add_argument("--mode", choices=["hybrid", "directshape"], default="hybrid", help="Preferred Revit import mode.")
    parser.add_argument("--write-session", action="store_true", help="Write workflow/revit2024/session/active_import.json.")
    return parser.parse_args()


def require_keys(obj: dict, keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(f"{context} is missing required keys: {', '.join(missing)}")


def validate_element(element: dict) -> None:
    require_keys(element, ["id", "category", "geometry", "dimensions", "revit_params"], f"element {element.get('id', '<unknown>')}")
    category = element["category"]
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")

    geometry = element["geometry"]
    if category == "wall":
        require_keys(geometry, ["base_line"], f"{element['id']}.geometry")
    elif category in {"floor", "ceiling", "roof", "door", "window", "stair"}:
        require_keys(geometry, ["profile"], f"{element['id']}.geometry")
    elif category == "column":
        require_keys(geometry, ["axis"], f"{element['id']}.geometry")
    elif category == "beam":
        require_keys(geometry, ["center_line"], f"{element['id']}.geometry")


def validate_payload(payload: dict) -> dict[str, object]:
    require_keys(payload, ["schema_version", "summary", "elements"], "payload")
    elements = payload["elements"]
    if not isinstance(elements, list) or not elements:
        raise ValueError("payload.elements must be a non-empty list")

    category_counts: dict[str, int] = {}
    for element in elements:
        if not isinstance(element, dict):
            raise ValueError("Every element must be an object")
        validate_element(element)
        category = element["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "schema_version": payload["schema_version"],
        "element_count": len(elements),
        "category_counts": category_counts,
    }


def write_session(input_path: Path, units: str, mode: str, stats: dict[str, object]) -> Path:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    session_payload = {
        "json_path": str(input_path.resolve()),
        "input_units": units,
        "preferred_mode": mode,
        "summary": stats,
    }
    SESSION_PATH.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    return SESSION_PATH


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    stats = validate_payload(payload)

    print(f"Validated JSON: {input_path}")
    print(f"Schema version: {stats['schema_version']}")
    print(f"Element count: {stats['element_count']}")
    print("Category counts:")
    for category, count in sorted(stats["category_counts"].items()):
        print(f"  - {category}: {count}")
    print(f"Units for Revit import: {args.units}")
    print(f"Preferred mode: {args.mode}")

    if args.write_session:
        session_path = write_session(input_path, args.units, args.mode, stats)
        print(f"Staged session: {session_path}")


if __name__ == "__main__":
    main()
