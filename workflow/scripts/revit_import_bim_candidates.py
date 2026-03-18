"""Import BIM candidate JSON into Revit as DirectShape geometry.

Target runtime:
- pyRevit CPython 3

Workflow:
1. Run `workflow/scripts/infer_pointcloud.py`
2. Run `workflow/scripts/export_bim_candidates.py`
3. Open this script inside pyRevit CPython 3 and run it

This importer is intentionally type-light: it creates DirectShape geometry in
appropriate Revit categories instead of trying to infer native system-family
types for every element.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


JSON_PATH = r""
INPUT_UNITS = "meters"  # meters | millimeters | feet | inches

UNIT_TO_FEET = {
    "meters": 3.280839895013123,
    "millimeters": 0.003280839895013123,
    "feet": 1.0,
    "inches": 1.0 / 12.0,
}



def load_revit_api_globals() -> None:
    import clr

    clr.AddReference("RevitAPI")
    from Autodesk.Revit.DB import (
        BuiltInCategory,
        BuiltInParameter,
        CurveLoop,
        DirectShape,
        ElementId,
        GeometryCreationUtilities,
        GeometryObject,
        Line,
        Transaction,
        XYZ,
    )
    from System.Collections.Generic import List

    globals().update(
        {
            "BuiltInCategory": BuiltInCategory,
            "BuiltInParameter": BuiltInParameter,
            "CurveLoop": CurveLoop,
            "DirectShape": DirectShape,
            "ElementId": ElementId,
            "GeometryCreationUtilities": GeometryCreationUtilities,
            "GeometryObject": GeometryObject,
            "Line": Line,
            "List": List,
            "Transaction": Transaction,
            "XYZ": XYZ,
        }
    )



def get_doc():
    try:
        return __revit__.ActiveUIDocument.Document
    except Exception:
        pass

    try:
        from RevitServices.Persistence import DocumentManager

        return DocumentManager.Instance.CurrentDBDocument
    except Exception:
        pass

    raise RuntimeError("Could not find an active Revit document. Run this inside pyRevit or Dynamo/Revit.")



def resolve_json_path() -> Path:
    if JSON_PATH:
        path = Path(JSON_PATH)
        if path.exists():
            return path
        raise FileNotFoundError("JSON_PATH does not exist: {}".format(path))

    try:
        import clr

        clr.AddReference("System.Windows.Forms")
        from System.Windows.Forms import DialogResult, OpenFileDialog

        dialog = OpenFileDialog()
        dialog.Filter = "BIM candidate JSON (*.json)|*.json"
        dialog.Title = "Select BIM candidate JSON"
        if dialog.ShowDialog() == DialogResult.OK:
            return Path(dialog.FileName)
    except Exception:
        pass

    raise RuntimeError("Set JSON_PATH at the top of the script or choose a file through the dialog.")



def to_internal_length(value: float) -> float:
    factor = UNIT_TO_FEET.get(INPUT_UNITS)
    if factor is None:
        raise ValueError("Unsupported INPUT_UNITS: {}".format(INPUT_UNITS))
    return float(value) * factor



def to_xyz(coords) -> object:
    return XYZ(
        to_internal_length(coords[0]),
        to_internal_length(coords[1]),
        to_internal_length(coords[2]),
    )



def to_xyz_direction(vec) -> object:
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if length == 0:
        raise ValueError("Zero-length direction vector.")
    return XYZ(float(vec[0] / length), float(vec[1] / length), float(vec[2] / length))



def make_curve_loop(points) -> object:
    loop = CurveLoop()
    for start, end in zip(points[:-1], points[1:]):
        loop.Append(Line.CreateBound(to_xyz(start), to_xyz(end)))
    return loop



def make_curve_loop_from_np(points_np) -> object:
    loop = CurveLoop()
    for start, end in zip(points_np[:-1], points_np[1:]):
        loop.Append(Line.CreateBound(to_xyz(start.tolist()), to_xyz(end.tolist())))
    return loop



def create_vertical_extrusion(profile_points, height) -> object:
    loops = List[CurveLoop]()
    loops.Add(make_curve_loop(profile_points))
    direction = XYZ.BasisZ
    extrusion_height = float(height)
    if extrusion_height < 0:
        direction = XYZ.BasisZ.Negate()
        extrusion_height = abs(extrusion_height)
    return GeometryCreationUtilities.CreateExtrusionGeometry(loops, direction, to_internal_length(max(extrusion_height, 0.001)))



def wall_profile_from_base_line(base_line, width) -> list[list[float]]:
    start = base_line["start"]
    end = base_line["end"]
    start_xy = [float(start[0]), float(start[1])]
    end_xy = [float(end[0]), float(end[1])]
    base_z = float(start[2])

    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        raise ValueError("Wall base_line has zero length.")

    ux = dx / length
    uy = dy / length
    nx = -uy
    ny = ux
    half_w = float(width) / 2.0

    p1 = [start_xy[0] - nx * half_w, start_xy[1] - ny * half_w, base_z]
    p2 = [end_xy[0] - nx * half_w, end_xy[1] - ny * half_w, base_z]
    p3 = [end_xy[0] + nx * half_w, end_xy[1] + ny * half_w, base_z]
    p4 = [start_xy[0] + nx * half_w, start_xy[1] + ny * half_w, base_z]
    return [p1, p2, p3, p4, p1]



def oriented_rect_profile(center_xyz, orientation_xy, width, depth, base_z) -> list[list[float]]:
    ux, uy = float(orientation_xy[0]), float(orientation_xy[1])
    norm = math.sqrt(ux * ux + uy * uy)
    if norm == 0:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = ux / norm, uy / norm
    vx, vy = -uy, ux

    half_w = float(width) / 2.0
    half_d = float(depth) / 2.0
    cx = float(center_xyz[0])
    cy = float(center_xyz[1])

    p1 = [cx - ux * half_w - vx * half_d, cy - uy * half_w - vy * half_d, base_z]
    p2 = [cx + ux * half_w - vx * half_d, cy + uy * half_w - vy * half_d, base_z]
    p3 = [cx + ux * half_w + vx * half_d, cy + uy * half_w + vy * half_d, base_z]
    p4 = [cx - ux * half_w + vx * half_d, cy - uy * half_w + vy * half_d, base_z]
    return [p1, p2, p3, p4, p1]



def create_beam_solid(center_line, section_width, section_depth) -> object:
    import numpy as np

    start = np.array(center_line["start"], dtype=float)
    end = np.array(center_line["end"], dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length == 0:
        raise ValueError("Beam center_line has zero length.")

    axis_dir = axis / length
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    side = np.cross(up, axis_dir)
    if np.linalg.norm(side) < 1e-8:
        side = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        side = side / np.linalg.norm(side)
    up_local = np.cross(axis_dir, side)
    up_local = up_local / max(np.linalg.norm(up_local), 1e-8)

    half_w = float(section_width) / 2.0
    half_d = float(section_depth) / 2.0

    p1 = start - side * half_w - up_local * half_d
    p2 = start + side * half_w - up_local * half_d
    p3 = start + side * half_w + up_local * half_d
    p4 = start - side * half_w + up_local * half_d
    profile = np.array([p1, p2, p3, p4, p1], dtype=float)

    loops = List[CurveLoop]()
    loops.Add(make_curve_loop_from_np(profile))
    return GeometryCreationUtilities.CreateExtrusionGeometry(
        loops,
        to_xyz_direction(axis_dir),
        to_internal_length(length),
    )



def category_to_bic(category: str):
    mapping = {
        "wall": BuiltInCategory.OST_Walls,
        "floor": BuiltInCategory.OST_Floors,
        "ceiling": BuiltInCategory.OST_Ceilings,
        "roof": BuiltInCategory.OST_Roofs,
        "column": BuiltInCategory.OST_StructuralColumns,
        "beam": BuiltInCategory.OST_StructuralFraming,
        "door": BuiltInCategory.OST_Doors,
        "window": BuiltInCategory.OST_Windows,
        "stair": BuiltInCategory.OST_Stairs,
    }
    bic = mapping.get(category)
    if bic is None:
        raise ValueError("Unsupported category: {}".format(category))
    return bic



def create_directshape(doc, category: str, solid, element_name: str):
    ds = DirectShape.CreateElement(doc, ElementId(category_to_bic(category)))
    ds.ApplicationId = "Scan2BIM"
    ds.ApplicationDataId = element_name

    shape = List[GeometryObject]()
    shape.Add(solid)
    ds.SetShape(shape)

    comment_param = ds.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
    if comment_param and not comment_param.IsReadOnly:
        comment_param.Set(element_name)

    return ds



def solid_from_element(element: dict):
    category = element["category"]

    if category == "wall":
        base_line = element["geometry"]["base_line"]
        width = float(element["dimensions"].get("thickness") or element["revit_params"].get("width") or 0.2)
        height = float(element["dimensions"].get("height") or element["revit_params"].get("unconnected_height") or 2.5)
        profile = wall_profile_from_base_line(base_line, width)
        return create_vertical_extrusion(profile, height)

    if category in {"floor", "ceiling", "roof"}:
        profile = element["geometry"]["profile"]
        thickness = float(element["dimensions"].get("thickness") or element["revit_params"].get("thickness") or 0.2)
        if category == "ceiling":
            thickness = -abs(thickness)
        return create_vertical_extrusion(profile, thickness)

    if category == "column":
        axis = element["geometry"]["axis"]
        center = axis["start"]
        orientation = element["geometry"].get("orientation_xy", [1.0, 0.0])
        width = float(element["dimensions"].get("width") or element["revit_params"].get("width") or 0.3)
        depth = float(element["dimensions"].get("depth") or element["revit_params"].get("depth") or 0.3)
        height = float(element["dimensions"].get("height") or 0.3)
        profile = oriented_rect_profile(center, orientation, width, depth, float(center[2]))
        return create_vertical_extrusion(profile, height)

    if category == "beam":
        center_line = element["geometry"]["center_line"]
        width = float(element["dimensions"].get("width") or element["revit_params"].get("section_width") or 0.3)
        depth = float(element["dimensions"].get("depth") or element["revit_params"].get("section_depth") or 0.3)
        return create_beam_solid(center_line, width, depth)

    if category in {"door", "window", "stair"}:
        profile = element["geometry"]["profile"]
        height = float(element["dimensions"].get("height") or element["revit_params"].get("height") or 2.0)
        return create_vertical_extrusion(profile, height)

    raise ValueError("Unsupported category: {}".format(category))



def main() -> None:
    load_revit_api_globals()
    doc = get_doc()
    json_path = resolve_json_path()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])
    if not elements:
        raise RuntimeError("No elements were found in {}".format(json_path))

    created = 0
    skipped = []

    transaction = Transaction(doc, "Import Scan2BIM Candidates")
    transaction.Start()
    try:
        for element in elements:
            element_id = element.get("id", "unnamed")
            try:
                solid = solid_from_element(element)
                create_directshape(doc, element["category"], solid, element_id)
                created += 1
            except Exception as exc:
                skipped.append("{}: {}".format(element_id, exc))
        transaction.Commit()
    except Exception:
        transaction.RollBack()
        raise

    print("Imported {} DirectShape elements from {}".format(created, json_path))
    if skipped:
        print("Skipped {} elements:".format(len(skipped)))
        for item in skipped[:20]:
            print("  - {}".format(item))
        if len(skipped) > 20:
            print("  ... and {} more".format(len(skipped) - 20))


if __name__ == "__main__":
    main()
