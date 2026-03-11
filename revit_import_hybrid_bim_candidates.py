"""Import BIM candidate JSON into Revit using native elements when possible.

Target runtime:
- pyRevit CPython 3

Strategy:
- wall / floor / ceiling: try native Revit creation first
- roof / column / beam / door / window / stair: DirectShape fallback
- if native creation fails, fall back to DirectShape automatically
"""

from __future__ import annotations

import json
import math
from pathlib import Path


JSON_PATH = r""
INPUT_UNITS = "meters"  # meters | millimeters | feet | inches
CREATE_LEVELS_IF_MISSING = True
LEVEL_TOLERANCE_FEET = 0.5

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
        Ceiling,
        CeilingType,
        CurveLoop,
        DirectShape,
        ElementId,
        FilteredElementCollector,
        Floor,
        FloorType,
        GeometryCreationUtilities,
        GeometryObject,
        Level,
        Line,
        Transaction,
        Wall,
        WallType,
        XYZ,
    )
    from System.Collections.Generic import List

    globals().update(
        {
            "BuiltInCategory": BuiltInCategory,
            "BuiltInParameter": BuiltInParameter,
            "Ceiling": Ceiling,
            "CeilingType": CeilingType,
            "CurveLoop": CurveLoop,
            "DirectShape": DirectShape,
            "ElementId": ElementId,
            "FilteredElementCollector": FilteredElementCollector,
            "Floor": Floor,
            "FloorType": FloorType,
            "GeometryCreationUtilities": GeometryCreationUtilities,
            "GeometryObject": GeometryObject,
            "Level": Level,
            "Line": Line,
            "List": List,
            "Transaction": Transaction,
            "Wall": Wall,
            "WallType": WallType,
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



def normalize3(vec):
    length = math.sqrt(sum(float(v) * float(v) for v in vec))
    if length == 0:
        raise ValueError("Zero-length vector.")
    return [float(v) / length for v in vec]



def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]



def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]



def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]



def scale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]



def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))



def make_curve_loop(points) -> object:
    loop = CurveLoop()
    for start, end in zip(points[:-1], points[1:]):
        loop.Append(Line.CreateBound(to_xyz(start), to_xyz(end)))
    return loop



def set_instance_comment(element, text: str) -> None:
    param = element.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
    if param and not param.IsReadOnly:
        param.Set(text)



def first_element_of_class(doc, cls):
    collector = FilteredElementCollector(doc).OfClass(cls)
    try:
        collector = collector.WhereElementIsElementType()
    except Exception:
        pass
    for element in collector:
        return element
    raise RuntimeError("No element found for class {}".format(cls))



def find_or_create_level(doc, elevation_feet: float):
    levels = list(FilteredElementCollector(doc).OfClass(Level))
    if levels:
        nearest = min(levels, key=lambda lvl: abs(lvl.Elevation - elevation_feet))
        if abs(nearest.Elevation - elevation_feet) <= LEVEL_TOLERANCE_FEET:
            return nearest

    if not CREATE_LEVELS_IF_MISSING:
        if levels:
            return nearest
        raise RuntimeError("No levels found in the model and level creation is disabled.")

    level = Level.Create(doc, elevation_feet)
    try:
        level.Name = "Scan2BIM_{:.3f}ft".format(elevation_feet)
    except Exception:
        pass
    return level



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
    return mapping[category]



def create_directshape(doc, category: str, solid, element_name: str):
    ds = DirectShape.CreateElement(doc, ElementId(category_to_bic(category)))
    ds.ApplicationId = "Scan2BIM"
    ds.ApplicationDataId = element_name
    shape = List[GeometryObject]()
    shape.Add(solid)
    ds.SetShape(shape)
    set_instance_comment(ds, element_name)
    return ds



def create_extrusion(profile_points, height, direction=None):
    loops = List[CurveLoop]()
    loops.Add(make_curve_loop(profile_points))
    direction_xyz = direction or XYZ.BasisZ
    return GeometryCreationUtilities.CreateExtrusionGeometry(
        loops,
        direction_xyz,
        max(to_internal_length(height), 0.001),
    )



def wall_profile_from_base_line(base_line, width):
    start = [float(v) for v in base_line["start"]]
    end = [float(v) for v in base_line["end"]]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        raise ValueError("Wall base line has zero length.")
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    half_w = float(width) / 2.0
    p1 = [start[0] - nx * half_w, start[1] - ny * half_w, start[2]]
    p2 = [end[0] - nx * half_w, end[1] - ny * half_w, start[2]]
    p3 = [end[0] + nx * half_w, end[1] + ny * half_w, start[2]]
    p4 = [start[0] + nx * half_w, start[1] + ny * half_w, start[2]]
    return [p1, p2, p3, p4, p1]



def create_beam_solid(center_line, section_width, section_depth):
    start = [float(v) for v in center_line["start"]]
    end = [float(v) for v in center_line["end"]]
    axis = sub(end, start)
    axis_dir = normalize3(axis)
    world_up = [0.0, 0.0, 1.0]
    side = cross(world_up, axis_dir)
    side_len = math.sqrt(sum(v * v for v in side))
    if side_len < 1e-8:
        side = [1.0, 0.0, 0.0]
    else:
        side = [v / side_len for v in side]
    up_local = normalize3(cross(axis_dir, side))

    half_w = float(section_width) / 2.0
    half_d = float(section_depth) / 2.0
    p1 = add(add(start, scale(side, -half_w)), scale(up_local, -half_d))
    p2 = add(add(start, scale(side, half_w)), scale(up_local, -half_d))
    p3 = add(add(start, scale(side, half_w)), scale(up_local, half_d))
    p4 = add(add(start, scale(side, -half_w)), scale(up_local, half_d))
    profile = [p1, p2, p3, p4, p1]
    direction_xyz = XYZ(axis_dir[0], axis_dir[1], axis_dir[2])
    return create_extrusion(profile, distance(start, end), direction_xyz)



def oriented_rect_profile(center_xyz, orientation_xy, width, depth, base_z):
    ux, uy = float(orientation_xy[0]), float(orientation_xy[1])
    norm = math.sqrt(ux * ux + uy * uy)
    if norm == 0:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = ux / norm, uy / norm
    vx, vy = -uy, ux
    half_w = float(width) / 2.0
    half_d = float(depth) / 2.0
    cx, cy = float(center_xyz[0]), float(center_xyz[1])

    p1 = [cx - ux * half_w - vx * half_d, cy - uy * half_w - vy * half_d, base_z]
    p2 = [cx + ux * half_w - vx * half_d, cy + uy * half_w - vy * half_d, base_z]
    p3 = [cx + ux * half_w + vx * half_d, cy + uy * half_w + vy * half_d, base_z]
    p4 = [cx - ux * half_w + vx * half_d, cy - uy * half_w + vy * half_d, base_z]
    return [p1, p2, p3, p4, p1]



def directshape_solid_from_element(element: dict):
    category = element["category"]

    if category == "wall":
        base_line = element["geometry"]["base_line"]
        width = float(element["dimensions"].get("thickness") or element["revit_params"].get("width") or 0.2)
        height = float(element["dimensions"].get("height") or element["revit_params"].get("unconnected_height") or 2.5)
        profile = wall_profile_from_base_line(base_line, width)
        return create_extrusion(profile, height)

    if category in {"floor", "ceiling", "roof"}:
        profile = element["geometry"]["profile"]
        thickness = float(element["dimensions"].get("thickness") or element["revit_params"].get("thickness") or 0.2)
        return create_extrusion(profile, thickness)

    if category == "column":
        axis = element["geometry"]["axis"]
        center = axis["start"]
        orientation = element["geometry"].get("orientation_xy", [1.0, 0.0])
        width = float(element["dimensions"].get("width") or element["revit_params"].get("width") or 0.3)
        depth = float(element["dimensions"].get("depth") or element["revit_params"].get("depth") or 0.3)
        height = float(element["dimensions"].get("height") or 3.0)
        profile = oriented_rect_profile(center, orientation, width, depth, float(center[2]))
        return create_extrusion(profile, height)

    if category == "beam":
        center_line = element["geometry"]["center_line"]
        width = float(element["dimensions"].get("width") or element["revit_params"].get("section_width") or 0.3)
        depth = float(element["dimensions"].get("depth") or element["revit_params"].get("section_depth") or 0.3)
        return create_beam_solid(center_line, width, depth)

    if category in {"door", "window", "stair"}:
        profile = element["geometry"]["profile"]
        height = float(element["dimensions"].get("height") or element["revit_params"].get("height") or 2.0)
        return create_extrusion(profile, height)

    raise ValueError("Unsupported category: {}".format(category))



def try_create_native_wall(doc, element: dict):
    wall_type = first_element_of_class(doc, WallType)
    base_line = element["geometry"]["base_line"]
    start = base_line["start"]
    end = base_line["end"]
    curve = Line.CreateBound(to_xyz(start), to_xyz(end))
    level = find_or_create_level(doc, to_internal_length(start[2]))
    height = float(element["dimensions"].get("height") or element["revit_params"].get("unconnected_height") or 2.5)
    wall = Wall.Create(
        doc,
        curve,
        wall_type.Id,
        level.Id,
        max(to_internal_length(height), 0.1),
        0.0,
        False,
        False,
    )
    set_instance_comment(wall, element["id"])
    return wall



def profile_on_level(profile_points, level_elevation_source_units):
    return [[float(p[0]), float(p[1]), float(level_elevation_source_units)] for p in profile_points]



def try_create_native_floor(doc, element: dict):
    floor_type = first_element_of_class(doc, FloorType)
    profile = element["geometry"]["profile"]
    elevation = float(element["geometry"].get("elevation", profile[0][2]))
    level = find_or_create_level(doc, to_internal_length(elevation))
    loops = List[CurveLoop]()
    loops.Add(make_curve_loop(profile_on_level(profile, elevation)))

    try:
        floor = Floor.Create(doc, loops, floor_type.Id, level.Id)
    except Exception:
        floor = Floor.Create(doc, loops, floor_type.Id, level.Id, False, None, 0.0)
    set_instance_comment(floor, element["id"])
    return floor



def try_create_native_ceiling(doc, element: dict):
    ceiling_type = first_element_of_class(doc, CeilingType)
    profile = element["geometry"]["profile"]
    elevation = float(element["geometry"].get("elevation", profile[0][2]))
    level = find_or_create_level(doc, to_internal_length(elevation))
    loops = List[CurveLoop]()
    loops.Add(make_curve_loop(profile_on_level(profile, elevation)))
    ceiling = Ceiling.Create(doc, loops, ceiling_type.Id, level.Id)
    set_instance_comment(ceiling, element["id"])
    return ceiling



def import_element(doc, element: dict):
    category = element["category"]
    if category == "wall":
        try:
            return "native", try_create_native_wall(doc, element)
        except Exception:
            solid = directshape_solid_from_element(element)
            return "directshape_fallback", create_directshape(doc, category, solid, element["id"])

    if category == "floor":
        try:
            return "native", try_create_native_floor(doc, element)
        except Exception:
            solid = directshape_solid_from_element(element)
            return "directshape_fallback", create_directshape(doc, category, solid, element["id"])

    if category == "ceiling":
        try:
            return "native", try_create_native_ceiling(doc, element)
        except Exception:
            solid = directshape_solid_from_element(element)
            return "directshape_fallback", create_directshape(doc, category, solid, element["id"])

    solid = directshape_solid_from_element(element)
    return "directshape", create_directshape(doc, category, solid, element["id"])



def main() -> None:
    load_revit_api_globals()
    doc = get_doc()
    json_path = resolve_json_path()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])
    if not elements:
        raise RuntimeError("No elements were found in {}".format(json_path))

    created_native = 0
    created_directshape = 0
    created_fallback = 0
    skipped = []

    transaction = Transaction(doc, "Import Scan2BIM Hybrid Candidates")
    transaction.Start()
    try:
        for element in elements:
            element_id = element.get("id", "unnamed")
            try:
                mode, _created = import_element(doc, element)
                if mode == "native":
                    created_native += 1
                elif mode == "directshape":
                    created_directshape += 1
                else:
                    created_fallback += 1
            except Exception as exc:
                skipped.append("{}: {}".format(element_id, exc))
        transaction.Commit()
    except Exception:
        transaction.RollBack()
        raise

    print("Imported from {}".format(json_path))
    print("  Native elements: {}".format(created_native))
    print("  DirectShape elements: {}".format(created_directshape))
    print("  DirectShape fallbacks: {}".format(created_fallback))
    if skipped:
        print("  Skipped: {}".format(len(skipped)))
        for item in skipped[:20]:
            print("    - {}".format(item))
        if len(skipped) > 20:
            print("    ... and {} more".format(len(skipped) - 20))


if __name__ == "__main__":
    main()
