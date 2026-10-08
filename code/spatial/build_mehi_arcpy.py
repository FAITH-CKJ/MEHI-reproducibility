"""Build annual MEHI grids and the 2000--2023 pMEHI grid with ArcGIS Pro.

The inputs are four polygon mosaics with a country/location field and the six
land-cover classes documented in Supplementary Note 2. All metric operations are carried
out in a metre-based equal-area coordinate system. Run ``--self-test`` to check
the bounded component equations without importing ArcPy.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


YEARS = (2000, 2010, 2020, 2023)
MANGROVE = 1
TIDAL_FLAT = 2
NON_MANGROVE_VEGETATION = 3
WATER = 4
BUILT_UP = 5
OTHER = 6
SOFT_CLASSES = (TIDAL_FLAT, NON_MANGROVE_VEGETATION, WATER)


def clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def capped_near_distance(distance: float | None, distance_cap_m: float) -> float:
    """Preserve zero adjacency; cap missing or out-of-search Near results."""
    value = -1.0 if distance is None else float(distance)
    if not math.isfinite(value) or value < 0:
        return distance_cap_m
    return clip(value, 0.0, distance_cap_m)


def component_values(
    edge_distance_m: float,
    contact_mangrove_area_m2: float,
    mangrove_area_m2: float,
    built_area_m2: float,
    soft_area_m2: float,
    other_area_m2: float,
    distance_cap_m: float,
) -> dict[str, float | str]:
    """Calculate the four bounded components and MEHI for one neighbourhood."""
    distance = clip(edge_distance_m, 0.0, distance_cap_m)
    p_component = clip(1.0 - 2.0 * distance / distance_cap_m)

    contact_fraction = 0.0
    if mangrove_area_m2 > 1.0:
        contact_fraction = clip(contact_mangrove_area_m2 / mangrove_area_m2, 0.0, 1.0)
    c_component = clip(2.0 * contact_fraction - 1.0)

    external_area = built_area_m2 + soft_area_m2 + other_area_m2
    if external_area > 1.0:
        b_component = clip(2.0 * built_area_m2 / external_area - 1.0)
        s_component = clip(1.0 - 2.0 * soft_area_m2 / external_area)
    else:
        b_component = 0.0
        s_component = 0.0

    mehi = clip((p_component + c_component + b_component + s_component) / 4.0)
    if mehi < -0.33:
        state = "soft_buffered"
    elif mehi > 0.33:
        state = "hard_urban_edge"
    else:
        state = "mixed"
    return {
        "P_COMP": p_component,
        "C_COMP": c_component,
        "B_COMP": b_component,
        "S_COMP": s_component,
        "MEHI": mehi,
        "MEHI_CLASS": state,
        "A_OBS_EXT_M2": external_area,
    }


def pmehi(mehi_2000: float, mehi_2023: float) -> float:
    return clip((float(mehi_2023) - float(mehi_2000)) / 2.0)


def self_test() -> None:
    soft = component_values(1000, 0, 1000, 0, 900, 100, 1000)
    hard = component_values(0, 1000, 1000, 900, 0, 100, 1000)
    assert soft["MEHI"] < -0.33
    assert hard["MEHI"] > 0.33
    assert -1.0 <= pmehi(float(soft["MEHI"]), float(hard["MEHI"])) <= 1.0
    assert capped_near_distance(0.0,1000.0)==0.0
    assert capped_near_distance(25.0,1000.0)==25.0
    assert capped_near_distance(1200.0,1000.0)==1000.0
    assert capped_near_distance(None,1000.0)==1000.0
    assert capped_near_distance(-1.0,1000.0)==1000.0
    assert capped_near_distance(float('nan'),1000.0)==1000.0
    assert component_values(0,0.5,0.5,0,0,0,1000)['C_COMP']==-1.0
    assert component_values(0,1.0,1.0,0,0,0,1000)['C_COMP']==-1.0
    assert component_values(0,1.1,1.1,0,0,0,1000)['C_COMP']==1.0
    print("MEHI equations, zero distance, missing distance and small-support checks: PASS")


def field_names(arcpy, feature_class: str) -> set[str]:
    return {field.name for field in arcpy.ListFields(feature_class)}


def add_field(arcpy, feature_class: str, name: str, field_type: str, length: int = 64) -> None:
    if name in field_names(arcpy, feature_class):
        return
    if field_type.upper() == "TEXT":
        arcpy.management.AddField(feature_class, name, field_type, field_length=length)
    else:
        arcpy.management.AddField(feature_class, name, field_type)


def require_fields(arcpy, feature_class: str, required: Iterable[str]) -> None:
    missing = sorted(set(required) - field_names(arcpy, feature_class))
    if missing:
        raise ValueError(f"{feature_class} is missing fields: {', '.join(missing)}")


def delete_if_exists(arcpy, path: str) -> None:
    if arcpy.Exists(path):
        arcpy.management.Delete(path)


def select_classes(arcpy, source: str, output: str, class_field: str, classes: Iterable[int]) -> str:
    values = ",".join(str(int(value)) for value in classes)
    delimiter = arcpy.AddFieldDelimiters(source, class_field)
    layer = f"select_{Path(output).name}"
    arcpy.management.MakeFeatureLayer(source, layer, f"{delimiter} IN ({values})")
    arcpy.management.CopyFeatures(layer, output)
    arcpy.management.Delete(layer)
    return output


def project_and_dissolve(
    arcpy,
    source: str,
    projected: str,
    dissolved: str,
    spatial_reference,
    country_field: str,
    class_field: str,
    countries: list[str],
) -> str:
    require_fields(arcpy, source, (country_field, class_field))
    source_for_projection = source
    if countries:
        quoted = ",".join(f"'{country.replace(chr(39), chr(39) * 2)}'" for country in countries)
        delimiter = arcpy.AddFieldDelimiters(source, country_field)
        layer = f"country_subset_{Path(projected).name}"
        arcpy.management.MakeFeatureLayer(source, layer, f"{delimiter} IN ({quoted})")
        source_for_projection = layer
    arcpy.management.Project(source_for_projection, projected, spatial_reference)
    if countries:
        arcpy.management.Delete(source_for_projection)
    arcpy.analysis.PairwiseDissolve(projected, dissolved, (country_field, class_field), multi_part="MULTI_PART")
    return dissolved


def build_boundary_grid(
    arcpy,
    mangrove: str,
    output_grid: str,
    grid_size_m: float,
    country_field: str,
    spatial_reference,
    densify_distance_m: float,
) -> str:
    boundary = f"{output_grid}_boundary"
    candidate = f"{output_grid}_candidate"
    delete_if_exists(arcpy, boundary)
    delete_if_exists(arcpy, candidate)
    delete_if_exists(arcpy, output_grid)
    arcpy.management.PolygonToLine(mangrove, boundary, "IGNORE_NEIGHBORS")
    if densify_distance_m > 0:
        arcpy.edit.Densify(boundary, "DISTANCE", f"{densify_distance_m} Meters")

    workspace = str(Path(output_grid).parent)
    name = Path(candidate).name
    arcpy.management.CreateFeatureclass(workspace, name, "POLYGON", spatial_reference=spatial_reference)
    add_field(arcpy, candidate, country_field, "TEXT", 12)
    add_field(arcpy, candidate, "GRID_X", "LONG")
    add_field(arcpy, candidate, "GRID_Y", "LONG")
    add_field(arcpy, candidate, "GRID_ID", "TEXT", 80)

    cells: set[tuple[str, int, int]] = set()
    with arcpy.da.SearchCursor(boundary, (country_field, "SHAPE@")) as cursor:
        for country, geometry in cursor:
            for part in geometry:
                for point in part:
                    if point is None:
                        continue
                    gx = math.floor(point.X / grid_size_m)
                    gy = math.floor(point.Y / grid_size_m)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            cells.add((str(country), gx + dx, gy + dy))

    fields = (country_field, "GRID_X", "GRID_Y", "GRID_ID", "SHAPE@")
    with arcpy.da.InsertCursor(candidate, fields) as cursor:
        for country, gx, gy in sorted(cells):
            xmin = gx * grid_size_m
            ymin = gy * grid_size_m
            polygon = arcpy.Polygon(
                arcpy.Array(
                    [
                        arcpy.Point(xmin, ymin),
                        arcpy.Point(xmin + grid_size_m, ymin),
                        arcpy.Point(xmin + grid_size_m, ymin + grid_size_m),
                        arcpy.Point(xmin, ymin + grid_size_m),
                    ]
                ),
                spatial_reference,
            )
            grid_id = f"{country}_{gx}_{gy}"
            cursor.insertRow((country, gx, gy, grid_id, polygon))

    layer = f"grid_intersection_{Path(output_grid).name}"
    arcpy.management.MakeFeatureLayer(candidate, layer)
    arcpy.management.SelectLayerByLocation(layer, "INTERSECT", boundary, selection_type="NEW_SELECTION")
    arcpy.management.CopyFeatures(layer, output_grid)
    arcpy.management.Delete(layer)
    add_field(arcpy, output_grid, "CELL_A_M2", "DOUBLE")
    with arcpy.da.UpdateCursor(output_grid, ("CELL_A_M2", "SHAPE@AREA")) as cursor:
        for row in cursor:
            row[0] = float(row[1])
            cursor.updateRow(row)
    return boundary


def aggregate_class_areas(
    arcpy,
    grid: str,
    land_cover: str,
    output_intersection: str,
    country_field: str,
    class_field: str,
) -> dict[str, dict[int, float]]:
    delete_if_exists(arcpy, output_intersection)
    arcpy.analysis.PairwiseIntersect((grid, land_cover), output_intersection, "ALL", output_type="INPUT")
    add_field(arcpy, output_intersection, "PART_A_M2", "DOUBLE")
    with arcpy.da.UpdateCursor(output_intersection, ("PART_A_M2", "SHAPE@AREA")) as cursor:
        for row in cursor:
            row[0] = float(row[1])
            cursor.updateRow(row)
    areas: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    with arcpy.da.SearchCursor(output_intersection, ("GRID_ID", country_field, class_field, "PART_A_M2")) as cursor:
        for grid_id, _country, land_class, area in cursor:
            areas[str(grid_id)][int(land_class)] += max(0.0, float(area or 0.0))
    return areas


def aggregate_contact_area(
    arcpy,
    grid: str,
    mangrove: str,
    built: str,
    output_prefix: str,
    contact_distance_m: float,
) -> dict[str, float]:
    built_buffer = f"{output_prefix}_built_buffer"
    contact_zone = f"{output_prefix}_contact_zone"
    contact_grid = f"{output_prefix}_contact_grid"
    for path in (built_buffer, contact_zone, contact_grid):
        delete_if_exists(arcpy, path)
    arcpy.analysis.PairwiseBuffer(built, built_buffer, f"{contact_distance_m} Meters", dissolve_option="ALL")
    arcpy.analysis.PairwiseIntersect((mangrove, built_buffer), contact_zone, "ALL", output_type="INPUT")
    arcpy.analysis.PairwiseIntersect((grid, contact_zone), contact_grid, "ALL", output_type="INPUT")
    add_field(arcpy, contact_grid, "CONTACT_M2", "DOUBLE")
    totals: dict[str, float] = defaultdict(float)
    with arcpy.da.UpdateCursor(contact_grid, ("GRID_ID", "CONTACT_M2", "SHAPE@AREA")) as cursor:
        for row in cursor:
            row[1] = float(row[2])
            totals[str(row[0])] += row[1]
            cursor.updateRow(row)
    return totals


def aggregate_edge_distance(
    arcpy,
    grid: str,
    boundary: str,
    built: str,
    output_segments: str,
    distance_cap_m: float,
) -> tuple[dict[str, float], dict[str, float]]:
    delete_if_exists(arcpy, output_segments)
    arcpy.analysis.PairwiseIntersect((grid, boundary), output_segments, "ALL", output_type="LINE")
    arcpy.analysis.Near(output_segments, built, f"{distance_cap_m} Meters", method="PLANAR")
    length_sum: dict[str, float] = defaultdict(float)
    distance_sum: dict[str, float] = defaultdict(float)
    with arcpy.da.SearchCursor(output_segments, ("GRID_ID", "SHAPE@LENGTH", "NEAR_DIST")) as cursor:
        for grid_id, length, distance in cursor:
            segment_length = max(0.0, float(length or 0.0))
            near_distance = capped_near_distance(distance,distance_cap_m)
            length_sum[str(grid_id)] += segment_length
            distance_sum[str(grid_id)] += segment_length * near_distance
    mean_distance = {
        grid_id: distance_sum[grid_id] / length
        for grid_id, length in length_sum.items()
        if length > 0
    }
    return length_sum, mean_distance


GRID_FIELDS = (
    ("A_M_M2", "DOUBLE"),
    ("A_BUILT_M2", "DOUBLE"),
    ("A_SOFT_M2", "DOUBLE"),
    ("A_OTHER_M2", "DOUBLE"),
    ("A_OBS_EXT_M2", "DOUBLE"),
    ("A_CONTACT_M2", "DOUBLE"),
    ("EDGE_LEN_M", "DOUBLE"),
    ("EDGE_DIST_M", "DOUBLE"),
    ("P_COMP", "DOUBLE"),
    ("C_COMP", "DOUBLE"),
    ("B_COMP", "DOUBLE"),
    ("S_COMP", "DOUBLE"),
    ("MEHI", "DOUBLE"),
    ("MEHI_CLASS", "TEXT"),
)


def populate_mehi_grid(
    arcpy,
    grid: str,
    class_areas: dict[str, dict[int, float]],
    contact_areas: dict[str, float],
    edge_lengths: dict[str, float],
    edge_distances: dict[str, float],
    distance_cap_m: float,
) -> None:
    for name, field_type in GRID_FIELDS:
        add_field(arcpy, grid, name, field_type, 24)
    names = [name for name, _ in GRID_FIELDS]
    with arcpy.da.UpdateCursor(grid, ["GRID_ID", *names]) as cursor:
        for row in cursor:
            grid_id = str(row[0])
            areas = class_areas.get(grid_id, {})
            mangrove_area = areas.get(MANGROVE, 0.0)
            built_area = areas.get(BUILT_UP, 0.0)
            soft_area = sum(areas.get(code, 0.0) for code in SOFT_CLASSES)
            other_area = areas.get(OTHER, 0.0)
            contact_area = contact_areas.get(grid_id, 0.0)
            edge_length = edge_lengths.get(grid_id, 0.0)
            edge_distance = edge_distances.get(grid_id, distance_cap_m)
            values = component_values(
                edge_distance,
                contact_area,
                mangrove_area,
                built_area,
                soft_area,
                other_area,
                distance_cap_m,
            )
            payload = {
                "A_M_M2": mangrove_area,
                "A_BUILT_M2": built_area,
                "A_SOFT_M2": soft_area,
                "A_OTHER_M2": other_area,
                "A_OBS_EXT_M2": values["A_OBS_EXT_M2"],
                "A_CONTACT_M2": contact_area,
                "EDGE_LEN_M": edge_length,
                "EDGE_DIST_M": edge_distance,
                "P_COMP": values["P_COMP"],
                "C_COMP": values["C_COMP"],
                "B_COMP": values["B_COMP"],
                "S_COMP": values["S_COMP"],
                "MEHI": values["MEHI"],
                "MEHI_CLASS": values["MEHI_CLASS"],
            }
            for index, name in enumerate(names, start=1):
                row[index] = payload[name]
            cursor.updateRow(row)


def export_grid_csv(arcpy, grid: str, output_csv: Path, country_field: str, year: int) -> None:
    fields = [
        country_field,
        "GRID_ID",
        "CELL_A_M2",
        "A_M_M2",
        "A_BUILT_M2",
        "A_SOFT_M2",
        "A_OTHER_M2",
        "A_OBS_EXT_M2",
        "A_CONTACT_M2",
        "EDGE_LEN_M",
        "EDGE_DIST_M",
        "P_COMP",
        "C_COMP",
        "B_COMP",
        "S_COMP",
        "MEHI",
        "MEHI_CLASS",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", *fields])
        with arcpy.da.SearchCursor(grid, fields) as cursor:
            for row in cursor:
                writer.writerow([year, *row])


def build_year(arcpy, args, year: int, gdb: str, tables: Path, spatial_reference) -> str:
    source = Path(args.input_dir) / args.input_template.format(year=year)
    if not source.exists():
        raise FileNotFoundError(source)
    projected = str(Path(gdb) / f"lc_{year}_projected")
    dissolved = str(Path(gdb) / f"lc_{year}_dissolved")
    mangrove_raw = str(Path(gdb) / f"mangrove_{year}_raw")
    mangrove = str(Path(gdb) / f"mangrove_{year}")
    built_raw = str(Path(gdb) / f"built_{year}_raw")
    built = str(Path(gdb) / f"built_{year}")
    grid = str(Path(gdb) / f"mehi_grid_{int(args.grid_size)}m_{year}")
    for path in (projected, dissolved, mangrove_raw, mangrove, built_raw, built, grid):
        delete_if_exists(arcpy, path)

    project_and_dissolve(
        arcpy,
        str(source),
        projected,
        dissolved,
        spatial_reference,
        args.country_field,
        args.class_field,
        args.countries,
    )
    select_classes(arcpy, dissolved, mangrove_raw, args.class_field, (MANGROVE,))
    arcpy.analysis.PairwiseDissolve(mangrove_raw, mangrove, args.country_field, multi_part="MULTI_PART")
    select_classes(arcpy, dissolved, built_raw, args.class_field, (BUILT_UP,))
    arcpy.analysis.PairwiseDissolve(built_raw, built, args.country_field, multi_part="MULTI_PART")

    boundary = build_boundary_grid(
        arcpy,
        mangrove,
        grid,
        args.grid_size,
        args.country_field,
        spatial_reference,
        args.densify_distance,
    )
    class_areas = aggregate_class_areas(
        arcpy,
        grid,
        dissolved,
        str(Path(gdb) / f"grid_class_{year}"),
        args.country_field,
        args.class_field,
    )
    contact_areas = aggregate_contact_area(
        arcpy,
        grid,
        mangrove,
        built,
        str(Path(gdb) / f"contact_{year}"),
        args.contact_distance,
    )
    edge_lengths, edge_distances = aggregate_edge_distance(
        arcpy,
        grid,
        boundary,
        built,
        str(Path(gdb) / f"edge_segments_{year}"),
        args.distance_cap,
    )
    populate_mehi_grid(
        arcpy,
        grid,
        class_areas,
        contact_areas,
        edge_lengths,
        edge_distances,
        args.distance_cap,
    )
    export_grid_csv(arcpy, grid, tables / f"mehi_grid_{year}.csv", args.country_field, year)
    return grid


def build_pmehi(arcpy, args, grids: dict[int, str], gdb: str, tables: Path, spatial_reference) -> str:
    if 2000 not in grids or 2023 not in grids:
        raise ValueError("pMEHI requires both 2000 and 2023 annual grids")

    fields = [
        args.country_field,
        "GRID_ID",
        "A_M_M2",
        "P_COMP",
        "C_COMP",
        "B_COMP",
        "S_COMP",
        "MEHI",
    ]
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    with arcpy.da.SearchCursor(grids[2000], fields) as cursor:
        for row in cursor:
            baseline[(str(row[0]), str(row[1]))] = dict(zip(fields[2:], map(float, row[2:])))

    output = str(Path(gdb) / f"pmehi_grid_{int(args.grid_size)}m_2000_2023")
    delete_if_exists(arcpy, output)
    workspace = str(Path(output).parent)
    arcpy.management.CreateFeatureclass(workspace, Path(output).name, "POLYGON", spatial_reference=spatial_reference)
    add_field(arcpy, output, args.country_field, "TEXT", 12)
    add_field(arcpy, output, "GRID_ID", "TEXT", 80)
    for name in (
        "weight_m2",
        "A_M_2000_M2",
        "A_M_2023_M2",
        "MEHI_2000",
        "MEHI_2023",
        "pMEHI",
        "dP_HALF",
        "dC_HALF",
        "dB_HALF",
        "dS_HALF",
    ):
        add_field(arcpy, output, name, "DOUBLE")

    output_fields = [
        "SHAPE@",
        args.country_field,
        "GRID_ID",
        "weight_m2",
        "A_M_2000_M2",
        "A_M_2023_M2",
        "MEHI_2000",
        "MEHI_2023",
        "pMEHI",
        "dP_HALF",
        "dC_HALF",
        "dB_HALF",
        "dS_HALF",
    ]
    with arcpy.da.InsertCursor(output, output_fields) as insert_cursor:
        with arcpy.da.SearchCursor(grids[2023], ["SHAPE@", *fields]) as cursor:
            for row in cursor:
                geometry, country, grid_id = row[0], str(row[1]), str(row[2])
                base = baseline.get((country, grid_id))
                if base is None:
                    continue
                current = dict(zip(fields[2:], map(float, row[3:])))
                weight = min(base["A_M_M2"], current["A_M_M2"])
                if weight <= 0:
                    continue
                insert_cursor.insertRow(
                    (
                        geometry,
                        country,
                        grid_id,
                        weight,
                        base["A_M_M2"],
                        current["A_M_M2"],
                        base["MEHI"],
                        current["MEHI"],
                        pmehi(base["MEHI"], current["MEHI"]),
                        (current["P_COMP"] - base["P_COMP"]) / 2.0,
                        (current["C_COMP"] - base["C_COMP"]) / 2.0,
                        (current["B_COMP"] - base["B_COMP"]) / 2.0,
                        (current["S_COMP"] - base["S_COMP"]) / 2.0,
                    )
                )

    export_fields = output_fields[1:]
    output_csv = tables / "pmehi_grid_2000_2023.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(export_fields)
        with arcpy.da.SearchCursor(output, export_fields) as cursor:
            writer.writerows(cursor)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-template", default="mosaic{year}join.shp")
    parser.add_argument("--country-field", default="GU_A3")
    parser.add_argument("--class-field", default="gridcode")
    parser.add_argument("--countries", nargs="*", default=[])
    parser.add_argument("--years", nargs="*", type=int, default=list(YEARS))
    parser.add_argument("--grid-size", type=float, default=150.0)
    parser.add_argument("--contact-distance", type=float, default=30.0)
    parser.add_argument("--distance-cap", type=float, default=1000.0)
    parser.add_argument("--densify-distance", type=float, default=0.0)
    parser.add_argument("--work-srid", type=int, default=54009)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.input_dir is None or args.output_dir is None:
        raise SystemExit("--input-dir and --output-dir are required")
    unknown_years = sorted(set(args.years) - set(YEARS))
    if unknown_years:
        raise SystemExit(f"Unsupported years: {unknown_years}")
    if args.grid_size <= 0 or args.contact_distance < 0 or args.distance_cap <= 0:
        raise SystemExit("Grid size and distance cap must be positive; contact distance must be non-negative")
    if args.densify_distance < 0:
        raise SystemExit("Densify distance must be non-negative")

    import arcpy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    gdb = args.output_dir / "mehi_work.gdb"
    if gdb.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to reuse it: {gdb}")
    if not arcpy.Exists(str(gdb)):
        arcpy.management.CreateFileGDB(str(args.output_dir), gdb.name)
    arcpy.env.overwriteOutput = bool(args.overwrite)
    arcpy.env.parallelProcessingFactor = "75%"
    spatial_reference = arcpy.SpatialReference(args.work_srid)

    grids: dict[int, str] = {}
    for year in args.years:
        print(f"Building MEHI for {year}", flush=True)
        grids[year] = build_year(arcpy, args, year, str(gdb), tables, spatial_reference)
    if 2000 in grids and 2023 in grids:
        output = build_pmehi(arcpy, args, grids, str(gdb), tables, spatial_reference)
        print(f"pMEHI output: {output}")
    print(f"MEHI geodatabase: {gdb}")


if __name__ == "__main__":
    main()
