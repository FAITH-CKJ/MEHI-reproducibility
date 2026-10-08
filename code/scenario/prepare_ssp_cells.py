"""Sample Gao--Pesaresi SSP netCDF grids on the MEHI coastal 1-km support."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def sample_variable(
    path: Path,
    variable: str,
    lon: np.ndarray,
    lat: np.ndarray,
    lon_name: str,
    lat_name: str,
) -> np.ndarray:
    with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as dataset:
        if variable not in dataset:
            raise KeyError(f"{variable} is not present in {path}")
        points = dataset[variable].sel(
            {
                lon_name: xr.DataArray(lon, dims="sample"),
                lat_name: xr.DataArray(lat, dims="sample"),
            },
            method="nearest",
        )
        return np.asarray(points.values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--lookup", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--external-root", required=True, type=Path)
    parser.add_argument("--lon-field", default="lon")
    parser.add_argument("--lat-field", default="lat")
    parser.add_argument("--netcdf-lon", default="lon")
    parser.add_argument("--netcdf-lat", default="lat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support = pd.read_csv(args.support)
    lookup = pd.read_csv(args.lookup)
    required_support = {
        "scenario_cell_id",
        "country",
        "sampled_area_m2",
        "U_obs_2023",
        args.lon_field,
        args.lat_field,
    }
    required_lookup = {
        "SSP",
        "year",
        "baseline_relative_path",
        "baseline_variable",
        "target_relative_path",
        "target_variable",
        "value_scale",
    }
    missing = required_support - set(support.columns)
    if missing:
        raise ValueError(f"Support table is missing {sorted(missing)}")
    missing = required_lookup - set(lookup.columns)
    if missing:
        raise ValueError(f"Lookup table is missing {sorted(missing)}")

    lon = support[args.lon_field].to_numpy(float)
    lat = support[args.lat_field].to_numpy(float)
    outputs: list[pd.DataFrame] = []
    for row in lookup.itertuples(index=False):
        baseline_path = args.external_root / str(row.baseline_relative_path)
        target_path = args.external_root / str(row.target_relative_path)
        baseline = sample_variable(
            baseline_path,
            str(row.baseline_variable),
            lon,
            lat,
            args.netcdf_lon,
            args.netcdf_lat,
        )
        target = sample_variable(
            target_path,
            str(row.target_variable),
            lon,
            lat,
            args.netcdf_lon,
            args.netcdf_lat,
        )
        scale = float(row.value_scale)
        frame = support.copy()
        frame.insert(2, "year", int(row.year))
        frame.insert(2, "SSP", str(row.SSP))
        frame["U_SSP_2020"] = np.clip(baseline * scale, 0.0, 1.0)
        frame["U_SSP_y"] = np.clip(target * scale, 0.0, 1.0)
        outputs.append(frame)
    result = pd.concat(outputs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression="gzip" if args.output.suffix == ".gz" else None)
    print(f"rows={len(result)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
