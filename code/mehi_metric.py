"""Calculate MEHI and pMEHI from analysis-ready component inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


D_MAX_M = 1000.0
CONTACT_M = 30.0
STATE_CUTPOINT = 0.33
CHANGE_THRESHOLD = 0.005


def calculate_components(
    distance_m: pd.Series,
    contact_fraction: pd.Series,
    built_area_m2: pd.Series,
    soft_area_m2: pd.Series,
    other_area_m2: pd.Series,
) -> pd.DataFrame:
    """Contact fraction is clipped A_CONTACT/A_M for A_M>1 m2, otherwise zero."""
    external = built_area_m2 + soft_area_m2 + other_area_m2
    if (pd.concat([built_area_m2,soft_area_m2,other_area_m2],axis=1)<0).any().any():
        raise ValueError("Land-cover areas must be non-negative")
    distance = distance_m.clip(lower=0, upper=D_MAX_M)
    contact = contact_fraction.clip(lower=0, upper=1)
    valid_external = external > 1.0
    built_share = (built_area_m2 / external.where(valid_external)).fillna(0)
    soft_share = (soft_area_m2 / external.where(valid_external)).fillna(0)
    out = pd.DataFrame(
        {
            "P": 1 - 2 * distance / D_MAX_M,
            "C": 2 * contact - 1,
            "B": (2 * built_share - 1).where(valid_external,0),
            "S": (1 - 2 * soft_share).where(valid_external,0),
        }
    )
    out["MEHI"] = out[["P", "C", "B", "S"]].mean(axis=1)
    return out


def state_class(values: pd.Series, cutpoint: float = STATE_CUTPOINT) -> pd.Series:
    return pd.Series(
        np.select(
            [values < -cutpoint, values > cutpoint],
            ["soft_buffered", "hard_urban_edge"],
            default="mixed",
        ),
        index=values.index,
    )


def change_class(values: pd.Series, threshold: float = CHANGE_THRESHOLD) -> pd.Series:
    return pd.Series(
        np.select(
            [values < -threshold, values > threshold],
            ["softening", "hardening"],
            default="near_neutral",
        ),
        index=values.index,
    )


def endpoint_change(mehi_2000: pd.Series, mehi_2023: pd.Series) -> pd.Series:
    return (mehi_2023 - mehi_2000) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distance", default="distance_to_built_m")
    parser.add_argument("--contact", default="contact_fraction_30m")
    parser.add_argument("--built-area", default="built_area_m2")
    parser.add_argument("--soft-area", default="soft_area_m2")
    parser.add_argument("--other-area", default="other_area_m2")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    components = calculate_components(
        data[args.distance], data[args.contact], data[args.built_area], data[args.soft_area], data[args.other_area]
    )
    result = pd.concat([data, components], axis=1)
    result["MEHI_class"] = state_class(result["MEHI"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
