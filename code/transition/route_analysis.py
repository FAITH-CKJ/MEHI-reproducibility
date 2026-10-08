"""Dominant hardening-route classification and packaged-table aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def value(row: pd.Series, name: str) -> float:
    if name not in row or pd.isna(row[name]):
        raise ValueError(f"Missing required route-classification field: {name}")
    return float(row[name])


def eligible_rows(data: pd.DataFrame) -> pd.Series:
    """Require valid endpoint support and a retained paired transition observation."""
    valid = data['valid_for_stats'].astype(str).str.lower().eq('true')
    return valid & data['observed_transition_area_m2'].gt(0) & data['weight_m2'].gt(0)


def classify_dominant_route(row: pd.Series, threshold: float = 0.30) -> str:
    """Classify a complete transition record, including class-resolved flow fields."""
    if value(row, 'observed_transition_area_m2') <= 0:
        raise ValueError('Route classification requires retained paired transition support')
    if value(row, "pMEHI") <= 0:
        return "not_hardening"
    changed = value(row, "changed_area_m2")
    soft_to_built = value(row, "soft_to_built_m2")
    # R is the sum of three nonnegative soft-to-built flows, so it is already
    # at least each of those constituent flows. Compare only independent
    # competing flows; this also avoids subtractive CSV-rounding comparisons
    # between a total and a constituent that is mathematically equal to it.
    candidates = [
        value(row, "soft_to_other_m2"),
        value(row, "other_to_built_m2"), value(row, "mangrove_to_built_m2"),
    ]
    if changed > 1 and soft_to_built >= max(candidates, default=0) and soft_to_built / changed >= threshold:
        return "built_up_replacement_route"
    if (
        value(row, "delta_P") > 0.05
        and abs(value(row, "delta_B")) <= 0.02
        and abs(value(row, "delta_C")) <= 0.02
        and abs(value(row, "delta_S")) <= 0.02
        and value(row, "soft_to_built_share_of_observed") < 0.05
    ):
        return "proximity_only_route"
    losses = {
        "tidal_flat_loss_route": max(0.0, -value(row, "delta_tidalflat_share")),
        "water_edge_narrowing_route": max(0.0, -value(row, "delta_water_share")),
        "vegetation_buffer_loss_route": max(0.0, -value(row, "delta_nonmangroveveg_share")),
    }
    best = max(losses, key=losses.get)
    if best == "tidal_flat_loss_route" and losses[best] > 0.01 and (
        value(row, 'delta_built_share') > 0 or value(row, 'delta_other_share') > 0
        or value(row, 'tidalflat_to_built_m2') > 1 or value(row, 'soft_to_other_m2') > 1
    ):
        return best
    if best == "water_edge_narrowing_route" and losses[best] > 0.01 and (
        value(row, 'delta_P') > 0.02 or value(row, 'delta_C') > 0.02
        or value(row, 'water_to_built_m2') > 1
    ):
        return best
    if best == "vegetation_buffer_loss_route" and losses[best] > 0.01:
        return best
    return "mixed_route"


def aggregate_existing_routes(data: pd.DataFrame) -> pd.DataFrame:
    hard = data.loc[eligible_rows(data) & data['pMEHI'].gt(0)].copy()
    assert hard['route'].ne('not_hardening').all()
    out = hard.groupby("route", as_index=False).agg(weight_m2=("weight_m2", "sum"), grid_count=("route", "size"))
    out["share_of_hardening_weight_percent"] = 100 * out["weight_m2"] / out["weight_m2"].sum()
    return out.sort_values("share_of_hardening_weight_percent", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    result = aggregate_existing_routes(data)
    print(result.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
