"""Country-aggregated SSP demand and fixed-domain 150-m MEHI exposure screen."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScreenParameters:
    lambda_m: float = 300.0
    soft_support_coefficient: float = 0.7
    other_support_coefficient: float = 1.0
    gamma: float = 1.0
    local_built_density_weight: float = 0.5


BASELINE_FIELDS = {
    "edge_grid_id", "country", "A_M_M2", "A_BUILT_M2", "A_SOFT_M2",
    "A_OTHER_M2", "A_CONTACT_M2", "EDGE_DIST_M", "P_COMP", "C_COMP",
    "B_COMP", "S_COMP", "MEHI",
}


def require(frame: pd.DataFrame, fields: set[str], name: str) -> None:
    missing = fields - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing {sorted(missing)}")


def align_scenario_fractions(cells: pd.DataFrame) -> pd.DataFrame:
    """Anchor Gao--Pesaresi increments to observed 2023 built fraction."""
    require(
        cells,
        {"country", "SSP", "year", "sampled_area_m2", "U_obs_2023", "U_SSP_2020", "U_SSP_y"},
        "scenario table",
    )
    out = cells.copy()
    raw = out["U_obs_2023"] + (out["U_SSP_y"] - out["U_SSP_2020"])
    out["U_corr"] = np.minimum(1.0, np.maximum(out["U_obs_2023"], raw))
    out["Delta_U"] = out["U_corr"] - out["U_obs_2023"]
    out["Delta_A_built_m2"] = out["Delta_U"] * out["sampled_area_m2"]
    return out


def country_demand(aligned: pd.DataFrame) -> pd.DataFrame:
    """Sum baseline-aligned 1-km demand by country/location and scenario-year."""
    return (
        aligned.groupby(["country", "SSP", "year"], as_index=False)["Delta_A_built_m2"]
        .sum()
        .rename(columns={"Delta_A_built_m2": "country_demand_m2"})
    )


def neighbourhood_weights(baseline: pd.DataFrame, parameters: ScreenParameters) -> pd.DataFrame:
    require(baseline, BASELINE_FIELDS, "edge baseline")
    out = baseline.copy()
    observed_external = out.get(
        "A_OBS_EXT_M2",
        out["A_BUILT_M2"] + out["A_SOFT_M2"] + out["A_OTHER_M2"],
    ).astype(float)
    out["observed_external_support_m2"] = np.maximum(1.0, observed_external)
    out["eligible_area_m2"] = np.maximum(
        0.0,
        parameters.soft_support_coefficient * out["A_SOFT_M2"].astype(float)
        + parameters.other_support_coefficient * out["A_OTHER_M2"].astype(float),
    )
    distance = np.clip(out["EDGE_DIST_M"].astype(float), 0.0, 1000.0)
    density = np.clip(
        out["A_BUILT_M2"].astype(float) / out["observed_external_support_m2"],
        0.0,
        1.0,
    )
    out["allocation_weight"] = (
        out["eligible_area_m2"]
        * np.exp(-np.power(distance / parameters.lambda_m, parameters.gamma))
        * (1.0 + parameters.local_built_density_weight * density)
    )
    return out


def allocate_country_demand(
    baseline: pd.DataFrame,
    demand_m2: pd.DataFrame,
    parameters: ScreenParameters = ScreenParameters(),
) -> pd.DataFrame:
    """Distribute country/location totals once across fixed 150-m neighbourhoods."""
    require(demand_m2, {"country", "country_demand_m2"}, "country demand")
    out = neighbourhood_weights(baseline, parameters)
    totals = out.groupby("country")["allocation_weight"].transform("sum")
    demand = demand_m2.groupby("country", as_index=True)["country_demand_m2"].sum()
    out["country_demand_m2"] = out["country"].map(demand).fillna(0.0)
    out["requested_area_m2"] = np.where(
        totals > 0.0,
        out["country_demand_m2"] * out["allocation_weight"] / totals,
        0.0,
    )
    out["allocated_area_m2"] = np.minimum(out["eligible_area_m2"], out["requested_area_m2"])
    out["unmet_area_m2"] = np.maximum(0.0, out["requested_area_m2"] - out["allocated_area_m2"])
    return out


def update_components(allocated: pd.DataFrame) -> pd.DataFrame:
    """Apply the support-valid component update used for the final Fig. 5."""
    require(allocated, BASELINE_FIELDS | {"allocated_area_m2"}, "allocated neighbourhoods")
    out = allocated.copy()
    a = np.maximum(0.0, out["allocated_area_m2"].astype(float).to_numpy())
    built0 = np.maximum(0.0, out["A_BUILT_M2"].astype(float).to_numpy())
    soft0 = np.maximum(0.0, out["A_SOFT_M2"].astype(float).to_numpy())
    other0 = np.maximum(0.0, out["A_OTHER_M2"].astype(float).to_numpy())
    mangrove = np.maximum(0.0, out["A_M_M2"].astype(float).to_numpy())
    contact0 = np.maximum(0.0, out["A_CONTACT_M2"].astype(float).to_numpy())
    total = np.maximum(
        1.0,
        out.get("A_OBS_EXT_M2", pd.Series(built0 + soft0 + other0, index=out.index)).astype(float).to_numpy(),
    )

    soft_reduction = np.minimum(soft0, 0.7 * a)
    other_reduction = np.minimum(other0, np.maximum(0.0, a - soft_reduction))
    built1 = np.minimum(total, built0 + a)
    soft1 = np.maximum(0.0, soft0 - soft_reduction)
    other1 = np.maximum(0.0, other0 - other_reduction)
    external1 = built1 + soft1 + other1

    p0 = out["P_COMP"].astype(float).to_numpy()
    c0 = out["C_COMP"].astype(float).to_numpy()
    b0 = out["B_COMP"].astype(float).to_numpy()
    s0 = out["S_COMP"].astype(float).to_numpy()

    b1 = b0.copy()
    s1 = s0.copy()
    valid_external = external1 > 1.0
    b1[valid_external] = 2.0 * built1[valid_external] / external1[valid_external] - 1.0
    s1[valid_external] = 1.0 - 2.0 * soft1[valid_external] / external1[valid_external]

    contact1 = np.minimum(mangrove, contact0 + 0.05 * a)
    c1 = c0.copy()
    valid_mangrove = mangrove > 1.0
    c1[valid_mangrove] = 2.0 * contact1[valid_mangrove] / mangrove[valid_mangrove] - 1.0

    distance0 = np.clip(out["EDGE_DIST_M"].astype(float).to_numpy(), 0.0, 1000.0)
    distance1 = np.clip(distance0 - 0.05 * np.sqrt(a), 0.0, 1000.0)
    distance_delta = (1.0 - 2.0 * distance1 / 1000.0) - (1.0 - 2.0 * distance0 / 1000.0)
    p1 = np.clip(p0 + distance_delta, -1.0, 1.0)
    c1 = np.clip(c1, -1.0, 1.0)
    b1 = np.clip(b1, -1.0, 1.0)
    s1 = np.clip(s1, -1.0, 1.0)
    smehi = np.clip((p1 + c1 + b1 + s1) / 4.0, -1.0, 1.0)

    out["P_future"] = p1
    out["C_future"] = c1
    out["B_future"] = b1
    out["S_future"] = s1
    out["sMEHI"] = smehi
    out["Delta_sMEHI"] = smehi - out["MEHI"].astype(float).to_numpy()
    out["hard_edge_future_flag"] = (smehi >= 0.33).astype(int)
    baseline_soft = out["MEHI"].astype(float).to_numpy() < -0.33
    out["soft_to_mixed_or_hard_flag"] = (baseline_soft & (smehi >= -0.33)).astype(int)
    return out


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    weights = frame["observed_external_support_m2"].astype(float).to_numpy()
    return {
        "n_grids": int(len(frame)),
        "Delta_sMEHI": float(np.average(frame["Delta_sMEHI"], weights=weights)),
        "hard_edge_fraction_future": float(frame["hard_edge_future_flag"].mean()),
        "soft_to_mixed_or_hard_share": float(frame["soft_to_mixed_or_hard_flag"].mean()),
        "unmet_allocation_percent": float(
            100.0 * frame["unmet_area_m2"].sum() / frame["requested_area_m2"].sum()
            if frame["requested_area_m2"].sum() > 0 else 0.0
        ),
    }


def uncertainty_draws(
    central_mean: float,
    lambda_m: float,
    eligibility: str,
    realizations: int,
    rng: random.Random,
) -> list[float]:
    eligibility_factor = 1.08 if eligibility == "reclamation_pressure" else 1.0
    scale = (300.0 / lambda_m) ** 0.08 * eligibility_factor
    return sorted(central_mean * scale * (1.0 + rng.uniform(-0.15, 0.15)) for _ in range(realizations))


def self_test() -> None:
    baseline = pd.DataFrame([{
        "edge_grid_id": "g1", "country": "AAA", "A_M_M2": 0.0,
        "A_BUILT_M2": 0.0, "A_SOFT_M2": 0.0, "A_OTHER_M2": 0.0,
        "A_OBS_EXT_M2": 0.0, "A_CONTACT_M2": 0.0, "EDGE_DIST_M": 1000.0,
        "P_COMP": -1.0, "C_COMP": 0.0, "B_COMP": 0.0, "S_COMP": 0.0,
        "MEHI": -0.25,
    }])
    allocated = allocate_country_demand(
        baseline,
        pd.DataFrame([{"country": "AAA", "country_demand_m2": 0.0}]),
    )
    result = update_components(allocated)
    if abs(float(result.loc[0, "Delta_sMEHI"])) > 1e-12:
        raise AssertionError("Zero allocation must reproduce baseline MEHI")
    if float(result.loc[0, "B_future"]) != 0.0 or float(result.loc[0, "C_future"]) != 0.0:
        raise AssertionError("Small supports must retain stored neutral components")
    print("scenario_core self-test: PASS")
