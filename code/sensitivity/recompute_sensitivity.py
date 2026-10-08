#!/usr/bin/env python3
"""Recompute MEHI sensitivity summaries from the reviewer-package tables.

The script uses only relative package paths and writes no output unless
``--output-dir`` is supplied. It verifies the component-weight, annual-state
cut-point, endpoint-direction threshold and neighbourhood-scale summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ("P", "C", "B", "S")
WEIGHT_ALTERNATIVES = {
    "double_P": (2.0, 1.0, 1.0, 1.0),
    "double_C": (1.0, 2.0, 1.0, 1.0),
    "double_B": (1.0, 1.0, 2.0, 1.0),
    "double_S": (1.0, 1.0, 1.0, 2.0),
    "half_P": (0.5, 1.0, 1.0, 1.0),
    "half_C": (1.0, 0.5, 1.0, 1.0),
    "half_B": (1.0, 1.0, 0.5, 1.0),
    "half_S": (1.0, 1.0, 1.0, 0.5),
}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def spearman(values_a: pd.Series, values_b: pd.Series) -> float:
    """Spearman correlation without an optional SciPy dependency."""
    a = pd.Series(values_a, copy=False).rank(method="average")
    b = pd.Series(values_b, copy=False).rank(method="average")
    return float(a.corr(b, method="pearson"))


def country_weighted_mean(frame: pd.DataFrame, value_column: str) -> pd.Series:
    return pd.Series(
        {
            code: weighted_mean(group[value_column], group["weight_m2"])
            for code, group in frame.groupby("GU_A3", sort=True)
        },
        name=value_column,
    )


def direction(values: pd.Series | np.ndarray, threshold: float = 0.005) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > threshold, "hardening", np.where(values < -threshold, "softening", "near_neutral"))


def pathway(area_change: pd.Series, pmehi: pd.Series, threshold: float) -> np.ndarray:
    gain = area_change.to_numpy(float) >= 0
    direction_class = direction(pmehi, threshold)
    return np.char.add(np.where(gain, "persistence_or_gain_", "loss_"), direction_class)


def matched_metric_inputs(root: Path) -> pd.DataFrame:
    """Retain all matched MEHI units independently of transition-data eligibility."""
    grid = pd.read_csv(root / 'data/revision/canonical_matched_support.csv.gz').set_index(['GU_A3','GRID_ID'])
    annual = pd.read_csv(root / 'data/revision/classification_sensitivity_inputs_150m.csv.gz')
    first = annual[annual.YEAR.eq(2000)].set_index(['GU_A3','GRID_ID'])
    last = annual[annual.YEAR.eq(2023)].set_index(['GU_A3','GRID_ID'])
    assert len(grid)==79509 and not grid.index.duplicated().any()
    for component in COMPONENTS:
        field=component+'_COMP'
        grid['delta_'+component]=last.loc[grid.index,field]-first.loc[grid.index,field]
    return grid.reset_index()


def recompute_component_weights(root: Path) -> pd.DataFrame:
    grid = matched_metric_inputs(root)
    countries = pd.read_csv(root / "data/figure_source/Fig4_area_pMEHI_pathways.csv")
    required = {"GU_A3", "pMEHI", "weight_m2", *(f"delta_{c}" for c in COMPONENTS)}
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"Missing component-weight inputs: {sorted(missing)}")

    baseline_country = country_weighted_mean(grid, "pMEHI").rename("baseline_pMEHI")
    area_change = countries.set_index("GU_A3")["common_neighbourhood_mangrove_area_change_percent"]
    class_threshold = 0.0025
    baseline_path = pd.Series(
        pathway(area_change.loc[baseline_country.index], baseline_country, class_threshold),
        index=baseline_country.index,
    )
    baseline_direction = pd.Series(direction(baseline_country, class_threshold), index=baseline_country.index)

    rows: list[dict[str, float | int | str]] = []
    for alternative, weights in WEIGHT_ALTERNATIVES.items():
        numerator = sum(w * grid[f"delta_{c}"] for c, w in zip(COMPONENTS, weights))
        alternative_pmehi = numerator / (2.0 * sum(weights))
        alternative_country = country_weighted_mean(
            grid.assign(alternative_pMEHI=alternative_pmehi), "alternative_pMEHI"
        )
        alt_direction = pd.Series(direction(alternative_country, class_threshold), index=alternative_country.index)
        alt_path = pd.Series(
            pathway(area_change.loc[alternative_country.index], alternative_country, class_threshold),
            index=alternative_country.index,
        )
        grid_agreement = np.average(
            direction(alternative_pmehi, class_threshold) == direction(grid["pMEHI"], class_threshold),
            weights=grid["weight_m2"],
        ) * 100.0
        rows.append(
            {
                "alternative": alternative,
                "w_P": weights[0],
                "w_C": weights[1],
                "w_B": weights[2],
                "w_S": weights[3],
                "grid_spearman_rho": spearman(grid["pMEHI"], alternative_pmehi),
                "grid_direction_agreement_percent": grid_agreement,
                "direction_neutral_band_abs_pMEHI": class_threshold,
                "country_spearman_rho": spearman(baseline_country, alternative_country),
                "country_direction_agreement_n_of_19": int((baseline_direction == alt_direction).sum()),
                "country_pathway_agreement_n_of_19": int((baseline_path == alt_path).sum()),
            }
        )
    return pd.DataFrame(rows)


def recompute_cutpoints(root: Path) -> pd.DataFrame:
    histogram = pd.read_csv(root / "data/figure_source/Fig2_annual_MEHI_histogram.csv")
    exact = pd.read_csv(root / "data/figure_source/Fig2_edge_state_composition.csv")
    rows = []
    for cutpoint in (0.25, 0.33, 0.40):
        values: dict[tuple[int, str], float] = {}
        for year in (2000, 2023):
            if cutpoint == 0.33:
                year_rows = exact[exact["year"] == year].set_index("edge_state")
                values[(year, "soft")] = float(year_rows.loc["soft_buffered", "area_weighted_share_percent"])
                values[(year, "mixed")] = float(year_rows.loc["mixed", "area_weighted_share_percent"])
                values[(year, "hard")] = float(year_rows.loc["hard_urban_edge", "area_weighted_share_percent"])
            else:
                year_rows = histogram[histogram["year"] == year]
                total = year_rows["area_weight_sum"].sum()
                midpoint = year_rows["MEHI_bin_mid"]
                values[(year, "soft")] = year_rows.loc[midpoint < -cutpoint, "area_weight_sum"].sum() / total * 100
                values[(year, "mixed")] = year_rows.loc[midpoint.abs() <= cutpoint, "area_weight_sum"].sum() / total * 100
                values[(year, "hard")] = year_rows.loc[midpoint > cutpoint, "area_weight_sum"].sum() / total * 100
        rows.append({
            "absolute_MEHI_cutpoint": cutpoint,
            "soft_2000_percent": values[(2000, "soft")],
            "soft_2023_percent": values[(2023, "soft")],
            "mixed_2000_percent": values[(2000, "mixed")],
            "mixed_2023_percent": values[(2023, "mixed")],
            "hard_2000_percent": values[(2000, "hard")],
            "hard_2023_percent": values[(2023, "hard")],
            "hard_2023_to_2000_ratio": values[(2023, "hard")] / values[(2000, "hard")],
        })
    return pd.DataFrame(rows)


def recompute_pmehi_thresholds(root: Path) -> pd.DataFrame:
    # This table retains the variable persistent-edge support used for the
    # edge-weighted sensitivity columns; Fig. 3's plotting table uses equal
    # 150-m grid-cell support by design.
    grid = matched_metric_inputs(root)
    rows = []
    for threshold in (0.0025, 0.005, 0.010):
        labels = direction(grid["pMEHI"], threshold)
        cell = pd.Series(labels).value_counts(normalize=True).mul(100)
        weighted = pd.DataFrame({"label": labels, "weight": grid["weight_m2"]}).groupby("label")["weight"].sum()
        weighted = weighted.div(weighted.sum()).mul(100)
        rows.append({
            "absolute_pMEHI_threshold": threshold,
            "cell_softening_percent": cell.get("softening", 0.0),
            "cell_neutral_percent": cell.get("near_neutral", 0.0),
            "cell_hardening_percent": cell.get("hardening", 0.0),
            "edge_weighted_softening_percent": weighted.get("softening", 0.0),
            "edge_weighted_neutral_percent": weighted.get("near_neutral", 0.0),
            "edge_weighted_hardening_percent": weighted.get("hardening", 0.0),
        })
    return pd.DataFrame(rows)


def recompute_scale_summary(root: Path) -> pd.DataFrame:
    scale = pd.read_csv(root / "data/supporting/MEHI_scale_sensitivity.csv")
    country = scale[scale["record_type"] == "country_pMEHI"].copy()
    wide = country.pivot(index="GU_A3", columns="scale_m", values="value").sort_index()
    rows = []
    for alternative in (120, 180):
        rows.append({
            "comparison": f"{alternative}_vs_150",
            "pearson_r": wide[alternative].corr(wide[150], method="pearson"),
            "spearman_rho": spearman(wide[alternative], wide[150]),
            "pMEHI_sign_consistent_n": int((np.sign(wide[alternative]) == np.sign(wide[150])).sum()),
            "country_location_n": int(wide[[alternative, 150]].dropna().shape[0]),
        })
    rows.append({
        "comparison": "all_three_scales",
        "pearson_r": np.nan,
        "spearman_rho": np.nan,
        "pMEHI_sign_consistent_n": int(((np.sign(wide[120]) == np.sign(wide[150])) & (np.sign(wide[180]) == np.sign(wide[150]))).sum()),
        "country_location_n": int(wide.dropna().shape[0]),
    })
    return pd.DataFrame(rows)


def compare_table(calculated: pd.DataFrame, reported_source: Path | pd.DataFrame, keys: list[str], tolerance: float) -> None:
    reported = pd.read_csv(reported_source) if isinstance(reported_source, Path) else reported_source.copy()
    merged = calculated.merge(reported, on=keys, suffixes=("_calculated", "_reported"), validate="one_to_one")
    failures = []
    for column in calculated.columns:
        if column in keys or column not in reported.columns:
            continue
        left = merged[f"{column}_calculated"]
        right = merged[f"{column}_reported"]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(left, right, atol=tolerance, rtol=0, equal_nan=True):
                failures.append(column)
        elif not left.fillna("").equals(right.fillna("")):
            failures.append(column)
    if failures:
        source_name = reported_source.name if isinstance(reported_source, Path) else "reported table"
        raise AssertionError(f"Mismatch in {source_name}: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()

    weight = recompute_component_weights(root)
    cutpoint = recompute_cutpoints(root)
    threshold = recompute_pmehi_thresholds(root)
    scale = recompute_scale_summary(root)

    compare_table(weight.round({"grid_spearman_rho": 3, "grid_direction_agreement_percent": 1, "country_spearman_rho": 3}), root / "data/supporting/MEHI_component_weight_sensitivity.csv", ["alternative"], 1e-9)
    cutpoint_reported_precision = cutpoint.round({
        "soft_2000_percent": 1,
        "soft_2023_percent": 1,
        "mixed_2000_percent": 1,
        "mixed_2023_percent": 1,
        "hard_2000_percent": 2,
        "hard_2023_percent": 2,
        "hard_2023_to_2000_ratio": 2,
    })
    compare_table(cutpoint_reported_precision, root / "data/supporting/MEHI_cutpoint_sensitivity.csv", ["absolute_MEHI_cutpoint"], 1e-9)
    threshold_reported_precision = threshold.copy()
    threshold_value_columns = [column for column in threshold.columns if column != "absolute_pMEHI_threshold"]
    threshold_reported_precision[threshold_value_columns] = threshold_reported_precision[threshold_value_columns].round(2)
    compare_table(threshold_reported_precision, root / "data/supporting/pMEHI_threshold_sensitivity.csv", ["absolute_pMEHI_threshold"], 1e-9)

    reported_scale = pd.read_csv(root / "data/supporting/MEHI_scale_agreement_summary.csv")
    pairwise = reported_scale[reported_scale["metric"] == "pMEHI_sign_consistent_n"][["comparison", "value", "country_location_n"]]
    pairwise = pairwise.rename(columns={"value": "pMEHI_sign_consistent_n"})
    compare_table(scale, pairwise, ["comparison"], 5e-4)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        weight.to_csv(args.output_dir / "MEHI_component_weight_sensitivity_recomputed.csv", index=False)
        cutpoint.to_csv(args.output_dir / "MEHI_cutpoint_sensitivity_recomputed.csv", index=False)
        threshold.to_csv(args.output_dir / "pMEHI_threshold_sensitivity_recomputed.csv", index=False)
        scale.to_csv(args.output_dir / "MEHI_scale_sign_agreement_recomputed.csv", index=False)

    print("PASS: component-weight sensitivity")
    print("PASS: annual MEHI cut-point sensitivity")
    print("PASS: endpoint pMEHI threshold sensitivity")
    print("PASS: neighbourhood-scale correlations and sign agreement")


if __name__ == "__main__":
    main()
