"""Run the country-aggregated, fixed-domain 150-m SSP exposure screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from scenario_core import (
    ScreenParameters,
    align_scenario_fractions,
    allocate_country_demand,
    country_demand,
    self_test,
    summarize,
    uncertainty_draws,
    update_components,
)


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, float_precision='round_trip')


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def parameters(config: dict) -> ScreenParameters:
    return ScreenParameters(**config.get("screen_parameters", {}))


def future(config: dict, config_dir: Path) -> None:
    baseline = read_table(resolve(config_dir, config["inputs"]["edge_baseline_2023"]))
    scenario = align_scenario_fractions(
        read_table(resolve(config_dir, config["inputs"]["scenario_cells"]))
    )
    out_dir = resolve(config_dir, config["output_dir"])
    grid_outputs: list[pd.DataFrame] = []
    global_rows: list[dict] = []
    country_rows: list[dict] = []

    for (ssp, year), cells in scenario.groupby(["SSP", "year"], sort=True):
        demand = country_demand(cells)
        demand = demand[(demand["SSP"] == ssp) & (demand["year"] == year)]
        allocated = allocate_country_demand(
            baseline,
            demand[["country", "country_demand_m2"]],
            parameters(config),
        )
        edge = update_components(allocated)
        edge.insert(0, "year", int(year))
        edge.insert(0, "SSP", str(ssp))
        grid_outputs.append(edge)
        global_rows.append({"SSP": ssp, "year": int(year), **summarize(edge)})
        for country, frame in edge.groupby("country", sort=True):
            country_rows.append({
                "country": country,
                "SSP": ssp,
                "year": int(year),
                **summarize(frame),
            })

    write_table(pd.concat(grid_outputs, ignore_index=True), out_dir / "future_grid_smehi_summary.csv.gz")
    write_table(pd.DataFrame(global_rows), out_dir / "future_global_smehi_summary.csv")
    write_table(pd.DataFrame(country_rows), out_dir / "future_country_smehi_summary.csv")
    print(f"future projection written to {out_dir}")


def ranks(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def backtest_metrics(predicted: pd.DataFrame, observed: pd.DataFrame, target_year: int) -> dict:
    target = observed.rename(columns={
        "MEHI": "MEHI_observed_target",
        "A_BUILT_M2": "A_BUILT_M2_target",
    })
    joined = predicted.merge(
        target[["edge_grid_id", "country", "MEHI_observed_target"]],
        on=["edge_grid_id", "country"],
        how="inner",
    )
    observed_delta = joined["MEHI_observed_target"] - joined["MEHI"]
    # pMEHI = Delta_MEHI / 2, so its primary +/-0.005 band becomes +/-0.01.
    direction = lambda x: np.where(x > 0.01, 1, np.where(x < -0.01, -1, 0))
    sign = direction(observed_delta) == direction(joined["Delta_sMEHI"])
    country = joined.assign(Delta_MEHI_observed=observed_delta).groupby("country", as_index=False).agg(
        Delta_MEHI_observed=("Delta_MEHI_observed", "mean"),
        Delta_MEHI_predicted=("Delta_sMEHI", "mean"),
    )
    top_observed = set(country.nlargest(5, "Delta_MEHI_observed")["country"])
    top_predicted = set(country.nlargest(5, "Delta_MEHI_predicted")["country"])
    return {
        "target_year": target_year,
        "grid_level_MEHI_MAE": float(np.mean(np.abs(joined["sMEHI"] - joined["MEHI_observed_target"]))),
        "country_level_spearman_rho_Delta_MEHI": float(
            ranks(country["Delta_MEHI_observed"]).corr(ranks(country["Delta_MEHI_predicted"]))
        ),
        "hard_edge_fraction_error": float(abs((joined["sMEHI"] >= 0.33).mean() - (joined["MEHI_observed_target"] >= 0.33).mean())),
        "direction_agreement_3class": float(sign.mean()),
        "neutral_band_Delta_MEHI": 0.01,
        "top_five_hardening_country_overlap": len(top_observed & top_predicted),
        "number_of_countries": int(country["country"].nunique()),
        "number_of_grids": int(len(joined)),
    }


def backtest(config: dict, config_dir: Path) -> None:
    out_dir = resolve(config_dir, config["output_dir"])
    rows: list[dict] = []
    for job in config.get("backtest_jobs", []):
        baseline = read_table(resolve(config_dir, job["edge_baseline"]))
        target = read_table(resolve(config_dir, job["observed_target_edges"]))
        merged = baseline.merge(
            target[["edge_grid_id", "country", "A_BUILT_M2"]].rename(columns={"A_BUILT_M2": "target_built"}),
            on=["edge_grid_id", "country"],
            how="inner",
        )
        merged["allocated_area_m2"] = np.maximum(0.0, merged["target_built"] - merged["A_BUILT_M2"])
        weighted = allocate_country_demand(
            merged.drop(columns=["allocated_area_m2"]),
            pd.DataFrame({"country": [], "country_demand_m2": []}),
            parameters(config),
        )
        weighted["requested_area_m2"] = merged["allocated_area_m2"].to_numpy()
        weighted["allocated_area_m2"] = np.minimum(weighted["eligible_area_m2"], weighted["requested_area_m2"])
        weighted["unmet_area_m2"] = weighted["requested_area_m2"] - weighted["allocated_area_m2"]
        predicted = update_components(weighted)
        rows.append(backtest_metrics(predicted, target, int(job["target_year"])))
    write_table(pd.DataFrame(rows), out_dir / "backtest_metrics.csv")
    print(f"backtest written to {out_dir}")


def uncertainty(config: dict, config_dir: Path) -> None:
    design = config["uncertainty"]
    out_dir = resolve(config_dir, config["output_dir"])
    global_table = read_table(out_dir / "future_global_smehi_summary.csv")
    rng = random.Random(int(design["random_seed"]))
    rows: list[dict] = []
    n = int(design["realizations"])
    for ssp in design["SSPs"]:
        central = float(global_table.loc[
            (global_table["SSP"] == ssp) & (global_table["year"] == int(design["year"])),
            "Delta_sMEHI",
        ].iloc[0])
        for lambda_m in design["lambda_m"]:
            for eligibility in design["eligibility"]:
                values = uncertainty_draws(central, float(lambda_m), eligibility, n, rng)
                rows.append({
                    "SSP": ssp,
                    "year": int(design["year"]),
                    "lambda": lambda_m,
                    "eligibility": eligibility,
                    "realization_count": n,
                    "global_mean_Delta_sMEHI": central,
                    "p05": values[int(0.05 * (n - 1))],
                    "p50": values[int(0.50 * (n - 1))],
                    "p95": values[int(0.95 * (n - 1))],
                })
    write_table(pd.DataFrame(rows), out_dir / "uncertainty_summary.csv")
    print(f"uncertainty summary written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["self-test", "future", "backtest", "uncertainty", "run-all"])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
        return
    if args.config is None:
        parser.error("--config is required outside self-test mode")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.mode in {"future", "run-all"}:
        future(config, config_path.parent)
    if args.mode in {"backtest", "run-all"} and config.get("backtest_jobs"):
        backtest(config, config_path.parent)
    if args.mode in {"uncertainty", "run-all"}:
        uncertainty(config, config_path.parent)


if __name__ == "__main__":
    main()
