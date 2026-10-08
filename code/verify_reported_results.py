"""Verify headline manuscript values from the packaged source tables."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figure_source"
SUP = ROOT / "data" / "supporting"
FAILURES: list[str] = []
FINAL_UNITS = {
    "AUS", "BHR", "BRA", "CHN", "COL", "GHA", "GIN", "IDN", "IND",
    "MEX", "MOZ", "MYS", "NGA", "NZL", "PAK", "PHL", "SGP", "THA", "USA",
}


def check(name: str, observed, expected, tolerance: float = 0.0) -> None:
    if isinstance(expected, (int, str, bool, set, tuple, list)):
        passed = observed == expected
    else:
        passed = math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance)
    status = "PASS" if passed else "FAIL"
    print(f"{status:4s} | {name}: observed={observed!r}, expected={expected!r}")
    if not passed:
        FAILURES.append(name)


def verify_fig2() -> None:
    annual = pd.read_csv(FIG / "Fig2_annual_MEHI_summary.csv")
    y2000 = annual.loc[annual["year"] == 2000].iloc[0]
    y2023 = annual.loc[annual["year"] == 2023].iloc[0]
    check("Fig2 soft-buffered share 2023 (%)", y2023["soft_buffered_area_share_percent"], 90.09590243, 1e-6)
    check("Fig2 hard-edge share 2000 (%)", y2000["hard_urban_edge_area_share_percent"], 0.67840562, 1e-6)
    check("Fig2 hard-edge share 2023 (%)", y2023["hard_urban_edge_area_share_percent"], 1.49963170, 1e-6)
    ratio = y2023["hard_urban_edge_area_share_percent"] / y2000["hard_urban_edge_area_share_percent"]
    check("Fig2 2023/2000 hard-edge ratio", ratio, 2.21052370, 1e-6)
    maps = pd.read_csv(FIG / "Fig2_edge_state_maps_2000_2023.csv.gz")
    check("Fig2 map rows 2000", int((maps["year"] == 2000).sum()), 115462)
    check("Fig2 map rows 2023", int((maps["year"] == 2023).sum()), 117546)
    check("Fig2 country/location coding", set(maps["location_code"].dropna().unique()), FINAL_UNITS)


def verify_fig3() -> None:
    data = pd.read_csv(FIG / "Fig3_matched_neighbourhoods.csv.gz")
    check("Fig3 matched neighbourhoods", len(data), 79509)
    classes = np.select(
        [data["pMEHI"] < -0.005, data["pMEHI"] > 0.005],
        ["softening", "hardening"],
        default="near_neutral",
    )
    shares = pd.Series(classes).value_counts(normalize=True) * 100
    check("Fig3 softening share (%)", shares["softening"], 44.50565345, 1e-6)
    check("Fig3 near-neutral share (%)", shares["near_neutral"], 14.94044699, 1e-6)
    check("Fig3 hardening share (%)", shares["hardening"], 40.55389956, 1e-6)
    weighted_mean = np.average(data["pMEHI"], weights=data["weight_m2"])
    check("Fig3 weighted mean pMEHI", weighted_mean, -0.0002084834, 1e-10)
    check("Fig3 country/location coding", set(data["GU_A3"].dropna().unique()), FINAL_UNITS)


def verify_fig4_and_fig5() -> None:
    pathways = pd.read_csv(FIG / "Fig4_area_pMEHI_pathways.csv")
    check("Fig4 country/location units", len(pathways), 19)
    ind = pathways.loc[pathways["GU_A3"] == "IND"].iloc[0]
    nga = pathways.loc[pathways["GU_A3"] == "NGA"].iloc[0]
    check("Fig4 India mangrove-area change (%)", ind["common_neighbourhood_mangrove_area_change_percent"], 97.6680, 1e-4)
    check("Fig4 Nigeria pMEHI", nga["pMEHI_aw"], 0.13635024, 1e-8)

    routes = pd.read_csv(FIG / "Fig5_global_route_shares.csv").set_index("route")
    replacement = routes.loc["built_up_replacement_route", "share_of_hardening_weight_percent"]
    combined = (
        routes.loc["proximity_only_route", "share_of_hardening_weight_percent"]
        + routes.loc["vegetation_buffer_loss_route", "share_of_hardening_weight_percent"]
    )
    check("Fig5 direct built-up replacement (%)", replacement, 4.33018925, 1e-6)
    check("Fig5 proximity plus vegetation-buffer loss (%)", combined, 72.80874788, 1e-6)
    grid = pd.read_csv(SUP / "Fig5_grid_route_classification.csv.gz", usecols=["GU_A3"])
    check("Fig5 country/location coding", set(grid["GU_A3"].dropna().unique()), FINAL_UNITS)


def verify_fig6() -> None:
    trajectory = pd.read_csv(FIG / "Fig6_global_trajectory.csv")
    final = trajectory.loc[trajectory["year"] == 2100].set_index("SSP")
    targets = {"SSP1": 0.0445884861, "SSP2": 0.1042439640, "SSP3": 0.0597968160, "SSP5": 0.1641945627}
    for ssp, expected in targets.items():
        check(f"Fig6 {ssp} 2100 Delta_sMEHI", final.loc[ssp, "Delta_sMEHI"], expected, 1e-10)
    ranking = pd.read_csv(FIG / "Fig6_country_ranking_all19_SSP235_2100.csv")
    check("Fig6 ranking rows", len(ranking), 57)
    check("Fig6 ranking country/location units", ranking["country"].nunique(), 19)
    zeros = set(map(tuple, ranking.loc[ranking["Delta_sMEHI"] == 0, ["country", "SSP"]].to_numpy()))
    check("Fig6 explicit zero combinations", zeros, {("BHR", "SSP2"), ("BHR", "SSP3"), ("GHA", "SSP5")})
    check("Fig6 country/location coding", set(ranking["country"].dropna().unique()), FINAL_UNITS)
    maps = pd.read_csv(FIG / "Fig6_maps_SSP2_SSP5_2100.csv.gz", usecols=["Delta_sMEHI_SSP2", "Delta_sMEHI_SSP5"])
    check("Fig6 SSP2 negative increments", int((maps["Delta_sMEHI_SSP2"] < -1e-12).sum()), 0)
    check("Fig6 SSP5 negative increments", int((maps["Delta_sMEHI_SSP5"] < -1e-12).sum()), 0)


def verify_classification() -> None:
    confusion = pd.read_csv(SUP / "landcover_confusion_matrix.csv").iloc[:6, 1:7].to_numpy(dtype=int)
    n = int(confusion.sum())
    oa = float(np.trace(confusion) / n)
    expected_agreement = float((confusion.sum(axis=1) * confusion.sum(axis=0)).sum() / n**2)
    kappa = (oa - expected_agreement) / (1 - expected_agreement)
    check("Land-cover validation samples", n, 9811)
    check("Land-cover overall accuracy", oa, 0.8824788503, 1e-10)
    check("Land-cover Kappa", kappa, 0.8518206481, 1e-10)
    annual = pd.read_csv(SUP / "landcover_accuracy_by_year.csv")
    year_metrics = annual.drop_duplicates("year").set_index("year")
    check("Land-cover 2000 overall accuracy (%)", year_metrics.loc[2000, "overall_accuracy_percent"], 84.49714536671059, 1e-9)
    check("Land-cover 2023 overall accuracy (%)", year_metrics.loc[2023, "overall_accuracy_percent"], 89.98628257887518, 1e-9)


def main() -> None:
    verify_fig2()
    verify_fig3()
    verify_fig4_and_fig5()
    verify_fig6()
    verify_classification()
    if FAILURES:
        print(f"\n{len(FAILURES)} verification check(s) failed: {', '.join(FAILURES)}", file=sys.stderr)
        raise SystemExit(1)
    print("\nAll reported-result checks passed.")


if __name__ == "__main__":
    main()
