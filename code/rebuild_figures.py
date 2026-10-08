"""Build compact diagnostic versions of final Figs. 2-6 from archive tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figure_source"
OUT = ROOT / "reproduced_figures" / "source_table_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2() -> None:
    maps = pd.read_csv(FIG / "Fig2_edge_state_maps_2000_2023.csv.gz")
    comp = pd.read_csv(FIG / "Fig2_edge_state_composition.csv")
    hist = pd.read_csv(FIG / "Fig2_annual_MEHI_histogram.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))
    for ax, year, col, title in [
        (axes[0, 0], 2000, "MEHI", "MEHI 2000"), (axes[0, 1], 2023, "MEHI", "MEHI 2023"),
        (axes[1, 0], 2000, "hard_edge_share_percent_0p5deg_bin", "Hard-edge intensity 2000"),
        (axes[1, 1], 2023, "hard_edge_share_percent_0p5deg_bin", "Hard-edge intensity 2023"),
    ]:
        d = maps.loc[maps["year"] == year]
        sc = ax.scatter(d["lon"], d["lat"], c=d[col], s=0.25, cmap="coolwarm", rasterized=True)
        ax.set_title(title); ax.set_xlim(-180, 180); ax.set_ylim(-45, 38); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    pivot = comp.pivot(index="year", columns="edge_state_label", values="area_weighted_share_percent")
    pivot.plot(kind="bar", stacked=True, ax=axes[0, 2], color=["#B64035", "#D8C9AD", "#2B8C7F"])
    axes[0, 2].set_ylabel("Area-weighted share (%)"); axes[0, 2].legend(fontsize=6)
    for year, d in hist.groupby("year"):
        axes[1, 2].plot(d["MEHI_bin_mid"], d["area_weight_sum"] / d["area_weight_sum"].sum(), label=str(year))
    axes[1, 2].set_xlabel("MEHI"); axes[1, 2].set_ylabel("Area-weighted density"); axes[1, 2].legend(frameon=False)
    fig.tight_layout(); save(fig, "Fig2_diagnostic_reconstruction")


def fig3() -> None:
    data = pd.read_csv(FIG / "Fig3_matched_neighbourhoods.csv.gz")
    country = pd.read_csv(FIG / "Fig3_country_summary.csv").sort_values("weighted_median_pMEHI")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    rng = np.random.default_rng(30)
    for i, (code, d) in enumerate(data.groupby("GU_A3")):
        sample = d.sample(min(len(d), 1200), random_state=30)
        axes[0].scatter(np.full(len(sample), i) + rng.normal(0, 0.08, len(sample)), sample["pMEHI"], s=1, alpha=0.25)
    axes[0].axhline(0, color="0.6", lw=0.6); axes[0].set_ylabel("pMEHI"); axes[0].set_title("Matched neighbourhoods")
    axes[1].hist(data["pMEHI"], bins=120, range=(-0.45, 0.45), weights=data["weight_m2"], color="#557A95")
    axes[1].axvline(0, color="0.4", lw=0.7); axes[1].set_xlabel("pMEHI"); axes[1].set_title("Global distribution")
    y = np.arange(len(country))
    axes[2].errorbar(country["weighted_median_pMEHI"], y, xerr=[country["weighted_median_pMEHI"]-country["weighted_p25"], country["weighted_p75"]-country["weighted_median_pMEHI"]], fmt="o", ms=3)
    axes[2].set_yticks(y, country["GU_A3"]); axes[2].axvline(0, color="0.6", lw=0.6); axes[2].set_title("Country/location summaries")
    fig.tight_layout(); save(fig, "Fig3_diagnostic_reconstruction")


def fig4() -> None:
    paths = pd.read_csv(FIG / "Fig4_area_pMEHI_pathways.csv")
    counts = pd.read_csv(FIG / "Fig4_pathway_counts.csv")
    comp = pd.read_csv(FIG / "Fig4_global_component_contributions.csv")
    heat = pd.read_csv(FIG / "Fig4_country_component_heatmap.csv").set_index("GU_A3")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for name, d in paths.groupby("pathway"):
        axes[0, 0].scatter(d["common_neighbourhood_mangrove_area_change_percent"], d["pMEHI_aw"], label=name, s=30)
    axes[0, 0].axhline(0, color="0.6", lw=0.6); axes[0, 0].axvline(0, color="0.6", lw=0.6); axes[0, 0].legend(fontsize=6)
    axes[0, 0].set_xlabel("Mangrove-area change (%)"); axes[0, 0].set_ylabel("pMEHI")
    axes[0, 1].barh(counts["pathway"], counts["country_location_code_count"], color="#6C8E7B"); axes[0, 1].set_title("Pathway counts")
    axes[1, 0].bar(comp["component"], comp["value"], color=["#3B6FB6", "#B6493A", "#D3A227", "#2B8C7F"]); axes[1, 0].axhline(0, color="0.5", lw=0.6); axes[1, 0].set_ylabel("Contribution to pMEHI")
    cols = [c for c in ["P_2023_aw", "C_2023_aw", "B_2023_aw", "S_2023_aw"] if c in heat.columns]
    im = axes[1, 1].imshow(heat[cols], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(cols)), [c[0] for c in cols]); axes[1, 1].set_yticks(range(len(heat)), heat.index, fontsize=6); fig.colorbar(im, ax=axes[1, 1], fraction=0.04)
    fig.tight_layout(); save(fig, "Fig4_diagnostic_reconstruction")


def fig5() -> None:
    transition = pd.read_csv(FIG / "Fig5_transition_matrix.csv")
    routes = pd.read_csv(FIG / "Fig5_global_route_shares.csv").sort_values("route_order")
    assoc = pd.read_csv(FIG / "Fig5_country_association.csv")
    matrix = transition.pivot(index="from_class", columns="to_class", values="share_of_all_transition_percent").fillna(0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    im = axes[0].imshow(matrix, cmap="YlOrRd"); axes[0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right", fontsize=6); axes[0].set_yticks(range(len(matrix.index)), matrix.index, fontsize=6); axes[0].set_title("Land-cover transitions"); fig.colorbar(im, ax=axes[0], fraction=0.04, label="All observed transition area (%)")
    axes[1].barh(routes["route_label"], routes["share_of_hardening_weight_percent"], color="#6C8E7B"); axes[1].invert_yaxis(); axes[1].set_xlabel("Hardening weight (%)")
    axes[2].scatter(assoc["soft_buffer_to_built_share_changed_area"], assoc["country_weighted_mean_pMEHI"], s=25)
    for _, r in assoc.iterrows(): axes[2].annotate(r["GU_A3"], (r["soft_buffer_to_built_share_changed_area"], r["country_weighted_mean_pMEHI"]), fontsize=5)
    axes[2].set_xlabel("Soft-buffer to built share"); axes[2].set_ylabel("Country/location pMEHI")
    fig.tight_layout(); save(fig, "Fig5_diagnostic_reconstruction")


def fig6() -> None:
    traj = pd.read_csv(FIG / "Fig6_global_trajectory.csv")
    maps = pd.read_csv(FIG / "Fig6_maps_SSP2_SSP5_2100.csv.gz")
    rank = pd.read_csv(FIG / "Fig6_country_ranking_all19_SSP235_2100.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ssp, d in traj.groupby("SSP"):
        axes[0, 0].plot(d["year"], d["Delta_sMEHI"], marker="o", label=ssp)
    axes[0, 0].set_ylabel("Delta sMEHI"); axes[0, 0].legend(frameon=False)
    for ax, ssp in [(axes[0, 1], "SSP2"), (axes[1, 0], "SSP5")]:
        sc = ax.scatter(maps["lon"], maps["lat"], c=maps[f"Delta_sMEHI_{ssp}"], s=0.3, cmap="viridis", rasterized=True)
        ax.set_xlim(-180, 180); ax.set_ylim(-45, 38); ax.set_title(f"{ssp}, 2100"); fig.colorbar(sc, ax=ax, fraction=0.035)
    pivot = rank.pivot(index="country", columns="SSP", values="Delta_sMEHI").sort_values("SSP5")
    pivot.plot(kind="barh", ax=axes[1, 1], width=0.8); axes[1, 1].set_xlabel("Delta sMEHI"); axes[1, 1].legend(fontsize=6)
    fig.tight_layout(); save(fig, "Fig6_diagnostic_reconstruction")


def main() -> None:
    fig2(); fig3(); fig4(); fig5(); fig6()
    print(f"Wrote diagnostic figures to {OUT}")


if __name__ == "__main__":
    main()
