"""Summarize the final Fig. 5 trajectory and country/location tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "figure_source"


def main() -> None:
    trajectory = pd.read_csv(SOURCE / "Fig6_global_trajectory.csv")
    ranking = pd.read_csv(SOURCE / "Fig6_country_ranking_all19_SSP235_2100.csv")
    final = trajectory.loc[trajectory["year"] == 2100, ["SSP", "Delta_sMEHI", "hard_edge_fraction_future", "soft_to_mixed_or_hard_share"]]
    zeros = ranking.loc[ranking["Delta_sMEHI"] == 0, ["country", "SSP", "n_grids"]]
    print("Global 2100 scenario summaries")
    print(final.to_string(index=False))
    print("\nExplicit zero-increment country/location-pathway combinations (n_grids is the final ranking-table support field)")
    print(zeros.to_string(index=False))


if __name__ == "__main__":
    main()
