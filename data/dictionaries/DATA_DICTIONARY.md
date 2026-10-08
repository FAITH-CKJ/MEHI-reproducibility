# Data dictionary

## Core metric fields

| Field | Meaning | Unit / range |
|---|---|---|
| `P` | Built-up proximity pressure | -1 to 1 |
| `C` | Direct built-up contact within 30 m | -1 to 1 |
| `B` | Built-up dominance in the external matrix | -1 to 1 |
| `S` | Soft-buffer deficit | -1 to 1 |
| `MEHI` | Equal-weight mean of P, C, B and S | -1 to 1 |
| `pMEHI` | Endpoint change `(MEHI_2023 - MEHI_2000) / 2` | -1 to 1 |
| `weight_m2` | Analysis-specific matched-neighbourhood support; see table-specific definitions below | m2 |
| `area_weight` | Mapped mangrove area (`A_M_M2`) within an annual neighbourhood | m2 |
| `GU_A3` / `location_code` | Three-letter country/location code used for reporting | text |
| `GRID_ID` / `GRID_UID` | Stable neighbourhood identifier within the relevant table | text / integer |

## State and change classes

- Annual state: soft-buffered when `MEHI < -0.33`, mixed when `-0.33 <= MEHI <= 0.33`, and hard urban when `MEHI > 0.33`.
- Endpoint direction: softening when `pMEHI < -0.005`, near-neutral when `abs(pMEHI) <= 0.005`, and hardening when `pMEHI > 0.005`.

## Table-specific weight semantics

- Fig. 2 uses equal 150-m grid-cell support (22,500 m2 per matched neighbourhood).
- Fig. 3 component and Fig. 4 route summaries use persistent mangrove-area support within matched edge neighbourhoods: `min(A_M_2000_m2, A_M_2023_m2)`.
- Fig. 1 annual summaries use `A_M_M2`; Fig. 5 scenario summaries use observed 2023 external-support area (`A_OBS_EXT_M2`).

## Fig. 5 fields

| Field | Meaning |
|---|---|
| `Delta_sMEHI` | Scenario-induced increment relative to the fixed 2023 edge baseline; theoretical range -2 to 2 |
| `sMEHI` | Scenario MEHI state after allocation |
| `hard_edge_future_flag` | Indicator for a future hard-edge state |
| `soft_to_mixed_or_hard_share` | Share of 2023 soft support moving into mixed or hard state |
| `unmet_allocation_percent` | Scenario urban increment not allocated under eligibility constraints |

Fig. 5 weights `Delta_sMEHI` by observed external support (`A_OBS_EXT_M2`) on
the fixed 2023 neighbourhood domain. Future hard-edge and soft-to-mixed/hard
fractions are neighbourhood shares on that same fixed domain. The spatial
screen first sums 1-km scenario demand by country/location, then distributes it
across 150-m neighbourhoods using effective support, distance and baseline
built share. Composition and contact retain their stored baseline values where
their respective supports are at most 1 m2.

The shared-grid Fig. 5 map table uses `_SSP2` and `_SSP5` suffixes for scenario-specific values.

## Sensitivity-analysis fields

`dataset_sensitivity_summary.csv` compares the GMW v3 envelope, the HGMF 2020 envelope and their union on a common 30-m analysis grid. The table reports global mangrove-envelope area, the mangrove area within 2 km of qualifying GUB 2018 urban boundaries, the corresponding sampled MEHI-domain area, sampled-domain coverage and differences from the primary GMW v3 definition. Areas are in km2 and coverage fields are percentages or percentage-point differences as named.

`MEHI_scale_sensitivity.csv` is a normalized long table with one scientific observation per row. `record_type` separates annual hard-edge shares, country/location pMEHI values and common-domain support. `scale_m` gives the evaluated neighbourhood width, `reference_scale_m` identifies the 150-m comparison scale where applicable, and `comparison` uses explicit labels such as `120_vs_150`, `180_vs_150` or `reference_150`.

`MEHI_scale_agreement_summary.csv` records pairwise correlations and sign agreement separately from agreement required across all three scales. `all_three_scales` therefore means that a country/location retained the same pMEHI sign, or the same joint mangrove-area--pMEHI pathway class, at 120, 150 and 180 m. The reported all-scale counts are 18 of 19 for pMEHI sign and 17 of 19 for joint pathway class.

The component-weight table uses the narrow endpoint band `abs(pMEHI) <= 0.0025` for direction and joint-pathway agreement. The primary `0.005` band and the wider `0.010` band are evaluated separately in `pMEHI_threshold_sensitivity.csv`. `code/sensitivity/recompute_sensitivity.py` reconstructs these summaries from the included analysis-ready source tables.

## Land-cover validation fields

The pooled confusion matrix uses mapped classes in rows and reference classes in columns. Producer's accuracy divides the diagonal by the reference total; user's accuracy divides the diagonal by the mapped total. The current input contains 53,452 records: 42,795 for training and 10,657 for validation. The validation table also served checkpoint selection. Final-map statistics count 9,811 unique covered coordinate-year points, after removing 192 concordant duplicate records and recording 654 out-of-coverage records. The point assignments and full annual matrices are in data/validation_current.

## Sample registers

`data/samples/training_samples.csv` retains four fields: `system.index` is
the source record identifier; `Landcover` is the reference class (1 mangrove,
2 tidal flat, 3 non-mangrove vegetation, 4 water, 5 built-up, 6 other); `Year`
is 2000, 2010, 2020 or 2023; and `.geo` is a GeoJSON Point with longitude
and latitude in geographic degrees. Validation reference fields are stored
once in `data/validation_current/map_validation_predictions.csv.gz`, as
`sample_row`, `reference_class`, `year` and `reference_coordinate`. Row order
preserves the source sample order. Consistently labelled repeated
coordinate-year records remain part of the training/validation input counts;
final-map accuracy counts each covered coordinate-year once.

Route statistics use 79,505 neighbourhoods with valid endpoint classification
support and retained paired-transition observations. The positive-pMEHI
route denominator is 35,619; the pMEHI > 0.005 denominator is 32,242.
Recorded intersections must exceed the original 1 m2 fragment threshold.
The general MEHI, endpoint-threshold and component-weight analyses retain
all 79,509 matched neighbourhoods through the canonical and annual tables.
