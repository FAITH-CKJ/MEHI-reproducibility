# Input schemas for spatial, classification and scenario workflows

All distances and areas used by the spatial and scenario routines are measured
in metres and square metres. Country/location codes use the final 19-unit coding
recorded in the manuscript and supporting tables.

## Four-date land-cover mosaics

`code/spatial/build_mehi_arcpy.py` reads polygon feature classes for 2000, 2010,
2020 and 2023. The default file template is `mosaic{year}join.shp`.

| Field | Type | Meaning |
|---|---|---|
| `GU_A3` | text | final country/location code |
| `gridcode` | integer | 1 mangrove; 2 tidal flat; 3 non-mangrove vegetation; 4 water; 5 built-up land; 6 other |
| geometry | polygon | 30-m land-cover mosaic polygons with a defined source CRS |

The script projects these mosaics to ESRI:54009, uses the existing dissolved
mangrove-boundary vertices with densification disabled, constructs the aligned 150-m edge grid and writes
annual MEHI and endpoint-matched pMEHI feature classes and CSV tables.

## Sample registers and classifier inputs

The training register contains `system.index`, `Landcover`, `Year` and `.geo`
(GeoJSON Point longitude/latitude in geographic degrees). The map-validation
assignment table stores `sample_row`, `reference_class`, `year` and
`reference_coordinate` for every validation record. Their order matches the
study sample record, supporting counts and coordinate-year membership.

For retraining or application, user-supplied predictor CSV tables contain `Landcover` followed by the
17 numerical predictor columns in this exact order:

`red, green, blue, nir, swir1, swir2, lst, ndvi, evi, ndbi, ndpi, bsi, ndwi, mndwi, cmri, mmri, dtw`.

`application.py` takes a co-registered 17-band raster in the same order and
writes the six-class byte raster. Class code 0 is reserved for source nodata.

## Present-edge baseline table

`code/scenario/run_scenario_projection.py` reads one row per fixed edge
neighbourhood.

| Field | Meaning |
|---|---|
| `edge_grid_id` | stable aligned edge-neighbourhood identifier |
| `country` | final country/location code |
| `A_OBS_EXT_M2` | observed external-support area used for Fig. 5 aggregation |
| `A_M_M2` | mangrove area |
| `A_BUILT_M2` | built-up external area |
| `A_SOFT_M2` | tidal-flat, water and non-mangrove-vegetation external area |
| `A_OTHER_M2` | other external area |
| `A_CONTACT_M2` | raw polygon-overlay mangrove area within the 30-m built-contact zone; the contact proportion is bounded to [0, 1] |
| `EDGE_DIST_M` | Minimum planar Near distance for each grid-clipped boundary line feature, weighted by feature length and capped at 1,000 m |
| `MEHI` | observed baseline MEHI |
| `P_COMP`, `C_COMP`, `B_COMP`, `S_COMP` | stored baseline components used by the validity-aware scenario update |

## Country-aggregated 150-m allocation

The scenario workflow sums baseline-aligned 1-km built-area increments by
country/location and scenario-year. It then joins each country/location total
to the present-edge baseline and distributes demand across its fixed 150-m
neighbourhoods. These aggregated support fields form the operational input to
the final Fig. 5 calculation.

Effective support is `0.7 * A_SOFT_M2 + A_OTHER_M2`. Allocation weights combine
this support with `EDGE_DIST_M` and baseline built share. Requested area is
normalized within country/location and capped by effective support. Component
updates preserve the stored composition values when valid external support is
at most 1 m2 and preserve stored contact when mangrove support is at most 1 m2.

## Gao--Pesaresi scenario-cell table

`prepare_ssp_cells.py` samples provider netCDF grids on a support table containing
`scenario_cell_id`, `country`, `sampled_area_m2`, `U_obs_2023`, `lon` and `lat`.
Its lookup table contains `SSP`, `year`, `baseline_relative_path`,
`baseline_variable`, `target_relative_path`, `target_variable` and `value_scale`.
The resulting scenario table adds `U_SSP_2020` and `U_SSP_y`; the projection
code calculates the baseline-aligned fraction and built-area demand.

## Historical backtest increments

Each backtest job links a 2000 edge baseline to an observed target-edge table.
The tables contain `edge_grid_id`, `country`, the component and support fields
listed above, `A_BUILT_M2` and `MEHI`. Positive observed built-area increments
are passed through the same support-valid component update, and reconstructed
states are compared with 2010, 2020 or 2023 observations.

Polygon-overlay area fields retain the source measurements. The observed
numerical excess over a nominal 150 m cell is below 0.01 m2, consistent
with the source geometry precision. Contact proportions apply their
documented [0, 1] bound before C is calculated.

## Transition-route summaries

Route statistics use 79,505 neighbourhoods with valid classified support at
both endpoints, positive minimum-endpoint mangrove-area weight and positive
retained paired-transition area. The complete metric and forest-area analyses
use all 79,509 matched neighbourhoods. Individual paired overlay pieces enter
the transition table when their area exceeds 1 m2.

Global route shares use pMEHI > 0 (35,619 neighbourhoods); the transition matrix
and country route compositions use pMEHI > 0.005 (32,242 neighbourhoods).
The matrix denominator is all recorded transition area in that subset.
In the country route table, `n_positive_pmehi` counts pMEHI > 0; the plotted
country support is the sum of `grid_count`, which counts pMEHI > 0.005.
For the country association, `soft_buffer_to_built_share_changed_area` is the
minimum-endpoint-area weighted mean of the individual neighbourhood ratios
`soft_to_built_share_of_changed`. Each ratio uses all changed area within that
neighbourhood. The same area weights aggregate pMEHI. This yields 19 paired
country/location summaries.

`data/supporting/route_classification_flow_inputs.csv.gz` supplies the nine
additional fields needed to rerun the complete route rule. Join it one-to-one
to the route table on `GU_A3, GRID_ID`. `delta_other_share` is the endpoint
change in the fraction of other land cover, taken from the complete endpoint
class-change table. The remaining fields are paired-transition areas in m2:
`tidalflat_to_built_m2`, `nonmangroveveg_to_built_m2`, `water_to_built_m2`,
`soft_to_other_m2`, `other_to_built_m2`, and `mangrove_to_built_m2`. They are
summed from retained paired-overlay pieces under the same >1 m2 rule.
`code/verify_route_statistics.py` joins these inputs, reclassifies all 79,505
records at thresholds 0.20, 0.30 and 0.40, and compares labels, counts and
weights with the supplied summaries. The 0.30 rule reproduces the primary
route labels. The sum of the three nonnegative soft-to-built flows already
dominates each constituent; the implementation compares that sum with the
independent competing flows and retains the exact replacement-ratio threshold.
