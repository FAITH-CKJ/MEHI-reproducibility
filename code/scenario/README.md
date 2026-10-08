# Fig. 5 scenario workflow

The final Fig. 5 uses a country-aggregated, fixed-domain 150-m exposure screen.
Gao--Pesaresi 1-km urban-fraction increments are aligned to the observed 2023
built fraction and summed by country/location and scenario-year. Each total is
distributed once across that unit's fixed 150-m MEHI neighbourhoods using
effective external support, edge-to-built distance and baseline built share.

`scenario_core.py` implements the allocation and the validity-aware component
update. Stored composition components are retained where external support is at
most 1 m2, and stored contact is retained where mangrove support is at most
1 m2. This is the same neutral-component convention used by the observed MEHI
geodatabase. `run_scenario_projection.py` adds future aggregation, historical
backtesting and the seeded global scenario-summary uncertainty envelope.

Run the equation and support checks with:

```powershell
python code\scenario\run_scenario_projection.py self-test
```

For a full spatial rerun, prepare the provider-managed Gao--Pesaresi grids and
the analysis-ready MEHI edge tables described in `workflow/INPUT_SCHEMAS.md`,
then set package-relative paths in a copy of `example_config.json`.
