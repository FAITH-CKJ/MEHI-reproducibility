# Mangrove Edge Hardening Index: source data and reproducibility

**Urban hardening of mangrove edges extends beyond forest loss**  
Kangjie Cao, Xiaoxue Shen, Xin Luo, Wei Li and Ruili Li  
Communications Earth & Environment revision · COMMSENV-26-4948-T · v1.0.0

This deposit supplies the numerical evidence and executable analyses for the
manuscript: four-date neighbourhood attributes, matched area–edge trajectories,
figure source values, sample registers, classifier weights, map-validation assignments,
classification sensitivity and urban-growth scenarios. The fixed release is
[v1.0.0](https://github.com/FAITH-CKJ/MEHI-reproducibility/releases/tag/v1.0.0).

## Start here

Use Python 3.11 in an environment with the packages in `code/requirements.txt`.
From this directory, run:

```sh
python -m pip install -r code/requirements.txt
python code/restore_tables.py
python code/verify_reported_results.py
python code/verify_current_validation.py
python code/audit_sample_tables.py
python code/verify_metric_conventions.py
python code/audit_result_tables.py
python code/verify_route_statistics.py
python code/sensitivity/recompute_sensitivity.py
python code/evaluate_temporal_uncertainty.py
python code/summarize_joint_area_edge.py
python code/verify_revision_analysis.py
python code/scenario/run_scenario_projection.py run-all --config code/scenario/example_config.json
python code/verify_all_outputs.py
python code/landcover_classifier/inference_ensemble.py --self-test
python code/redraw_cee_r1_figures.py --figures 1 2 5
```

Restoration writes ordinary CSV/CSV.gz tables into `data/` and checks every
column against its stored checksum. The deposit preserves all numeric bits,
text values, row order and column order. Repeated attributes are stored once;
exact arithmetic residuals retain the original floating-point values when
columns are reconstructed. `workflow/STORAGE_FORMAT.md` describes the format,
and `code/lossless_frame_codec.py` supplies its reader and writer.

Numerical results are written under `reproduced/`, scenario results under
`outputs/scenario/`, and rebuilt figures under `reproduced_figures/`. Allow
approximately 2 GB of disk space for expanded tables and generated outputs.
Restoration and numerical checks run on a CPU. The seven supplied PyTorch
checkpoints also support CPU inference.

## Data and result map

| Evidence | Source tables after restoration | Reproduction |
|---|---|---|
| Annual edge states, 2000/2010/2020/2023 | `data/revision/classification_sensitivity_inputs_150m.csv.gz` (474,003 rows); `data/figure_source/Fig2_*` | `evaluate_temporal_uncertainty.py`; current Figure 1 |
| Matched forest area and edge change | `data/revision/canonical_matched_support.csv.gz` (79,509 rows); `data/figure_source/Fig3_*` and `Fig4_*` | `summarize_joint_area_edge.py`; current Figures 2–3 |
| Mapped transition routes | `data/supporting/Fig5_grid_route_classification.csv.gz` (79,505 eligible rows); `data/figure_source/Fig5_*` | `code/transition/route_analysis.py`; current Figure 4 |
| Conditional future exposure and backtests | `data/scenario/`; `data/figure_source/Fig6_*`; `data/supporting/Fig6_*` | `code/scenario/`; current Figure 5 |
| Classifier and map evaluation | `data/samples/`; `data/validation_current/`; `code/landcover_classifier/models/` | `verify_current_validation.py`; model card |
| Additional scale, envelope, threshold and weight comparisons | `data/supporting/`; `data/dictionaries/` | source summaries and `code/sensitivity/recompute_sensitivity.py` |

Source-table figure prefixes retain stable analytical identifiers. The current
manuscript figure mapping is in `workflow/FIGURE_NUMBERING.txt`; the concept
diagram appears as Figure 6 in Methods. The deposit contains the numerical
figure sources; publication figure files accompany the manuscript.

The training register documents 42,795 records through their identifiers,
reference labels, observation years and coordinates. The map-validation table
contains the corresponding reference fields for all 10,657 validation records,
together with final-map assignments. These fields support sample accounting,
coordinate-year membership and final-map validation. The classifier code
specifies the 17-variable schema for user-supplied predictor tables.
The validation table serves checkpoint selection and final-map evaluation.
Deduplication and map-coverage screening yield 9,811 unique covered observations
for final-map agreement (88.247885%, kappa 0.8518206481). The complete record
flow, annual confusion matrices and class metrics are supplied.

## Spatial inputs, versions and attribution

`workflow/REVISION_DATA_GUIDE.txt` and the dictionaries define units, weights,
thresholds and table roles. `workflow/EXTERNAL_DATA.md` gives provider records
and dataset identifiers. The numerical workflows use the included derived
tables. Full spatial construction uses the inventoried four annual
full-resolution land-cover mosaics and ArcGIS Pro/arcpy; the classifier and
spatial source code, predictor order and input schemas are included.

The verified analytical environment uses Python 3.11.7, NumPy 1.26.4,
pandas 2.1.4, SciPy 1.11.4, Matplotlib 3.8.0 and PyTorch 2.9.1.
Supplementary Note 2 and the model card describe task-specific versions and
measured CPU inference resources. `metadata/verification.json` records the
numerical checks performed on a clean restoration of this release.

When using these materials, cite the manuscript and this versioned deposit.
Provider products retain their original attribution and use terms; Natural
Earth basemap terms are at
https://www.naturalearthdata.com/about/terms-of-use/ .
`CITATION.cff` identifies the deposit authors and version; `CHECKSUMS.tsv`
records the SHA-256 and byte size of every distributed file.

The complete spatial archive comprises the four-date land-cover vector mosaics and the data used to generate them. Together, these files exceed 1 TB and are available from the corresponding author, Ruili Li (liruili@pkusz.edu.cn), who will arrange transfer of the requested files. The public deposit provides the compact numerical inputs and source tables for reproducing the reported results and figures.
