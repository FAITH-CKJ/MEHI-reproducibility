# Sensitivity table guide

| Table | Scientific question | Recomputed from | Verification code |
|---|---|---|---|
| `dataset_sensitivity_summary.csv` | How do mapped urban-interface area and sampled-domain coverage vary among the GMW v3, HGMF 2020 and union mangrove envelopes? | Source-product overlays and geodesic pixel-area summaries documented in Supplementary Note 2 | Supplementary Note 2 dataset-sensitivity workflow and reported-value checks |
| `MEHI_component_weight_sensitivity.csv` | Do endpoint ranks and classes persist when one of P, C, B or S is doubled or halved, using the declared `abs(pMEHI) <= 0.0025` agreement band? | `Fig5_grid_route_classification.csv.gz` and `Fig4_area_pMEHI_pathways.csv` | `code/sensitivity/recompute_sensitivity.py` |
| `MEHI_cutpoint_sensitivity.csv` | Does the rise in hard-edge share persist at annual MEHI cut-points of 0.25, 0.33 and 0.40? | Fig. 1 annual histogram and exact 0.33 state-composition tables | `code/sensitivity/recompute_sensitivity.py` |
| `pMEHI_threshold_sensitivity.csv` | How do endpoint direction shares change at pMEHI bands of 0.0025, 0.005 and 0.010? | `Fig5_grid_route_classification.csv.gz` | `code/sensitivity/recompute_sensitivity.py` |
| `MEHI_scale_sensitivity.csv` | How do annual hard-edge share, country/location pMEHI and common support vary at 120, 150 and 180 m? | Scale-specific MEHI and pMEHI geodatabase summaries | `code/sensitivity/recompute_sensitivity.py` |
| `MEHI_scale_agreement_summary.csv` | Which correlation, pairwise agreement or all-scale agreement does each reported count represent? | Country/location records in `MEHI_scale_sensitivity.csv` and the joint pathway comparison | `code/sensitivity/recompute_sensitivity.py` |

The all-scale pathway count requires the same joint mangrove-area--pMEHI pathway at all three neighbourhood widths. Pairwise pMEHI sign counts are reported separately and are not substituted for this joint criterion.
