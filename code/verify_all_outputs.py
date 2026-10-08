"""Compare independently rerun release analyses with the adopted source tables."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def compare(name,a,b,keys,columns):
 a=a.set_index(keys).sort_index();b=b.set_index(keys).loc[a.index]
 assert not a.index.duplicated().any() and not b.index.duplicated().any()
 delta=0.0
 for c in columns:
  np.testing.assert_allclose(a[c],b[c],rtol=0,atol=1e-12,equal_nan=True,err_msg=name+' '+c)
  if len(a):delta=max(delta,float((a[c]-b[c]).abs().max()))
 checks.append({'comparison':name,'rows':len(a),'columns':columns,'max_absolute_difference':delta})
fig=ROOT/'data/figure_source';out=ROOT/'outputs/scenario';sup=ROOT/'data/supporting'
compare('Sixteen global future scenario values',pd.read_csv(out/'future_global_smehi_summary.csv'),pd.read_csv(fig/'Fig6_global_trajectory.csv'),['SSP','year'],['Delta_sMEHI','hard_edge_fraction_future','soft_to_mixed_or_hard_share','unmet_allocation_percent'])
expected=pd.read_csv(fig/'Fig6_country_ranking_all19_SSP235_2100.csv').rename(columns={'hard_edge_fraction':'hard_edge_fraction_future'})
actual=pd.read_csv(out/'future_country_smehi_summary.csv').query('year==2100 and SSP in ["SSP2","SSP3","SSP5"]')
compare('All 57 country-location scenario summaries',actual,expected,['country','SSP','year'],['n_grids','Delta_sMEHI','hard_edge_fraction_future','soft_to_mixed_or_hard_share'])
compare('Seeded summary-sensitivity percentiles',pd.read_csv(out/'uncertainty_summary.csv'),pd.read_csv(sup/'Fig6_uncertainty_summary.csv'),['SSP','year','lambda','eligibility'],['realization_count','global_mean_Delta_sMEHI','p05','p50','p95'])
compare('Historical reconstruction with explicit three-category direction',pd.read_csv(out/'backtest_metrics.csv'),pd.read_csv(sup/'Fig6_backtest_metrics.csv'),['target_year'],['grid_level_MEHI_MAE','country_level_spearman_rho_Delta_MEHI','hard_edge_fraction_error','direction_agreement_3class','neutral_band_Delta_MEHI','top_five_hardening_country_overlap','number_of_grids'])
for p in (ROOT/'reproduced/analysis').glob('*.csv'):
 ref=ROOT/'data/revision'/p.name
 if not ref.exists():continue
 a=pd.read_csv(p);b=pd.read_csv(ref)
 pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=0,atol=1e-12)
 checks.append({'comparison':'Reproduced '+p.name,'rows':len(a),'columns':len(a.columns)})
report={'status':'passed','absolute_numeric_tolerance':1e-12,'comparisons':checks,'scope':'Current classification/composition and joint analyses, all global scenario values, all published country rankings, seed-based summary sensitivity and historical reconstruction.'}
(ROOT/'reproduced/qa/numerical_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
