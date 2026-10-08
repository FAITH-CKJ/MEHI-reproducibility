"""Check the semantics of the numerical inputs used by the reported results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def read(rel):return pd.read_csv(ROOT/rel,float_precision='round_trip')
def finite(frame):assert np.isfinite(frame.select_dtypes('number').to_numpy()).all()
def unique(frame,keys):assert not frame.duplicated(keys).any(),keys
def bounds(series,lo,hi,tol=1e-12):assert series.between(lo-tol,hi+tol).all(),series.name

a=read('data/revision/classification_sensitivity_inputs_150m.csv.gz')
assert len(a)==474003 and not a.isna().any().any()
finite(a);unique(a,['YEAR','GU_A3','GRID_ID'])
assert a.YEAR.isin([2000,2010,2020,2023]).all()
assert a.GU_A3.nunique()==19
for c in ['A_M_M2','A_OBS_EXT_M2']+[c for c in a if c.startswith('area_')]:
    # Source polygon-overlay areas retain their measured floating-point precision.
    bounds(a[c],0,22500.01)
bounds(a.EDGE_DIST_M,0,1000)
for c in ['P_COMP','C_COMP','B_COMP','S_COMP','MEHI']:bounds(a[c],-1,1)
np.testing.assert_allclose(a.MEHI,a[['P_COMP','C_COMP','B_COMP','S_COMP']].mean(axis=1),atol=1e-14,rtol=0)
np.testing.assert_array_equal(a.A_M_M2,a.area_mangrove_m2)
np.testing.assert_allclose(a.A_OBS_EXT_M2,a[[c for c in a if c.startswith('area_') and c!='area_mangrove_m2']].sum(axis=1),atol=1e-10,rtol=0)
checks.append({'table':'annual_neighbourhoods','rows':len(a),'years':[2000,2010,2020,2023],
               'checks':'Unique country-grid-year keys; finite measurements; nonnegative m2 areas; 0–1000 m distances; bounded components and their mean; class-area identities',
               'maximum_overlay_area_excess_m2':float(max(0,a.A_OBS_EXT_M2.max()-22500)),
               'area_note':'Polygon overlay measurements retain source precision; maximum cell-area excess is below 0.01 m2.'})
s=read('data/scenario/ssp_aligned_inputs.csv.gz');finite(s)
assert len(s)==135280 and not s.isna().any().any()
unique(s,['country','SSP','year','ssp_grid_id'])
assert set(s.SSP)=={'SSP1','SSP2','SSP3','SSP5'} and s.year.isin([2030,2050,2070,2100]).all()
assert (s.sampled_area_m2>0).all()
for c in ['U_obs_2023','U_SSP_2020','U_SSP_y']:bounds(s[c],0,1)
checks.append({'table':'scenario_inputs','rows':len(s),'checks':'Unique scenario-year-cell keys; positive sampled m2; finite urban fractions between 0 and 1; valid scenario and year domain'})
c=read('data/revision/canonical_matched_support.csv.gz');finite(c);unique(c,['GU_A3','GRID_ID'])
assert len(c)==79509 and not c.isna().any().any()
assert (c[['weight_m2','A_M_2000_m2','A_M_2023_m2']]>0).all().all()
bounds(c.pMEHI,-1,1)
np.testing.assert_array_equal(c.weight_m2,np.minimum(c.A_M_2000_m2,c.A_M_2023_m2))
end=a[a.YEAR.isin([2000,2023])].pivot(index=['GU_A3','GRID_ID'],columns='YEAR',values='MEHI')
matched=c.set_index(['GU_A3','GRID_ID'])
np.testing.assert_array_equal(matched.pMEHI,((end[2023]-end[2000])/2).loc[matched.index])
checks.append({'table':'matched_support','rows':len(c),'checks':'Positive persistent endpoint support; exact minimum-area weights; exact half-change from annual MEHI'})
r=read('data/supporting/Fig5_grid_route_classification.csv.gz');finite(r);unique(r,['GU_A3','GRID_ID'])
assert len(r)==79505 and not r.isna().any().any()
assert r.valid_for_stats.all() and r.observed_transition_area_m2.gt(0).all()
rr=r.set_index(['GU_A3','GRID_ID'])
np.testing.assert_allclose(rr.pMEHI,matched.loc[rr.index].pMEHI,atol=1e-12,rtol=0)
np.testing.assert_allclose(rr.weight_m2,matched.loc[rr.index].weight_m2,atol=1e-10,rtol=0)
for x in ['soft_to_built_share_of_changed','soft_to_built_share_of_observed']:bounds(r[x],0,1)
for x in ['delta_P','delta_C','delta_B','delta_S']:bounds(r[x],-2,2)
for x in ['delta_tidalflat_share','delta_water_share','delta_nonmangroveveg_share','delta_built_share','delta_soft_share']:bounds(r[x],-1,1)
for x in ['soft_to_built_m2','changed_area_m2','observed_transition_area_m2','built_gain_from_nonbuilt_m2']:bounds(r[x],0,22501)
checks.append({'table':'transition_routes','rows':len(r),'checks':'Unique joins to matched support; pMEHI agrees within 1e-12 and area weights within 1e-10 m2 across independently exported tables; valid fraction and component-change domains; nonnegative transition areas'})
p=read('data/validation_current/map_validation_predictions.csv.gz')
assert len(p)==10657 and p.sample_row.tolist()==list(range(10657))
assert p.year.isin([2000,2010,2020,2023]).all() and p.reference_class.isin(range(1,7)).all()
assert p.mapped_class.dropna().isin(range(1,7)).all()
assert p.loc[p.mapped_class.isna(),'status'].eq('outside_map').all()
assert p.mapped_class.isna().sum()==654
assert p.evaluation_status.isin(['valid','outside_map','duplicate_record']).all()
checks.append({'table':'map_validation_assignments','rows':len(p),'missing_map_assignments':654,
               'checks':'Reference year/class domains; source row order; mapped class domain; missing map labels explicitly outside coverage'})
train=read('data/samples/training_samples.csv')
assert train.columns.tolist()==['system.index','Landcover','Year','.geo']
assert not (ROOT/'data/packed/train.ndcol.zst').exists() and not (ROOT/'data/packed/test.ndcol.zst').exists()
manifest=json.loads((ROOT/'metadata/storage_manifest.json').read_text(encoding='utf8'))
assert not any('classifier_production' in item['path'] for item in manifest['direct_tables'])
checks.append({'table':'public_sample_scope','rows':len(train),'checks':'Only the four source-reference fields are distributed in the training register; validation reference fields are stored once with map assignments'})
report={'status':'passed','scope':'Actual distributed numerical result inputs and reference metadata','checks':checks}
out=ROOT/'reproduced/qa/result_table_semantics.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
