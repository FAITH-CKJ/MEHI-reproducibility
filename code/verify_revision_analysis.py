from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root=Path(__file__).resolve().parents[1]
rev=root/'reproduced'
(rev/'qa').mkdir(parents=True,exist_ok=True)
support=pd.read_csv(root/'data/figure_source/Fig4_area_pMEHI_pathways.csv').set_index('GU_A3')
country=pd.read_csv(rev/'analysis/R2_2_country_error_sensitivity.csv')
reference=country.query('scenario == "observed"').set_index('location_code').loc[support.index]
np.testing.assert_allclose(reference.area_weighted_pMEHI,support.pMEHI_aw,atol=1e-12)
records=[]
gain=support.common_neighbourhood_mangrove_area_change_percent>=0
for scenario,f in country.groupby('scenario',sort=False):
 f=f.set_index('location_code').loc[support.index]
 values=f.area_weighted_pMEHI
 records.append({'scenario':scenario,'countries_locations':len(f),'same_sign_as_observed':int((np.sign(values)==np.sign(reference.area_weighted_pMEHI)).sum()),'country_rank_spearman':float(spearmanr(values,reference.area_weighted_pMEHI).statistic),'persistence_gain_with_positive_pMEHI':int((gain & (values>0)).sum()),'persistence_gain_with_pMEHI_above_0p005':int((gain & (values>.005)).sum()),'positive_persistent_codes':','.join(values[gain & (values>0)].index),'near_neutral_persistent_codes':','.join(values[gain & (abs(values)<=.005)].index)})
pd.DataFrame(records).to_csv(rev/'analysis/R2_2_country_pathway_sensitivity.csv',index=False)
# Independent matrix-direction and fixed-support checks.
cm=pd.read_csv(root/'data/validation_current/landcover_confusion_matrix.csv')
names=['mangrove','tidal_flat','non_mangrove_vegetation','water','built_up','other']
q=cm.set_index('mapped_class').loc[names,names].to_numpy(float)
assert q.sum()==9811
expected=q/q.sum(axis=1)[:,None]
saved=pd.read_csv(rev/'analysis/R2_2_error_transition_matrices.csv')
for year in [2000,2010,2020,2023]:
 v=saved.query('year == @year and scenario == "pooled_composition"').pivot(index='mapped_class',columns='reference_class',values='probability').loc[names,names].to_numpy()
 np.testing.assert_allclose(v,expected,atol=1e-14)
fixed=pd.read_csv(rev/'analysis/R2_2_fixed_support_comparison.csv').pivot(index='scenario',columns='year',values='hard_edge_share_percent')
assert ((fixed[2023]-fixed[2000])>0).all()
report={'status':'passed','checks':['Original country pMEHI reproduced to 1e-12','Pooled confusion matrix has 9811 unique coordinate-year observations and correct mapped-to-reference orientation in all four outputs','All four fixed-support endpoint comparisons increase','Country pathway sensitivity evaluated under raw sign and primary 0.005 band'],'country_results':records,'rounded_area_examples':{c:round(float(support.loc[c,'common_neighbourhood_mangrove_area_change_percent']),1) for c in ['NGA','PHL','IND','SGP']}}
(rev/'qa/analysis_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
