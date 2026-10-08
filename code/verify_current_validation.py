"""Reproduce current map validation directly from packaged point assignments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'data/validation_current'
names=['mangrove','tidal_flat','non_mangrove_vegetation','water','built_up','other']
r=json.loads((V/'validation_record.json').read_text(encoding='utf8'))
p=pd.read_csv(V/'map_validation_predictions.csv.gz')
train=pd.read_csv(ROOT/'data/samples/training_samples.csv')
assert len(train)==r['training_n'] and len(p)==r['validation_n']
assert p.sample_row.nunique()==len(p)
assert p.groupby(['year','reference_coordinate']).reference_class.nunique().max()==1
dup=p.duplicated(['year','reference_coordinate'])
np.testing.assert_array_equal(dup,p.duplicate_coordinate_year)
assert int(dup.sum())==r['duplicate_validation_records_n']
assert p.evaluation_status.eq('outside_map').sum()==r['outside_map_n']
assert p.evaluation_status.eq('valid').sum()==r['map_evaluation_n']
assert len(p)==r['duplicate_validation_records_n']+r['outside_map_n']+r['map_evaluation_n']
annual=pd.read_csv(V/'landcover_confusion_matrix_by_year.csv')
pooled=pd.read_csv(V/'landcover_confusion_matrix.csv').set_index('mapped_class').loc[names,names].to_numpy()
sum_annual=np.zeros((6,6),dtype=int)
results=[]
for year,g in p.groupby('year'):
 g=g[g.evaluation_status.eq('valid')]
 q=np.zeros((6,6),dtype=int)
 np.add.at(q,(g.mapped_class.to_numpy(dtype=int)-1,g.reference_class.to_numpy(dtype=int)-1),1)
 expected=annual[annual.year.eq(year)].pivot(index='mapped_class',columns='reference_class',values='count').loc[names,names].to_numpy()
 np.testing.assert_array_equal(q,expected)
 sum_annual+=q
 results.append({'year':int(year),'n':len(g),'agreement_percent':100*np.trace(q)/q.sum()})
np.testing.assert_array_equal(sum_annual,pooled)
n=pooled.sum();oa=np.trace(pooled)/n;pe=pooled.sum(axis=0)@pooled.sum(axis=1)/n**2
assert abs(oa*100-r['map_accuracy']['overall_accuracy_percent'])<1e-10
assert abs((oa-pe)/(1-pe)-r['map_accuracy']['kappa'])<1e-10
acc=pd.read_csv(V/'landcover_class_accuracy.csv').set_index('class').loc[names]
np.testing.assert_allclose(np.diag(pooled)/pooled.sum(axis=1)*100,acc.user_accuracy_percent,atol=1e-12)
np.testing.assert_allclose(np.diag(pooled)/pooled.sum(axis=0)*100,acc.producer_accuracy_percent,atol=1e-12)
out=ROOT/'reproduced/qa';out.mkdir(parents=True,exist_ok=True)
report={'status':'passed','records':len(p),'unique_covered':int(n),'pooled_agreement_percent':float(100*oa),'kappa':float((oa-pe)/(1-pe)),'annual':results,'orientation':'rows=mapped, columns=reference','checks':['record and duplicate flow','point assignments reproduce each annual matrix','annual matrices sum to pooled matrix','producer/user accuracy use correct denominators']}
(out/'current_validation_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
