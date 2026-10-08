"""Check component and support conventions against all four annual input tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from mehi_metric import calculate_components
from spatial.build_mehi_arcpy import capped_near_distance, component_values

ROOT=Path(__file__).resolve().parents[1]
records=[]
for year in [2000,2010,2020,2023]:
    frame=pd.read_csv(ROOT/f'data/scenario/edge_grid_{year}.csv.gz',float_precision='round_trip')
    contact=(frame.A_CONTACT_M2/frame.A_M_M2.where(frame.A_M_M2>1)).fillna(0).clip(0,1)
    computed=calculate_components(frame.EDGE_DIST_M,contact,frame.A_BUILT_M2,frame.A_SOFT_M2,frame.A_OTHER_M2)
    differences={}
    for name in ['P','C','B','S','MEHI']:
        expected=frame[name if name=='MEHI' else name+'_COMP']
        np.testing.assert_allclose(computed[name],expected,atol=3e-15,rtol=0)
        differences[name]=float((computed[name]-expected).abs().max())
    zero=frame.EDGE_DIST_M.eq(0)
    assert computed.loc[zero,'P'].eq(1).all()
    assert computed.loc[frame.A_M_M2<=1,'C'].eq(-1).all()
    assert computed.loc[frame.A_OBS_EXT_M2<=1,['B','S']].eq(0).all().all()
    records.append({'year':year,'rows':len(frame),'zero_distance_records':int(zero.sum()),
                    'small_mangrove_support_records':int((frame.A_M_M2<=1).sum()),
                    'max_absolute_difference':differences})
for value,expected in [(0,0),(-1,1000),(None,1000),(float('nan'),1000),(50,50),(1001,1000)]:
    assert capped_near_distance(value,1000)==expected
assert component_values(0,0.5,0.5,0,0,0,1000)['C_COMP']==-1
assert component_values(0,1,1,0,0,0,1000)['C_COMP']==-1
assert component_values(0,2,2,0,0,0,1000)['C_COMP']==1
report={'status':'passed','rows':sum(x['rows'] for x in records),'annual':records,
        'checks':['Vector component helper agrees with stored four-date fields to floating-point tolerance',
                  'Zero Near distance is preserved as contact proximity',
                  'Negative or missing Near results take the distance cap',
                  'Mangrove support <=1 m2 gives C=-1; external support <=1 m2 gives B=S=0']}
out=ROOT/'reproduced/qa/metric_conventions.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
