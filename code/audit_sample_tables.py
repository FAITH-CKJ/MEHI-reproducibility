"""Verify sample accounting, years, reference labels, coordinates and split membership."""
from pathlib import Path
import argparse, hashlib, json
import torch
import numpy as np
import pandas as pd

YEARS=[2000,2010,2020,2023]

def audit(root):
    root=Path(root)
    report={'status':'passed','scope':'Public sample reference fields and train/validation membership',
            'valid_observation_years':YEARS,'tables':{}}
    keys={}
    for label,count in [('train',42795),('test',10657)]:
        if label=='train':
            path=root/'data/samples/training_samples.csv'
            frame=pd.read_csv(path,float_precision='round_trip',keep_default_na=False)
            assert frame.columns.tolist()==['system.index','Landcover','Year','.geo']
        else:
            path=root/'data/validation_current/map_validation_predictions.csv.gz'
            raw=pd.read_csv(path,float_precision='round_trip',keep_default_na=False)
            frame=raw[['sample_row','reference_class','year','reference_coordinate']].rename(columns={
                'sample_row':'system.index','reference_class':'Landcover','year':'Year','reference_coordinate':'.geo'})
            frame['system.index']=frame['system.index'].astype(str)
        assert len(frame)==count
        assert frame.Year.isin(YEARS).all(),f'{label}: invalid year'
        assert frame.Landcover.isin(range(1,7)).all(),f'{label}: invalid class'
        assert frame['system.index'].str.len().gt(0).all()
        coordinates=[]
        for value in frame['.geo']:
            item=json.loads(value)
            assert item['type']=='Point'
            xy=item['coordinates']
            assert len(xy)==2 and all(isinstance(x,(int,float)) and np.isfinite(x) for x in xy)
            assert -180<=xy[0]<=180 and -90<=xy[1]<=90
            coordinates.append(tuple(xy))
        check=pd.DataFrame({'coordinate':coordinates,'year':frame.Year,'class':frame.Landcover})
        groups=check.groupby(['coordinate','year'],sort=False)
        assert groups['class'].nunique().max()==1,f'{label}: conflicting repeated labels'
        keys[label]=set(zip(coordinates,frame.Year))
        report['tables'][label]={
            'rows':len(frame),'columns':len(frame.columns),
            'year_counts':{int(y):int((frame.Year==y).sum()) for y in YEARS},
            'class_counts':{int(c):int((frame.Landcover==c).sum()) for c in range(1,7)},
            'invalid_years':0,'invalid_classes':0,'invalid_coordinates':0,
            'missing_reference_values':int(frame.isna().sum().sum()),
            'duplicate_coordinate_year_additional_records':len(frame)-len(keys[label]),
            'conflicting_duplicate_labels':0,
        }
    overlap=keys['train']&keys['test']
    assert not overlap,f'Training/validation coordinate-year overlap: {len(overlap)}'
    report['shared_training_validation_coordinate_year_pairs']=0
    report['duplicate_record_handling']='Sample reference fields preserve source record membership. Final-map accuracy counts each covered coordinate-year once; the validation assignment table records each observation and its evaluation status.'
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(args.root)
    target=args.output or args.root/'reproduced/qa/sample_table_audit.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='tables'},indent=2))
    for name,item in result['tables'].items():
        print(name,json.dumps({k:v for k,v in item.items() if k!='predictor_ranges'},indent=2))
