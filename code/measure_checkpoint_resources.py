"""Measure CPU inference resources for the retained seven-checkpoint ensemble."""
from pathlib import Path
import sys, time, json, platform, statistics, os, argparse
from datetime import date
import torch
import numpy as np
import pandas as pd
import psutil
try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
        cpu=winreg.QueryValueEx(k,'ProcessorNameString')[0].strip()
except (OSError,ImportError):
    cpu=platform.processor()

PKG=Path(__file__).resolve().parents[1]
CODE=PKG/'code/landcover_classifier'
sys.path.insert(0,str(CODE))
from dataset import read_predictor_table,PREDICTOR_COLUMNS
from inference_ensemble import load_ensemble,predict_numpy
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-csv',type=Path,required=True,help='User-supplied labelled CSV with all 17 predictors')
args=parser.parse_args()
torch.set_num_threads(4)
start=time.perf_counter()
test=read_predictor_table(args.input_csv)
features=test[PREDICTOR_COLUMNS].to_numpy(np.float32)
read_seconds=time.perf_counter()-start
start=time.perf_counter()
models=load_ensemble(CODE/'models',torch.device('cpu'))
load_seconds=time.perf_counter()-start
times=[];reference=None
for n in range(5):
    start=time.perf_counter()
    prediction=predict_numpy(features,models,torch.device('cpu'),batch_size=1024)
    times.append(time.perf_counter()-start)
    if reference is None:reference=prediction.copy()
    else:assert np.array_equal(reference,prediction)
record={
 'task':f'CPU inference with seven retained checkpoints on {len(test):,} supplied records',
 'scope':'Measured evaluation task in the revision reproducibility environment; saved weights used directly',
 'date':date.today().isoformat(),'cpu':cpu,'os':platform.system()+' '+platform.release(),
 'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(logical=True),
 'installed_memory_gib':round(psutil.virtual_memory().total/1024**3,2),
 'torch_intraop_threads':torch.get_num_threads(),'device':'cpu','gpu_used':False,
 'inference_batch_size':1024,'records':len(test),'checkpoints':len(models),
 'python':platform.python_version(),'pytorch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
 'measurement_excludes':'Interpreter and module import startup; data loading and checkpoint loading reported separately',
 'csv_loading_seconds':read_seconds,'checkpoint_loading_seconds':load_seconds,
 'inference_seconds_five_runs':times,'inference_median_seconds':statistics.median(times),
 'inference_min_seconds':min(times),'inference_max_seconds':max(times),'predictions_identical_across_runs':True,
}
out=PKG/'reproduced/checkpoint_resource_measurement.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(record,ensure_ascii=False,indent=2))
