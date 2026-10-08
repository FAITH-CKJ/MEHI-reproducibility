"""Restore the full-precision MEHI source tables and verify every stored column."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

# Load PyTorch first when installed; this also supports Windows BLAS runtimes.
try:
    import torch
except ImportError:
    pass
import numpy as np
import pandas as pd
import zstandard as zstd
from lossless_frame_codec import read_frame, _sha, _string_sha

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frame(frame, specs):
    assert frame.columns.tolist() == [s['name'] for s in specs]
    for spec in specs:
        values = frame[spec['name']].to_numpy()
        assert str(values.dtype) == spec['dtype'], (spec['name'], str(values.dtype), spec['dtype'])
        digest = _sha(values) if values.dtype.kind in 'iufb' else _string_sha(values)
        assert digest == spec['sha256'], f"Column checksum failed: {spec['name']}"


def restore(root=ROOT):
    manifest = json.loads((ROOT / 'metadata/storage_manifest.json').read_text(encoding='utf8'))
    tables = {}
    results = []

    def save(path, frame):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, compression={'method':'gzip', 'mtime':0} if target.suffix=='.gz' else None)
        results.append({'path':path, 'rows':len(frame), 'columns':len(frame.columns), 'status':'passed'})
        print(f"Restored {path}: {len(frame):,} rows", flush=True)

    for item in manifest['direct_tables']:
        frame = read_frame(ROOT / item['storage'])
        save(item['path'], frame)
        if item['path'].endswith('classification_sensitivity_inputs_150m.csv.gz'):
            tables['annual'] = frame

    annual = tables['annual']
    for item in manifest['relational_tables']:
        stored = read_frame(ROOT / item['storage'])
        reference = annual.iloc[stored['annual_row'].to_numpy()]
        end = annual.iloc[stored['annual_end_row'].to_numpy()] if 'annual_end_row' in stored else None
        columns = {}
        # Direct columns precede formulas that refer to them, independently of CSV order.
        for spec in item['columns']:
            if spec['source']=='constant':
                columns[spec['name']] = np.full(len(stored), spec['value'], dtype=spec['dtype'])
            elif spec['source']=='payload':
                columns[spec['name']] = stored[spec['column']].to_numpy().astype(spec['dtype'])
        for spec in item['columns']:
            if spec['source'] not in ['annual', 'formula']:
                continue
            if spec['source']=='annual':
                terms = spec['terms']
                values = reference[terms[0]].to_numpy().copy()
                for term in terms[1:]:
                    values = values + reference[term].to_numpy()
            else:
                recipe = spec['recipe']
                op = recipe['op']
                if op=='state_coordinate':
                    state = tables['state_coordinates']
                    keys = pd.MultiIndex.from_arrays([reference.GU_A3, reference.GRID_ID])
                    values = state.loc[keys, recipe['axis']].to_numpy()
                elif op=='annual_plus_column':
                    values = reference[recipe['annual']].to_numpy() + columns[recipe['column']]
                elif op=='half_endpoint_change':
                    values = (end.MEHI.to_numpy() - reference.MEHI.to_numpy()) / 2
                elif op=='minimum_endpoint_area':
                    values = np.minimum(reference.A_M_M2.to_numpy(), end.A_M_M2.to_numpy())
                elif op=='endpoint_area':
                    values = end.A_M_M2.to_numpy()
                elif op=='component_change':
                    values = end[recipe['column']].to_numpy() - reference[recipe['column']].to_numpy()
                elif op=='cover_share_change':
                    terms = recipe['terms']
                    def fraction(frame):
                        a = frame[terms[0]].to_numpy().copy()
                        for term in terms[1:]:
                            a = a + frame[term].to_numpy()
                        den = frame.A_OBS_EXT_M2.to_numpy()
                        return np.divide(a,den,out=np.zeros_like(a),where=den>1)
                    values = fraction(end) - fraction(reference)
                else:
                    raise ValueError(f'Unrecognized storage formula: {op}')
            values = np.asarray(values,dtype=spec['dtype'])
            if spec['residual'] is not None:
                values = (values.view(np.uint64) + stored[spec['residual']].to_numpy().view(np.uint64)).view(np.float64)
            columns[spec['name']] = values
        frame = pd.DataFrame({spec['name']:columns[spec['name']] for spec in item['columns']})
        verify_frame(frame,item['columns'])
        save(item['path'],frame)
        if item['path'].endswith('Fig2_edge_state_maps_2000_2023.csv.gz'):
            tables['state_coordinates'] = frame.query('year == 2023').set_index(['location_code','GRID_ID'])[['lon','lat']]

    from restore_route_table import read_route
    item=manifest['route_table']
    save(item['path'],read_route(annual,ROOT/item['storage'],ROOT/item['recipe']))

    for item in manifest['compressed_files']:
        target = root/item['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(zstd.ZstdDecompressor().decompress((ROOT/item['storage']).read_bytes()))
        assert sha(target)==item['sha256']
    for item in manifest['aliases']:
        target = root/item['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/item['source'],target)
        assert sha(target)==item['sha256']
    out = root/'reproduced/qa/restoration_verification.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'status':'passed','tables':results,'verification':'All reconstructed columns retain their stored dtypes and exact numeric bits or UTF-8 string values.'},indent=2),encoding='utf8')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT,help='Directory to receive expanded data tables (default: this package).')
    args=parser.parse_args()
    restore(args.output_root.resolve())
