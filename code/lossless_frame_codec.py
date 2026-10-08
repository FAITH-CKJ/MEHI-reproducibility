"""Lossless columnar storage for a numeric/string pandas DataFrame.

Requires Python 3.10+, numpy, pandas and zstandard. Floating-point columns retain
their IEEE-754 bits. Optional arithmetic recipes are accepted only when they
reconstruct every input bit; no rounding, precision reduction or float32 cast is
used. Data columns and row order are preserved. The DataFrame index must be the
default RangeIndex (store another index as an explicit column).

The file contains a compressed JSON manifest and independent zstd blocks.
Recipes are a small, explicit operation tree; loading never evaluates Python code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import zstandard as zstd

MAGIC = b"NDCOL001"
_ALLOWED_DTYPES = {f"{kind}{size}" for kind in "iu" for size in [1, 2, 4, 8]} | {"f4", "f8", "b1"}

def _dtype(name):
    d = np.dtype(name)
    if d.kind + str(d.itemsize) not in _ALLOWED_DTYPES:
        raise ValueError(f"Unsupported numeric dtype: {name}")
    return d.newbyteorder("<")

def _canonical(a):
    return np.ascontiguousarray(a, dtype=_dtype(a.dtype))

def _bits(a):
    a = _canonical(np.asarray(a))
    return a.view(np.dtype(f"<u{a.dtype.itemsize}"))

def _sha(a):
    return hashlib.sha256(_canonical(np.asarray(a)).tobytes()).hexdigest()

def _string_sha(values):
    h = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        h.update(struct.pack("<Q", len(encoded)))
        h.update(encoded)
    return h.hexdigest()

def _same(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.dtype.kind in "iufb":
        return aa.dtype == bb.dtype and aa.shape == bb.shape and np.array_equal(_bits(aa), _bits(bb))
    return aa.shape == bb.shape and all(x == y for x, y in zip(aa, bb))

def evaluate_recipe(recipe, columns, n):
    """Evaluate a restricted arithmetic tree using separate NumPy operations."""
    if isinstance(recipe, dict) and set(recipe) == {"column"}:
        return columns[recipe["column"]]
    if isinstance(recipe, (int, float)) and not isinstance(recipe, bool):
        return recipe
    if not isinstance(recipe, list) or not recipe:
        raise ValueError("Malformed recipe")
    op, *args = recipe
    if op == "group_cumcount1":
        if len(args) != 1:
            raise ValueError("group_cumcount1 requires one column")
        group = evaluate_recipe(args[0], columns, n)
        return pd.Series(group).groupby(pd.Series(group), sort=False).cumcount().to_numpy() + 1
    vals = [evaluate_recipe(x, columns, n) for x in args]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "add" and len(vals) == 2: return np.add(vals[0], vals[1])
        if op == "subtract" and len(vals) == 2: return np.subtract(vals[0], vals[1])
        if op == "multiply" and len(vals) == 2: return np.multiply(vals[0], vals[1])
        if op == "divide" and len(vals) == 2: return np.divide(vals[0], vals[1])
        if op == "greater" and len(vals) == 2: return np.greater(vals[0], vals[1])
        if op == "where" and len(vals) == 3: return np.where(vals[0], vals[1], vals[2])
    raise ValueError(f"Unsupported recipe operation: {op}")

def _narrow_integer(a):
    lo, hi = int(a.min(initial=0)), int(a.max(initial=0))
    kinds = "u" if lo >= 0 else "i"
    for size in [1, 2, 4, 8]:
        d = np.dtype(f"<{kinds}{size}")
        lim = np.iinfo(d)
        if lo >= lim.min and hi <= lim.max:
            return np.asarray(a, dtype=d)
    raise ValueError("Integer outside supported range")

def _encode_layout(values, layout):
    a = _canonical(values)
    n, itemsize = len(a), a.dtype.itemsize
    b = a.view(np.uint8).reshape(n, itemsize)
    if layout == "plain":
        return b.tobytes()
    if layout == "byte_shuffle":
        return b.T.tobytes()
    if layout == "bit_shuffle":
        planes = np.unpackbits(b, axis=1, bitorder="little").T
        return np.packbits(planes, axis=1, bitorder="little").tobytes()
    raise ValueError(layout)

def _decode_layout(data, layout, dtype, n):
    dtype = _dtype(dtype)
    k = dtype.itemsize
    if layout == "plain":
        expected = n * k
        if len(data) != expected: raise ValueError("Block size mismatch")
        return np.frombuffer(data, dtype=dtype).copy()
    if layout == "byte_shuffle":
        if len(data) != n*k: raise ValueError("Block size mismatch")
        b = np.frombuffer(data, dtype=np.uint8).reshape(k, n).T.copy()
    elif layout == "bit_shuffle":
        row_bytes = (n+7)//8
        if len(data) != 8*k*row_bytes: raise ValueError("Block size mismatch")
        planes = np.frombuffer(data, dtype=np.uint8).reshape(8*k, row_bytes)
        bits = np.unpackbits(planes, axis=1, count=n, bitorder="little").T
        b = np.packbits(bits, axis=1, bitorder="little")
    else:
        raise ValueError(layout)
    return np.ascontiguousarray(b).view(dtype).reshape(n)

def _candidate_streams(a):
    a = _canonical(a)
    bits = _bits(a)
    for layout in ["plain", "byte_shuffle", "bit_shuffle"]:
        yield {"predictor":"none", "layout":layout}, _encode_layout(a, layout)
    if len(a):
        d = np.empty_like(bits)
        d[0] = bits[0]
        if a.dtype.kind == "f":
            d[1:] = bits[1:] ^ bits[:-1]
            predictor = "xor_previous"
        else:
            with np.errstate(over="ignore"):
                d[1:] = bits[1:] - bits[:-1]
            predictor = "delta_previous"
        for layout in ["byte_shuffle", "bit_shuffle"]:
            yield {"predictor":predictor, "layout":layout}, _encode_layout(d, layout)
        unique, counts = np.unique(bits, return_counts=True)
        common = unique[np.argmax(counts)]
        if int(counts.max()) >= len(a)//10:
            mask = bits != common
            mask_bytes = np.packbits(mask, bitorder="little").tobytes()
            for layout in ["byte_shuffle", "bit_shuffle"]:
                raw = mask_bytes + _encode_layout(a[mask], layout)
                yield {"predictor":"sparse_mode","layout":layout,
                       "mode_bits":int(common),"non_mode_count":int(mask.sum()),
                       "mask_bytes":len(mask_bytes)}, raw

def _untransform(data, meta, dtype, n):
    dtype = _dtype(dtype)
    udtype = np.dtype(f"<u{dtype.itemsize}")
    if meta["predictor"] == "sparse_mode":
        size = meta["mask_bytes"]
        mask = np.unpackbits(np.frombuffer(data[:size],dtype=np.uint8),
                             count=n,bitorder="little").astype(bool)
        if int(mask.sum()) != meta["non_mode_count"]: raise ValueError("Mask count mismatch")
        values = _decode_layout(data[size:],meta["layout"],dtype,meta["non_mode_count"])
        output = np.full(n,meta["mode_bits"],dtype=udtype)
        output[mask] = _bits(values)
        return output.view(dtype)
    a = _decode_layout(data,meta["layout"],dtype,n)
    if meta["predictor"] == "xor_previous":
        a = np.bitwise_xor.accumulate(a.view(udtype),dtype=udtype).view(dtype)
    elif meta["predictor"] == "delta_previous":
        a = np.cumsum(a.view(udtype),dtype=udtype).view(dtype)
    elif meta["predictor"] != "none":
        raise ValueError("Unknown predictor")
    return a

def _compress_block(job, probe_level=9, final_level=19):
    name, a = job
    a = _canonical(a)
    compressor = zstd.ZstdCompressor(level=probe_level)
    trials, best = [], None
    for method, raw in _candidate_streams(a):
        blob = compressor.compress(raw)
        trial = dict(method,compressed_bytes=len(blob),raw_bytes=len(raw))
        trials.append(trial)
        if best is None or len(blob) < best[0]:
            best = (len(blob),method,raw)
    _, method, raw = best
    start = time.perf_counter()
    blob = zstd.ZstdCompressor(level=final_level).compress(raw)
    meta = {"name":name,"dtype":a.dtype.str,"n":len(a),**method,
            "raw_bytes":len(raw),"compressed_bytes":len(blob),
            "decoded_sha256":_sha(a),
            "compressed_sha256":hashlib.sha256(blob).hexdigest()}
    # Check the selected transform immediately, including bitwise float equality.
    restored = _untransform(zstd.ZstdDecompressor().decompress(blob),meta,a.dtype,len(a))
    if not _same(a,restored): raise AssertionError(f"Block round-trip mismatch: {name}")
    return meta, blob, {"name":name,"trials":trials,"selected":method,
                       "final_bytes":len(blob),"final_compression_seconds":time.perf_counter()-start}

def write_frame(df, filename, *, recipes=None, final_level=19, workers=4):
    """Write a lossless frame. Returns metadata and a per-block encoding report."""
    if not isinstance(df.index,pd.RangeIndex) or not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("Store a non-default index as an explicit column")
    if not df.columns.is_unique or any(not isinstance(c,str) for c in df.columns):
        raise ValueError("Unique string column names are required")
    recipes = recipes or OrderedDict()
    original = {c:df[c].to_numpy() for c in df}
    columns, arrays, derived = [], OrderedDict(), []
    for c in df:
        a = original[c]
        spec = {"name":c,"original_dtype":str(a.dtype)}
        if a.dtype.kind in "iufb":
            spec["sha256"] = _sha(a)
            if c in recipes:
                pred = np.asarray(evaluate_recipe(recipes[c],original,len(df)),dtype=a.dtype)
                if pred.ndim == 0: pred = np.full(len(df),pred,dtype=a.dtype)
                # If arithmetic differs by any bit, store signed modulo-bit residuals.
                residual = (_bits(a)-_bits(pred)).view(f"<i{a.dtype.itemsize}")
                spec.update(kind="derived",recipe=recipes[c])
                if np.any(residual):
                    key=c+"__bit_residual"
                    arrays[key]=residual
                    spec["residual_block"]=key
                else:
                    spec["residual_block"]=None
                derived.append(c)
            else:
                key=c
                arrays[key] = _narrow_integer(a) if a.dtype.kind in "iub" else _canonical(a)
                spec.update(kind="numeric",block=key)
        elif a.dtype.kind in "OUS":
            if not all(isinstance(v,str) for v in a):
                raise ValueError("String columns must contain strings; encode missing values explicitly")
            spec["sha256"] = _string_sha(a)
            pairs = df[c].str.extract(r"^(-?\d+)_(-?\d+)$")
            if pairs.notna().all().all():
                left=pairs[0].to_numpy(dtype=np.int64)
                right=pairs[1].to_numpy(dtype=np.int64)
                restored=[f"{x}_{y}" for x,y in zip(left,right)]
                exact=all(x==y for x,y in zip(a,restored))
            else: exact=False
            if exact:
                keys=[c+"__left",c+"__right"]
                arrays[keys[0]]=_narrow_integer(left)
                arrays[keys[1]]=_narrow_integer(right)
                spec.update(kind="integer_pair",blocks=keys,separator="_")
            else:
                codes, values=pd.factorize(a,sort=False)
                if np.any(codes<0): raise ValueError("Missing string code")
                key=c+"__codes"
                arrays[key]=_narrow_integer(codes)
                spec.update(kind="dictionary",block=key,values=values.tolist())
        else:
            raise ValueError(f"Unsupported column {c}: {a.dtype}")
        columns.append(spec)
    # Decode order must respect recipe dependencies.
    available=set(original)-set(derived)
    order=[]
    def refs(tree):
        if isinstance(tree,dict) and "column" in tree:return {tree["column"]}
        if isinstance(tree,list):
            return set().union(*(refs(v) for v in tree[1:]))
        return set()
    while len(order)<len(derived):
        candidates=[c for c in derived if c not in order and refs(recipes[c])<=available]
        if not candidates:raise ValueError("Circular or missing recipe dependency")
        order.extend(candidates);available.update(candidates)
    jobs=list(arrays.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(lambda job:_compress_block(job,final_level=final_level),jobs))
    manifest={"format":"ndcol","version":1,"rows":len(df),"columns":columns,
              "derive_order":order,"blocks":[],"integer_arithmetic":"modulo 2^(8*itemsize)",
              "float_arithmetic":"separate IEEE-754 NumPy operations; exact-bit residuals stored where needed"}
    offset=0
    for meta,blob,_ in results:
        meta["offset"]=offset;offset+=len(blob)
        manifest["blocks"].append(meta)
    packed=zstd.ZstdCompressor(level=final_level).compress(
        json.dumps(manifest,ensure_ascii=False,separators=(",",":")).encode("utf-8"))
    with Path(filename).open("wb") as f:
        f.write(MAGIC);f.write(struct.pack("<Q",len(packed)));f.write(packed)
        for _,blob,_ in results:f.write(blob)
    return manifest, {"block_trials":[v[2] for v in results],"manifest_compressed_bytes":len(packed),
                      "file_bytes":Path(filename).stat().st_size}

def read_frame(filename, *, verify=True):
    """Read a stored frame and, by default, check every reconstructed column hash."""
    with Path(filename).open("rb") as f:
        if f.read(8)!=MAGIC:raise ValueError("Not an ndcol version-1 file")
        packed_n=struct.unpack("<Q",f.read(8))[0]
        if packed_n>128*1024*1024:raise ValueError("Manifest size exceeds limit")
        manifest=json.loads(zstd.ZstdDecompressor().decompress(f.read(packed_n),max_output_size=256*1024*1024))
        if manifest.get("version")!=1:raise ValueError("Unsupported version")
        base=f.tell();n=manifest["rows"];blocks={}
        for meta in manifest["blocks"]:
            f.seek(base+meta["offset"]);blob=f.read(meta["compressed_bytes"])
            if verify and hashlib.sha256(blob).hexdigest()!=meta["compressed_sha256"]:
                raise ValueError(f"Compressed block checksum failed: {meta['name']}")
            raw=zstd.ZstdDecompressor().decompress(blob,max_output_size=meta["raw_bytes"])
            a=_untransform(raw,meta,meta["dtype"],meta["n"])
            if verify and _sha(a)!=meta["decoded_sha256"]:
                raise ValueError(f"Decoded block checksum failed: {meta['name']}")
            blocks[meta["name"]]=a
    cols={}
    specs={x["name"]:x for x in manifest["columns"]}
    for name,spec in specs.items():
        kind=spec["kind"]
        if kind=="derived":continue
        if kind=="numeric":
            cols[name]=blocks[spec["block"]].astype(_dtype(spec["original_dtype"]))
        elif kind=="dictionary":
            values=np.asarray(spec["values"],dtype=object)
            cols[name]=values[blocks[spec["block"]].astype(np.int64)]
        elif kind=="integer_pair":
            left,right=[blocks[k] for k in spec["blocks"]]
            cols[name]=np.asarray([f"{x}{spec['separator']}{y}" for x,y in zip(left,right)],dtype=object)
        else:raise ValueError(f"Unsupported column kind: {kind}")
    for name in manifest["derive_order"]:
        spec=specs[name];dtype=_dtype(spec["original_dtype"])
        pred=np.asarray(evaluate_recipe(spec["recipe"],cols,n),dtype=dtype)
        if pred.ndim==0:pred=np.full(n,pred,dtype=dtype)
        if spec["residual_block"] is not None:
            residual=blocks[spec["residual_block"]]
            pred=(_bits(pred)+_bits(residual)).view(dtype)
        cols[name]=pred
    if verify:
        for name,spec in specs.items():
            actual=_string_sha(cols[name]) if spec["kind"] in ["dictionary","integer_pair"] else _sha(cols[name])
            if actual!=spec["sha256"]:raise ValueError(f"Reconstructed column checksum failed: {name}")
    return pd.DataFrame({name:cols[name] for name in specs})

def verify_frame(original, restored):
    """Return exact row, column, dtype, value and IEEE-754-bit comparisons."""
    if list(original.columns)!=list(restored.columns) or len(original)!=len(restored):
        raise AssertionError("Table shape/order mismatch")
    result=[]
    for name in original:
        a,b=original[name].to_numpy(),restored[name].to_numpy()
        exact=_same(a,b)
        row={"column":name,"dtype":str(a.dtype),"rows":len(a),"exact":bool(exact)}
        if a.dtype.kind in "iufb":
            row.update(bit_mismatches=int(np.count_nonzero(_bits(a)!=_bits(b))),
                       original_sha256=_sha(a),decoded_sha256=_sha(b))
        else:
            row.update(value_mismatches=sum(x!=y for x,y in zip(a,b)),
                       original_sha256=_string_sha(a),decoded_sha256=_string_sha(b))
        result.append(row)
        if not exact:raise AssertionError(f"Exact restore failed: {name}")
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    dec=sub.add_parser("decode");dec.add_argument("input");dec.add_argument("output")
    ver=sub.add_parser("verify");ver.add_argument("input");ver.add_argument("csv")
    args=parser.parse_args()
    frame=read_frame(args.input)
    if args.action=="decode":
        # %.17g is enough to round-trip each float64 when read with round_trip.
        frame.to_csv(args.output,index=False,float_format="%.17g")
        print(f"Restored {len(frame):,} rows and {len(frame.columns)} columns.")
    else:
        source=pd.read_csv(args.csv,float_precision="round_trip")
        checks=verify_frame(source,frame)
        print(json.dumps({"rows":len(frame),"columns":len(checks),
                          "all_exact":all(x["exact"] for x in checks)},indent=2))
if __name__=="__main__":main()
