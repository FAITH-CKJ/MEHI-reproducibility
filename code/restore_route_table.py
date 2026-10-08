"""Reconstruct the complete route table from annual rows and exact-bit residuals.

Dependencies: numpy, pandas, and lossless_frame_codec (which uses zstandard).
All 25 source columns, dtypes, row order, string values and IEEE-754 bits are
validated against the column hashes in the recipe JSON. No Python expressions
are evaluated from the JSON; formula kinds are explicitly implemented below.
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
import pandas as pd
from lossless_frame_codec import read_frame

def safe_div(a,b,threshold=0):
    return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=np.float64)),where=np.asarray(b)>threshold)

def prediction(spec,a0,a1,out):
    typ=spec["kind"]
    if typ=="annual_component_delta":return a1[spec["field"]]-a0[spec["field"]]
    if typ=="pMEHI":return (a1["MEHI"]-a0["MEHI"])/2
    if typ=="min_mangrove":return np.minimum(a0["A_M_M2"],a1["A_M_M2"])
    if typ=="class_delta":
        key=spec["field"];d0=a0["A_M_M2"]+a0["A_OBS_EXT_M2"];d1=a1["A_M_M2"]+a1["A_OBS_EXT_M2"]
        mode=spec["mode"]
        if mode=="annual_total_shares":return safe_div(a1[key],d1)-safe_div(a0[key],d0)
    if typ=="observed_area":
        x=a0["A_M_M2"]+a0["A_OBS_EXT_M2"];y=a1["A_M_M2"]+a1["A_OBS_EXT_M2"]
        mode=spec["mode"]
        if mode=="minimum":return np.minimum(x,y)
    if typ=="local_ratio":
        return safe_div(out[spec["numerator"]],out[spec["denominator"]],spec["threshold"])
    if typ=="soft_sum":
        f=spec["fields"];return (out[f[0]]+out[f[1]])+out[f[2]]
    if typ=="evidence":
        losses=[np.maximum(0.,-out[k]) for k in ["delta_tidalflat_share","delta_water_share","delta_nonmangroveveg_share"]]
        best=np.maximum.reduce(losses)
        route=out["route"]
        evidence=np.select([
            route=="not_hardening",route=="built_up_replacement_route",
            route=="proximity_only_route",route=="tidal_flat_loss_route",
            route=="water_edge_narrowing_route",route=="vegetation_buffer_loss_route"],
            [np.zeros(len(route)),out["soft_to_built_share_of_changed"],out["delta_P"],
             losses[0],losses[1],losses[2]],default=best)
        return evidence
    if typ=="constant":return np.full(len(a0["MEHI"]),spec["value"],dtype=np.float64)
    raise ValueError(spec)


def _column_hash(values):
    a=np.asarray(values)
    h=hashlib.sha256()
    if a.dtype.kind in "OUS":
        for value in a:
            if not isinstance(value,str):raise ValueError("Unexpected non-string label")
            encoded=value.encode("utf-8")
            h.update(struct.pack("<Q",len(encoded)));h.update(encoded)
    else:
        a=np.ascontiguousarray(a,dtype=a.dtype.newbyteorder("<"))
        h.update(a.tobytes())
    return h.hexdigest()

def read_route(annual_dataframe, payload_path, recipe_path):
    """Return the complete source route table, checking every restored column."""
    recipe=json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    if recipe.get("format")!="route-annual-relative" or recipe.get("version")!=1:
        raise ValueError("Unsupported route recipe")
    payload=read_frame(payload_path)
    n=recipe["rows"]
    if len(payload)!=n:raise ValueError("Route payload row count mismatch")
    i0=payload["annual_row"].to_numpy(dtype=np.int64)
    i1=payload["annual_end_row"].to_numpy(dtype=np.int64)
    if np.any(i0<0) or np.any(i1<0) or np.any(i0>=len(annual_dataframe)) or np.any(i1>=len(annual_dataframe)):
        raise ValueError("Annual row reference outside input table")
    needed=["YEAR","GU_A3","GRID_ID","A_M_M2","A_OBS_EXT_M2","MEHI",
            "P_COMP","C_COMP","B_COMP","S_COMP","area_tidal_flat_m2",
            "area_water_m2","area_non_mangrove_vegetation_m2","area_built_up_m2"]
    a0={c:annual_dataframe[c].to_numpy()[i0] for c in needed}
    a1={c:annual_dataframe[c].to_numpy()[i1] for c in needed}
    if not np.all(a0["YEAR"]==recipe["annual_years"][0]) or not np.all(a1["YEAR"]==recipe["annual_years"][1]):
        raise ValueError("Annual endpoint years do not match recipe")
    for key in ["GU_A3","GRID_ID"]:
        if not np.array_equal(a0[key],a1[key]):raise ValueError(f"Endpoint keys differ: {key}")
    columns=recipe["route_columns"]
    result={c:payload[c].to_numpy() for c in columns if c in payload}
    result.update(GU_A3=a0["GU_A3"],GRID_ID=a0["GRID_ID"])
    for c,entry in recipe["derivations"].items():
        pred=np.asarray(prediction(entry["spec"],a0,a1,result),dtype="<f8")
        if pred.shape!=(n,):raise ValueError(f"Invalid prediction shape: {c}")
        if entry["residual_column"] is not None:
            r=payload[entry["residual_column"]].to_numpy(dtype="<i8")
            pred=(pred.view("<u8")+r.view("<u8")).view("<f8")
        result[c]=pred
    frame=pd.DataFrame({c:result[c] for c in columns})
    for c in columns:
        if str(frame[c].dtype)!=recipe["route_dtypes"][c]:
            raise ValueError(f"Route dtype mismatch: {c}")
        if _column_hash(frame[c].to_numpy())!=recipe["column_sha256"][c]:
            raise ValueError(f"Route column checksum mismatch: {c}")
    return frame

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("annual");p.add_argument("payload");p.add_argument("recipe");p.add_argument("--output")
    a=p.parse_args()
    annual=read_frame(a.annual)
    route=read_route(annual,a.payload,a.recipe)
    if a.output:route.to_csv(a.output,index=False,float_format="%.17g")
    print(f"Restored {len(route):,} rows and {len(route.columns)} columns; all column hashes verified.")
if __name__=="__main__":main()
