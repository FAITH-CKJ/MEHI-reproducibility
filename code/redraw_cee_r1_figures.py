"""CEE R1: improve figure legibility from accepted scientific source tables.

Run with the project science Python. All paths are resolved from this file;
the accepted package and root figures are read only. No model is refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
from cmcrameri import cm as scientific_cmaps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figure_source"
OUT = ROOT / "reproduced_figures"
OUT.mkdir(parents=True, exist_ok=True)
W = 167.0
SOFT, NEUTRAL, HARD = "#138F8F", "#D5D0C7", "#D94A26"
SSPC = {"SSP1":"#3B8F5A", "SSP2":"#2F73C5", "SSP3":"#D19A2A", "SSP5":"#A43A63"}
LAND = ROOT / "data/cartography/ne_50m_land.geojson"
INPUTS, CHECKS, OUTPUTS = {}, {}, {}
plt.rcParams.update({
    "font.family":"Arial", "font.size":8,
    "axes.labelsize":8, "axes.titlesize":8.5,
    "xtick.labelsize":7.5, "ytick.labelsize":7.5,
    "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
    "axes.linewidth":0.55, "axes.spines.top":False,
    "axes.spines.right":False, "axes.unicode_minus":True,
    "text.color":"#222222", "axes.labelcolor":"#222222",
    "xtick.color":"#333333", "ytick.color":"#333333",
    "savefig.facecolor":"white",
})


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(name):
    p = DATA / name
    INPUTS[str(p.relative_to(ROOT))] = sha(p)
    return pd.read_csv(p)


def canvas(height):
    f = plt.figure(figsize=(W / 25.4, height / 25.4), dpi=150)
    return f, height


def axes(f, height, x, y, w, h):
    return f.add_axes([x / W, y / height, w / W, h / height])


def label(f, height, letter, text, x, y):
    f.text(x / W, y / height, letter, weight="bold", fontsize=10,
           ha="left", va="baseline")
    f.text((x + 5) / W, y / height, text, fontsize=8.5,
           ha="left", va="baseline")


def clean(ax):
    ax.grid(axis="y", color="#E9E9E9", lw=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.4, width=.5, pad=2)
    for sp in ax.spines.values():
        sp.set_color("#999999")


def save(fig, name, height):
    pdf, png = OUT / (name + ".pdf"), OUT / (name + ".png")
    # Fixed physical canvas: never tight-crop or rescale type after export.
    fig.savefig(pdf, dpi=450)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    with fitz.open(pdf) as doc:
        p = doc[0]
        fonts = [s["size"] for b in p.get_text("dict")["blocks"]
                 if "lines" in b for l in b["lines"] for s in l["spans"]
                 if s["text"].strip()]
        p.get_pixmap(matrix=fitz.Matrix(150/72,150/72)).save(OUT/(name+"_print_preview.png"))
        OUTPUTS[name] = {"width_mm":p.rect.width*25.4/72,
                         "height_mm":p.rect.height*25.4/72,
                         "minimum_extracted_font_pt":min(fonts) if fonts else None,
                         "pdf_sha256":sha(pdf), "png_sha256":sha(png)}
    print("Exported", name, flush=True)


def land(ax, extent=(-177,180,-40,36)):
    if str(LAND.relative_to(ROOT)) not in INPUTS:
        INPUTS[str(LAND.relative_to(ROOT))] = sha(LAND)
    feats = json.loads(LAND.read_text(encoding="utf-8"))["features"]
    patches = []
    for feat in feats:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            arr = np.asarray(poly[0])
            if arr[:,0].max() < extent[0] or arr[:,0].min() > extent[1] or arr[:,1].max() < extent[2] or arr[:,1].min() > extent[3]:
                continue
            patches.append(Polygon(arr, closed=True))
    ax.add_collection(PatchCollection(patches, facecolor="#F4F1EA", edgecolor="#BDB7AC", lw=.25, zorder=1))
    ax.set(xlim=extent[:2], ylim=extent[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-120,-60,0,60,120], ["120°W","60°W","0°","60°E","120°E"])
    ax.set_yticks([-30,0,30], ["30°S","0°","30°N"])
    # Matplotlib ticks can expand limits: enforce geographic bounds after ticks.
    ax.set(xlim=extent[:2], ylim=extent[2:])
    ax.grid(color="#EAEAEA", linewidth=.35, zorder=0)
    ax.tick_params(length=2, width=.5, pad=1.6, labelsize=7.3)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#B7B7B0"); sp.set_linewidth(.5)


def fig2():
    maps = read("Fig2_edge_state_maps_2000_2023.csv.gz")
    comp = read("Fig2_edge_state_composition.csv")
    hist = read("Fig2_annual_MEHI_histogram.csv")
    read("Fig2_annual_MEHI_summary.csv")
    assert len(maps) == 233008
    assert not maps[["MEHI","hard_edge_share_percent_0p5deg_bin","area_weight"]].isna().any().any()
    f,H = canvas(200)
    breaks = [-1,-.95,-.9,-.85,-.8,-.75,-.7,-.65,-.6,-.55,-.5,-.45,-.4,-.33,-.2,-.05,.05,.2,.33,.55,.75,1]
    cols = ["#071738","#08255F","#0B3B86","#1558A6","#2377B8","#2A92BF","#1FA9B1","#18B99E","#4BC797","#7FD19D","#ADD8A6","#CAD8AF","#D7D8C3","#E2DED1","#ECE6DC","#DDD9CF","#F0C98E","#EF9B50","#D94A26","#B73522","#7F1D1D"]
    cm = scientific_cmaps.vik; nm = Normalize(-1,1)
    hb = [0,.323401,2.737395,6.467173,13.914683,29.648819,58.807458,100]
    hc = ListedColormap(scientific_cmaps.lajolla_r(np.linspace(.10,.94,7)))
    hn = BoundaryNorm(hb,hc.N)
    positions = [("a",2000,"MEHI",161),("b",2023,"MEHI",125),
                 ("c",2000,"hard_edge_share_percent_0p5deg_bin",87),
                 ("d",2023,"hard_edge_share_percent_0p5deg_bin",51)]
    for letter, year, col, y in positions:
        ax = axes(f,H,10,y,134,28.53); land(ax)
        sub = maps[maps.year.eq(year)]
        cmap,norm = (cm,nm) if col == "MEHI" else (hc,hn)
        ax.scatter(sub.lon,sub.lat,c=sub[col],s=5.0,cmap=cmap,norm=norm,
                   edgecolors="none",alpha=.9,rasterized=True,zorder=3)
        # The inset is geographic and uses exactly the same rows/values/scale.
        ia = ax.inset_axes([.06,.09,.21,.48])
        ex = (-84.8,-79.2,24.2,30.5)
        land(ia,ex); ia.set_xticks([]);ia.set_yticks([])
        m = sub.lon.between(ex[0],ex[1]) & sub.lat.between(ex[2],ex[3])
        ia.scatter(sub.loc[m,"lon"],sub.loc[m,"lat"],c=sub.loc[m,col],s=3.1,
                   cmap=cmap,norm=norm,lw=0,rasterized=True,zorder=3)
        ax.add_patch(Rectangle((ex[0],ex[2]),ex[1]-ex[0],ex[3]-ex[2],fill=False,lw=.6,ec="#444444",zorder=4))
        ia.text(.03,.98,"*",transform=ia.transAxes,va="top",fontsize=8)
        label(f,H,letter,("MEHI state" if col=="MEHI" else "Hard urban edge share")+f" · {year}",5,y+30)
    for cmap,norm,y,h,lbl,ticks in [(cm,nm,125,64,"MEHI",[-1,-.8,-.6,-.33,0,.33,1]),
                                    (hc,hn,51,64,"Hard urban edge (%)",[0,2.737395,13.914683,58.807458,100])]:
        cb=f.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=norm),cax=axes(f,H,149,y,2.4,h))
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks] if lbl=="MEHI" else ["0","2.7","14","59","100"])
        cb.ax.tick_params(labelsize=7.3,length=2,pad=1.8)
        cb.set_label(lbl,fontsize=8,labelpad=4)
    # A short unified key avoids repeating long inset text four times.
    f.text(.985,47.3/H,"* Florida–Bahamas",fontsize=7,ha="right")
    label(f,H,"e","Edge-state composition",5,43)
    ae=axes(f,H,16,10,61,27)
    years=[2000,2010,2020,2023]; left=np.zeros(4)
    for state,color,name in [("soft_buffered",SOFT,"Soft-buffered"),("mixed",NEUTRAL,"Mixed"),("hard_urban_edge",HARD,"Hard")]:
        vals=np.array([comp.loc[comp.year.eq(y)&comp.edge_state.eq(state),"area_weighted_share_percent"].item() for y in years])
        ae.barh(np.arange(4),vals,left=left,color=color,height=.6,label=name,lw=0);left+=vals
    assert np.allclose(left,100,atol=1e-10)
    ae.set_yticks(range(4),years);ae.invert_yaxis();ae.set_xlim(0,100)
    ae.set_xlabel("Mangrove-area share (%)",labelpad=2)
    ae.set_xticks([0,50,100]);ae.tick_params(axis="y",length=0)
    ae.legend(ncol=3,fontsize=7.3,frameon=False,loc="lower center",bbox_to_anchor=(.48,1.01),
              handlelength=.8,handletextpad=.35,columnspacing=.65,borderpad=0)
    clean(ae)
    label(f,H,"f","Annual MEHI distributions",91,43)
    af=axes(f,H,105,10,57,27)
    yc={2000:"#137E8C",2010:"#CC9827",2020:"#8C759D",2023:"#CF533B"}
    endpoints=[]
    for y,ls in zip(years,["-","--",":","-."]):
        ss=hist[hist.year.eq(y)].sort_values("MEHI_bin_right")
        yy=np.cumsum(ss.area_weight_sum)/ss.area_weight_sum.sum()*100
        af.step(np.r_[-1,ss.MEHI_bin_right],np.r_[0,yy],where="post",color=yc[y],ls=ls,lw=1,label=str(y))
        endpoints.append(float(yy.iloc[-1]))
    af.set(xlim=(-1,1),ylim=(0,100),xticks=[-1,0,1],yticks=[0,50,100])
    af.set_xlabel("MEHI",labelpad=2);af.set_ylabel("Cumulative area (%)",labelpad=2)
    af.legend(loc="lower right",ncol=2,frameon=False,fontsize=7.3,handlelength=1.5,columnspacing=.65,handletextpad=.4)
    clean(af)
    CHECKS["Fig2"]={"map_rows":len(maps),"years":maps.groupby("year").size().to_dict(),
                    "composition_sums_percent":left.tolist(),"cdf_endpoints":endpoints,
                    "cdf_bin_width":.025,"map_MEHI_aggregation":"none; original neighbourhood values",
                    "hard_edge_map":"canonical 0.5 degree area share field; unchanged",
                    "palettes":"Crameri vik (MEHI); Crameri lajolla_r (hard share; accepted bins)",
                    "distribution":"cumulative canonical mangrove-area histogram, no smoothing/log transform"}
    save(f,"Fig1_MEHI_state_R1",H)


def fig3():
    raw=read("Fig3_matched_neighbourhoods.csv.gz")
    summary=read("Fig3_country_summary.csv")
    glob=read("Fig3_global_summary.csv").iloc[0]
    assert len(raw)==79509 and raw.GU_A3.nunique()==19
    assert raw.weight_m2.eq(22500).all()
    order=summary.GU_A3.tolist()
    assert summary.n_grid.sum()==len(raw)
    rng=np.random.default_rng(6148)
    f,H=canvas(195)
    label(f,H,"a","Matched neighbourhoods by country/location",5,190)
    aa=axes(f,H,23,94,125,91)
    aa.axvspan(-.005,.005,color=NEUTRAL,alpha=.5,zorder=0)
    aa.axvline(0,color="#777777",lw=.6,zorder=1)
    aa.text(1.07,1.025,"n",transform=aa.transAxes,fontsize=8,ha="center")
    for i,country in enumerate(order):
        s=raw.loc[raw.GU_A3.eq(country),"pMEHI"].to_numpy()
        row=summary[summary.GU_A3.eq(country)].iloc[0]
        assert len(s)==row.n_grid
        colors=np.where(s<-.005,SOFT,np.where(s>.005,HARD,NEUTRAL))
        jitter=rng.uniform(-.27,.27,len(s))
        aa.scatter(s,np.full(len(s),i)+jitter,c=colors,s=1.0 if len(s)>=100 else 4.5,
                   lw=0,alpha=.4 if len(s)>=100 else .8,rasterized=True,zorder=2)
        aa.text(1.05,i,f"{len(s):,}",ha="left",va="center",fontsize=7.5,
                transform=aa.get_yaxis_transform(),clip_on=False)
    aa.set_yticks(range(19),[c+("†" if summary.loc[summary.GU_A3.eq(c),"n_grid"].item()<100 else "") for c in order])
    aa.set_ylim(18.65,-.65);aa.set_xlim(-1,1);aa.set_xticks([-1,-.5,0,.5,1])
    aa.set_xlabel("pMEHI (2000–2023)",labelpad=2)
    aa.tick_params(axis="y",length=0,labelsize=8);aa.tick_params(axis="x",labelsize=7.6)
    aa.spines["left"].set_visible(False)
    for i in range(19):aa.axhline(i,color="#EEEEEE",lw=.35,zorder=0)
    f.text(5/W,83/H,"All 79,509 matched cells · equal-cell support · † fewer than 100 cells",fontsize=7.3)
    label(f,H,"b","Global distribution",5,77)
    ab=axes(f,H,17,36,66,35)
    edges=np.linspace(-1,1,201)
    counts,_=np.histogram(raw.pMEHI,bins=edges)
    assert counts.sum()==79509
    centers=(edges[:-1]+edges[1:])/2
    # Stack each bin by the true per-cell class, rather than assigning a whole
    # bin by its centre (the +/-0.005 boundaries bisect the 0.01-wide bins).
    bottom=np.zeros(len(centers))
    class_masks=[raw.pMEHI.lt(-.005),raw.pMEHI.abs().le(.005),raw.pMEHI.gt(.005)]
    class_counts=[]
    for mask,color in zip(class_masks,[SOFT,NEUTRAL,HARD]):
        cc,_=np.histogram(raw.loc[mask,"pMEHI"],bins=edges)
        vv=cc/len(raw)*100
        ab.bar(centers,vv,bottom=bottom,width=np.diff(edges)*.95,color=color,lw=0)
        bottom+=vv;class_counts.append(int(cc.sum()))
    assert np.array_equal(np.rint(bottom/100*len(raw)).astype(int),counts)
    ab.axvline(0,color="#6D6D6D",lw=.65)
    ab.set(xlim=(-1,1),ylim=(0,max(counts/len(raw)*100)*1.12),xticks=[-1,-.5,0,.5,1])
    ab.set_xlabel("pMEHI",labelpad=2);ab.set_ylabel("Neighbourhood share (%)",labelpad=3)
    ab.text(.04,.95,"Mean −0.0002\nMedian 0.0000",transform=ab.transAxes,va="top",fontsize=7.5)
    clean(ab)
    ash=axes(f,H,17,16,66,5)
    shares=[glob.softening_share_percent,glob.near_neutral_share_percent,glob.hardening_share_percent]
    left=0
    for v,c in zip(shares,[SOFT,NEUTRAL,HARD]):
        ash.barh([0],[v],left=left,color=c,height=.8,lw=0)
        ash.text(left+v/2,0,f"{v:.1f}%",ha="center",va="center",fontsize=7.7,
                 color="white" if c!=NEUTRAL else "#222222");left+=v
    ash.set(xlim=(0,100),ylim=(-.5,.5),xticks=[],yticks=[])
    for sp in ash.spines.values():sp.set_visible(False)
    hs=[Line2D([0],[0],color=c,lw=4,label=l) for c,l in zip([SOFT,NEUTRAL,HARD],["Softening","Near-neutral","Hardening"])]
    ash.legend(handles=hs,ncol=3,frameon=False,fontsize=7.3,loc="lower center",bbox_to_anchor=(.5,1.15),
               handlelength=.75,columnspacing=.65,handletextpad=.4)
    ash.set_xlabel("Matched neighbourhoods (%)",labelpad=3)
    label(f,H,"c","Country/location summaries",94,77)
    ac=axes(f,H,108,13,53,59)
    ac.axvspan(-.005,.005,color=NEUTRAL,alpha=.5,zorder=0)
    ac.axvline(0,color="#888888",lw=.65,zorder=1)
    for i,r in summary.iterrows():
        color=HARD if r.weighted_median_pMEHI>.005 else SOFT if r.weighted_median_pMEHI<-.005 else "#999185"
        ac.plot([r.weighted_p25,r.weighted_p75],[i,i],c="#A0A0A0",lw=1,zorder=2)
        ac.scatter([r.weighted_median_pMEHI],[i],s=12,c=color,lw=.3,edgecolors="white",zorder=3)
    ac.set_yticks(range(19),order);ac.set_ylim(18.65,-.65);ac.set_xlim(-.225,.215)
    ac.set_xticks([-.2,-.1,0,.1,.2]);ac.set_xlabel("pMEHI median and IQR",labelpad=2)
    ac.tick_params(axis="y",length=0,labelsize=7.6);clean(ac);ac.grid(axis="y",visible=False)
    CHECKS["Fig3"]={"n_neighbourhoods":len(raw),"n_countries":19,"weight_per_cell_m2":22500,
                    "jitter_axis":"country-row direction only; pMEHI coordinates unmodified",
                    "jitter_seed":6148,"pMEHI_full_range":[float(raw.pMEHI.min()),float(raw.pMEHI.max())],
                    "histogram_count_sum":int(counts.sum()),"histogram_bin_width":.01,
                    "histogram_class_counts":class_counts,"histogram_colors":"true per-cell class counts stacked within each bin",
                    "class_shares_percent":shares,"country_median_IQR":"unchanged canonical source table"}
    save(f,"Fig2_pMEHI_trajectories_R1",H)


def scenario_weights(maps):
    weight_path=DATA/"Fig6_2023_external_area_weights.csv.gz"
    w=pd.read_csv(weight_path)
    keys=["country","GRID_ID","GRID_UID"]
    assert not w.duplicated(keys).any() and not maps.duplicated(keys).any()
    j=maps.merge(w,on=keys,how="left",validate="one_to_one",indicator=True)
    assert len(j)==117546 and j._merge.eq("both").all()
    assert j.A_OBS_EXT_M2.notna().all() and j.A_OBS_EXT_M2.ge(0).all()
    digest=hashlib.sha256(w.sort_values(keys).to_csv(index=False,float_format="%.17g").encode()).hexdigest()
    INPUTS[str(weight_path.relative_to(ROOT))]=sha(weight_path)
    return j.drop(columns="_merge")


def fig6():
    traj=read("Fig6_global_trajectory.csv")
    maps=read("Fig6_maps_SSP2_SSP5_2100.csv.gz")
    rank=read("Fig6_country_ranking_all19_SSP235_2100.csv")
    maps=scenario_weights(maps)
    maps["scenario_support_m2"]=maps.A_OBS_EXT_M2.clip(lower=1.0)
    core=ROOT/"code/scenario/scenario_core.py"
    INPUTS[str(core.relative_to(ROOT))]=sha(core)
    assert len(rank)==57 and rank.country.nunique()==19
    assert not rank.duplicated(["country","SSP"]).any() and rank.Delta_sMEHI.notna().all()
    maps["lon_bin"]=np.floor(maps.lon/.5)*.5+.25
    maps["lat_bin"]=np.floor(maps.lat/.5)*.5+.25
    binned=[]
    for ssp in ["SSP2","SSP5"]:
        maps["weighted_value"]=maps[f"Delta_sMEHI_{ssp}"]*maps.scenario_support_m2
        b=maps.groupby(["lon_bin","lat_bin"],as_index=False).agg(
            weighted_sum=("weighted_value","sum"),external_support_m2=("scenario_support_m2","sum"),
            n_neighbourhoods=("GRID_UID","size"))
        assert b.external_support_m2.gt(0).all()
        b["Delta_sMEHI"]=b.weighted_sum/b.external_support_m2;b["SSP"]=ssp;b["year"]=2100
        binned.append(b)
        target=traj.loc[traj.SSP.eq(ssp)&traj.year.eq(2100),"Delta_sMEHI"].item()
        aggregate=np.average(maps[f"Delta_sMEHI_{ssp}"],weights=maps.scenario_support_m2)
        assert np.isclose(target,aggregate,rtol=0,atol=1e-10),(ssp,target,aggregate)
    bins=pd.concat(binned,ignore_index=True)
    bins.to_csv(OUT/"Fig6_conditional_allocation_bins_R1.csv",index=False)
    norm=Normalize(0,np.ceil(bins.Delta_sMEHI.max()*10)/10)
    cmap=plt.get_cmap("cividis")
    f,H=canvas(195)
    label(f,H,"a","Global exposure at evaluated SSP years",5,188)
    aa=axes(f,H,18,156,132,27)
    for ssp in ["SSP1","SSP2","SSP3","SSP5"]:
        d=traj[traj.SSP.eq(ssp)].sort_values("year")
        aa.plot(d.year,d.Delta_sMEHI,"o--",color=SSPC[ssp],lw=.9,ms=4,mec="white",mew=.5)
        dy={"SSP1":-.003,"SSP2":0,"SSP3":.002,"SSP5":0}[ssp]
        aa.text(2103,d.iloc[-1].Delta_sMEHI+dy,ssp,fontsize=8,color=SSPC[ssp],va="center",weight="bold")
    aa.axhline(0,color="#999999",lw=.55)
    aa.set(xlim=(2027,2110),ylim=(0,.18),xticks=[2030,2050,2070,2100],yticks=[0,.08,.16])
    aa.set_xlabel("Evaluated year",labelpad=1);aa.set_ylabel("Area-weighted ΔsMEHI",labelpad=3)
    clean(aa)
    for letter,ssp,y in [("b","SSP2",115),("c","SSP5",76)]:
        ax=axes(f,H,10,y,135,28.74);land(ax)
        b=bins[bins.SSP.eq(ssp)]
        ax.scatter(b.lon_bin,b.lat_bin,c=b.Delta_sMEHI,cmap=cmap,norm=norm,s=8.5,
                   marker="o",edgecolor="white",lw=.18,zorder=3,rasterized=True)
        label(f,H,letter,f"{ssp}, 2100 · conditional spatial allocation",5,y+31)
    cax=axes(f,H,151,76,2.5,67.7)
    cb=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    cb.ax.tick_params(labelsize=7.3,length=2,pad=2);cb.set_label("ΔsMEHI (0.5° bins)",fontsize=8,labelpad=3)
    f.text(10/W,69.5/H,"Illustrative local screen · 2023 external-support weights",fontsize=7.3)
    label(f,H,"d","Country/location exposure in 2100",5,65)
    ad=axes(f,H,17,10,141,50)
    order=rank.groupby("country").Delta_sMEHI.max().sort_values(ascending=False).index.tolist()
    ys=np.arange(19)
    for ssp,off in [("SSP2",-.23),("SSP3",0),("SSP5",.23)]:
        v=rank[rank.SSP.eq(ssp)].set_index("country").reindex(order).Delta_sMEHI.to_numpy()
        assert not np.isnan(v).any()
        ad.hlines(ys+off,0,v,color=SSPC[ssp],lw=1.1,zorder=2)
        zero=np.isclose(v,0,atol=1e-12)
        ad.scatter(v[~zero],(ys+off)[~zero],s=8,c=SSPC[ssp],edgecolor="white",lw=.25,zorder=3)
        ad.scatter(v[zero],(ys+off)[zero],s=9,facecolor="white",edgecolor=SSPC[ssp],lw=.6,zorder=4,clip_on=False)
    ad.set_yticks(ys,order);ad.set_ylim(18.7,-.7);ad.set_xlim(-.006,.7)
    ad.set_xticks([0,.1,.2,.3,.4,.5,.6,.7]);ad.set_xlabel("Area-weighted ΔsMEHI",labelpad=2)
    ad.tick_params(axis="y",labelsize=7.5,length=0,pad=3)
    ad.grid(axis="x",color="#EAEAEA",lw=.45);ad.set_axisbelow(True)
    handles=[Line2D([0],[0],color=SSPC[s],lw=2,label=s) for s in ["SSP2","SSP3","SSP5"]]
    handles.append(Line2D([0],[0],marker="o",lw=0,mfc="white",mec="#777777",ms=3,label="Zero increment"))
    ad.legend(handles=handles,frameon=False,ncol=4,fontsize=7.2,loc="lower right",bbox_to_anchor=(1,1.015),
              handlelength=1,handletextpad=.4,columnspacing=.75,borderpad=0)
    CHECKS["Fig6"]={"map_join_rows":len(maps),"join_keys":["country","GRID_ID","GRID_UID"],
                    "unmatched_map_rows":0,"weights":"max(A_OBS_EXT_M2,1 m2), mehi_grid_150m_2023; accepted scenario_core implementation",
                    "n_at_numeric_floor":int(maps.A_OBS_EXT_M2.le(1).sum()),
                    "bin_degrees":.5,"bins_per_scenario":int(len(binned[0])),
                    "aggregation":"sum(Delta_sMEHI * max(A_OBS_EXT_M2,1))/sum(max(A_OBS_EXT_M2,1))",
                    "map_aggregate_matches_global":True,"color_limits":[norm.vmin,norm.vmax],
                    "map_color_max_clipping":False,"ranking_rows":57,"ranking_countries":19,
                    "global_trajectory_rows":16,"scenario_values":"unchanged canonical source tables"}
    save(f,"Fig5_projected_urban_edge_exposure_R1",H)



def main():
    p=argparse.ArgumentParser();p.add_argument("--figures",nargs="+",choices=["1","2","5"],default=["1","2","5"])
    a=p.parse_args()
    funcs={"1":fig2,"2":fig3,"5":fig6}
    for x in a.figures:funcs[x]()
    report=OUT/"figure_revision_checks.json"
    existing=json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"inputs":{},"checks":{},"outputs":{}}
    existing["inputs"].update(INPUTS);existing["checks"].update(CHECKS);existing["outputs"].update(OUTPUTS)
    existing["code_sha256"]=sha(__file__)
    report.write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
