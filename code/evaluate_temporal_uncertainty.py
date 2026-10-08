"""CEE R1: conditional land-cover error sensitivity and temporal support checks.

This is an explicit sensitivity evaluation, not an accuracy-adjusted population
estimator or a confidence interval for a true map change. No spatial or temporal
reference-label covariance is inferred from pooled marginal validation tables.
"""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduced/analysis'
INPUT = ROOT / 'data/revision'
PKG = ROOT
VALIDATION = ROOT / 'data/validation_current'
CLASSES = ['mangrove','tidal_flat','non_mangrove_vegetation','water','built_up','other']
YEARS = [2000,2010,2020,2023]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def share(m,w):return float(np.average(np.asarray(m)>0.33,weights=w)*100)
def cls(m):return np.where(m>0.005,1,np.where(m < -0.005,-1,0))
def dump(name,records):
    frame=records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT/name,index=False)

def build_inputs():
    return pd.read_csv(INPUT/'classification_sensitivity_inputs_150m.csv.gz',keep_default_na=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    x=build_inputs()
    cm=pd.read_csv(VALIDATION/'landcover_confusion_matrix.csv').set_index('mapped_class').loc[CLASSES,CLASSES].to_numpy(float)
    by_year=pd.read_csv(VALIDATION/'landcover_accuracy_by_year.csv')
    annual_counts=pd.read_csv(VALIDATION/'landcover_confusion_matrix_by_year.csv')
    pooled=cm/cm.sum(axis=1,keepdims=True)
    off=cm.copy();np.fill_diagonal(off,0);off=off/off.sum(axis=1,keepdims=True)
    annual=[]; matrices=[]; model_values={}; margins=[]; deps=[]; residual=[]
    baseline_table=pd.read_csv(PKG/'data/figure_source/Fig2_annual_MEHI_summary.csv').set_index('year')
    for year in YEARS:
        f=x[x.YEAR==year].copy().reset_index(drop=True)
        a=f[['area_'+c+'_m2' for c in CLASSES]].to_numpy(float)
        w=f.A_M_M2.to_numpy(float);m=f.MEHI.to_numpy(float)
        assert np.all(np.isfinite(a)) and np.min(a)>-1e-7 and np.sum(w)>0
        np.testing.assert_allclose(f[['P_COMP','C_COMP','B_COMP','S_COMP']].mean(axis=1),m,atol=1e-12)
        observed_share=share(m,w)
        assert abs(observed_share-baseline_table.loc[year,'hard_urban_edge_area_share_percent'])<1e-5
        residual.append({'year':year,'observed_gdb_share_percent':observed_share,'published_share_percent':float(baseline_table.loc[year,'hard_urban_edge_area_share_percent'])})
        ua=1-by_year[by_year.year==year].set_index('class').loc[CLASSES,'mapped_sample_error_percent'].to_numpy(float)/100
        cm_year=annual_counts.query('year == @year').pivot(index='mapped_class',columns='reference_class',values='count').loc[CLASSES,CLASSES].to_numpy(float)
        q_year=cm_year/cm_year.sum(axis=1,keepdims=True)
        models={'observed':np.eye(6),'pooled_composition':pooled,'year_specific_composition':q_year}
        # A partial sensitivity: preserve observed geometries, contact, distances,
        # annual domain and observed mangrove-area weights for every scenario.
        for name,q in models.items():
            np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
            np.testing.assert_allclose(a.sum(axis=1),(a@q).sum(axis=1),rtol=1e-12,atol=1e-5)
            aq=a@q;ext=aq[:,1:].sum(axis=1)
            valid=(f.A_OBS_EXT_M2.to_numpy()>1)&(ext>1)
            B=f.B_COMP.to_numpy(float).copy();S=f.S_COMP.to_numpy(float).copy()
            B[valid]=2*aq[valid,4]/ext[valid]-1
            S[valid]=1-2*aq[valid,1:4].sum(axis=1)/ext[valid]
            mm=(f.P_COMP.to_numpy()+f.C_COMP.to_numpy()+B+S)/4
            if name=='observed':np.testing.assert_allclose(mm,m,atol=1e-10)
            f[name]=mm
            annual.append({'scenario':name,'year':year,'grid_count':len(f),'observed_mangrove_weight_m2':w.sum(),'hard_edge_share_percent':share(mm,w),'soft_edge_share_percent':float(np.average(mm < -.33,weights=w)*100),'weighted_MEHI_mean':float(np.average(mm,weights=w)),'scope':'fixed observed geometry/domain/weights; composition only'})
            for i,ci in enumerate(CLASSES):
                for j,cj in enumerate(CLASSES):matrices.append({'scenario':name,'year':year,'mapped_class':ci,'reference_class':cj,'probability':q[i,j]})
        f['three_block']=(f.P_COMP+f.C_COMP+(f.B_COMP+f.S_COMP)/2)/3
        annual.append({'scenario':'three_block','year':year,'grid_count':len(f),'observed_mangrove_weight_m2':w.sum(),'hard_edge_share_percent':share(f.three_block,w),'soft_edge_share_percent':float(np.average(f.three_block < -.33,weights=w)*100),'weighted_MEHI_mean':float(np.average(f.three_block,weights=w)),'scope':'specification sensitivity: proximity, contact and combined composition each one-third'})
        denom=f.A_OBS_EXT_M2.to_numpy();ok=denom>1
        identity=np.abs(f.S_COMP.to_numpy()[ok]-f.B_COMP.to_numpy()[ok]-2*a[ok,5]/denom[ok])
        deps.append({'year':year,'valid_composition_grids':int(ok.sum()),'B_S_spearman_rho':float(spearmanr(f.B_COMP[ok],f.S_COMP[ok]).statistic),'S_minus_B_minus_2_other_max_abs':float(identity.max()),'three_block_MEHI_spearman_rho':float(spearmanr(m,f.three_block).statistic)})
        for eps in [0,.005,.01,.015,.03,.05,.075,.1,.15,.2]:
            margins.append({'year':year,'max_absolute_MEHI_perturbation':eps,'lower_hard_share_percent':share(m-eps,w),'upper_hard_share_percent':share(m+eps,w),'interpretation':'deterministic margin bound, not a confidence interval or calibrated map-error model'})
        for shift in [-60,-30,30,60]:
            pp=1-2*np.clip(f.EDGE_DIST_M.to_numpy()+shift,0,1000)/1000
            mm=m+(pp-f.P_COMP.to_numpy())/4
            annual.append({'scenario':f'distance_shift_{shift:+d}m','year':year,'grid_count':len(f),'observed_mangrove_weight_m2':w.sum(),'hard_edge_share_percent':share(mm,w),'soft_edge_share_percent':float(np.average(mm < -.33,weights=w)*100),'weighted_MEHI_mean':float(np.average(mm,weights=w)),'scope':'distance-only stress test; other observed components fixed'})
        model_values[year]=f
        print('Evaluated conditional models',year,flush=True)

    a=model_values[2000].set_index(['GU_A3','GRID_ID']);b=model_values[2023].set_index(['GU_A3','GRID_ID'])
    raw_common=a.index.intersection(b.index)
    canonical=pd.read_csv(INPUT/'canonical_matched_support.csv.gz',keep_default_na=False).set_index(['GU_A3','GRID_ID'])
    assert not len(canonical.index.difference(raw_common))
    common=canonical.index;a=a.loc[common];b=b.loc[common]
    assert len(common)==79509
    persistent=np.minimum(a.A_M_M2.to_numpy(),b.A_M_M2.to_numpy())
    baseline_delta=(b.observed.to_numpy()-a.observed.to_numpy())/2
    np.testing.assert_allclose(baseline_delta,canonical.pMEHI.to_numpy(),atol=1e-10)
    np.testing.assert_allclose(persistent,canonical.weight_m2.to_numpy(),atol=1e-5)
    (OUT/'common_support_selection.json').write_text(json.dumps({'raw_annual_grid_join_rows':len(raw_common),'canonical_persistent_rows':len(common),'excluded_from_endpoint':len(raw_common)-len(common),'reason':'Use the previously adopted persistent-edge GDB support. Four annual grid matches have zero mangrove support in 2000 and do not enter the persistent-edge product.'},indent=2),encoding='utf-8')
    endpoints=[];countries=[];support=[]
    names=['observed','pooled_composition','year_specific_composition','three_block']
    for name in names:
        delta=(b[name].to_numpy()-a[name].to_numpy())/2
        endpoints.append({'scenario':name,'matched_grids':len(common),'mean_pMEHI_equal_cell':float(delta.mean()),'mean_pMEHI_persistent_area':float(np.average(delta,weights=persistent)),'hardening_percent_equal_cell':float(np.mean(delta>.005)*100),'softening_percent_equal_cell':float(np.mean(delta<-.005)*100),'neutral_percent_equal_cell':float(np.mean(abs(delta)<=.005)*100),'three_class_agreement_percent':float(np.mean(cls(delta)==cls(baseline_delta))*100),'pMEHI_spearman_rho':float(spearmanr(delta,baseline_delta).statistic)})
        area_nondecreasing=b.A_M_M2.to_numpy() >= a.A_M_M2.to_numpy()
        hardening=delta>.005
        joint=area_nondecreasing & hardening
        endpoints[-1].update({'area_nondecreasing_hardening_n':int(joint.sum()),'hardening_without_area_loss_percent_of_hardening_cells':float(100*joint.sum()/hardening.sum()),'hardening_percent_of_area_nondecreasing_cells':float(100*joint.sum()/area_nondecreasing.sum()),'joint_percent_all_cells':float(100*joint.mean()),'joint_percent_all_minimum_area_weight':float(100*persistent[joint].sum()/persistent.sum())})
        for code in common.get_level_values(0).unique():
            mask=common.get_level_values(0)==code
            countries.append({'scenario':name,'location_code':code,'n':int(mask.sum()),'area_weighted_pMEHI':float(np.average(delta[mask],weights=persistent[mask])),'median_pMEHI_equal_cell':float(np.median(delta[mask]))})
        for year,frame in [(2000,a),(2023,b)]:
            support.append({'scenario':name,'year':year,'support':'common endpoint cells, fixed minimum mangrove-area weights','n':len(common),'hard_edge_share_percent':share(frame[name],persistent)})
    # Smallest symmetric per-year absolute index bound allowing reversal.
    m0=model_values[2000].MEHI.to_numpy();w0=model_values[2000].A_M_M2.to_numpy()
    m1=model_values[2023].MEHI.to_numpy();w1=model_values[2023].A_M_M2.to_numpy()
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if share(m1-mid,w1)-share(m0+mid,w0)>0:lo=mid
        else:hi=mid
    tipping={'symmetric_per_year_MEHI_error_bound_at_first_possible_trend_reversal':hi,'adverse_2000_hard_share_percent':share(m0+hi,w0),'adverse_2023_hard_share_percent':share(m1-hi,w1),'meaning':'Uniform worst-direction index perturbation in opposite directions between endpoints; diagnostic only. No empirically justified upper bound on spatial classification-induced index error is available.'}
    annual=pd.DataFrame(annual)
    deltas=[]
    for name,d in annual.groupby('scenario',sort=False):
        d=d.set_index('year');v0=float(d.loc[2000,'hard_edge_share_percent']);v1=float(d.loc[2023,'hard_edge_share_percent'])
        deltas.append({'scenario':name,'hard_share_2000_percent':v0,'hard_share_2023_percent':v1,'change_percentage_points':v1-v0,'ratio_2023_to_2000':v1/v0})
    dump('R2_2_annual_error_sensitivity.csv',annual)
    dump('R2_2_temporal_changes.csv',deltas)
    dump('R2_2_error_transition_matrices.csv',matrices)
    dump('R2_2_endpoint_error_sensitivity.csv',endpoints)
    dump('R2_2_country_error_sensitivity.csv',countries)
    dump('R2_2_fixed_support_comparison.csv',support)
    dump('R2_2_hard_class_margin_bounds.csv',margins)
    dump('R2_7_component_dependence.csv',deps)
    dump('baseline_annual_reproduction.csv',residual)
    (OUT/'R2_2_margin_tipping_point.json').write_text(json.dumps(tipping,indent=2),encoding='utf-8')
    sources=[VALIDATION/'landcover_confusion_matrix_by_year.csv',VALIDATION/'landcover_confusion_matrix.csv',VALIDATION/'landcover_accuracy_by_year.csv',PKG/'data/figure_source/Fig2_annual_MEHI_summary.csv',INPUT/'classification_sensitivity_inputs_150m.csv.gz']
    status={'status':'complete','years':YEARS,'annual_rows':len(x),'matched_rows':len(common),'input_sha256':{p.name:digest(p) for p in sources},'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyogrio':pyogrio.__version__,'assumptions':['Mapped-class conditional reference probabilities are applied to area fractions as an explicit sensitivity model, not as a design-unbiased correction.','Each year-specific model uses its full mapped-to-reference confusion matrix sampled from final maps at retained validation coordinates.','Observed boundary geometry, proximity, contact, sample membership and area weights are fixed during composition sensitivity.','No independent-pixel assumption, spatial bootstrap, calibrated confidence interval or ecological response is inferred.','Distance offsets and uniform index bounds are stress tests, not empirical error estimates.'],'temporal_changes':deltas,'endpoint_sensitivity':endpoints,'tipping':tipping}
    (OUT/'analysis_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
