"""Recompute published route summaries from the eligible neighbourhood records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from transition.route_analysis import aggregate_existing_routes, eligible_rows, classify_dominant_route

ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return pd.read_csv(ROOT / 'data' / name, float_precision='round_trip')

def close(actual, expected, kind='fraction'):
    tolerance = {'fraction': 1e-10, 'area': 1e-6, 'count': 0}[kind]
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)

r = read('supporting/Fig5_grid_route_classification.csv.gz')
assert len(r) == 79505 and eligible_rows(r).all()
assert r.GU_A3.nunique() == 19
assert int(r.pMEHI.gt(0).sum()) == 35619
assert int(r.pMEHI.gt(.005).sum()) == 32242
g = aggregate_existing_routes(r).set_index('route').sort_index()
reported = read('figure_source/Fig5_global_route_shares.csv').set_index('route').sort_index()
assert g.index.equals(reported.index)
close(g.weight_m2, reported.weight_m2_sum, 'area')
close(g.grid_count, reported.grid_count, 'count')
close(g.share_of_hardening_weight_percent, reported.share_of_hardening_weight_percent)
close(reported.share_of_hardening_weight_percent.sum(), 100)

c = read('figure_source/Fig5_country_route_shares.csv').set_index(['GU_A3', 'route']).sort_index()
hard = r[r.pMEHI.gt(.005)]
cg = hard.groupby(['GU_A3', 'route']).agg(weight=('weight_m2', 'sum'), n=('route', 'size')).sort_index()
assert cg.index.equals(c.index)
close(cg.weight, c.weight_m2_sum, 'area')
close(cg.n, c.grid_count, 'count')
denom = hard.groupby('GU_A3').weight_m2.sum()
close(c.country_hardening_weight_m2, c.index.get_level_values(0).map(denom), 'area')
close(c.share_of_country_hardening_weight_percent, 100 * cg.weight / c.country_hardening_weight_m2)

assoc = read('figure_source/Fig5_country_association.csv').set_index('GU_A3').sort_index()
country_rows = []
for country, sub in r.groupby('GU_A3'):
    country_rows.append({'GU_A3': country, 'pMEHI': np.average(sub.pMEHI, weights=sub.weight_m2),
                         'replacement': np.average(sub.soft_to_built_share_of_changed, weights=sub.weight_m2),
                         'weight': sub.weight_m2.sum(), 'n': len(sub)})
calculated = pd.DataFrame(country_rows).set_index('GU_A3').sort_index()
assert calculated.index.equals(assoc.index)
close(calculated.pMEHI, assoc.country_weighted_mean_pMEHI)
close(calculated.replacement, assoc.soft_buffer_to_built_share_changed_area)
close(calculated.weight, assoc.total_weight_m2, 'area')
close(calculated.n, assoc.n_grid, 'count')
rho = calculated.pMEHI.corr(calculated.replacement, method='spearman')
close(rho, 0.6807017543859648)

end = read('figure_source/Fig5_endpoint_groups.csv')
counts = r.pmehi_group.value_counts()
assert counts.to_dict() == {'softening': 35386, 'hardening': 32242, 'near_neutral': 11877}
for label, sub in end.groupby('group'):
    if label == 'all_valid':
        assert sub.n_grid.eq(len(r)).all()
        assert sub.total_weight_m2.isna().all()  # Correlation rows do not report a group area.
        continue
    assert sub.n_grid.eq(counts[label]).all()
    close(sub.total_weight_m2, r.loc[r.pmehi_group.eq(label), 'weight_m2'].sum(), 'area')

t = read('figure_source/Fig5_transition_matrix.csv')
area_col = next(c for c in ['area_m2', 'transition_area_m2', 'area_m2_sum'] if c in t.columns)
close(t.share_of_all_transition_percent, 100 * t[area_col] / t[area_col].sum())
close(t.share_of_all_transition_percent.sum(), 100)

threshold = read('supporting/route_threshold_global_shares.csv')
base = threshold[threshold.route_threshold.eq(.3)].set_index('route').sort_index()
assert base.index.equals(reported.index)
close(base.weight_m2_sum, reported.weight_m2_sum, 'area')
close(base.grid_count, reported.grid_count, 'count')
close(base.share_of_hardening_weight_percent, reported.share_of_hardening_weight_percent)
grid = read('supporting/route_threshold_grid_reclassification.csv')
flows = read('supporting/route_classification_flow_inputs.csv.gz')
assert len(flows) == len(r) and not flows.duplicated(['GU_A3', 'GRID_ID']).any()
assert np.isfinite(flows.select_dtypes('number')).all().all()
assert flows.delta_other_share.between(-1, 1).all()
assert flows.filter(regex='_m2$').ge(0).all().all()
full = r.merge(flows, on=['GU_A3', 'GRID_ID'], how='left', validate='one_to_one')
assert len(full) == len(r) and not full.isna().any().any()
close(full.soft_to_built_m2, full[['tidalflat_to_built_m2', 'nonmangroveveg_to_built_m2', 'water_to_built_m2']].sum(axis=1), 'area')
reclassified = {}
for tau, sub in threshold.groupby('route_threshold'):
    check = grid[grid.route_threshold.eq(tau)].set_index('route')
    sh = sub.set_index('route')
    labels = full.apply(lambda row: classify_dominant_route(row, float(tau)), axis=1)
    if tau == .3:
        np.testing.assert_array_equal(labels.to_numpy(), full.route.to_numpy())
    classified = full.assign(route=labels)
    generated = aggregate_existing_routes(classified).set_index('route').loc[sh.index]
    close(generated.grid_count, sh.grid_count, 'count')
    close(generated.weight_m2, sh.weight_m2_sum, 'area')
    changed = int(labels.ne(full.route).sum())
    assert check.number_of_changed_route_labels.eq(changed).all()
    reclassified[str(tau)] = {'rows': len(labels), 'changed_from_primary': changed,
                              'counts_and_weights_agree': True}
    assert int(sub.grid_count.sum()) == 35619
    close(sub.share_of_hardening_weight_percent.sum(), 100)
    close(sh.weight_m2_sum, check.loc[sh.index, 'global_weight_m2'], 'area')
    close(check.global_weight_m2.sum(), r.weight_m2.sum(), 'area')
    close(check.global_route_share_percent.sum(), 100)
assert grid.loc[grid.route_threshold.eq(.3), 'number_of_changed_route_labels'].eq(0).all()

summary = read('supporting/route_analysis_summary.csv').set_index('result_id')
close(summary.loc['valid_grid_count', 'value'], len(r), 'count')
for route, row in reported.iterrows():
    close(summary.loc['global_route_share_' + route, 'value'], row.share_of_hardening_weight_percent)

report = {'status': 'passed', 'route_neighbourhoods': len(r), 'positive_pmehi_neighbourhoods': 35619,
          'pmehi_above_0p005_neighbourhoods': 32242, 'country_count': len(assoc),
          'country_association_spearman_rho': rho,
          'country_association_definition': 'Minimum-endpoint-area weighted mean of neighbourhood replacement fractions',
          'matrix_denominator': 'All observed paired-transition area in pMEHI > 0.005 neighbourhoods',
          'general_matched_neighbourhoods': 79509,
          'complete_rule_reclassification': reclassified,
          'checks': ['Global route counts, weights and percentages', 'Country route support and weights',
                     'Country weighted means and Spearman association', 'Endpoint group support',
                     'Transition matrix denominator', 'Three threshold summaries and consistent primary threshold',
                     'Analysis-summary values']}
out = ROOT / 'reproduced/qa/route_statistics.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
