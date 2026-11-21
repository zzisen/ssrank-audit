"""Explicitly label conditional vs no-tie-assumption score bounds; no input I/O."""
import copy
def present(candidate):
    if candidate.get('status')!='ok':return {'status':'unavailable_candidate','conditional':None,'without_no_tie_assumption':None}
    b=copy.deepcopy(candidate.get('conditional_known_weight_sensitivity',{}));status=b.get('status')
    if status=='unavailable_observed_constant_features':
        return {'status':'unavailable_full_score_undefined','conditional':b,'without_no_tie_assumption':None}
    if status not in ('conditional_strict_ranks','unavailable_function_ties'):
        return {'status':'unavailable_bound_status','conditional':b,'without_no_tie_assumption':None}
    if status=='conditional_strict_ranks' and (b.get('lower') is None or b.get('upper') is None or not -1<=b['lower']<=b['upper']<=1):raise ValueError('invalid conditional rank bounds')
    return {'status':'reported_with_assumptions','conditional':b,
            'without_no_tie_assumption':{'status':'uninformative','lower':-1.,'upper':1.},
            'limits':['not a confidence interval','score target and missing-gene assumptions remain conditional','NA weights not recovered']}
