"""Fixed-cohort descriptive log-function slope; no file access."""
import numpy as np

def descriptive_effect(score,function):
    s=np.asarray(score,float);y=np.asarray(function,float)
    if s.ndim!=1 or y.shape!=s.shape or s.size<3 or not np.isfinite(s).all() or not np.isfinite(y).all():return {'status':'unavailable_invalid'}
    if np.any(y<=0):return {'status':'unavailable_nonpositive'}
    spread=float(np.quantile(s,.75)-np.quantile(s,.25))
    if spread<=0:return {'status':'unavailable_zero_iqr'}
    i,j=np.triu_indices(len(s),1);dx=s[j]-s[i];eligible=dx!=0
    if not eligible.any():return {'status':'unavailable_equal_scores'}
    slopes=(np.log(y[j[eligible]])-np.log(y[i[eligible]]))/dx[eligible]
    if not np.isfinite(slopes).all():return {'status':'unavailable_nonfinite_slopes'}
    slope=float(np.median(slopes));scaled=slope*spread
    if not np.isfinite(scaled) or scaled>np.log(np.finfo(float).max) or scaled<np.log(np.finfo(float).tiny):
        return {'status':'unavailable_exponent_range','pairs_used':int(eligible.sum())}
    return {'status':'ok','log_slope_per_score_unit':slope,'score_iqr':spread,
            'log_change_per_score_iqr':float(scaled),'function_ratio_per_score_iqr':float(np.exp(scaled)),
            'pairs_used':int(eligible.sum()),'pairs_excluded_equal_scores':int((~eligible).sum()),
            'minimum_nonzero_score_gap':float(np.min(np.abs(dx[eligible]))),
            'interpretation':'fixed-conditions descriptive association, no causal claim or population uncertainty'}
