"""Endpoint-specific descriptive magnitude from prevalidated condition records."""
import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('effect131_for_b6',ROOT/'simulations/p41_effect_size_131.py');effect=importlib.util.module_from_spec(s);s.loader.exec_module(effect)

def calculate(records):
    endpoints=sorted({r['assay'] for r in records});result={}
    for endpoint in endpoints:
        all_rows=[r for r in records if r['assay']==endpoint]
        if any('missing_or_invalid_selected_measurement' in r.get('reasons',[]) for r in all_rows):
            result[endpoint]={'status':'unavailable_selected_measurement','drugs':[]};continue
        candidates=[r for r in all_rows if r.get('status')=='eligible']
        drugs=[]
        for drug in sorted({r['drug'] for r in candidates}):
            vals=[r for r in candidates if r['drug']==drug]
            ranks=[r['rank'] for r in vals]
            if len(ranks)!=len(set(ranks)):raise ValueError('duplicate condition rank')
            if len(vals)<3:raise ValueError('unvalidated fewer than three ranks')
            x=np.array([r['mean_score'] for r in vals],float);y=np.array([r['assay_value'] for r in vals],float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('unvalidated nonfinite records')
            spread=float(np.quantile(x,.75)-np.quantile(x,.25))
            if spread<=0:
                drugs.append({'drug':drug,'status':'unavailable_zero_iqr'});continue
            i,j=np.triu_indices(len(x),1);dx=x[j]-x[i];use=dx!=0
            if not use.any():drugs.append({'drug':drug,'status':'unavailable_equal_scores'});continue
            slopes=(y[j[use]]-y[i[use]])/dx[use]
            if not np.isfinite(slopes).all():drugs.append({'drug':drug,'status':'unavailable_nonfinite_slopes'});continue
            raw=float(np.median(slopes))
            drugs.append({'drug':drug,'status':'ok','n_ranks':len(vals),'raw_slope_per_score_unit':raw,
                          'raw_change_per_score_iqr':raw*spread,'unit':'assay source unit not established',
                          'log_sensitivity':effect.descriptive_effect(x,y),'excluded_equal_score_pairs':int((~use).sum())})
        result[endpoint]={'status':'ok' if drugs else 'unavailable_no_eligible_drugs','drugs':drugs,
                          'interpretation':'within drug and assay only; no causal or dose-scale interpretation'}
    return result
