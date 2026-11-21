"""Pure V3 B6 plan and descriptive summaries; no file loading or authorization."""
import collections
import math
import re
import numpy as np
from scipy.stats import rankdata

BLOCK_FIELDS=('LibraryID','Line','Batch','Treatment','Rank','Time','Block')

def build_plan(samples,assays):
    ids=set();blocks=collections.defaultdict(list);names=set()
    for s in samples:
        if any(not isinstance(s.get(k),str) or not s[k] for k in BLOCK_FIELDS):raise ValueError('missing block metadata')
        if s.get('Excluded') not in ('TRUE','FALSE'):raise ValueError('invalid author flag')
        sid=s.get('sample_id')
        if not sid or sid in ids:raise ValueError('duplicate sample ID')
        ids.add(sid);names.add(s['Treatment'])
        match=re.fullmatch(r'Con([1-9][0-9]*)',s['Rank'])
        if not match:raise ValueError('invalid rank')
        blocks[tuple(s[k] for k in BLOCK_FIELDS)].append(s)
    by_condition=collections.defaultdict(list)
    for key,rows in blocks.items():
        if len(rows)!=3:raise ValueError('expected complete original triplicate metadata')
        library,line,batch,drug,rank,time,block=key
        by_condition[(drug,rank[3:])].append({'id':key,'retained':sorted(r['sample_id'] for r in rows if r['Excluded']=='FALSE')})
    by_assay=collections.defaultdict(list);ambiguous=set()
    for index,row in enumerate(assays):
        rank=str(row.get('rank',''))
        endpoint=row.get('assay')
        if not re.fullmatch(r'[1-9][0-9]*',rank) or not endpoint:raise ValueError('invalid assay metadata')
        candidates={row.get(k) for k in ('treatment','label') if row.get(k) in names}
        if len(candidates)>1:
            for name in candidates:ambiguous.add((endpoint,name,rank))
        elif len(candidates)==1:by_assay[(endpoint,next(iter(candidates)),rank)].append(index)
    endpoints=sorted({row['assay'] for row in assays})
    plan=[]
    for drug,rank in sorted(by_condition):
        for endpoint in endpoints:
            k=(endpoint,drug,rank);blocklist=sorted(by_condition[(drug,rank)],key=lambda v:v['id'])
            matches=by_assay[k];reasons=[]
            if k in ambiguous:reasons.append('ambiguous_assay_name')
            if not matches:reasons.append('missing_assay')
            if len(matches)>1:reasons.append('duplicate_assay')
            if any(len(b['retained'])<2 for b in blocklist):reasons.append('insufficient_retained_block')
            if len({(b['id'][1],b['id'][2],b['id'][5]) for b in blocklist})>1:reasons.append('line_batch_time_ambiguous')
            plan.append({'assay':endpoint,'drug':drug,'rank':rank,'blocks':blocklist,
                         'assay_index':matches[0] if len(matches)==1 else None,
                         'status':'unavailable' if reasons else 'eligible', 'reasons':sorted(reasons)})
    return plan

def enforce_min_ranks(plan,minimum=3):
    counts=collections.defaultdict(set)
    for row in plan:
        if row['status']=='eligible':counts[(row['assay'],row['drug'])].add(row['rank'])
    out=[]
    for original in plan:
        row={**original,'reasons':list(original['reasons'])}
        if row['status']=='eligible' and len(counts[(row['assay'],row['drug'])])<minimum:
            row['status']='unavailable';row['reasons'].append('fewer_than_minimum_eligible_ranks')
        out.append(row)
    return out

def summarize(plan,scores,assay_values):
    result=[]
    for item in plan:
        record={'assay':item['assay'],'drug':item['drug'],'rank':item['rank'],
                'status':item['status'],'reasons':list(item['reasons'])}
        if item['status']=='eligible':
            try:
                blocks=[[float(scores[sid]) for sid in b['retained']] for b in item['blocks']]
                assay=float(assay_values[item['assay_index']])
                if not math.isfinite(assay) or any(not math.isfinite(x) for block in blocks for x in block):raise ValueError
            except (KeyError,IndexError,ValueError,TypeError):
                record['status']='unavailable';record['reasons']=['missing_or_invalid_selected_measurement']
            else:
                record['mean_score']=sum(sum(x)/len(x) for x in blocks)/len(blocks)
                record['assay_value']=assay
                record['unit']='one_assay_per_condition'
        result.append(record)
    return result

def endpoint_summary(records):
    def rho(a,b):
        x=rankdata(a);y=rankdata(b)
        return None if len(x)<2 or np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
    outputs={}
    for endpoint in sorted({r['assay'] for r in records}):
        rows=[r for r in records if r['assay']==endpoint]
        if any(r['reasons']==['missing_or_invalid_selected_measurement'] for r in rows):
            outputs[endpoint]={'status':'unavailable_selected_measurement','drugs':[]};continue
        groups=collections.defaultdict(list)
        for row in rows:
            if row['status']=='eligible':groups[row['drug']].append(row)
        drugs=[]
        for drug,vals in sorted(groups.items()):
            vals=sorted(vals,key=lambda r:int(r['rank']))
            drugs.append({'drug':drug,'ranks':[v['rank'] for v in vals],
                          'score_mean':float(np.mean([v['mean_score'] for v in vals])),
                          'assay_mean':float(np.mean([v['assay_value'] for v in vals])),
                          'within_rank_rho':rho([v['mean_score'] for v in vals],[v['assay_value'] for v in vals])})
        outputs[endpoint]={'status':'ok' if drugs else 'unavailable_no_eligible_drugs',
                           'drug_mean_rho':rho([d['score_mean'] for d in drugs],[d['assay_mean'] for d in drugs]) if drugs else None,
                           'drugs':drugs,
                           'interpretation':'condition-level descriptive only; unequal dose ranges and assay identity remain limitations'}
    return outputs
