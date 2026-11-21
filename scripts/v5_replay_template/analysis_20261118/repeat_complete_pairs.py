"""Post-QC repair: exact pre-existing paired keys; never cross-experiment pair wells."""
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1]
def assemble(rows,labels,expected):
    times=('04H','24H');keys=defaultdict(list);wells=Counter()
    for r in rows:
        if r['label'] not in labels:continue
        if r['time'] not in times or not r['well'] or not r['plate']:raise ValueError('invalid metadata')
        k=(r['plate'],r['label']);keys[(r['time'],k)].append(r)
        wells[(r['time'],r['plate'],r['well'])]+=1
    if any(n!=1 for n in wells.values()):raise ValueError('duplicate selected well')
    for key,entries in keys.items():
        if len(entries)!=1:raise ValueError('duplicate selected condition/time')
    allkeys={key for _,key in keys};complete={key for key in allkeys if all((t,key) in keys for t in times)}
    if complete!=expected:raise ValueError('complete identity keys differ from fixed metadata')
    result=defaultdict(list)
    for key in sorted(complete):
        a,b=(keys[(t,key)][0] for t in times)
        if a['well']!=b['well']:raise ValueError('within-experiment timepoint well mismatch')
        for r in (a,b):
            if not isinstance(r['value'],(int,float)) or isinstance(r['value'],bool) or not math.isfinite(r['value']):raise ValueError('nonfinite force')
        result[key[1]].append(float(b['value']))
    if set(result)!=set(labels):raise ValueError('missing treatment')
    orphans={t:sum((t,k) in keys and (other,k) not in keys for k in allkeys) for t,other in [('04H','24H'),('24H','04H')]}
    return {'means':{k:float(np.mean(result[k])) for k in labels},'counts':{k:len(result[k]) for k in labels},
            'complete_keys':len(complete),'orphan_keys':orphans}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-real',action='store_true');z=parser.parse_args()
    if not z.run_real:raise ValueError('explicit mode required')
    manifest=json.loads((ROOT/'theory/V5_3D_BIO_MANIFEST.json').read_text(encoding='utf8'))
    workbook=Path(manifest['inputs']['function']['path']);expected_hash=manifest['inputs']['function']['sha256']
    if sha(workbook)!=expected_hash:raise ValueError('workbook hash changed')
    primary=ROOT/'reports/real_analysis/V5_3D_BIO_DESCRIPTIVE_2026-11-18.json';d=json.loads(primary.read_text(encoding='utf8'))
    labels=d['original']['labels'];metadata=ROOT/'prior_art/cardiopedia_repeat_overlap_47_metadata_2026-11-16.json'
    if sha(metadata)!=manifest['inputs']['repeat_metadata']['sha256']:raise ValueError('repeat design changed')
    anon=json.loads(metadata.read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
    expected={(p,labels[i]) for i in range(len(labels)) for p in anon[f'FAKE_L{i:02d}']}
    if len(expected)!=167 or len(labels)!=47:raise ValueError('unexpected fixed design')
    rows=[];wb=load_workbook(workbook,read_only=True,data_only=True)
    for time,sheet in [('04H','Tab 1 Functional data 04H'),('24H','Tab 2 Functional data 24H')]:
        ws=wb[sheet];ws.reset_dimensions();it=ws.iter_rows(values_only=True);h=next(it);ix={v:i for i,v in enumerate(h)}
        for row in it:
            def get(k):return row[ix[k]] if ix[k]<len(row) else None
            ds,name,well,pt,val=(get(k) for k in ('Dataset','Treatment','Wellname','Plate_timepoint','Force'))
            if ds!='Independent_Experiment' or not isinstance(name,str) or name.strip() not in labels:continue
            if not isinstance(well,str) or not isinstance(pt,str):raise ValueError('bad identity')
            plate=pt.strip().split('_',1)[0]
            if pt.strip()!=plate+'_'+time:raise ValueError('time mismatch')
            rows.append({'time':time,'plate':plate,'label':name.strip(),'well':well.strip(),'value':val})
    wb.close();data=assemble(rows,labels,expected)
    scores=np.asarray(d['original']['main']['scores_by_plate_and_treatment'],float).mean(axis=0)
    vals=[data['means'][l] for l in labels];r=float(spearmanr(scores,vals).statistic)
    out={'scope':'post-QC exploratory repeat function, same labels, no independent RNA; not preregistered',
         'source_function_sha256':expected_hash,'source_primary_result_sha256':sha(primary),'repeat_design_sha256':sha(metadata),
         'n_treatments':len(labels),'n_complete_paired_keys':data['complete_keys'],'orphan_keys':data['orphan_keys'],
         'spearman_rho':r,'by_treatment':[{'label':l,'score_mean':float(x),'repeat_function_mean_24h':v,'n_repeat_plates':data['counts'][l]} for l,x,v in zip(labels,scores,vals)],
         'limitations':['post-QC exploratory repair','not independent RNA, donor, dose or vehicle verification','not same tissue paired across experiments']}
    dest=ROOT/'reports/real_analysis/V5_REPEAT_COMPLETE_PAIRS_EXPLORATORY_2026-11-18.json'
    with dest.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k!='by_treatment'},indent=2))
if __name__=='__main__':main()
