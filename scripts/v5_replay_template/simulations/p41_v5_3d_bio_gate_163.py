"""Gated V5 3D descriptive analysis. Never execute on real data without user authorization."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('primary118_for154','simulations/p41_v3_primary_runner_118.py')
v4=load('candidate134_for154','simulations/p41_signed_coverage_candidate_134.py')
recal=load('common161_for163','simulations/p41_common_feature_recal_161.py')
presentation=load('presentation153_for154','simulations/p41_bound_presentation_153.py')
base=load('qa144_for154_code','simulations/p41_qc_only_gate_144.py')
PROFILE={'scope':'P41_V5_3D_BIO_DESCRIPTIVE','roles':{'counts','function','signature','programs','incidence','repeat_metadata'},
         'code':base.COMMON|{'simulations/p41_v5_3d_bio_gate_163.py','simulations/p41_signed_coverage_candidate_134.py',
              'simulations/p41_effect_size_131.py','simulations/p41_rank_bound_fast_127.py',
              'simulations/p41_recalibration_139.py','simulations/p41_common_feature_recal_161.py','simulations/p41_bound_presentation_153.py',
              'theory/P4_1_SAP_V5_POST_QA_REVIEW_DRAFT_2026-11-17.md',
              'theory/v5_sap_post_qa_local_lock.json','theory/P4_1_V5_BIO_GATES_PROTOCOL_163_164_2026-11-17.md'},
         'manifest':ROOT/'theory/V5_3D_BIO_MANIFEST.json','permit':ROOT/'seal/REAL_SCIENTIFIC_USER_AUTHORIZATION_V5_3D.json'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def checked():
    permit=PROFILE['permit'];manifest=PROFILE['manifest']
    if not permit.is_file():raise PermissionError('V5 real biological analysis not authorized')
    token=json.loads(permit.read_text(encoding='utf8'))
    if token.get('scope')!=PROFILE['scope'] or token.get('explicit_user_biological_authorization') is not True or token.get('manifest_sha256')!=digest(manifest):raise PermissionError('V5 biological authorization invalid')
    doc=json.loads(manifest.read_text(encoding='utf8'))
    if doc.get('scope')!=PROFILE['scope'] or set(doc.get('inputs',{}))!=PROFILE['roles'] or set(doc.get('implementation',{}))!=PROFILE['code']:raise ValueError('V5 biological manifest mismatch')
    for name,sha in doc['implementation'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or digest(path)!=sha:raise ValueError('implementation mismatch')
    paths={}
    for role,item in doc['inputs'].items():
        path=Path(item['path']).resolve()
        if not path.is_file() or path.stat().st_size!=item['size_bytes'] or digest(path)!=item['sha256']:raise ValueError('input mismatch '+role)
        paths[role]=path
    return paths

def evaluate():
    paths=checked();labels=sorted(json.loads(paths['incidence'].read_text(encoding='utf8'))['noncontrol_labels_on_all_five'])
    repeats=json.loads(paths['repeat_metadata'].read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
    oldgate,oldpaths,oldload=p.real_gate,p.real_paths,p.adapter.load;captured={}
    def guarded():checked()
    def approved_paths():return checked()
    def capture(*args,**kwargs):
        audit,arrays=oldload(*args,**kwargs);captured['arrays']=arrays;return audit,arrays
    p.real_gate,p.real_paths,p.adapter.load=guarded,approved_paths,capture
    try:original=p.evaluate(paths,labels,repeats,p.REAL_PROGRAMS,'real')
    finally:p.real_gate,p.real_paths,p.adapter.load=oldgate,oldpaths,oldload
    if original['status']!='completed_descriptive' or original['main']['status']!='ok' or captured.get('arrays') is None:
        return {'status':'unavailable_original','original':original,'scope':PROFILE['scope']}
    counts,late,controls,ids,symbols,biotypes,repeat,early=captured['arrays']
    names,weights=p.sig.load_known_weights(paths['signature']);mapped=p.adapter.resolve(ids,symbols,biotypes,names);by_id={v:i for i,v in enumerate(ids)}
    present=np.array([r['strict'] is not None for r in mapped],dtype=bool);ix=[by_id[r['strict']] for r in mapped if r['strict'] is not None]
    log=p.core.log_cpm(counts);groups=np.zeros((47,235))
    for i in range(47):groups[i,i::47]=.2
    candidate=v4.evaluate(log[:,ix],weights,present,groups,late.mean(axis=0))
    if candidate['status']!='ok':return {'status':'unavailable_v4_candidate','candidate':candidate,'scope':PROFILE['scope']}
    original_means=np.asarray(original['main']['scores_by_plate_and_treatment']).mean(axis=0)
    if not np.allclose(candidate['score_means'],original_means,atol=1e-11,rtol=1e-11) or not np.isclose(candidate['rho_observed_feature_target'],original['main']['description']['effect']['primary_rho']):raise ValueError('old/new score inconsistency')
    for name,value in candidate['audit']['coverage_known_weights'].items():
        old=original['main']['audit']['coverage_known_finite_weights'][name]
        if (value is None)!=(old is None) or (value is not None and not np.isclose(value,old)):raise ValueError('old/new coverage inconsistency')
    x=log[:,ix]
    diagnosis=recal.diagnose(x.reshape(5,47,-1),weights,present)
    return {'status':'completed_bounded_descriptive','scope':PROFILE['scope'],'original':original,'candidate':candidate,
            'recalibration':diagnosis,'bound_presentation':presentation.present(candidate),
            'limitations':['retrospective fixed-cohort only','no causal, independent donor, physical force or population-inference claim']}

def main():
    p0=argparse.ArgumentParser();p0.add_argument('--output',type=Path,required=True);a=p0.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to((ROOT/'reports/real_analysis').resolve()):raise PermissionError('analysis output outside directory')
    if out.exists():raise FileExistsError('no overwrite')
    result=evaluate();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'status':result['status'],'scope':PROFILE['scope']}))
if __name__=='__main__':main()
