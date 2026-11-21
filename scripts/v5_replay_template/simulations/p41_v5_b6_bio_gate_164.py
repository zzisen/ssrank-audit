"""Gated V5 B6 descriptive analysis. Real execution requires later user authorization."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('b119_for155','simulations/p41_v3_b6_runner_119.py')
mag=load('mag133_for155','simulations/p41_b6_magnitude_133.py')
base=load('qa144_for155_code','simulations/p41_qc_only_gate_144.py')
PROFILE={'scope':'P41_V5_B6_BIO_DESCRIPTIVE','roles':{'counts','samples','assays','signature','programs','gtf'},
         'code':base.COMMON|{'simulations/p41_v5_b6_bio_gate_164.py','simulations/p41_b6_magnitude_133.py','simulations/p41_effect_size_131.py',
                             'theory/P4_1_SAP_V5_POST_QA_REVIEW_DRAFT_2026-11-17.md','theory/v5_sap_post_qa_local_lock.json',
                             'theory/P4_1_V5_BIO_GATES_PROTOCOL_163_164_2026-11-17.md'},
         'manifest':ROOT/'theory/V5_B6_BIO_MANIFEST.json','permit':ROOT/'seal/REAL_SCIENTIFIC_USER_AUTHORIZATION_V5_B6.json'}

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
    paths=checked();oldgate,oldpaths=b.real_gate,b.real_paths
    def guarded():checked()
    def approved_paths():return checked()
    b.real_gate,b.real_paths=guarded,approved_paths
    try:original=b.evaluate(paths,b.REAL_PROGRAMS,'real')
    finally:b.real_gate,b.real_paths=oldgate,oldpaths
    descriptions={}
    for name,program in original['programs'].items():
        descriptions[name]=mag.calculate(program['conditions']) if program['status']=='ok' else {'status':'unavailable_score'}
    return {'status':'completed_bounded_descriptive','scope':PROFILE['scope'],'original':original,'magnitude':descriptions,
            'limitations':['retrospective condition-level description','no same-well or causal claim',
                           'function time, line, dose and donor independence may be unidentified']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to((ROOT/'reports/real_analysis').resolve()):raise PermissionError('analysis output outside directory')
    if out.exists():raise FileExistsError('no overwrite')
    result=evaluate();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'status':result['status'],'scope':PROFILE['scope']}))
if __name__=='__main__':main()
