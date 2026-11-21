"""Separate, fixed-file quality-control gate. Does not implement biological analysis."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
q3=load('qa142_for144','simulations/p41_rna_qc_synthetic_142.py')
q2=load('qa144_generic_b6','simulations/p41_b6_qc_generic_144.py')
PROGRAMS=('structural_exclusive','conduction_exclusive','ecm','hypoxia','oxphos','upr','fibroblast_marker','structural_original','conduction_original')
COMMON={'simulations/p41_qc_only_gate_144.py','simulations/p41_rna_qc_synthetic_142.py',
        'simulations/p41_b6_qc_generic_144.py','simulations/p41_feature_qc_141.py',
        'simulations/p41_gate_adapter_86.py','simulations/p41_qc_entry_105_known_weights.py',
        'simulations/p41_v3_b6_runner_119.py','simulations/p41_v3_primary_runner_118.py',
        'simulations/p41_v3_b6_plan_116.py','simulations/p41_v3_primary_core_114.py',
        'simulations/p41_qc_entry_102.py','simulations/p41_qc_entry_103.py','simulations/p41_qc_entry_104.py',
        'simulations/p41_qc_entry_99.py','simulations/p41_qc_preflight_98.py',
        'simulations/p41_effect_diagnostics_84.py','simulations/p41_early_sensitivity_90.py',
        'simulations/p41_within_between_94.py','simulations/p41_burden_100.py'}
PROFILES={
 '3d':{'scope':'P41_V4_QA_3D_ONLY','roles':{'counts','signature','incidence'},'code':COMMON,
       'manifest':ROOT/'theory/V4_QA_3D_MANIFEST.json','permit':ROOT/'seal/REAL_QA_USER_AUTHORIZATION_V4_3D.json'},
 'b6':{'scope':'P41_V4_QA_B6_ONLY','roles':{'counts','samples','signature','programs','gtf'},'code':COMMON,
       'program_names':PROGRAMS,'manifest':ROOT/'theory/V4_QA_B6_MANIFEST.json','permit':ROOT/'seal/REAL_QA_USER_AUTHORIZATION_V4_B6.json'}}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def checked(name):
    if name not in PROFILES:raise PermissionError('unknown QC scope')
    profile=PROFILES[name];permit_path=Path(profile['permit']);manifest_path=Path(profile['manifest'])
    if not permit_path.is_file():raise PermissionError('QA permission absent')
    permit=json.loads(permit_path.read_text(encoding='utf8'))
    if permit.get('scope')!=profile['scope'] or permit.get('explicit_user_qc_authorization') is not True or permit.get('manifest_sha256')!=digest(manifest_path):raise PermissionError('QA permission invalid')
    doc=json.loads(manifest_path.read_text(encoding='utf8'))
    if doc.get('scope')!=profile['scope'] or set(doc.get('inputs',{}))!=profile['roles'] or set(doc.get('implementation',{}))!=profile['code']:raise ValueError('QA manifest roles or implementation')
    if name=='b6' and tuple(doc.get('program_names',()))!=tuple(profile['program_names']):raise ValueError('QA program names mismatch')
    for rel,expected in doc['implementation'].items():
        path=(ROOT/rel).resolve()
        if not path.is_relative_to(ROOT) or digest(path)!=expected:raise ValueError('QA implementation hash mismatch')
    paths={}
    for role,item in doc['inputs'].items():
        path=Path(item['path']).resolve()
        if not path.is_file() or path.stat().st_size!=item['size_bytes'] or digest(path)!=item['sha256']:raise ValueError('QA input hash mismatch '+role)
        paths[role]=path
    return paths

def execute(name):
    paths=checked(name)
    if name=='3d':
        labels=json.loads(paths['incidence'].read_text(encoding='utf8'))['noncontrol_labels_on_all_five']
        labels=sorted(labels)
        if len(labels)!=47 or len(set(labels))!=47 or any(not isinstance(x,str) or not x for x in labels):raise ValueError('QA label design')
        oldsafe,oldlabels=q3.safe,q3.LABELS
        allowed={paths['counts'],paths['signature']}
        def safe(path):
            path=Path(path).resolve()
            if path not in allowed:raise PermissionError('unapproved QA input')
            return path
        q3.safe,q3.LABELS=safe,tuple(labels)
        try:result=q3.inspect(paths['counts'],paths['signature'])
        finally:q3.safe,q3.LABELS=oldsafe,oldlabels
    else:
        oldsafe,oldnames=q2.safe,q2.PROGRAM_NAMES
        allowed=set(paths.values())
        def safe(path):
            path=Path(path).resolve()
            if path not in allowed:raise PermissionError('unapproved QA input')
            return path
        q2.safe,q2.PROGRAM_NAMES=safe,tuple(PROFILES[name]['program_names'])
        try:result=q2.inspect(paths)
        finally:q2.safe,q2.PROGRAM_NAMES=oldsafe,oldnames
    result['status']='completed_real_qc_only';result['synthetic_only']=False
    result['scope']=PROFILES[name]['scope'];result['score_or_function_evaluated']=False
    result['warning']='QA measures file/feature computability only; not control validity, donor independence or biological analysis'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('3d','b6'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to((ROOT/'reports/real_preflight').resolve()):raise PermissionError('QA output outside preflight')
    if out.exists():raise FileExistsError('no overwrite')
    result=execute(a.scope)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'status':result['status'],'scope':result['scope'],'score_or_function_evaluated':False}))
if __name__=='__main__':main()
