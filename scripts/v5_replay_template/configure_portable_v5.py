"""Configure a local analysis replay with source paths and exact byte checks."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,ROOT/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
if len(sys.argv)!=2:raise SystemExit('Pass a JSON file with source input paths')
conf=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'));spec=json.loads((ROOT/'input_spec.json').read_text(encoding='utf8'))
mods={'3d':load('portable3d','simulations/p41_v5_3d_bio_gate_163.py'),'b6':load('portableb6','simulations/p41_v5_b6_bio_gate_164.py')}
prepared=[]
for name,mod in mods.items():
    p=mod.PROFILE;fixed=spec[name]
    if set(conf[name])!=p['roles'] or set(fixed['inputs'])!=p['roles'] or set(fixed['implementation'])!=p['code']:raise ValueError('role mismatch')
    inputs={}
    for role,value in conf[name].items():
        path=Path(value).resolve();expected=fixed['inputs'][role]
        if not path.is_file() or path.stat().st_size!=expected['size_bytes'] or sha(path)!=expected['sha256']:raise ValueError('source mismatch '+name+' '+role)
        inputs[role]={'path':str(path),'size_bytes':path.stat().st_size,'sha256':expected['sha256']}
    for rel,value in fixed['implementation'].items():
        if sha(ROOT/rel)!=value:raise ValueError('code mismatch '+rel)
    if p['manifest'].exists() or p['permit'].exists():raise FileExistsError('replay is already configured')
    doc={'scope':p['scope'],'inputs':inputs,'implementation':fixed['implementation'],'local_replay_only':True}
    prepared.append((p,doc))
for p,doc in prepared:
    with p['manifest'].open('x',encoding='utf8') as f:json.dump(doc,f,indent=2)
    token={'scope':p['scope'],'explicit_user_biological_authorization':True,'manifest_sha256':sha(p['manifest']),'note':'Local replay initiated by operator; not external registration or source permission.'}
    with p['permit'].open('x',encoding='utf8') as f:json.dump(token,f,indent=2)
print('Configured both V5 replays after validating inputs and implementation')
