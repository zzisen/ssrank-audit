#!/usr/bin/env python3
"""Rebuild the four V5 seed tables from verified public inputs.

This retrospective replay copies the original hash-bound V5 implementation to
an isolated temporary workspace, validates its input and code hashes, and runs
the historical 3D, B6, and repeat-function calculations. Its generated local
path manifests and operator markers stay in the temporary workspace; they are
not independent preregistration or evidence of prospective blinding. Public
source terms remain applicable. Existing output files are never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[1]
FILES=('3d_treatment_summary.csv','3d_plate_treatment_values.csv',
       'b6_all_program_condition_statuses.csv','b6_drug_summary.csv')

def run(*command):
    print('+', ' '.join(str(x) for x in command),flush=True)
    subprocess.run([str(x) for x in command],check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(os.environ.get('CARDIAC_DATA_DIR',REPO/'data')))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();data=args.data_dir.resolve();out=args.output_dir.resolve()
    existing=[str(out/f) for f in FILES if (out/f).exists()]
    if existing:raise FileExistsError('Refusing to overwrite seed tables: '+', '.join(existing))
    template=REPO/'scripts/v5_replay_template'
    with tempfile.TemporaryDirectory(prefix='ssrank_v5_replay_') as temp:
        work=Path(temp)/'v5';shutil.copytree(template,work)
        (work/'seal').mkdir(exist_ok=True)
        results=work/'reports/real_analysis';results.mkdir(parents=True,exist_ok=True)
        source_programs=data/'reference/theory/fixed_programs_2026-11-17/fixed_programs.gmt'
        if not source_programs.is_file():raise FileNotFoundError('Run prepare_gene_sets.py to reconstruct the fixed GMT')
        programs=work/'theory/fixed_programs_2026-11-17/fixed_programs.gmt'
        programs.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_programs,programs)
        signature=data/'datasets/ReHeaT2_publication/41467_2025_62219_MOESM10_ESM.csv'
        config={
          '3d':{'counts':data/'datasets/GSE314310/GSE314310_RSEM_Counts_Cardiopedia-Ligand.xlsx',
                'function':data/'datasets/GSE314310/Cardiopedia_Table_S2_normalized_function.xlsx',
                'incidence':work/'prior_art/cardiopedia_treatment_plate_incidence_2026-11-16.json',
                'programs':programs,
                'repeat_metadata':work/'prior_art/cardiopedia_repeat_overlap_47_metadata_2026-11-16.json',
                'signature':signature},
          'b6':{'counts':data/'datasets/GSE244740/GSE244740_processed_data_counts.txt.gz',
                'samples':data/'datasets/GSE244740/iemt_a_2273378_sm6218.txt',
                'assays':data/'datasets/GSE244740/iemt_a_2273378_sm6211.txt',
                'gtf':data/'reference/Homo_sapiens.GRCh38.104.gtf.gz',
                'programs':programs,'signature':signature}}
        missing=[str(p) for paths in config.values() for p in paths.values() if not p.is_file()]
        if missing:raise FileNotFoundError('Missing replay inputs; run prepare_inputs.py and prepare_gene_sets.py: '+', '.join(sorted(set(missing))))
        confpath=work/'inputs.local.json'
        confpath.write_text(json.dumps({k:{role:str(path) for role,path in v.items()} for k,v in config.items()},indent=2),encoding='utf-8')
        run(sys.executable,work/'configure_portable_v5.py',confpath)
        run(sys.executable,work/'simulations/p41_v5_3d_bio_gate_163.py','--output',results/'V5_3D_BIO_DESCRIPTIVE_2026-11-18.json')
        run(sys.executable,work/'simulations/p41_v5_b6_bio_gate_164.py','--output',results/'V5_B6_BIO_DESCRIPTIVE_2026-11-18.json')
        run(sys.executable,work/'analysis_20261118/repeat_complete_pairs.py','--run-real')
        run(sys.executable,REPO/'scripts/export_v5_seed_tables.py','--replay-dir',results,'--output-dir',out)
    print('Rebuilt V5 seed tables in',out,flush=True)

if __name__=='__main__':main()
