"""QC only: allow explicit NA weights and report coverage among known weights only.

Created after the real format audit; no score, association, fitting, or prediction.
Never equate NA with zero, and never interpret conditional coverage as total coverage.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('entry104', ROOT / 'simulations/p41_qc_entry_104.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
q = entry.q
signature_audit = {}

def load_known_weights(path):
    names, weights, all_names, missing = [], [], [], 0
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or reader.fieldnames.count('gene') != 1 or reader.fieldnames.count('mol_score') != 1:
            raise ValueError('signature header')
        for row in reader:
            name, raw = row.get('gene'), row.get('mol_score')
            if not name or name.strip() != name:
                raise ValueError('signature gene')
            all_names.append(name)
            if raw == 'NA':
                missing += 1
                continue
            try:
                weight = float(raw)
            except (ValueError, TypeError):
                raise ValueError('invalid signature weight') from None
            if not np.isfinite(weight):
                raise ValueError('nonfinite signature weight')
            names.append(name)
            weights.append(weight)
    if len(set(all_names)) != len(all_names) or not names or not np.any(weights):
        raise ValueError('signature duplicated or no nonzero known weights')
    signature_audit.clear()
    signature_audit.update({'total_rows': len(all_names), 'explicit_NA_rows': missing,
                            'finite_weight_rows': len(names),
                            'interpretation': 'Coverage denominators include known finite weights only. NA is unknown, not zero; total-original-weight coverage is not identifiable.'})
    return names, np.asarray(weights)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    q.real_gate()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / 'reports/real_preflight').resolve()):
        raise ValueError('real output outside QC directory')
    try:
        paths = q.checked_real_paths()
        q.load_signature = load_known_weights
        q.stream_qc = entry.m.m.p99.strict_stream_qc
        result = q.run(paths, 'real')
        with paths['b6_samples'].open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        result['b6']['author_excluded_flag_counts'] = entry.m.m.exclusion_counts(rows)
        result['b6']['author_excluded_flag_interpretation'] = 'Counts only; no category automatically treated as included or excluded.'
        result['b6']['gtf_caution'] = 'Paper reports GRCh38.p13 and Ensembl 104; exact GTF bytes and gene identity remain unverified.'
        result['status'] = 'completed_qc_known_weights_only'
        result['signature_weight_format_audit'] = dict(signature_audit)
        result['coverage_interpretation'] = 'Known finite weight mass only; original NA weights are unknown and are not set to zero.'
        result = entry.m.enrich(result, rows, 'real')
        result = entry.add_cardio_keys(result, paths['cardio_counts'], paths['cardio_function'])
    except Exception as exc:
        entry.m.m.safe_error(out, 'real', exc)
        raise SystemExit(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'scientific_association_performed': False,
                      'cardio_main_status': result['cardiopedia']['audit']['main_status'],
                      'b6_samples': result['b6']['sample_count'],
                      'signature_format': result['signature_weight_format_audit']}, indent=2))

if __name__ == '__main__':
    main()
