"""Analyze completed K-MNAD warm-up experiments and select a fine interval."""
import argparse
import csv
import json
from pathlib import Path


parser = argparse.ArgumentParser(description='Analyze K-MNAD warm-up AUC results')
parser.add_argument('--result-specs', nargs='+', required=True,
                    help='entries in the form WARMUP=/absolute/path/auc_results.csv')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--plan-output', type=Path,
                    help='optional JSON plan for the best adjacent coarse interval')
args = parser.parse_args()


def read_result(spec):
    warmup_text, path_text = spec.split('=', 1)
    warmup = int(warmup_text)
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f'Missing completed result: {path}')
    with path.open(newline='', encoding='utf-8') as stream:
        values = {row['method']: float(row['frame_auc_percent'])
                  for row in csv.DictReader(stream)}
    required = {
        'K-MNAD MNAD-R (aligned)': 'k_mnad_r_auc',
        'K-MNAD MNAD-P': 'k_mnad_p_auc',
        'K-MNAD R+P fusion': 'k_fusion_auc',
        'K-MNAD R+P fusion + Gaussian': 'k_fusion_gaussian_auc',
    }
    missing = [name for name in required if name not in values]
    if missing:
        raise KeyError(f'{path} is missing: {missing}')
    return {
        'warmup_epochs': warmup,
        **{column: values[name] for name, column in required.items()},
        'source': str(path),
    }


rows = sorted((read_result(spec) for spec in args.result_specs),
              key=lambda row: row['warmup_epochs'])
if len({row['warmup_epochs'] for row in rows}) != len(rows):
    raise ValueError('Warm-up values must be unique')

args.output.parent.mkdir(parents=True, exist_ok=True)
columns = ['warmup_epochs', 'k_mnad_r_auc', 'k_mnad_p_auc',
           'k_fusion_auc', 'k_fusion_gaussian_auc', 'source']
with args.output.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

best = max(rows, key=lambda row: row['k_fusion_gaussian_auc'])
print(f"Best observed warm-up: {best['warmup_epochs']} "
      f"(AUC={best['k_fusion_gaussian_auc']:.4f}%)")
print('Saved:', args.output)

if args.plan_output:
    if len(rows) < 2:
        raise ValueError('At least two coarse points are required for refinement')
    pairs = list(zip(rows[:-1], rows[1:]))
    left, right = max(
        pairs,
        key=lambda pair: (pair[0]['k_fusion_gaussian_auc'] +
                          pair[1]['k_fusion_gaussian_auc']) / 2.0)
    fine_warmups = list(range(left['warmup_epochs'] + 1,
                              right['warmup_epochs']))
    plan = {
        'selection_rule': 'largest mean final AUC among adjacent coarse points',
        'coarse_left': left['warmup_epochs'],
        'coarse_left_auc': left['k_fusion_gaussian_auc'],
        'coarse_right': right['warmup_epochs'],
        'coarse_right_auc': right['k_fusion_gaussian_auc'],
        'fine_warmups': fine_warmups,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print('Selected refinement interval:',
          f"{left['warmup_epochs']} to {right['warmup_epochs']}")
    print('New unit-step warm-ups:', ' '.join(map(str, fine_warmups)))
    print('Plan saved:', args.plan_output)
