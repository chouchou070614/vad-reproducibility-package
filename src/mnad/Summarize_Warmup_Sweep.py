"""Create one CSV for a K-means warm-up-length ablation matrix."""
import argparse
import csv
from pathlib import Path


parser = argparse.ArgumentParser(description='Summarize K-MNAD warm-up sweep')
parser.add_argument('--experiment-root', type=Path, required=True)
parser.add_argument('--warmups', type=int, nargs='+', required=True)
parser.add_argument('--run-prefix', required=True,
                    help='run tag prefix, e.g. ped2_kmeans_warmup')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

rows = []
for warmup in args.warmups:
    result_dir = args.experiment_root / (
        f'mmm_ped2_{args.run_prefix}_{warmup}_4090_results')
    source = result_dir / 'auc_results.csv'
    if not source.is_file():
        raise FileNotFoundError(f'Missing completed result: {source}')
    with source.open(newline='', encoding='utf-8') as stream:
        records = {row['method']: row['frame_auc_percent']
                   for row in csv.DictReader(stream)}
    rows.append([
        warmup,
        records.get('K-MNAD MNAD-R (aligned)', ''),
        records.get('K-MNAD MNAD-P', ''),
        records.get('K-MNAD R+P fusion', ''),
        records.get('K-MNAD R+P fusion + Gaussian', ''),
    ])

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['warmup_epochs', 'k_mnad_r_auc', 'k_mnad_p_auc',
                     'k_fusion_auc', 'k_fusion_gaussian_auc'])
    writer.writerows(rows)

print('Warm-up sweep summary')
for row in rows:
    print(', '.join(map(str, row)))
print('Saved:', args.output)
