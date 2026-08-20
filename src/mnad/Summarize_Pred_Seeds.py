"""Summarize repeated MNAD-P runs from their saved test-score files."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='Summarize MNAD-P seed stability')
parser.add_argument('--scores', nargs='+', required=True,
                    help='NPZ score files, one per independent seed')
parser.add_argument('--output', required=True)
args = parser.parse_args()

rows = []
reference_labels = None
for score_path in args.scores:
    data = np.load(score_path)
    labels = data['labels']
    if reference_labels is None:
        reference_labels = labels
    elif not np.array_equal(reference_labels, labels):
        raise ValueError(f'Labels differ: {score_path}')
    auc = roc_auc_score(labels, data['scores']) * 100.0
    rows.append((Path(score_path).stem, auc, score_path))

values = np.asarray([row[1] for row in rows], dtype=np.float64)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['run', 'frame_auc_percent', 'score_file'])
    writer.writerows([[run, f'{auc:.4f}', path] for run, auc, path in rows])
    writer.writerow(['mean', f'{values.mean():.4f}', ''])
    writer.writerow(['std', f'{values.std(ddof=0):.4f}', ''])
    writer.writerow(['min', f'{values.min():.4f}', ''])
    writer.writerow(['max', f'{values.max():.4f}', ''])

for run, auc, _ in rows:
    print(f'{run}: {auc:.4f}%')
print(f'mean: {values.mean():.4f}%')
print(f'std:  {values.std(ddof=0):.4f}%')
print(f'range: {values.min():.4f}% to {values.max():.4f}%')
print('Saved:', output)
