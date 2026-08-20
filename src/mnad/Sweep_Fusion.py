"""Report R/P score-fusion sensitivity without retraining either branch.

This utility is deliberately analysis-only: it reads the saved R and P score
files and writes a CSV of AUC values for a list of beta values.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='Sweep MNAD R/P fusion weights')
parser.add_argument('--recons-scores', required=True)
parser.add_argument('--pred-scores', required=True)
parser.add_argument('--betas', type=float, nargs='+',
                    default=[round(x / 10, 1) for x in range(11)])
parser.add_argument('--pred-offset', type=int, default=4)
parser.add_argument('--sigma', type=float, default=None,
                    help='also report per-video Gaussian-smoothed AUC')
parser.add_argument('--output', required=True)
args = parser.parse_args()

if any(beta < 0.0 or beta > 1.0 for beta in args.betas):
    parser.error('Every beta must be between 0 and 1')
if args.sigma is not None and args.sigma <= 0:
    parser.error('--sigma must be greater than 0')


def split_scores(path):
    data = np.load(path)
    names = data['video_names'].astype(str).tolist()
    lengths = data['lengths'].astype(int).tolist()
    scores, labels = data['scores'], data['labels']
    by_video_scores, by_video_labels = [], []
    start = 0
    for length in lengths:
        end = start + length
        by_video_scores.append(scores[start:end])
        by_video_labels.append(labels[start:end])
        start = end
    if start != len(scores) or start != len(labels):
        raise ValueError(f'Invalid score file: {path}')
    return str(data['dataset']), names, by_video_scores, by_video_labels


r_dataset, r_names, r_scores, r_labels = split_scores(args.recons_scores)
p_dataset, p_names, p_scores, p_labels = split_scores(args.pred_scores)
if r_dataset != p_dataset or r_names != p_names:
    raise ValueError('R/P score files do not describe the same videos')

aligned_r, aligned_p, aligned_labels = [], [], []
for name, r_score, p_score, r_label, p_label in zip(
        r_names, r_scores, p_scores, r_labels, p_labels):
    if len(r_score) - len(p_score) != args.pred_offset:
        raise ValueError(f'{name}: unexpected R/P frame offset')
    r_score = r_score[args.pred_offset:]
    r_label = r_label[args.pred_offset:]
    if not np.array_equal(r_label, p_label):
        raise ValueError(f'{name}: R/P labels are not aligned')
    aligned_r.append(r_score)
    aligned_p.append(p_score)
    aligned_labels.append(p_label)

r = np.concatenate(aligned_r)
p = np.concatenate(aligned_p)
labels = np.concatenate(aligned_labels)
r_auc = roc_auc_score(labels, r) * 100.0
p_auc = roc_auc_score(labels, p) * 100.0

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
rows = []
for beta in args.betas:
    fused_by_video = [
        beta * one_r + (1.0 - beta) * one_p
        for one_r, one_p in zip(aligned_r, aligned_p)]
    fused = np.concatenate(fused_by_video)
    fused_auc = roc_auc_score(labels, fused) * 100.0
    smooth_auc = None
    if args.sigma is not None:
        smoothed = np.concatenate([
            gaussian_filter1d(one_video, sigma=args.sigma)
            for one_video in fused_by_video])
        smooth_auc = roc_auc_score(labels, smoothed) * 100.0
    rows.append((beta, r_auc, p_auc, fused_auc, smooth_auc))

with output.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    header = ['beta_for_R', 'R_aligned_auc', 'P_auc', 'fused_auc']
    if args.sigma is not None:
        header.append(f'fused_gaussian_auc_sigma_{args.sigma:g}')
    writer.writerow(header)
    for beta, one_r_auc, one_p_auc, fused_auc, smooth_auc in rows:
        row = [f'{beta:.2f}', f'{one_r_auc:.4f}', f'{one_p_auc:.4f}',
               f'{fused_auc:.4f}']
        if smooth_auc is not None:
            row.append(f'{smooth_auc:.4f}')
        writer.writerow(row)

for beta, _, _, fused_auc, smooth_auc in rows:
    line = f'beta={beta:.2f}: fused AUC={fused_auc:.4f}%'
    if smooth_auc is not None:
        line += f', Gaussian AUC={smooth_auc:.4f}%'
    print(line)
best_fused = max(rows, key=lambda row: row[3])
print(f'Best exploratory fusion: beta={best_fused[0]:.2f}, AUC={best_fused[3]:.4f}%')
if args.sigma is not None:
    best_smooth = max(rows, key=lambda row: row[4])
    print(f'Best exploratory Gaussian: beta={best_smooth[0]:.2f}, '
          f'AUC={best_smooth[4]:.4f}%, sigma={args.sigma:g}')
print('Saved:', output)
