import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='Fuse MNAD-R and MNAD-P anomaly scores')
parser.add_argument('--recons_scores', required=True, help='scores_recons.npz')
parser.add_argument('--pred_scores', required=True, help='scores_pred.npz')
parser.add_argument('--beta', type=float, default=0.5,
                    help='fusion weight: beta * R + (1-beta) * P')
parser.add_argument('--pred_offset', type=int, default=4,
                    help='number of initial frames skipped by MNAD-P')
parser.add_argument('--output', default=None, help='path used to save fused scores')
args = parser.parse_args()

if not 0.0 <= args.beta <= 1.0:
    parser.error('--beta must be between 0 and 1')


def load_scores(path):
    data = np.load(path)
    names = data['video_names'].astype(str).tolist()
    lengths = data['lengths'].astype(int).tolist()
    flat_scores = data['scores'].astype(np.float32)
    flat_labels = data['labels'].astype(np.int8)

    scores_by_video = []
    labels_by_video = []
    start = 0
    for length in lengths:
        end = start + length
        scores_by_video.append(flat_scores[start:end])
        labels_by_video.append(flat_labels[start:end])
        start = end

    if start != len(flat_scores) or start != len(flat_labels):
        raise ValueError(f'Invalid score file: {path}')

    return str(data['dataset']), names, scores_by_video, labels_by_video


r_dataset, r_names, r_scores, r_labels = load_scores(args.recons_scores)
p_dataset, p_names, p_scores, p_labels = load_scores(args.pred_scores)

if r_dataset != p_dataset:
    raise ValueError(f'Dataset mismatch: R={r_dataset}, P={p_dataset}')
if r_names != p_names:
    raise ValueError('Video names or video order do not match')

fused_by_video = []
aligned_r_by_video = []
labels_by_video = []

for name, one_r_score, one_p_score, one_r_label, one_p_label in zip(
        r_names, r_scores, p_scores, r_labels, p_labels):
    if len(one_r_score) - len(one_p_score) != args.pred_offset:
        raise ValueError(
            f'{name}: R has {len(one_r_score)} frames, P has {len(one_p_score)} frames; '
            f'expected an offset of {args.pred_offset}')

    aligned_r_score = one_r_score[args.pred_offset:]
    aligned_r_label = one_r_label[args.pred_offset:]

    if not np.array_equal(aligned_r_label, one_p_label):
        raise ValueError(f'{name}: R and P labels are not aligned')

    fused_score = args.beta * aligned_r_score + (1.0 - args.beta) * one_p_score
    aligned_r_by_video.append(aligned_r_score)
    fused_by_video.append(fused_score)
    labels_by_video.append(one_p_label)

aligned_r_scores = np.concatenate(aligned_r_by_video)
pred_scores = np.concatenate(p_scores)
fused_scores = np.concatenate(fused_by_video)
labels = np.concatenate(labels_by_video)

r_auc = roc_auc_score(labels, aligned_r_scores)
p_auc = roc_auc_score(labels, pred_scores)
fused_auc = roc_auc_score(labels, fused_scores)

if args.output is None:
    args.output = os.path.join(
        os.path.dirname(os.path.abspath(args.pred_scores)), 'scores_fused.npz')
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

np.savez(
    args.output,
    dataset=np.asarray(r_dataset),
    branch=np.asarray('fused'),
    video_names=np.asarray(r_names),
    lengths=np.asarray([len(scores) for scores in fused_by_video], dtype=np.int64),
    scores=fused_scores,
    labels=labels,
    beta=np.asarray(args.beta, dtype=np.float32),
    pred_offset=np.asarray(args.pred_offset, dtype=np.int64))

print(f'MNAD-R AUC (aligned): {r_auc * 100:.2f}%')
print(f'MNAD-P AUC:           {p_auc * 100:.2f}%')
print(f'Fused AUC:            {fused_auc * 100:.2f}%')
print(f'Formula: {args.beta:.2f} * R + {1.0 - args.beta:.2f} * P')
print('Fused scores saved to:', args.output)
