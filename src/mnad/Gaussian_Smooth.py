import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='Apply Gaussian smoothing to anomaly scores')
parser.add_argument('--input', required=True, help='input score file, normally scores_fused.npz')
parser.add_argument('--sigma', type=float, default=3.0, help='standard deviation of the Gaussian kernel')
parser.add_argument('--output', default=None, help='path used to save smoothed scores')
args = parser.parse_args()

if args.sigma <= 0:
    parser.error('--sigma must be greater than 0')

data = np.load(args.input)
video_names = data['video_names'].astype(str)
lengths = data['lengths'].astype(np.int64)
scores = data['scores'].astype(np.float32)
labels = data['labels'].astype(np.int8)

if lengths.sum() != len(scores) or lengths.sum() != len(labels):
    raise ValueError('The stored video lengths do not match the scores or labels')

smoothed_by_video = []
start = 0
for video_name, length in zip(video_names, lengths):
    end = start + int(length)
    one_video_scores = scores[start:end]
    if len(one_video_scores) == 0:
        raise ValueError(f'{video_name}: no scores were found')

    smoothed_scores = gaussian_filter1d(one_video_scores, sigma=args.sigma)
    smoothed_by_video.append(smoothed_scores.astype(np.float32))
    start = end

smoothed_scores = np.concatenate(smoothed_by_video)
auc_before = roc_auc_score(labels, scores)
auc_after = roc_auc_score(labels, smoothed_scores)

if args.output is None:
    args.output = os.path.join(
        os.path.dirname(os.path.abspath(args.input)), 'scores_fused_smooth.npz')
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

output_data = {
    'dataset': data['dataset'],
    'branch': np.asarray('fused_gaussian'),
    'video_names': video_names,
    'lengths': lengths,
    'scores': smoothed_scores,
    'labels': labels,
    'sigma': np.asarray(args.sigma, dtype=np.float32),
}
if 'beta' in data.files:
    output_data['beta'] = data['beta']
if 'pred_offset' in data.files:
    output_data['pred_offset'] = data['pred_offset']

np.savez(args.output, **output_data)

print(f'AUC before smoothing: {auc_before * 100:.2f}%')
print(f'AUC after smoothing:  {auc_after * 100:.2f}%')
print(f'Sigma: {args.sigma}')
print('Smoothed scores saved to:', args.output)
