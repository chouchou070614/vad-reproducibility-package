"""Build K-means centers and sample normal-training queries."""

from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

if os.name == "nt":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from trajectory_sampling import DeterministicPrioritySampler


def build_kmeans_memory(
    model,
    loader,
    device,
    memory_size,
    input_channels=None,
    seed=2020,
    trace_sample_count=4096,
    trace_seed=None,
) -> Tuple[torch.Tensor, int, Dict[str, np.ndarray], torch.Tensor]:
    """Use every encoder query once and return centers plus an audit sample."""
    was_training = model.training
    model.eval()

    kmeans = MiniBatchKMeans(
        n_clusters=memory_size,
        random_state=seed,
        n_init=10,
        batch_size=4096,
    )

    sampler = None
    query_count = 0
    dataset_offset = 0
    effective_trace_seed = int(seed) + 104729 if trace_seed is None else int(trace_seed)
    with torch.no_grad():
        for batch_index, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            if input_channels is not None:
                imgs = imgs[:, :input_channels]

            features = model.encoder(imgs)[0]
            queries = F.normalize(features, dim=1)
            batch_size, feature_dim, spatial_h, spatial_w = queries.shape
            queries = queries.permute(0, 2, 3, 1).contiguous()
            queries = queries.view(-1, queries.size(-1))
            queries_numpy = queries.cpu().numpy()

            # Update centers in query order before sampling for analysis.
            kmeans.partial_fit(queries_numpy)

            if sampler is None:
                sampler = DeterministicPrioritySampler(
                    capacity=int(trace_sample_count),
                    feature_dim=int(feature_dim),
                    seed=effective_trace_seed,
                )
            per_item = int(spatial_h * spatial_w)
            local = np.arange(queries_numpy.shape[0], dtype=np.int64)
            batch_item_index = local // per_item
            spatial_flat = local % per_item
            sampler.observe(
                queries_numpy,
                {
                    "global_query_index": query_count + local,
                    "dataset_index": dataset_offset + batch_item_index,
                    "batch_index": np.full(local.shape, batch_index, dtype=np.int64),
                    "batch_item_index": batch_item_index,
                    "spatial_y": spatial_flat // int(spatial_w),
                    "spatial_x": spatial_flat % int(spatial_w),
                },
            )
            query_count += int(queries.size(0))
            dataset_offset += int(batch_size)

            if (batch_index + 1) % 100 == 0:
                print("K-means: processed %d queries" % query_count)

    raw_cluster_centers = torch.as_tensor(
        kmeans.cluster_centers_, dtype=torch.float32, device=device
    )
    centers = F.normalize(raw_cluster_centers, dim=1)
    if sampler is None:
        raise RuntimeError("normal-training query traversal produced no queries")
    trace = sampler.finalize(centers.detach().cpu().numpy())

    if was_training:
        model.train()

    print("K-means finished: all %d queries were used" % query_count)
    return centers, query_count, trace, raw_cluster_centers
