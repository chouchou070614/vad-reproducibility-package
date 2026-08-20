import os
import warnings

if os.name == 'nt':
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
    warnings.filterwarnings('ignore', message='Could not find the number of physical cores')

import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


def build_kmeans_memory(model, loader, device, memory_size, input_channels=None, seed=2020):
    """Use every encoder query once to build the initial memory items."""
    was_training = model.training
    model.eval()

    kmeans = MiniBatchKMeans(
        n_clusters=memory_size,
        random_state=seed,
        n_init=10,
        batch_size=4096)

    query_count = 0
    with torch.no_grad():
        for batch_index, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=device.type == 'cuda')
            if input_channels is not None:
                imgs = imgs[:, :input_channels]

            features = model.encoder(imgs)[0]
            queries = F.normalize(features, dim=1)
            queries = queries.permute(0, 2, 3, 1).contiguous()
            queries = queries.view(-1, queries.size(-1))

            kmeans.partial_fit(queries.cpu().numpy())
            query_count += queries.size(0)

            if (batch_index + 1) % 100 == 0:
                print(f'K-means: processed {query_count} queries')

    centers = torch.as_tensor(
        kmeans.cluster_centers_, dtype=torch.float32, device=device)
    centers = F.normalize(centers, dim=1)

    if was_training:
        model.train()

    print(f'K-means finished: all {query_count} queries were used')
    return centers, query_count
