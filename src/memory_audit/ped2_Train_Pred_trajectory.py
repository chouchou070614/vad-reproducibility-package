"""Train the Ped2 prediction branch and save memory snapshots along its trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.utils_pred import DataLoader
from model.Prediction import *
from ped2_kmeans_memory_trajectory import build_kmeans_memory
from trajectory_sampling import save_query_trace
from wandb_compat import wandb


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description="MNAD Ped2-P Memory trajectory")
parser.add_argument("--gpus", nargs="+", type=str, help="gpus")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--scheduler_t_max", type=int, default=None)
parser.add_argument("--loss_compact", type=float, default=0.1)
parser.add_argument("--loss_separate", type=float, default=0.1)
parser.add_argument("--h", type=int, default=256)
parser.add_argument("--w", type=int, default=256)
parser.add_argument("--c", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--t_length", type=int, default=5)
parser.add_argument("--fdim", type=int, default=512)
parser.add_argument("--mdim", type=int, default=512)
parser.add_argument("--msize", type=int, default=10)
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--num_workers_test", type=int, default=1)
parser.add_argument("--dataset_type", type=str, default="ped2")
parser.add_argument("--dataset_path", type=str, default="./dataset/")
parser.add_argument("--exp_dir", type=str, default="log_abbey")
parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--kmeans_init", action="store_true")
parser.add_argument("--kmeans_warmup_epochs", type=int, default=26)
parser.add_argument("--kmeans_seed", type=int, default=2020)
parser.add_argument("--freeze_memory_after_kmeans", action="store_true")
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--trajectory_sample_count", type=int, default=4096)
parser.add_argument("--trajectory_trace_seed", type=int, default=None)
args = parser.parse_args()

if args.scheduler_t_max is None:
    args.scheduler_t_max = args.epochs
if args.scheduler_t_max < args.epochs:
    parser.error("--scheduler_t_max must be greater than or equal to --epochs")
if not args.kmeans_init:
    parser.error("--kmeans_init required")
if not 1 <= args.kmeans_warmup_epochs < args.epochs:
    parser.error("--kmeans_warmup_epochs must be at least 1 and smaller than --epochs")
if args.save_every < 0:
    parser.error("--save_every must be non-negative")
if args.freeze_memory_after_kmeans:
    parser.error("continued memory updates required")
if args.dataset_type != "ped2":
    parser.error("dataset_type=ped2 required")
if args.trajectory_sample_count <= 0:
    parser.error("--trajectory_sample_count must be positive")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)

torch.backends.cudnn.enabled = True
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is unavailable")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("Using device:", device)
wandb.init(project="mnad", mode="online" if args.wandb else "disabled")

train_folder = os.path.join(args.dataset_path, args.dataset_type, "training", "frames")
test_folder = os.path.join(args.dataset_path, args.dataset_type, "testing", "frames")
train_dataset = DataLoader(
    train_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1,
)
test_dataset = DataLoader(
    test_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1,
)
print("Train Size:" + str(len(train_dataset)))
print("Test Size:" + str(len(test_dataset)))
train_batch = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
    pin_memory=device.type == "cuda",
)
kmeans_batch = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=False,
    pin_memory=device.type == "cuda",
)

model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_count = sum([np.prod(p.size()) for p in model_parameters])
print(params_count)
params = list(model.encoder.parameters()) + list(model.decoder.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max)
model.to(device)
if args.wandb:
    wandb.watch(model)
print(model)

log_dir = Path("./exp") / args.dataset_type / args.exp_dir
log_dir.mkdir(parents=True, exist_ok=True)
trajectory_dir = log_dir / "memory_trajectory"
trajectory_dir.mkdir(parents=True, exist_ok=True)
trajectory_files = {
    "random_initial_memory": "01_random_initial_memory.pt",
    "warmup_pre_kmeans_memory": "02_warmup_%d_pre_kmeans_memory.pt"
    % args.kmeans_warmup_epochs,
    "actual_kmeans_centers": "03_warmup_%d_actual_kmeans_centers.pt"
    % args.kmeans_warmup_epochs,
    "raw_kmeans_centers_before_normalization":
    "03b_warmup_%d_raw_kmeans_centers_before_normalization.pt"
    % args.kmeans_warmup_epochs,
    "post_replacement_memory": "04_warmup_%d_post_replacement_memory.pt"
    % args.kmeans_warmup_epochs,
    "final_memory": "05_final_memory.pt",
    "warmup_query_trace": "warmup_%d_normal_query_trace.npz"
    % args.kmeans_warmup_epochs,
    "final_checkpoint": "../model.pth",
}
trajectory_state = {
    "schema_version": "memory-trajectory-v2",
    "task_id": args.exp_dir,
    "dataset": "ped2",
    "branch": "P",
    "family": "KMemory",
    "memory_update_policy": "Continue",
    "query_source_split": "normal_training",
    "kmeans_uses_test_labels": False,
    "same_training_trajectory": True,
    "warmup_epoch": args.kmeans_warmup_epochs,
    "files": trajectory_files,
    "sha256": {},
    "completed_stages": [],
}

loss_func_mse = nn.MSELoss(reduction="none")
m_items = F.normalize(
    torch.rand((args.msize, args.mdim), dtype=torch.float, device=device), dim=1
)
torch.save(
    m_items.detach().cpu().clone(),
    trajectory_dir / trajectory_files["random_initial_memory"],
)
trajectory_state["completed_stages"].append("random_initial_memory")
write_json(trajectory_dir / "trajectory_state.json", trajectory_state)

query_count = None
for epoch in range(args.epochs):
    epoch_start = time.time()
    model.train()
    train_loss = AverageMeter()
    step_0 = time.time()

    for j, imgs in enumerate(train_batch):
        imgs = Variable(imgs).to(device, non_blocking=device.type == "cuda")
        memory_before_step = m_items
        (
            outputs,
            _,
            _,
            updated_m_items,
            softmax_score_query,
            softmax_score_memory,
            separateness_loss,
            compactness_loss,
        ) = model.forward(imgs[:, 0:12], m_items, True)
        # Continue memory updates after the K-means replacement.
        memory_is_frozen = (
            args.freeze_memory_after_kmeans and epoch >= args.kmeans_warmup_epochs
        )
        m_items = memory_before_step if memory_is_frozen else updated_m_items

        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
        loss = (
            loss_pixel
            + args.loss_compact * compactness_loss
            + args.loss_separate * separateness_loss
        )
        train_loss.update(loss.item(), imgs.size(0))
        if j % 100 == 0:
            step_100 = time.time()
            print("[%d] Train loss  %s Time 100 steps: %s" % (
                j, train_loss.avg, step_100 - step_0
            ))
            step_0 = step_100
        loss.backward(retain_graph=False)
        wandb.log({
            "Loss": loss,
            "Compactness loss": compactness_loss,
            "Separateness Loss": separateness_loss,
        })
        optimizer.step()

    completed_epochs = epoch + 1
    if args.save_every and completed_epochs % args.save_every == 0:
        torch.save(model, log_dir / ("%d_model.pth" % completed_epochs))
        torch.save(m_items, log_dir / ("%d_keys.pt" % completed_epochs))

    if args.kmeans_init and completed_epochs == args.kmeans_warmup_epochs:
        print("Starting K-means memory initialization...")
        torch.save(
            m_items.detach().cpu().clone(),
            trajectory_dir / trajectory_files["warmup_pre_kmeans_memory"],
        )
        trajectory_state["completed_stages"].append("warmup_pre_kmeans_memory")
        write_json(trajectory_dir / "trajectory_state.json", trajectory_state)

        kmeans_centers, query_count, query_trace, raw_cluster_centers = build_kmeans_memory(
            model=model,
            loader=kmeans_batch,
            device=device,
            memory_size=args.msize,
            input_channels=3 * (args.t_length - 1),
            seed=args.kmeans_seed,
            trace_sample_count=args.trajectory_sample_count,
            trace_seed=args.trajectory_trace_seed,
        )
        # Save both the normalized memory centers and the raw K-means centers.
        torch.save(
            kmeans_centers.detach().cpu().clone(),
            trajectory_dir / trajectory_files["actual_kmeans_centers"],
        )
        torch.save(
            raw_cluster_centers.detach().cpu().clone(),
            trajectory_dir
            / trajectory_files["raw_kmeans_centers_before_normalization"],
        )
        m_items = kmeans_centers
        torch.save(
            m_items.detach().cpu().clone(),
            trajectory_dir / trajectory_files["post_replacement_memory"],
        )
        effective_trace_seed = (
            args.kmeans_seed + 104729
            if args.trajectory_trace_seed is None
            else args.trajectory_trace_seed
        )
        # Record the source video, frame, and spatial position of each query.
        sampled_dataset_indices = query_trace["dataset_index"].astype(np.int64)
        source_paths = [train_dataset.samples[int(index)] for index in sampled_dataset_indices]
        video_names = [os.path.basename(os.path.dirname(path)) for path in source_paths]
        frame_positions = {
            video_name: {path: position for position, path in enumerate(info["frame"])}
            for video_name, info in train_dataset.videos.items()
        }
        target_paths = []
        for source_path, video_name in zip(source_paths, video_names):
            position = frame_positions[video_name][source_path]
            target_paths.append(
                train_dataset.videos[video_name]["frame"][position + args.t_length - 1]
            )
        query_trace["video_name"] = np.asarray(video_names, dtype=np.unicode_)
        query_trace["sequence_start_filename"] = np.asarray(
            [os.path.basename(path) for path in source_paths], dtype=np.unicode_
        )
        query_trace["prediction_target_filename"] = np.asarray(
            [os.path.basename(path) for path in target_paths], dtype=np.unicode_
        )
        query_trace["sequence_start_relative_path"] = np.asarray(
            [os.path.relpath(path, args.dataset_path) for path in source_paths],
            dtype=np.unicode_,
        )
        query_trace["prediction_target_relative_path"] = np.asarray(
            [os.path.relpath(path, args.dataset_path) for path in target_paths],
            dtype=np.unicode_,
        )
        save_query_trace(
            trajectory_dir / trajectory_files["warmup_query_trace"],
            query_trace,
            query_count=query_count,
            memory_size=args.msize,
            feature_dim=args.mdim,
            sample_seed=effective_trace_seed,
            kmeans_seed=args.kmeans_seed,
            warmup_epoch=args.kmeans_warmup_epochs,
        )
        torch.save(m_items, log_dir / ("%d_kmeans_keys.pt" % completed_epochs))
        trajectory_state["query_count_full"] = int(query_count)
        trajectory_state["query_sample_count"] = int(len(query_trace["queries"]))
        trajectory_state["actual_kmeans_centers_representation"] = (
            "L2-normalized K-means centers"
        )
        trajectory_state["raw_kmeans_centers_representation"] = (
            "MiniBatchKMeans.cluster_centers_ before normalization"
        )
        trajectory_state["post_replacement_memory_representation"] = (
            "L2-normalized K-means centers"
        )
        trajectory_state["completed_stages"].extend([
            "actual_kmeans_centers",
            "post_replacement_memory",
            "warmup_query_trace",
        ])
        write_json(trajectory_dir / "trajectory_state.json", trajectory_state)
        print(
            "Memory replaced after epoch %d; queries used: %d"
            % (completed_epochs, query_count)
        )

    scheduler.step()
    epoch_end = time.time()
    print("----------------------------------------")
    print("Epoch:", epoch + 1)
    print("Epoch average total loss: {:.9f}".format(train_loss.avg))
    print("Learning rate: {:.12g}".format(optimizer.param_groups[0]["lr"]))
    print(
        "Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}".format(
            loss_pixel.item(), compactness_loss.item(), separateness_loss.item()
        )
    )
    print("Epoch time taken:", str(epoch_end - epoch_start))
    print("----------------------------------------")

print("Training is finished")
print(log_dir)
torch.save(model, log_dir / "model.pth")
torch.save(m_items, log_dir / "keys.pt")
torch.save(
    m_items.detach().cpu().clone(), trajectory_dir / trajectory_files["final_memory"]
)
trajectory_state["completed_stages"].append("final_memory")
trajectory_state["completed_stages"].append("final_checkpoint")
trajectory_state["training_complete"] = True
trajectory_state["epochs"] = args.epochs
trajectory_state["seed"] = args.seed
trajectory_state["kmeans_seed"] = args.kmeans_seed
trajectory_state["trajectory_sample_seed"] = (
    args.kmeans_seed + 104729
    if args.trajectory_trace_seed is None
    else args.trajectory_trace_seed
)
trajectory_state["sha256"] = {
    name: sha256_file((trajectory_dir / relative).resolve())
    for name, relative in trajectory_files.items()
}
write_json(trajectory_dir / "trajectory_state.json", trajectory_state)
wandb.finish()
