import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.datasets import load_dataset
from lars import LARSWrapper
from model.model import encoder
from patchsketch import (
    gather_selected_embeddings,
    patchsketch_loss,
    reshape_patch_embeddings,
    select_representative_patches,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PatchSketch Training")
    parser.add_argument(
        "--sketch_size",
        type=int,
        default=10,
        help="Frequent Directions sketch size (default: 10)",
    )
    parser.add_argument(
        "--selected_patches",
        type=int,
        default=None,
        help="Number of selected patches per image (default: sketch_size)",
    )
    parser.add_argument(
        "--cov_weight",
        type=float,
        default=1.0,
        help="Weight for the covariance decorrelation loss (default: 1.0)",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=100,
        help="number of patches used in PatchSketch training (default: 100)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18-cifar",
        help="network architecture (default: resnet18-cifar)",
    )
    parser.add_argument("--bs", type=int, default=100, help="batch size (default: 100)")
    parser.add_argument(
        "--lr", type=float, default=0.3, help="learning rate (default: 0.3)"
    )
    parser.add_argument(
        "--msg",
        type=str,
        default="NONE",
        help="additional message for description (default: NONE)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="PatchSketch-Training",
        help="directory name (default: PatchSketch-Training)",
    )
    parser.add_argument(
        "--data", type=str, default="cifar10", help="data (default: cifar10)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="max number of epochs to finish (default: 30)",
    )
    return parser.parse_args()


def validate_args(args):
    if not 2 <= args.sketch_size <= args.num_patches:
        raise ValueError("sketch_size must satisfy 2 <= sketch_size <= num_patches")

    if args.selected_patches is None:
        args.selected_patches = args.sketch_size

    if not 2 <= args.selected_patches <= args.num_patches:
        raise ValueError(
            "selected_patches must satisfy 2 <= selected_patches <= num_patches"
        )

    return args


def default_num_workers(data_name):
    if data_name in {"imagenet100", "imagenet"}:
        return 8
    return 16


def load_train_dataset(args):
    dataset_name = "imagenet" if args.data in {"imagenet100", "imagenet"} else args.data
    return load_dataset(dataset_name, train=True, num_patch=args.num_patches)


def build_train_dataloader(args, num_workers=None):
    train_dataset = load_train_dataset(args)
    if num_workers is None:
        num_workers = default_num_workers(args.data)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return train_dataset, dataloader


def build_model(args, device):
    net = encoder(arch=args.arch)
    if device.type == "cuda":
        net = nn.DataParallel(net)
    return net.to(device)


def build_optimizer(model, lr):
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    return LARSWrapper(optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True)


def build_scheduler(optimizer, args):
    if args.data in {"imagenet100", "imagenet"}:
        num_converge = (150000 // args.bs) * args.epoch
    else:
        num_converge = (50000 // args.bs) * args.epoch

    return lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_converge, eta_min=0, last_epoch=-1
    )


def checkpoint_state_dict(model):
    state_dict = model.state_dict()
    if isinstance(model, nn.DataParallel):
        return state_dict

    return {f"module.{name}": value for name, value in state_dict.items()}


def train_one_epoch(model, dataloader, optimizer, scheduler, device, args):
    model.train()

    total_loss_sum = 0.0
    inv_loss_sum = 0.0
    cov_loss_sum = 0.0
    score_sum = 0.0
    num_steps = 0

    for step, (patch_views, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(patch_views) != args.num_patches:
            raise ValueError(
                f"Expected {args.num_patches} patch views, got {len(patch_views)}"
            )

        batch_size = labels.size(0)
        optimizer.zero_grad()

        flat_patches = torch.cat(patch_views, dim=0).to(device)
        flat_embeddings = model(flat_patches)
        batch_embeddings = reshape_patch_embeddings(
            flat_embeddings, batch_size=batch_size, num_patches=args.num_patches
        )

        with torch.no_grad():
            selected_indices, selected_scores = select_representative_patches(
                batch_embeddings.detach(),
                sketch_size=args.sketch_size,
                selected_patches=args.selected_patches,
            )

        selected_embeddings = gather_selected_embeddings(batch_embeddings, selected_indices)
        total_loss, inv_loss, cov_loss = patchsketch_loss(
            selected_embeddings, cov_weight=args.cov_weight
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss_sum += total_loss.item()
        inv_loss_sum += inv_loss.item()
        cov_loss_sum += cov_loss.item()
        score_sum += selected_scores.mean().item()
        num_steps += 1

    if num_steps == 0:
        raise RuntimeError("Training dataloader yielded no steps")

    return {
        "loss": total_loss_sum / num_steps,
        "loss_inv": inv_loss_sum / num_steps,
        "loss_cov": cov_loss_sum / num_steps,
        "mean_topk_score": score_sum / num_steps,
    }


def save_checkpoint(model, args, epoch):
    dir_name = (
        f"./logs/{args.dir}/sketch{args.sketch_size}_topk{args.selected_patches}"
        f"_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}_{args.msg}"
    )
    model_dir = os.path.join(dir_name, "save_models")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"{epoch}.pt")
    torch.save(checkpoint_state_dict(model), checkpoint_path)
    return checkpoint_path


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    args = validate_args(parse_args())
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dataloader = build_train_dataloader(args)
    model = build_model(args, device)
    optimizer = build_optimizer(model, args.lr)
    scheduler = build_scheduler(optimizer, args)

    for epoch in range(args.epoch):
        metrics = train_one_epoch(model, dataloader, optimizer, scheduler, device, args)
        checkpoint_path = save_checkpoint(model, args, epoch)

        print(
            f"At epoch: {epoch} "
            f"loss is {metrics['loss']:.6f}, "
            f"loss_inv is {metrics['loss_inv']:.6f}, "
            f"loss_cov is {metrics['loss_cov']:.6f}, "
            f"mean_topk_score is {metrics['mean_topk_score']:.6f}, "
            f"learning rate is {optimizer.param_groups[0]['lr']:.6f}, "
            f"checkpoint is {checkpoint_path}"
        )


if __name__ == "__main__":
    main()
