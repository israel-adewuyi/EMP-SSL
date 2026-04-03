import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid

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
    parser.add_argument(
        "--hist_every_n_steps",
        type=int,
        default=100,
        help="log TensorBoard histograms every N steps; 0 disables histogram logging",
    )
    parser.add_argument(
        "--image_every_n_steps",
        type=int,
        default=0,
        help="log patch preview images every N steps; 0 disables image logging",
    )
    parser.add_argument(
        "--disable_tensorboard",
        action="store_true",
        help="disable TensorBoard event logging",
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


def run_dir(args):
    return (
        f"./logs/{args.dir}/sketch{args.sketch_size}_topk{args.selected_patches}"
        f"_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}_{args.msg}"
    )


def checkpoint_state_dict(model):
    state_dict = model.state_dict()
    if isinstance(model, nn.DataParallel):
        return state_dict

    return {f"module.{name}": value for name, value in state_dict.items()}


def build_summary_writer(args, log_dir):
    if args.disable_tensorboard:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install it or pass --disable_tensorboard."
        ) from exc

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("run/args", str(args))
    return writer


def total_grad_norm(model):
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().norm(2).item()
        squared_norm += grad_norm * grad_norm
    return squared_norm ** 0.5


def total_param_norm(model):
    squared_norm = 0.0
    for parameter in model.parameters():
        param_norm = parameter.detach().norm(2).item()
        squared_norm += param_norm * param_norm
    return squared_norm ** 0.5


def should_log_interval(interval, step):
    return interval > 0 and (step + 1) % interval == 0


def use_tqdm():
    return sys.stderr.isatty()


def normalize_for_tensorboard(images):
    return torch.clamp(images.detach().cpu() * 0.5 + 0.5, 0.0, 1.0)


def log_patch_previews(writer, global_step, patch_views, selected_indices):
    sample_idx = 0
    candidate_patches = torch.stack([view[sample_idx] for view in patch_views], dim=0)
    selected_patches = candidate_patches[selected_indices[sample_idx].detach().cpu()]

    writer.add_image(
        "patches/all_candidates",
        make_grid(normalize_for_tensorboard(candidate_patches), nrow=min(10, candidate_patches.size(0))),
        global_step,
    )
    writer.add_image(
        "patches/selected",
        make_grid(normalize_for_tensorboard(selected_patches), nrow=min(10, selected_patches.size(0))),
        global_step,
    )


def log_step_metrics(
    writer,
    global_step,
    scalars,
    selection_diagnostics,
    batch_embeddings,
    selected_embeddings,
    patch_views,
    selected_indices,
    args,
):
    if writer is None:
        return

    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step)

    if should_log_interval(args.hist_every_n_steps, global_step):
        writer.add_histogram(
            "train_hist/all_patch_scores",
            selection_diagnostics["all_scores"].reshape(-1).detach().cpu(),
            global_step,
        )
        writer.add_histogram(
            "train_hist/selected_patch_scores",
            selection_diagnostics["selected_scores"].reshape(-1).detach().cpu(),
            global_step,
        )
        writer.add_histogram(
            "train_hist/selection_margin",
            selection_diagnostics["selection_margin"].reshape(-1).detach().cpu(),
            global_step,
        )
        writer.add_histogram(
            "train_hist/score_spread",
            selection_diagnostics["score_spread"].reshape(-1).detach().cpu(),
            global_step,
        )
        writer.add_histogram(
            "train_hist/sketch_singular_values",
            selection_diagnostics["sketch_singular_values"].reshape(-1).detach().cpu(),
            global_step,
        )
        writer.add_histogram(
            "train_hist/embedding_norms",
            batch_embeddings.detach().norm(dim=2).reshape(-1).cpu(),
            global_step,
        )
        writer.add_histogram(
            "train_hist/selected_embedding_norms",
            selected_embeddings.detach().norm(dim=2).reshape(-1).cpu(),
            global_step,
        )

    if should_log_interval(args.image_every_n_steps, global_step):
        log_patch_previews(writer, global_step, patch_views, selected_indices)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, args, writer, global_step):
    model.train()

    metric_sums = {
        "loss": 0.0,
        "loss_inv": 0.0,
        "loss_cov": 0.0,
        "mean_topk_score": 0.0,
        "min_topk_score": 0.0,
        "max_topk_score": 0.0,
        "selection_margin": 0.0,
        "sketch_effective_rank": 0.0,
        "sketch_nonzero_rows": 0.0,
        "consensus_direction_norm": 0.0,
        "selected_embedding_variance": 0.0,
        "batch_embedding_variance": 0.0,
        "grad_norm": 0.0,
        "param_norm": 0.0,
        "step_time": 0.0,
        "samples_per_sec": 0.0,
        "patches_per_sec": 0.0,
    }
    num_steps = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not use_tqdm())
    for step, (patch_views, labels) in progress_bar:
        step_start = time.perf_counter()
        if len(patch_views) != args.num_patches:
            raise ValueError(
                f"Expected {args.num_patches} patch views, got {len(patch_views)}"
            )

        batch_size = labels.size(0)
        optimizer.zero_grad()

        flat_patches = torch.cat(patch_views, dim=0).to(device, non_blocking=device.type == "cuda")
        flat_embeddings = model(flat_patches)
        batch_embeddings = reshape_patch_embeddings(
            flat_embeddings, batch_size=batch_size, num_patches=args.num_patches
        )

        with torch.no_grad():
            selected_indices, selected_scores, selection_diagnostics = select_representative_patches(
                batch_embeddings.detach(),
                sketch_size=args.sketch_size,
                selected_patches=args.selected_patches,
                return_diagnostics=True,
            )

        selected_embeddings = gather_selected_embeddings(batch_embeddings, selected_indices)
        total_loss, inv_loss, cov_loss = patchsketch_loss(
            selected_embeddings, cov_weight=args.cov_weight
        )

        total_loss.backward()
        grad_norm = total_grad_norm(model)
        optimizer.step()
        scheduler.step()

        step_time = time.perf_counter() - step_start
        param_norm = total_param_norm(model)
        selected_embedding_variance = (
            selected_embeddings.detach().var(dim=1, unbiased=False).mean().item()
        )
        batch_embedding_variance = (
            batch_embeddings.detach().var(dim=1, unbiased=False).mean().item()
        )
        mean_topk_score = selected_scores.mean().item()
        min_topk_score = selected_scores.min().item()
        max_topk_score = selected_scores.max().item()
        selection_margin = selection_diagnostics["selection_margin"].mean().item()
        sketch_effective_rank = (
            selection_diagnostics["sketch_effective_rank"].float().mean().item()
        )
        sketch_nonzero_rows = (
            selection_diagnostics["sketch_nonzero_rows"].float().mean().item()
        )
        consensus_direction_norm = (
            selection_diagnostics["consensus_direction_norm"].float().mean().item()
        )
        samples_per_sec = batch_size / max(step_time, 1e-12)
        patches_per_sec = (batch_size * args.num_patches) / max(step_time, 1e-12)

        step_scalars = {
            "train/loss_total": total_loss.item(),
            "train/loss_inv": inv_loss.item(),
            "train/loss_cov": cov_loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/step_time": step_time,
            "train/samples_per_sec": samples_per_sec,
            "train/patches_per_sec": patches_per_sec,
            "train/mean_selected_score": mean_topk_score,
            "train/min_selected_score": min_topk_score,
            "train/max_selected_score": max_topk_score,
            "train/selection_margin": selection_margin,
            "train/sketch_effective_rank": sketch_effective_rank,
            "train/sketch_nonzero_rows": sketch_nonzero_rows,
            "train/consensus_direction_norm": consensus_direction_norm,
            "train/selected_embedding_variance": selected_embedding_variance,
            "train/batch_embedding_variance": batch_embedding_variance,
            "train/grad_norm": grad_norm,
            "train/param_norm": param_norm,
        }
        selection_diagnostics["selected_scores"] = selected_scores.detach()
        log_step_metrics(
            writer,
            global_step,
            step_scalars,
            selection_diagnostics,
            batch_embeddings,
            selected_embeddings,
            patch_views,
            selected_indices,
            args,
        )

        metric_sums["loss"] += total_loss.item()
        metric_sums["loss_inv"] += inv_loss.item()
        metric_sums["loss_cov"] += cov_loss.item()
        metric_sums["mean_topk_score"] += mean_topk_score
        metric_sums["min_topk_score"] += min_topk_score
        metric_sums["max_topk_score"] += max_topk_score
        metric_sums["selection_margin"] += selection_margin
        metric_sums["sketch_effective_rank"] += sketch_effective_rank
        metric_sums["sketch_nonzero_rows"] += sketch_nonzero_rows
        metric_sums["consensus_direction_norm"] += consensus_direction_norm
        metric_sums["selected_embedding_variance"] += selected_embedding_variance
        metric_sums["batch_embedding_variance"] += batch_embedding_variance
        metric_sums["grad_norm"] += grad_norm
        metric_sums["param_norm"] += param_norm
        metric_sums["step_time"] += step_time
        metric_sums["samples_per_sec"] += samples_per_sec
        metric_sums["patches_per_sec"] += patches_per_sec
        num_steps += 1
        global_step += 1

        if use_tqdm():
            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                inv=f"{inv_loss.item():.4f}",
                cov=f"{cov_loss.item():.4f}",
                score=f"{mean_topk_score:.4f}",
                margin=f"{selection_margin:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.5f}",
            )

    if num_steps == 0:
        raise RuntimeError("Training dataloader yielded no steps")

    metrics = {name: value / num_steps for name, value in metric_sums.items()}
    return metrics, global_step


def save_checkpoint(model, args, epoch):
    model_dir = os.path.join(run_dir(args), "save_models")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"{epoch}.pt")
    torch.save(checkpoint_state_dict(model), checkpoint_path)
    return checkpoint_path


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    args = validate_args(parse_args())
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = run_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    _, dataloader = build_train_dataloader(args)
    model = build_model(args, device)
    optimizer = build_optimizer(model, args.lr)
    scheduler = build_scheduler(optimizer, args)
    writer = build_summary_writer(args, log_dir)
    global_step = 0

    for epoch in range(args.epoch):
        metrics, global_step = train_one_epoch(
            model, dataloader, optimizer, scheduler, device, args, writer, global_step
        )
        checkpoint_path = save_checkpoint(model, args, epoch)
        if writer is not None:
            writer.add_scalar("epoch/loss_total", metrics["loss"], epoch)
            writer.add_scalar("epoch/loss_inv", metrics["loss_inv"], epoch)
            writer.add_scalar("epoch/loss_cov", metrics["loss_cov"], epoch)
            writer.add_scalar("epoch/mean_selected_score", metrics["mean_topk_score"], epoch)
            writer.add_scalar("epoch/selection_margin", metrics["selection_margin"], epoch)
            writer.add_scalar("epoch/sketch_effective_rank", metrics["sketch_effective_rank"], epoch)
            writer.add_scalar(
                "epoch/selected_embedding_variance",
                metrics["selected_embedding_variance"],
                epoch,
            )
            writer.flush()

        print(
            f"At epoch: {epoch} "
            f"loss is {metrics['loss']:.6f}, "
            f"loss_inv is {metrics['loss_inv']:.6f}, "
            f"loss_cov is {metrics['loss_cov']:.6f}, "
            f"mean_topk_score is {metrics['mean_topk_score']:.6f}, "
            f"selection_margin is {metrics['selection_margin']:.6f}, "
            f"sketch_effective_rank is {metrics['sketch_effective_rank']:.6f}, "
            f"selected_embedding_variance is {metrics['selected_embedding_variance']:.6f}, "
            f"batch_embedding_variance is {metrics['batch_embedding_variance']:.6f}, "
            f"grad_norm is {metrics['grad_norm']:.6f}, "
            f"param_norm is {metrics['param_norm']:.6f}, "
            f"learning rate is {optimizer.param_groups[0]['lr']:.6f}, "
            f"checkpoint is {checkpoint_path}"
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
