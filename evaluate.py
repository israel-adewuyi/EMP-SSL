############
## Import ##
############
import argparse
import json
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from func import WeightedKNNClassifier, linear

######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--test_patches', type=int, default=128,
                    help='number of patches used in testing (default: 128)')  

parser.add_argument('--data', type=str, default="cifar10",
                    help='dataset (default: cifar10)')  
parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')
parser.add_argument('--norm', type=str, choices=['batch', 'layer'], default='batch',
                    help='normalization used by the trained encoder (default: batch)')

parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate for linear eval (default: 0.03)')        
parser.add_argument('--linear', type=bool, default=True,
                    help='use linear eval or not')
parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
parser.add_argument('--model_path', type=str, default="",
                    help='model directory for eval')
parser.add_argument('--results_json', type=str, default="",
                    help='optional path to write structured evaluation results as JSON')
parser.add_argument('--device', type=str, default="auto", choices=["auto", "cuda", "cpu"],
                    help='device to use: auto, cuda, or cpu (default: auto)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of dataloader workers (default: 0)')

            
args = parser.parse_args()















######################
## Testing Accuracy ##
######################
test_patches = args.test_patches

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

knn_classifier = WeightedKNNClassifier()


def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


def resolve_device(requested_device):
    if requested_device == "cpu":
        return torch.device("cpu")

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        if requested_device == "cuda":
            raise RuntimeError("CUDA was requested, but CUDA initialization failed.") from exc
        print(f"CUDA initialization failed, falling back to CPU: {exc}")
        return torch.device("cpu")

    if requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested, but no usable CUDA device is available. "
                "Check the NVIDIA driver and the PyTorch CUDA build."
            )
        return torch.device("cuda")

    if cuda_available:
        return torch.device("cuda")

    print("CUDA is unavailable, using CPU.")
    return torch.device("cpu")


def build_dataloader(dataset, batch_size, shuffle, drop_last, num_workers, device):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


def test(net, train_loader, test_loader, device):
    results = {}
    
    train_z_full_list, train_y_list, test_z_full_list, test_y_list = [], [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(train_loader):

            x = torch.cat(x, dim = 0)
            x = x.to(device, non_blocking=device.type == "cuda")
            
            z_proj, z_pre = net(x, is_test=True)

            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
            
            
            train_z_full_list.append(z_pre)
            
            
            knn_classifier.update(train_features = z_pre, train_targets = y)

            train_y_list.append(y)
                
        for x, y in tqdm(test_loader):
            x = torch.cat(x, dim = 0)
            x = x.to(device, non_blocking=device.type == "cuda")
            
            z_proj, z_pre = net(x, is_test=True)

            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
           
            test_z_full_list.append(z_pre)
       
            knn_classifier.update(test_features = z_pre, test_targets = y)

            test_y_list.append(y)
                
            
    train_features_full, train_labels, test_features_full, test_labels = torch.cat(train_z_full_list,dim=0), torch.cat(train_y_list,dim=0), torch.cat(test_z_full_list,dim=0), torch.cat(test_y_list,dim=0)
   
    if args.data == "cifar10":
        num_classes = 10
    elif args.data == "cifar100":
        num_classes = 100
    elif args.data == "tinyimagenet200":
        num_classes = 200
    elif args.data == "imagenet100":
        num_classes = 100
    elif args.data == "imagenet":
        num_classes = 1000
        
    if args.linear:
        print("Using Linear Eval to evaluate accuracy")
        results["linear"] = linear(
            train_features_full,
            train_labels,
            test_features_full,
            test_labels,
            lr=args.lr,
            num_classes=num_classes,
        )
    
    if args.knn:
        print("Using KNN to evaluate accuracy")
        top1, top5 = knn_classifier.compute()
        print("KNN (top1/top5):", top1, top5)
        results["knn"] = {
            "top1": float(top1),
            "top5": float(top5),
        }

    return results
    
def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


torch.multiprocessing.set_sharing_strategy('file_system')
device = resolve_device(args.device)
print(f"Using device: {device}")


#Get Dataset
if args.data == "imagenet100" or args.data == "imagenet":
        
    memory_dataset = load_dataset(args.data, train=True, num_patch = test_patches)
    memory_loader = build_dataloader(
        memory_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        device=device,
    )

    test_data = load_dataset(args.data, train=False, num_patch = test_patches)
    test_loader = build_dataloader(
        test_data,
        batch_size=50,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=device,
    )

else:
    memory_dataset = load_dataset(args.data, train=True, num_patch = test_patches)
    memory_loader = build_dataloader(
        memory_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        device=device,
    )

    test_data = load_dataset(args.data, train=False, num_patch = test_patches)
    test_loader = build_dataloader(
        test_data,
        batch_size=50,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=device,
    )

# Load Model and Checkpoint
net = encoder(arch=args.arch, norm=args.norm)
if device.type == "cuda" and torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
save_dict = torch.load(args.model_path, map_location=device)
net.load_state_dict(save_dict,strict=False)
net = net.to(device)
net.eval()
results = test(net, memory_loader, test_loader, device)

if args.results_json:
    results_dir = os.path.dirname(args.results_json)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "data": args.data,
        "arch": args.arch,
        "norm": args.norm,
        "test_patches": args.test_patches,
        "linear_lr": args.lr,
        "used_linear": bool(args.linear),
        "used_knn": bool(args.knn),
        "results": results,
    }
    with open(args.results_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Saved evaluation results to {args.results_json}")



