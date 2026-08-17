import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at each spatial location."""

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def getmodel(arch, norm="batch"):
    if norm == "batch":
        conv_norm = nn.BatchNorm2d
    elif norm == "layer":
        conv_norm = LayerNorm2d
    else:
        raise ValueError(f"Unsupported normalization: {norm}")

    if arch == "resnet18-cifar":
        backbone = resnet18(norm_layer=conv_norm)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        return backbone, 512

    if arch == "resnet18-imagenet":
        backbone = resnet18(norm_layer=conv_norm)
        backbone.fc = nn.Identity()
        return backbone, 512

    if arch == "resnet18-tinyimagenet":
        backbone = resnet18(norm_layer=conv_norm)
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Identity()
        return backbone, 512

    raise NameError(f"{arch} not found in network architecture")


class encoder(nn.Module):
    def __init__(
        self,
        z_dim=1024,
        hidden_dim=4096,
        norm_p=2,
        arch="resnet18-cifar",
        norm="batch",
    ):
        super().__init__()

        backbone, feature_dim = getmodel(arch, norm=norm)
        self.backbone = backbone
        self.norm_p = norm_p
        feature_norm = nn.BatchNorm1d if norm == "batch" else nn.LayerNorm
        self.pre_feature = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            feature_norm(hidden_dim),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            feature_norm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x, is_test=False):
        feature = self.backbone(x)
        feature = self.pre_feature(feature)
        z = F.normalize(self.projection(feature), p=self.norm_p)

        if is_test:
            return z, feature
        return z
