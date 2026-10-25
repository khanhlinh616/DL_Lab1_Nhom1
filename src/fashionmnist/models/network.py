"""
network.py — Neural network architectures for FashionMNIST classification.

Architectures:
  - BaselineMLP  : Flatten → Linear → ReLU → Linear.  Simple baseline.
  - SimpleCNN    : 2 Conv blocks + GAP + FC. Main model.

Design rules:
  - Output is 10 raw logits (no Softmax); use CrossEntropyLoss externally.
  - Support CPU and GPU (device management done by caller).
  - No hardcoded input size beyond 1×28×28.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline: Fully-Connected MLP
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """
    Minimal MLP baseline.

    Architecture:
        Flatten → Linear(784, hidden) → ReLU → Dropout → Linear(hidden, 10)

    Chosen as baseline to set a performance floor before CNN experiments.
    Does not exploit spatial structure of images.
    """

    def __init__(self, hidden_units: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, 28, 28).
        Returns:
            Logits tensor of shape (B, 10).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Main Model: Small CNN
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Small convolutional network for FashionMNIST.

    Architecture:
        [Conv Block 1]: Conv(1→32, 3×3) → BN → ReLU → MaxPool(2×2)
        [Conv Block 2]: Conv(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
        [Classifier]:   AdaptiveAvgPool(1×1) → Flatten → Linear(64, fc_hidden)
                        → ReLU → Dropout → Linear(fc_hidden, 10)

    Rationale:
      - Two conv blocks capture local patterns (textures, edges, shapes).
      - BatchNorm stabilizes training; MaxPool reduces spatial dimensions.
      - Global Average Pooling before the FC reduces parameters and overfitting.
      - Dropout in FC layer for further regularization.
      - Outputs 10 logits; Softmax is not applied (use CrossEntropyLoss).
    """

    def __init__(
        self,
        hidden_channels: list[int] | None = None,
        fc_hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]
        c1, c2 = hidden_channels[0], hidden_channels[1]

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28×28 → 14×14
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14×14 → 7×7
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  # (B, c2, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, 28, 28).
        Returns:
            Logits tensor of shape (B, 10).
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Construct a model from configuration.

    Args:
        cfg: Merged configuration dict.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If architecture name is unknown.
    """
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("architecture", "SimpleCNN")

    if arch == "BaselineMLP":
        return BaselineMLP(
            hidden_units=model_cfg.get("hidden_units", 256),
            dropout=model_cfg.get("dropout", 0.0),
        )
    elif arch == "SimpleCNN":
        return SimpleCNN(
            hidden_channels=model_cfg.get("hidden_channels", [32, 64]),
            fc_hidden=model_cfg.get("fc_hidden", 256),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        raise ValueError(
            f"Unknown architecture: '{arch}'. Choose from: BaselineMLP, SimpleCNN."
        )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
