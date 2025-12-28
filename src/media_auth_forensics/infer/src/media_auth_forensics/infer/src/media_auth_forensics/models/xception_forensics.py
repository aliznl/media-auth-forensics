import torch
from torch import nn
from torchvision import models


class ForensicsBinaryClassifier(nn.Module):
    """
    Binary classifier for manipulated vs real (logit output).

    Note:
      True Xception requires a dedicated implementation. This module provides a
      stable backbone interface using torchvision models (EfficientNet-B3 default)
      so training/inference is immediately usable.

    Replace backbone with a verified Xception implementation when ready.
    """

    def __init__(self, backbone: str = "efficientnet_b3", pretrained: bool = True):
        super().__init__()

        name = backbone.lower()
        if name == "efficientnet_b3":
            m = models.efficientnet_b3(pretrained=pretrained)
            in_features = m.classifier[1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
        elif name == "resnet18":
            m = models.resnet18(pretrained=pretrained)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits (shape: [N]).
        """
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(1)
        return logits
