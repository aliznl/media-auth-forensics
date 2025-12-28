import torch
from torch import nn
from torchvision import models


class GeneratorIdentifier(nn.Module):
    """
    Multi-class classifier that estimates WHICH generator/editing family likely created the manipulation.

    This requires labeled training data per generator family (e.g., StyleGAN2, FaceSwap, Deepfakes, SD-edit, etc.).
    Output is probabilities over classes, not a ground-truth proof.
    """

    def __init__(self, num_classes: int = 8, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            m = models.resnet18(pretrained=pretrained)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError("Only resnet18 is configured by default for identifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class logits (shape: [N, C]).
        """
        feats = self.backbone(x)
        return self.head(feats)
