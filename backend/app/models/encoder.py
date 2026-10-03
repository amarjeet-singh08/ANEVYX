from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.models as models

class FoundationEncoderAdapter(nn.Module, ABC):
    """
    Abstract adapter for pluggable foundation vision models.
    """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Returns the primary bottleneck representation."""
        pass

    @abstractmethod
    def encode_multiscale(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Returns multiscale feature maps."""
        pass
        
    @abstractmethod
    def embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Returns a flat global embedding for memory bank comparisons."""
        pass

    @abstractmethod
    def feature_maps(self, image: torch.Tensor) -> torch.Tensor:
        """Returns the raw spatial feature maps."""
        pass


class ResNetEncoder(FoundationEncoderAdapter):
    """
    A real pre-trained ResNet18. 
    Uses supervised ImageNet pretrained weights to act as a robust vision foundation.
    (Point 8: Do not claim self-supervised learning unless it is actually self-supervised.)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Adapt for 1-channel input by summing weights across RGB channels, 
        # preserving the pre-trained features instead of starting from scratch.
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.backbone.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            
        # We don't need the classification head
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def feature_maps(self, image: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(image)
        
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.feature_maps(image)

    def encode_multiscale(self, image: torch.Tensor) -> list[torch.Tensor]:
        # Extract from multiple layers
        x = image
        features = []
        for name, module in self.backbone.named_children():
            if name in ['fc', 'avgpool']: continue
            x = module(x)
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                features.append(x)
        return features
        
    def embedding(self, image: torch.Tensor) -> torch.Tensor:
        fmap = self.feature_maps(image)
        emb = self.pool(fmap)
        return torch.flatten(emb, 1)
