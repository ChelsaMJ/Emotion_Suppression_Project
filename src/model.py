import torch
import torch.nn as nn
from torchvision import models

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)

        # Freeze backbone (optional during fine-tuning)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Replace classifier head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)