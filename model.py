"""
Model definition: HybridCNNViT
- Combines a ViT-Large backbone with a small CNN head and a final classifier.
"""

import torch
import torch.nn as nn
import timm

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNViT, self).__init__()
        # Using ViT-Large architecture
        self.vit = timm.create_model("vit_large_patch16_224", pretrained=False) 
        self.vit.head = nn.Identity()
        
        # Simple CNN head
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)), nn.Flatten()
        )
        
        # ViT-Large outputs 1024 features
        vit_out = 1024 
        cnn_out = 64 * 7 * 7
        self.fc = nn.Linear(vit_out + cnn_out, num_classes)

    def forward(self, x):
        vit_feat = self.vit(x)
        cnn_feat = self.cnn(x)
        combined = torch.cat((vit_feat, cnn_feat), dim=1)
        out = self.fc(combined)
        return out