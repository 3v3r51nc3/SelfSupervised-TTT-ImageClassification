"""
Downstream classifier.

Scenarios:
- linear probe (encoder frozen),
- fine-tune (encoder trainable).
"""

import torch.nn as nn


class LinearClassifier(nn.Module):
    # a single linear layer: 192-dim embedding → 10 class scores
    # used on top of a frozen encoder during linear probe

    def __init__(self, embed_dim=192, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class FineTuneModel(nn.Module):
    # encoder + classifier together, both are trained (fine-tune mode)
    # unlike linear probe, the encoder weights are not frozen here

    def __init__(self, encoder, embed_dim=192, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = LinearClassifier(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))
