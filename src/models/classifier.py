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


class TTTModel(nn.Module):
    # Y-shape model for Sun 2020 TTT:
    # shared encoder feeds both the supervised classifier and a 4-way rotation head.
    # forward(x)         → classification logits (used at inference)
    # forward_rotation(x) → rotation logits (used as SSL auxiliary at train and adapt time)

    def __init__(self, encoder, embed_dim: int = 192, num_classes: int = 10, num_rotations: int = 4):
        super().__init__()
        self.encoder = encoder
        self.classifier = LinearClassifier(embed_dim, num_classes)
        self.rotation_head = nn.Linear(embed_dim, num_rotations)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def forward_rotation(self, x):
        return self.rotation_head(self.encoder(x))
