"""
SimCLR model.

Composition:
- encoder (ViT),
- projection head (MLP),
- methods for features and projections.
"""

import torch.nn as nn


class SimCLRModel(nn.Module):

    def __init__(self, encoder, embed_dim=192, projection_dim=128):
        super().__init__()
        self.encoder = encoder

        # the projector is a small 2-layer network on top of the encoder
        # it is only used during SSL training — NT-Xent loss is computed on its output
        # after training, the projector is discarded and only the encoder is kept
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )

    def forward(self, view_a, view_b):
        # pass both augmented views through encoder + projector
        # z_a and z_b are vectors in the contrastive space, used to compute NT-Xent
        z_a = self.projector(self.encoder(view_a))
        z_b = self.projector(self.encoder(view_b))
        return z_a, z_b

    def get_features(self, x):
        # skip the projector — used during linear probe and fine-tuning
        # at this point the encoder is already trained, we just extract the embedding
        return self.encoder(x)
