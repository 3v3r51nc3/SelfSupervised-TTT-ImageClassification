"""
Backbone model (ViT).

Design idea:
- keep a dedicated class for ViT config under CIFAR-10 constraints,
- make tiny/small variants easy to swap.
"""

import timm

# timm is a library with hundreds of ready-to-use vision model architectures
# we use it to avoid writing ViT from scratch


class ViTBackboneBuilder:

    def __init__(self, variant="vit_tiny", patch_size=4, image_size=32):
        self.variant = variant
        self.patch_size = patch_size
        self.image_size = image_size

    def build(self):
        # ViT splits the image into small patches and processes them
        # like words in a sentence (same idea as a text Transformer)
        # patch_size=4 on a 32x32 image gives 64 patches (8x8 grid)
        # num_classes=0 removes the classification head — we only need
        # the raw embedding vector (192 numbers), we'll attach the head separately
        model_name = f"{self.variant}_patch{self.patch_size}_{self.image_size}"
        encoder = timm.create_model(model_name, pretrained=False, num_classes=0)
        return encoder
