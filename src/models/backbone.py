"""
Backbone model (ViT).

Design idea:
- keep a dedicated class for ViT config under CIFAR-10 constraints,
- make tiny/small variants easy to swap.
"""

from timm.models.vision_transformer import VisionTransformer

# timm provides the ViT implementation, but its model registry only exposes
# ImageNet-oriented names such as vit_tiny_patch16_224. For CIFAR-sized runs we
# build the transformer directly so patch_size=4 and img_size=32 remain valid.


class ViTBackboneBuilder:
    _VARIANT_SPECS = {
        "vit_tiny": {"depth": 12, "num_heads": 3},
        "vit_small": {"depth": 12, "num_heads": 6},
        "vit_base": {"depth": 12, "num_heads": 12},
    }

    def __init__(self, variant="vit_tiny", patch_size=4, image_size=32, embed_dim=192):
        self.variant = variant
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

    def build(self):
        if self.variant not in self._VARIANT_SPECS:
            supported = ", ".join(sorted(self._VARIANT_SPECS))
            raise ValueError(f"Unsupported ViT variant '{self.variant}'. Supported variants: {supported}")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for ViT patch embedding.")

        # ViT splits the image into small patches and processes them
        # like words in a sentence (same idea as a text Transformer)
        # patch_size=4 on a 32x32 image gives 64 patches (8x8 grid)
        # num_classes=0 removes the classification head — we only need
        # the raw embedding vector (192 numbers), we'll attach the head separately
        spec = self._VARIANT_SPECS[self.variant]
        return VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=spec["depth"],
            num_heads=spec["num_heads"],
            num_classes=0,
        )
