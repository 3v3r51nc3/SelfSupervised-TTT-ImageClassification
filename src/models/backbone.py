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

    def __init__(
        self,
        variant: str = "vit_tiny",
        patch_size: int = 4,
        image_size: int = 32,
        embed_dim: int = 192,
        drop_path_rate: float = 0.0,
    ) -> None:
        self.variant = variant
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.drop_path_rate = drop_path_rate

    def build(self):
        if self.variant not in self._VARIANT_SPECS:
            supported = ", ".join(sorted(self._VARIANT_SPECS))
            raise ValueError(f"Unsupported ViT variant '{self.variant}'. Supported variants: {supported}")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for ViT patch embedding.")

        # patch_size=4 on a 32x32 image gives 64 tokens (8x8 grid).
        # num_classes=0 returns the raw embedding (the classifier head is attached separately).
        spec = self._VARIANT_SPECS[self.variant]
        return VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=spec["depth"],
            num_heads=spec["num_heads"],
            num_classes=0,
            drop_path_rate=self.drop_path_rate,
        )
