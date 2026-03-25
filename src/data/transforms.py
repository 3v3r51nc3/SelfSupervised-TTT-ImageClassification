"""
Augmentation factory.

Separate modes are required:
- simclr_aug: two strong views of one image,
- eval_aug: minimal processing for evaluation.
"""

from torchvision import transforms

# cifar10 mean/std (precomputed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SimCLRTwoViewTransform:
    """Applies two different random augmentations to the same image."""

    def __init__(self, image_size=32):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.aug(x), self.aug(x)


class EvalTransform:
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)


class TransformFactory:
    def __init__(self, image_size=32):
        self.image_size = image_size

    def build_simclr(self):
        return SimCLRTwoViewTransform(self.image_size)

    def build_eval(self):
        return EvalTransform(self.image_size)
