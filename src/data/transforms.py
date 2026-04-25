"""
Augmentation factory.

Separate modes are required:
- simclr: two strong views of one image (SimCLR pretraining),
- supervised_train: standard CIFAR augmentations + RandAugment for finetune,
- eval: minimal processing for evaluation / linear probe / val.
"""

from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SimCLRTwoViewTransform:
    """Applies two different random augmentations to the same image."""

    def __init__(self, image_size: int = 32) -> None:
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.aug(x), self.aug(x)


class SupervisedTrainTransform:
    """RandomCrop + flip + RandAugment + RandomErasing — standard CIFAR finetune recipe."""

    def __init__(self, image_size: int = 32, randaug_n: int = 2, randaug_m: int = 9) -> None:
        ops = [
            transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug_n > 0:
            ops.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        self.transform = transforms.Compose(ops)

    def __call__(self, x):
        return self.transform(x)


class EvalTransform:
    def __init__(self, image_size: int = 32) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)


class TransformFactory:
    def __init__(
        self,
        image_size: int = 32,
        randaug_n: int = 2,
        randaug_m: int = 9,
    ) -> None:
        self.image_size = image_size
        self.randaug_n = randaug_n
        self.randaug_m = randaug_m

    def build_simclr(self):
        return SimCLRTwoViewTransform(self.image_size)

    def build_supervised_train(self, augment: bool = True):
        if not augment:
            return self.build_eval()
        return SupervisedTrainTransform(
            image_size=self.image_size,
            randaug_n=self.randaug_n,
            randaug_m=self.randaug_m,
        )

    def build_eval(self):
        return EvalTransform(self.image_size)
