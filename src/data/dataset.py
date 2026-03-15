"""
Data module (CIFAR-10).

Expected responsibilities:
- load train/val/test splits,
- build DataLoaders for SSL and supervised stages,
- optionally support CIFAR-10-C for robustness/TTT.
"""


class CIFARDataModule:
    # TODO: Prepare datasets and data loaders.
    def prepare_data(self) -> None:
        pass

    # TODO: Return data loader for SimCLR pretraining.
    def train_ssl_loader(self):
        pass

    # TODO: Return data loaders for linear probe and fine-tune.
    def supervised_loaders(self):
        pass

    # TODO: Return test/corruption loader for final evaluation.
    def test_loader(self):
        pass
