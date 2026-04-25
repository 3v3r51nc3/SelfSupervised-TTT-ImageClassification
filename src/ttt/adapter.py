"""
Test-Time Training adapter (TENT-style).

Implements entropy-minimization on a (corruption, severity) test stream.
Only LayerNorm affine parameters are updated by default — this matches
the TENT recipe and the `adapt_scope: norm_only` config flag.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestTimeAdapter:
    """Online test-time adaptation by entropy minimization."""

    def __init__(
        self,
        model: nn.Module,
        steps: int = 1,
        learning_rate: float = 1.0e-4,
        adapt_scope: str = "norm_only",
    ) -> None:
        if adapt_scope not in {"norm_only", "all"}:
            raise ValueError("adapt_scope must be 'norm_only' or 'all'.")
        if steps < 1:
            raise ValueError("steps must be >= 1.")

        self.model = model
        self.steps = steps
        self.learning_rate = learning_rate
        self.adapt_scope = adapt_scope

        self._trainable_params: list[nn.Parameter] = self._configure_params()
        self._initial_state = self._snapshot_state()
        self.optimizer = torch.optim.Adam(self._trainable_params, lr=self.learning_rate)
        self._initial_optim_state = copy.deepcopy(self.optimizer.state_dict())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore weights to their pre-adaptation snapshot."""
        self.model.load_state_dict(self._initial_state, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(self._initial_optim_state))

    @torch.enable_grad()
    def adapt_and_predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run K adaptation steps on `images`, then return final logits."""
        self._set_eval_with_trainable_norms()

        for _ in range(self.steps):
            logits = self.model(images)
            loss = self._entropy(logits)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            logits = self.model(images)
        return logits

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _configure_params(self) -> list[nn.Parameter]:
        if self.adapt_scope == "all":
            for p in self.model.parameters():
                p.requires_grad = True
            return list(self.model.parameters())

        # norm_only: enable grads on LayerNorm/BatchNorm affine, freeze the rest.
        for p in self.model.parameters():
            p.requires_grad = False

        params: list[nn.Parameter] = []
        for module in self.model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    module.weight.requires_grad = True
                    params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    params.append(module.bias)
        if not params:
            raise ValueError("No normalization layers found for 'norm_only' TTT scope.")
        return params

    def _set_eval_with_trainable_norms(self) -> None:
        """Eval mode for everything except (Layer/Batch/Group)Norm — those stay updatable."""
        self.model.eval()
        if self.adapt_scope == "norm_only":
            for module in self.model.modules():
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    module.train()

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -(probs * log_probs).sum(dim=1).mean()


def collect_norm_params(model: nn.Module) -> Iterable[nn.Parameter]:
    """Helper exposed for tests / external callers."""
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if module.weight is not None:
                yield module.weight
            if module.bias is not None:
                yield module.bias
