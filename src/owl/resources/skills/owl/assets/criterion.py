from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module

from owl.data.dataset import DatasetBatch
from owl.schemas.calls.context import TrainCallContext
from owl.schemas.outputs.criterion import CriterionOutput


class SimpleMaskCriterion(Module):
    """Compute pixel-level binary cross-entropy loss."""

    def __init__(self) -> None:
        super().__init__()

        self.loss_function = BCEWithLogitsLoss()

    def forward(
        self,
        loss_output: Tensor,
        batch: DatasetBatch,
        context: TrainCallContext,
    ) -> CriterionOutput:
        """Compute loss from model logits and ground-truth masks."""
        del context

        target = batch["gt"].float()
        loss = self.loss_function(
            loss_output,
            target,
        )

        return {
            "loss": loss,
        }