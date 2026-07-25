from torch.nn import BCEWithLogitsLoss, Module
from torch import Tensor
from owl.schemas.calls.context import TrainCallContext

class SimpleMaskCriterion(Module):
    """Compute pixel-level binary cross-entropy loss."""

    def __init__(self) -> None:
        super().__init__()

        self.loss_function = BCEWithLogitsLoss()

    def forward(
        self,
        loss_output: Tensor,
        batch: dict[str, Tensor],
        context: TrainCallContext,
    ) -> dict[str, Tensor]:
        """Compute loss from model logits and ground-truth masks."""
        del context

        target = batch["gt"].float()
        loss = self.loss_function(
            loss_output,
            target,
        )

        return {
            "loss": loss,
            "metric:loss_bce": loss.detach().item()
        }
