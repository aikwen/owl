from torch.nn import Conv2d, Module, ReLU, Sequential
from torch import Tensor
from owl.schemas.calls.context import TrainCallContext

class SimpleMaskModel(Module):
    """Map an RGB image to one-channel logits at the same resolution."""

    def __init__(self) -> None:
        super().__init__()

        self.network = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            ReLU(inplace=True),
            Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        context: TrainCallContext | None = None,
    ) -> dict[str, Tensor]:
        """Execute either a training or inference model call."""
        del context

        logits = self.network(batch["tp"])

        return {
            "loss": logits,
            "eval": logits,
            "metric:logit_mean": logits.detach().mean().item()
        }