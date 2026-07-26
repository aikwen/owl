from model import SimpleMaskModel
from criterion import SimpleMaskCriterion
from pathlib import Path
import torch
from owl import invoke
from owl.data.augment import infer as infer_transform
from owl.data.augment import train as train_transform
from owl.data.entry import OwlV1EntrySource
from owl.invocation.components.components import TrainComponents
from owl.invocation.data.infer import InferData
from owl.invocation.data.train import TrainData
from owl.invocation.execution.checkpoint import CheckpointSave
from owl.invocation.execution.train import TrainExecution
from owl.invocation.train import TrainInference, TrainInvocation
from owl.optim import adamw, poly


DATASET_ROOT = Path(r"D:\example")
IMAGE_SIZE = (512, 512)



def build_invocation() -> TrainInvocation:
    """Build the complete Owl training declaration."""
    training_data = TrainData(
        sources=DATASET_ROOT,
        default_entry_source=OwlV1EntrySource,
        augment=train_transform(IMAGE_SIZE),
        loader={
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        },
    )

    validation_data = InferData(
        sources={
            "example": DATASET_ROOT,
        },
        default_entry_source=OwlV1EntrySource,
        augment=infer_transform(IMAGE_SIZE),
        loader={
            "batch_size": 2,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        },
    )

    components = TrainComponents(
        model=SimpleMaskModel,
        criterion=SimpleMaskCriterion,
        optimizer=adamw(
            lr=1e-3,
            weight_decay=1e-4,
        ),
        scheduler=poly(
            power=0.9,
        ),
        checkpoint=None,
    )

    execution = TrainExecution(
        total_epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        workspace=None,
        checkpoint=CheckpointSave(
            autosave=True,
        ),
    )

    return TrainInvocation(
        components=components,
        data=training_data,
        execution=execution,
        inference=TrainInference(
            data=validation_data,
        ),
    )


def main() -> None:
    """Execute training through Owl orchestration."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"dataset root does not exist: {DATASET_ROOT}"
        )

    invocation = build_invocation()

    invoke(invocation)


if __name__ == "__main__":
    main()