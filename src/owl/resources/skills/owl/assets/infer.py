from pathlib import Path

import torch

from owl import invoke
from owl.data.augment import infer as infer_transform
from owl.data.entry import OwlV1EntrySource
from owl.invocation.components.checkpoint import CheckpointLoad
from owl.invocation.components.components import InferComponents
from owl.invocation.data.infer import InferData
from owl.invocation.execution.infer import InferExecution
from owl.invocation.infer import InferInvocation

from model import SimpleMaskModel


DATASET_ROOT = Path(r"D:\example")

# 修改为刚才训练生成的具体 checkpoint 路径。
CHECKPOINT_PATH = Path(
    r"D:\owl_test\workspace-20260725202320925\checkpoints\epoch-0004.pt"
)

IMAGE_SIZE = (512, 512)



def build_invocation() -> InferInvocation:
    """Build the standalone inference declaration."""
    components = InferComponents(
        model=SimpleMaskModel,
        checkpoint=CheckpointLoad(
            path=CHECKPOINT_PATH,
            model_only=False,
            strict=True,
        ),
    )

    data = InferData(
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

    execution = InferExecution(
        device="cuda" if torch.cuda.is_available() else "cpu",
        workspace=None,
    )

    return InferInvocation(
        components=components,
        data=data,
        execution=execution,
    )


def main() -> None:
    """Execute standalone inference through Owl orchestration."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"dataset root does not exist: {DATASET_ROOT}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"checkpoint does not exist: {CHECKPOINT_PATH}"
        )

    results = invoke(build_invocation())

    print("Inference results:")

    if results is None:
        print("No evaluation results were returned.")
        return

    for dataset_name, metrics in results.items():
        print(f"{dataset_name}:")

        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")


if __name__ == "__main__":
    main()