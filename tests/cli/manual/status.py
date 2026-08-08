"""Manually simulate workspace snapshots for testing ``owl status``."""

import random
import shutil
import time
from pathlib import Path

from owl.workspace.snapshot import _SnapshotStore


_TOTAL_EPOCHS = 5
_TOTAL_BATCHES = 60

_MIN_BATCH_SECONDS = 0.7
_MAX_BATCH_SECONDS = 1.0

_MIN_EVALUATION_SECONDS = 4.0
_MAX_EVALUATION_SECONDS = 6.0

_FINAL_DISPLAY_SECONDS = 2.0

_DEMO_DIRECTORY_NAME = ".owl-status-demo"


def main() -> None:
    """Generate live training snapshots in an isolated demo workspace."""
    root = Path.cwd() / _DEMO_DIRECTORY_NAME
    snapshot_path = root / "snapshot"

    # Remove stale data left by a previously force-terminated demo.
    shutil.rmtree(
        root,
        ignore_errors=True,
    )

    root.mkdir(parents=True)

    store = _SnapshotStore(
        root=root,
        mode="train",
    )

    try:
        store.start()

        for epoch in range(1, _TOTAL_EPOCHS + 1):
            store.set_stage("train")

            for batch in range(1, _TOTAL_BATCHES + 1):
                total_steps = _TOTAL_EPOCHS * _TOTAL_BATCHES
                current_step = (
                    (epoch - 1) * _TOTAL_BATCHES
                    + batch
                )
                progress = current_step / total_steps

                loss = max(
                    0.05,
                    1.2 * (1.0 - progress)
                    + random.uniform(-0.03, 0.03),
                )

                learning_rate = 1e-4 * (
                    1.0 - progress * 0.8
                )

                store.update_train(
                    current_epoch=epoch,
                    total_epoch=_TOTAL_EPOCHS,
                    current_batch=batch,
                    total_batch=_TOTAL_BATCHES,
                    loss=loss,
                    learning_rates=[learning_rate],
                )

                time.sleep(
                    random.uniform(
                        _MIN_BATCH_SECONDS,
                        _MAX_BATCH_SECONDS,
                    )
                )

            if epoch < _TOTAL_EPOCHS:
                store.set_stage("infer")

                time.sleep(
                    random.uniform(
                        _MIN_EVALUATION_SECONDS,
                        _MAX_EVALUATION_SECONDS,
                    )
                )

        store.complete()

        time.sleep(_FINAL_DISPLAY_SECONDS)

    except KeyboardInterrupt:
        store.interrupt(
            "Manual status demo interrupted."
        )
        time.sleep(_FINAL_DISPLAY_SECONDS)

    finally:
        shutil.rmtree(
            root,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()