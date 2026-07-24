"""Tests for workspace artifact persistence."""

import json
from pathlib import Path

import pytest

from owl.workspace.workspace import Workspace


def _read_json(path: Path) -> object:
    """Read and decode one JSON document."""
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[object]:
    """Read and decode every non-empty JSON Lines record."""
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def test_train_workspace_persists_artifacts(
    tmp_path: Path,
) -> None:
    """Persist metadata, snapshots, and history for a train workspace."""
    with Workspace(
        parent=tmp_path,
        mode="train",
    ) as workspace:
        root = workspace.root
        workspace_id = workspace.workspace_id

        workspace.update_train_snapshot(
            current_epoch=1,
            total_epoch=10,
            current_batch=2,
            total_batch=100,
            loss=0.25,
            learning_rates=[1e-4, 5e-5],
        )
        workspace.append_train_history(
            epoch=1,
            batch=2,
            loss=0.25,
            learning_rates=[1e-4, 5e-5],
        )
        workspace.append_model_metric_history(
            epoch=1,
            batch=2,
            metrics={
                "accuracy": 0.9,
            },
        )
        workspace.append_criterion_metric_history(
            epoch=1,
            batch=2,
            metrics={
                "loss_bce": 0.2,
            },
        )

    assert root.exists()
    assert root.parent == tmp_path
    assert root.name == workspace_id

    metadata = _read_json(root / "metadata.json")
    assert metadata["schema_version"] == 1
    assert metadata["workspace_id"] == workspace_id
    assert metadata["mode"] == "train"
    assert metadata["owl_version"] == "0.0.2"
    assert isinstance(metadata["created_at"], str)

    lifecycle = _read_json(
        root / "snapshot" / "lifecycle.json"
    )
    assert lifecycle == {
        "status": "completed",
        "stage": "train",
        "message": "",
    }

    train_snapshot = _read_json(
        root / "snapshot" / "train.json"
    )
    assert train_snapshot == {
        "epoch": {
            "current": 1,
            "total": 10,
        },
        "batch": {
            "current": 2,
            "total": 100,
        },
        "loss": 0.25,
        "learning_rates": [1e-4, 5e-5],
    }

    train_history = _read_jsonl(
        root / "history" / "train.jsonl"
    )
    assert train_history == [
        {
            "epoch": 1,
            "batch": 2,
            "loss": 0.25,
            "learning_rates": [1e-4, 5e-5],
        }
    ]

    model_metrics = _read_jsonl(
        root / "history" / "metric" / "model.jsonl"
    )
    assert model_metrics == [
        {
            "epoch": 1,
            "batch": 2,
            "metrics": {
                "accuracy": 0.9,
            },
        }
    ]

    criterion_metrics = _read_jsonl(
        root / "history" / "metric" / "criterion.jsonl"
    )
    assert criterion_metrics == [
        {
            "epoch": 1,
            "batch": 2,
            "metrics": {
                "loss_bce": 0.2,
            },
        }
    ]


def test_infer_workspace_persists_evaluation(
    tmp_path: Path,
) -> None:
    """Persist evaluation results without an epoch for standalone infer."""
    with Workspace(
        parent=tmp_path,
        mode="infer",
    ) as workspace:
        root = workspace.root

        workspace.append_evaluation_history(
            results={
                "casia_v1": {
                    "f1": 0.72,
                    "auc": 0.91,
                },
                "coverage": {
                    "f1": 0.68,
                    "auc": 0.88,
                },
            },
        )

    lifecycle = _read_json(
        root / "snapshot" / "lifecycle.json"
    )
    assert lifecycle == {
        "status": "completed",
        "stage": "infer",
        "message": "",
    }

    evaluation_history = _read_jsonl(
        root / "history" / "evaluation.jsonl"
    )
    assert evaluation_history == [
        {
            "results": {
                "casia_v1": {
                    "f1": 0.72,
                    "auc": 0.91,
                },
                "coverage": {
                    "f1": 0.68,
                    "auc": 0.88,
                },
            },
        }
    ]
    assert "epoch" not in evaluation_history[0]

    assert not (root / "snapshot" / "train.json").exists()
    assert not (root / "history" / "train.jsonl").exists()


def test_train_evaluation_includes_epoch(
    tmp_path: Path,
) -> None:
    """Persist the one-based epoch for evaluation during training."""
    with Workspace(
        parent=tmp_path,
        mode="train",
    ) as workspace:
        root = workspace.root

        workspace.set_stage("infer")
        workspace.append_evaluation_history(
            epoch=3,
            results={
                "validation": {
                    "f1": 0.75,
                },
            },
        )

    lifecycle = _read_json(
        root / "snapshot" / "lifecycle.json"
    )
    assert lifecycle["status"] == "completed"
    assert lifecycle["stage"] == "infer"

    evaluation_history = _read_jsonl(
        root / "history" / "evaluation.jsonl"
    )
    assert evaluation_history == [
        {
            "epoch": 3,
            "results": {
                "validation": {
                    "f1": 0.75,
                },
            },
        }
    ]


def test_workspace_records_runtime_interruption(
    tmp_path: Path,
) -> None:
    """Publish interrupted lifecycle and preserve the runtime exception."""
    workspace = Workspace(
        parent=tmp_path,
        mode="train",
    )

    with pytest.raises(
        ValueError,
        match="training failed",
    ):
        with workspace:
            root = workspace.root

            workspace.update_train_snapshot(
                current_epoch=1,
                total_epoch=10,
                current_batch=1,
                total_batch=100,
                loss=1.0,
                learning_rates=[1e-4],
            )

            raise ValueError("training failed")

    lifecycle = _read_json(
        root / "snapshot" / "lifecycle.json"
    )
    assert lifecycle == {
        "status": "interrupted",
        "stage": "train",
        "message": "ValueError: training failed",
    }

    train_snapshot = _read_json(
        root / "snapshot" / "train.json"
    )
    assert train_snapshot["batch"]["current"] == 1


def test_workspace_creates_output_directories_lazily(
    tmp_path: Path,
) -> None:
    """Create checkpoint and visual directories only when requested."""
    with Workspace(
        parent=tmp_path,
        mode="train",
    ) as workspace:
        root = workspace.root

        assert not (root / "checkpoints").exists()
        assert not (root / "visual").exists()

        checkpoint_dir = workspace.checkpoint_dir()
        visual_dir = workspace.visual_dir()

        assert checkpoint_dir == root / "checkpoints"
        assert visual_dir == root / "visual"
        assert checkpoint_dir.is_dir()
        assert visual_dir.is_dir()


def test_workspace_cannot_be_entered_twice(
    tmp_path: Path,
) -> None:
    """Reject reuse of a workspace instance."""
    workspace = Workspace(
        parent=tmp_path,
        mode="train",
    )

    with workspace:
        pass

    with pytest.raises(
        RuntimeError,
        match="workspace has already been entered",
    ):
        workspace.__enter__()