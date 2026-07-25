"""Tests for the default binary mask evaluator."""

import math

import torch
from torch import Tensor

from owl.processors.evaluator.binary_mask import BinaryMaskEvaluator


def _to_logits(probability: Tensor) -> Tensor:
    """Convert test probabilities to logits."""
    return torch.logit(probability)


def test_binary_mask_evaluator_computes_batch_metrics() -> None:
    """A batch of perfect predictions should produce perfect metrics."""
    target = torch.tensor(
        [
            [[[0, 0], [1, 1]]],
            [[[0, 1], [0, 1]]],
        ],
        dtype=torch.float32,
    )
    probability = torch.tensor(
        [
            [[[0.1, 0.2], [0.8, 0.9]]],
            [[[0.1, 0.9], [0.2, 0.8]]],
        ],
    )

    evaluator = BinaryMaskEvaluator()
    evaluator.update(
        eval_output=_to_logits(probability),
        batch={"gt": target},
    )

    result = evaluator.compute()

    assert result["pixel_f1"] == 1.0
    assert result["pixel_auc"] == 1.0


def test_binary_mask_evaluator_weights_samples_across_batches() -> None:
    """Dataset metrics should be averaged by sample rather than by batch."""
    perfect_target = torch.tensor(
        [[[[0, 0], [1, 1]]]],
        dtype=torch.float32,
    )
    perfect_probability = torch.tensor(
        [[[[0.1, 0.2], [0.8, 0.9]]]],
    )

    reversed_target = perfect_target.repeat(3, 1, 1, 1)
    reversed_probability = torch.tensor(
        [[[[0.9, 0.8], [0.2, 0.1]]]],
    ).repeat(3, 1, 1, 1)

    evaluator = BinaryMaskEvaluator()

    evaluator.update(
        eval_output=_to_logits(perfect_probability),
        batch={"gt": perfect_target},
    )
    evaluator.update(
        eval_output=_to_logits(reversed_probability),
        batch={"gt": reversed_target},
    )

    result = evaluator.compute()

    assert result["pixel_f1"] == 0.25
    assert result["pixel_auc"] == 0.25


def test_binary_mask_evaluator_includes_empty_target_in_f1() -> None:
    """An empty target should contribute zero to the pixel F1 average."""
    target = torch.tensor(
        [
            [[[0, 0], [1, 1]]],
            [[[0, 0], [0, 0]]],
        ],
        dtype=torch.float32,
    )
    probability = torch.tensor(
        [
            [[[0.1, 0.2], [0.8, 0.9]]],
            [[[0.1, 0.2], [0.3, 0.4]]],
        ],
    )

    evaluator = BinaryMaskEvaluator()
    evaluator.update(
        eval_output=_to_logits(probability),
        batch={"gt": target},
    )

    result = evaluator.compute()

    assert result["pixel_f1"] == 0.5
    assert result["pixel_auc"] == 1.0


def test_binary_mask_evaluator_excludes_undefined_auc() -> None:
    """Single-class targets should not contribute to the AUC average."""
    target = torch.tensor(
        [
            [[[0, 0], [1, 1]]],
            [[[0, 0], [0, 0]]],
            [[[1, 1], [1, 1]]],
        ],
        dtype=torch.float32,
    )
    probability = torch.tensor(
        [
            [[[0.1, 0.2], [0.8, 0.9]]],
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.6, 0.7], [0.8, 0.9]]],
        ],
    )

    evaluator = BinaryMaskEvaluator()
    evaluator.update(
        eval_output=_to_logits(probability),
        batch={"gt": target},
    )

    result = evaluator.compute()

    assert result["pixel_auc"] == 1.0


def test_binary_mask_evaluator_returns_nan_without_valid_auc() -> None:
    """AUC should be NaN when every target contains only one class."""
    target = torch.tensor(
        [
            [[[0, 0], [0, 0]]],
            [[[1, 1], [1, 1]]],
        ],
        dtype=torch.float32,
    )
    probability = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.6, 0.7], [0.8, 0.9]]],
        ],
    )

    evaluator = BinaryMaskEvaluator()
    evaluator.update(
        eval_output=_to_logits(probability),
        batch={"gt": target},
    )

    result = evaluator.compute()

    assert math.isnan(result["pixel_auc"])


def test_binary_mask_evaluator_returns_nan_before_update() -> None:
    """Both metrics should be NaN before any samples are accumulated."""
    evaluator = BinaryMaskEvaluator()

    result = evaluator.compute()

    assert math.isnan(result["pixel_f1"])
    assert math.isnan(result["pixel_auc"])


def test_binary_mask_evaluator_reset_clears_accumulated_state() -> None:
    """Reset should remove metrics accumulated for a previous dataloader."""
    target = torch.tensor(
        [[[[0, 0], [1, 1]]]],
        dtype=torch.float32,
    )
    perfect_probability = torch.tensor(
        [[[[0.1, 0.2], [0.8, 0.9]]]],
    )
    reversed_probability = torch.tensor(
        [[[[0.9, 0.8], [0.2, 0.1]]]],
    )

    evaluator = BinaryMaskEvaluator()

    evaluator.update(
        eval_output=_to_logits(perfect_probability),
        batch={"gt": target},
    )
    evaluator.reset()
    evaluator.update(
        eval_output=_to_logits(reversed_probability),
        batch={"gt": target},
    )

    result = evaluator.compute()

    assert result["pixel_f1"] == 0.0
    assert result["pixel_auc"] == 0.0


def test_binary_mask_evaluator_uses_configured_threshold() -> None:
    """The configured probability threshold should control F1 binarization."""
    target = torch.tensor(
        [[[[0, 1], [0, 1]]]],
        dtype=torch.float32,
    )
    probability = torch.tensor(
        [[[[0.1, 0.7], [0.2, 0.9]]]],
    )

    evaluator = BinaryMaskEvaluator(threshold=0.8)
    evaluator.update(
        eval_output=_to_logits(probability),
        batch={"gt": target},
    )

    result = evaluator.compute()

    torch.testing.assert_close(
        torch.tensor(result["pixel_f1"]),
        torch.tensor(2.0 / 3.0),
    )
    assert result["pixel_auc"] == 1.0
