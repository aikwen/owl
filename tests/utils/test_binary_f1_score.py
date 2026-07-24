"""Tests for image-level binary F1 score calculation."""

import pytest
import torch

from owl.utils.metric import binary_f1_score


def test_binary_f1_score_for_batch() -> None:
    """The function should compute one F1 score for each batch item."""
    prediction = torch.tensor(
        [
            [[[1, 1], [0, 0]]],
            [[[1, 0], [1, 0]]],
        ],
    )
    target = torch.tensor(
        [
            [[[1, 1], [0, 0]]],
            [[[1, 1], [0, 0]]],
        ],
    )

    result = binary_f1_score(prediction, target)

    expected = torch.tensor([1.0, 0.5])

    torch.testing.assert_close(result, expected)


def test_binary_f1_score_for_empty_masks() -> None:
    """Two masks without positive pixels should produce an F1 score of zero."""
    prediction = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 1, 4, 4)

    result = binary_f1_score(prediction, target)

    expected = torch.tensor([0.0, 0.0])

    torch.testing.assert_close(result, expected)


def test_binary_f1_score_for_completely_wrong_prediction() -> None:
    """A prediction with no correctly predicted positive pixels returns zero."""
    prediction = torch.ones(1, 1, 2, 2)
    target = torch.zeros(1, 1, 2, 2)

    result = binary_f1_score(prediction, target)

    expected = torch.tensor([0.0])

    torch.testing.assert_close(result, expected)


def test_binary_f1_score_requires_four_dimensions() -> None:
    """Both tensors must use the [B, C, H, W] layout."""
    prediction = torch.zeros(1, 2, 2)
    target = torch.zeros(1, 2, 2)

    with pytest.raises(
        ValueError,
        match=r"must have shape \[B, C, H, W\]",
    ):
        binary_f1_score(prediction, target)


def test_binary_f1_score_requires_matching_shapes() -> None:
    """Prediction and target must have identical shapes."""
    prediction = torch.zeros(2, 1, 2, 2)
    target = torch.zeros(1, 1, 2, 2)

    with pytest.raises(
        ValueError,
        match="must have the same shape",
    ):
        binary_f1_score(prediction, target)
