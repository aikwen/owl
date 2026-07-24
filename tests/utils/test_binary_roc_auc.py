"""Tests for image-level pixel ROC AUC calculation."""

import pytest
import torch

from owl.utils.metric import binary_roc_auc


def test_binary_roc_auc_for_perfect_ranking() -> None:
    """Perfectly ranked positive pixels should produce an AUC of one."""
    target = torch.tensor(
        [
            [[[0, 0], [1, 1]]],
            [[[0, 1], [0, 1]]],
        ],
    )
    prediction = torch.tensor(
        [
            [[[0.1, 0.2], [0.8, 0.9]]],
            [[[0.1, 0.9], [0.2, 0.8]]],
        ],
    )

    result = binary_roc_auc(target, prediction)

    expected = torch.tensor([1.0, 1.0])

    torch.testing.assert_close(result, expected)


def test_binary_roc_auc_for_reversed_ranking() -> None:
    """Completely reversed ranking should produce an AUC of zero."""
    target = torch.tensor(
        [[[[0, 0], [1, 1]]]],
    )
    prediction = torch.tensor(
        [[[[0.9, 0.8], [0.2, 0.1]]]],
    )

    result = binary_roc_auc(target, prediction)

    expected = torch.tensor([0.0])

    torch.testing.assert_close(result, expected)


def test_binary_roc_auc_matches_v1_example() -> None:
    """The implementation should preserve the v1 IMDLBenCo convention."""
    target = torch.tensor(
        [[[[0, 0], [1, 1]]]],
    )
    prediction = torch.tensor(
        [[[[0.1, 0.4], [0.3, 0.8]]]],
    )

    result = binary_roc_auc(target, prediction)

    expected = torch.tensor([0.75])

    torch.testing.assert_close(result, expected)


def test_binary_roc_auc_returns_nan_for_all_negative_target() -> None:
    """A target containing only negatives has an undefined ROC AUC."""
    target = torch.zeros(1, 1, 2, 2)
    prediction = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]]]],
    )

    result = binary_roc_auc(target, prediction)

    assert torch.isnan(result[0])


def test_binary_roc_auc_returns_nan_for_all_positive_target() -> None:
    """A target containing only positives has an undefined ROC AUC."""
    target = torch.ones(1, 1, 2, 2)
    prediction = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]]]],
    )

    result = binary_roc_auc(target, prediction)

    assert torch.isnan(result[0])


def test_binary_roc_auc_preserves_nan_inside_batch() -> None:
    """Undefined samples should remain in their original batch positions."""
    target = torch.tensor(
        [
            [[[0, 0], [1, 1]]],
            [[[0, 0], [0, 0]]],
        ],
        dtype=torch.float32,
    )
    prediction = torch.tensor(
        [
            [[[0.1, 0.2], [0.8, 0.9]]],
            [[[0.1, 0.2], [0.3, 0.4]]],
        ],
    )

    result = binary_roc_auc(target, prediction)

    torch.testing.assert_close(result[0], torch.tensor(1.0))
    assert torch.isnan(result[1])


def test_binary_roc_auc_requires_four_dimensions() -> None:
    """Both tensors must use the [B, C, H, W] layout."""
    target = torch.zeros(1, 2, 2)
    prediction = torch.zeros(1, 2, 2)

    with pytest.raises(
        ValueError,
        match=r"must have shape \[B, C, H, W\]",
    ):
        binary_roc_auc(target, prediction)


def test_binary_roc_auc_requires_matching_shapes() -> None:
    """Prediction and target must have identical shapes."""
    target = torch.zeros(2, 1, 2, 2)
    prediction = torch.zeros(1, 1, 2, 2)

    with pytest.raises(
        ValueError,
        match="must have the same shape",
    ):
        binary_roc_auc(target, prediction)