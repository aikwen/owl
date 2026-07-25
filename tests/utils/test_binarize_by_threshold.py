"""Tests for threshold-based tensor binarization."""

import torch

from owl.utils.metric import binarize_by_threshold


def test_binarize_probability_values() -> None:
    """Values equal to the threshold should belong to the positive class."""
    values = torch.tensor([0.2, 0.5, 0.8])

    result = binarize_by_threshold(
        values,
        threshold=0.5,
    )

    expected = torch.tensor([0.0, 1.0, 1.0])

    torch.testing.assert_close(result, expected)


def test_binarize_grayscale_values() -> None:
    """The function should support numeric ranges other than probabilities."""
    values = torch.tensor(
        [0, 127, 128, 255],
        dtype=torch.uint8,
    )

    result = binarize_by_threshold(
        values,
        threshold=128,
    )

    expected = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.uint8,
    )

    assert torch.equal(result, expected)


def test_binarize_batched_mask() -> None:
    """A batched mask should be binarized without changing its properties."""
    values = torch.tensor(
        [
            [[[0.1, 0.5], [0.7, 0.3]]],
            [[[0.9, 0.4], [0.5, 0.0]]],
        ],
        dtype=torch.float32,
    )

    result = binarize_by_threshold(
        values,
        threshold=0.5,
    )

    expected = torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(result, expected)
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert result.device == values.device


def test_binarize_does_not_modify_input() -> None:
    """Thresholding should return a new tensor without changing the input."""
    values = torch.tensor([0.2, 0.5, 0.8])
    original = values.clone()

    result = binarize_by_threshold(
        values,
        threshold=0.5,
    )

    assert torch.equal(values, original)
    assert result.data_ptr() != values.data_ptr()
