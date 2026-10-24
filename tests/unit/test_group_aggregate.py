import math

import pytest
import torch

from art.utils.group_aggregate import Reduction, group_aggregate


def _reference(values, by, reduce, broadcast):
    """Reduce each group independently, without scatter operations."""
    groups = torch.unique(by, sorted=True)
    batches = []
    for batch in range(values.shape[0]):
        reduced = []
        for group in groups:
            selected = values[batch, by[batch] == group]
            if selected.shape[0] == 0:
                fill = float("inf") if reduce == "min" else float("-inf")
                reduced.append(values.new_full(values.shape[2:], fill))
            elif reduce == "min":
                reduced.append(selected.amin(dim=0))
            elif reduce == "max":
                reduced.append(selected.amax(dim=0))
            elif reduce == "sum":
                reduced.append(selected.sum(dim=0))
            else:
                reduced.append(selected.sum(dim=0) / (selected.shape[0] + 1e-8))
        if broadcast:
            reduced = [reduced[groups.tolist().index(label)] for label in by[batch]]
        batches.append(torch.stack(reduced))
    return torch.stack(batches)


@pytest.mark.parametrize("reduce", ["min", "max", "sum", "mean"])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("feature_shape", [(), (3,), (2, 3)])
def test_group_aggregate_matches_reference(
    reduce: Reduction, broadcast, dtype, feature_shape
):
    by = torch.tensor([[7, -3, 7, 42, -3, 42], [42, 7, -3, 7, 42, -3]])
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(2, 6, *feature_shape, generator=generator, dtype=dtype)
    values.requires_grad_()
    reference_values = values.detach().clone().requires_grad_()

    actual = group_aggregate(values, by=by, reduce=reduce, broadcast=broadcast)
    expected = _reference(reference_values, by, reduce, broadcast)
    torch.testing.assert_close(actual, expected)

    weights = torch.arange(1, actual.numel() + 1, dtype=dtype).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(values.grad, reference_values.grad)


@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_group_aggregate_extrema_ties_and_missing_groups(
    reduce: Reduction, broadcast, dtype
):
    # Positive minima / negative maxima must not be clipped to zero. Group 7
    # occurs only in the second batch, and tied extrema share gradients.
    values = torch.tensor([[2, 2, 5, 5], [3, 3, 4, 4]], dtype=dtype)
    if reduce == "max":
        values = -values
    values.requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    by = torch.tensor([[-3, -3, 42, 42], [7, 7, 42, 42]])

    actual = group_aggregate(values, by=by, reduce=reduce, broadcast=broadcast)
    expected = _reference(reference_values, by, reduce, broadcast)
    torch.testing.assert_close(actual, expected)
    if not broadcast:
        fill = math.inf if reduce == "min" else -math.inf
        assert actual[0, 1].item() == fill
        assert actual[1, 0].item() == fill

    actual[torch.isfinite(actual)].sum().backward()
    expected[torch.isfinite(expected)].sum().backward()
    torch.testing.assert_close(values.grad, reference_values.grad)
