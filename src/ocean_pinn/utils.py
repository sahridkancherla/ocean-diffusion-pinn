"""Device selection and mini-batching helpers."""

from __future__ import annotations

from collections.abc import Iterator

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """`"auto"` picks CUDA when available, otherwise the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def minibatches(
    n: int,
    batch_size: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """Index tensors covering range(n) in batches, shuffled if a generator is given.

    The last batch may be smaller than `batch_size`.
    """
    order = torch.arange(n) if generator is None else torch.randperm(n, generator=generator)
    yield from order.to(device).split(batch_size)
