"""Shuffled batches from an :class:`~sextante.activations.ActivationStore`."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import torch
from safetensors.torch import load_file
from torch import Tensor

from ..activations import ActivationRows, ActivationStore

RowFilter = Literal["all", "image", "text"]


def _filter(rows: ActivationRows, which: RowFilter) -> ActivationRows:
    if which == "all":
        return rows
    return rows.select(rows.is_image if which == "image" else ~rows.is_image)


class ActivationLoader:
    """Iterates over one hook point of a store in shuffled batches of activations.

    Shards are read ``buffer_shards`` at a time (memory-mapped), concatenated, shuffled
    and cut into batches; leftovers roll into the next buffer. Each pass over the loader
    is one epoch that visits every (filtered) row exactly once, except the incomplete
    last batch when ``drop_last``.

    Args:
        rows: Keep ``"all"`` rows, only ``"image"`` rows or only ``"text"`` rows.
        buffer_shards: Shards mixed together. More means better shuffling, more RAM.
    """

    def __init__(
        self,
        store: ActivationStore,
        point: str,
        batch_size: int,
        *,
        rows: RowFilter = "all",
        shuffle: bool = True,
        buffer_shards: int = 8,
        drop_last: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        if rows not in ("all", "image", "text"):
            raise ValueError(f"rows must be 'all', 'image' or 'text', got {rows!r}")
        self.store = store
        self.point = point
        self.batch_size = batch_size
        self.rows = rows
        self.shuffle = shuffle
        self.buffer_shards = max(1, buffer_shards)
        self.drop_last = drop_last
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        self.epoch = 0
        self.shards = store.shard_paths(point)
        if not self.shards:
            raise ValueError(f"{point!r} has no rows in {store.path}")

    @property
    def d(self) -> int:
        return self.store.d(self.point)

    def __len__(self) -> int:
        """Batches per epoch (exact for ``rows='all'``, an upper bound otherwise)."""
        n = self.store.n_rows(self.point)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Tensor]:
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(len(self.shards), generator=gen) if self.shuffle else None
        shard_ids = order.tolist() if order is not None else list(range(len(self.shards)))
        carry: Tensor | None = None
        for start in range(0, len(shard_ids), self.buffer_shards):
            blocks = [
                _filter(ActivationRows(**load_file(str(self.shards[i]))), self.rows).acts
                for i in shard_ids[start : start + self.buffer_shards]
            ]
            if carry is not None:
                blocks.insert(0, carry)
            buffer = torch.cat(blocks)
            if self.shuffle:
                buffer = buffer[torch.randperm(len(buffer), generator=gen)]
            n_full = len(buffer) // self.batch_size * self.batch_size
            for i in range(0, n_full, self.batch_size):
                yield self._to_device(buffer[i : i + self.batch_size])
            carry = buffer[n_full:] if n_full < len(buffer) else None
        if carry is not None and len(carry) and not self.drop_last:
            yield self._to_device(carry)

    def _to_device(self, x: Tensor) -> Tensor:
        return x.to(self.device, self.dtype, non_blocking=True)

    def forever(self) -> Iterator[Tensor]:
        """Loop over epochs indefinitely (reshuffling each time)."""
        while True:
            empty = True
            for batch in self:
                empty = False
                yield batch
            if empty:
                raise ValueError(
                    f"No batches: fewer than batch_size={self.batch_size} rows match "
                    f"rows={self.rows!r}"
                )


@torch.no_grad()
def estimate_norm_scale(
    store: ActivationStore, point: str, *, rows: RowFilter = "all", max_rows: int = 100_000
) -> float:
    """Scale ``s`` such that ``s * x`` has mean L2 norm ``sqrt(d)`` (SAELens convention)."""
    norms, seen = [], 0
    for shard in store.iter_shards(point):
        acts = _filter(shard, rows).acts.float()
        norms.append(acts.norm(dim=-1))
        seen += len(acts)
        if seen >= max_rows:
            break
    if not seen:
        raise ValueError(f"No rows for {point!r} with rows={rows!r}")
    mean_norm = torch.cat(norms)[:max_rows].mean().item()
    return math.sqrt(store.d(point)) / mean_norm
