"""One pass of an SAE over a store: per-sample maxima and per-feature statistics.

Everything downstream (top examples, metadata statistics, autointerp) reads from a
:class:`FeatureScan`, so the expensive part (encoding every row) happens once.

Memory: ``sample_max`` is a dense ``[n_samples, d_sae]`` float32 matrix (10k samples x
16k latents = 640 MB). It is saved as float16.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from ..activations import ActivationStore
from ..sae import SAE
from ..sae.data import RowFilter

_TENSORS = ("sample_max", "fire_count", "act_sum", "image_act_sum", "image_fire_count")


@dataclass
class FeatureScan:
    """Result of :func:`scan_features`.

    Attributes:
        sample_max: ``[n_samples, d_sae]`` max activation of each feature in each sample.
        fire_count: ``[d_sae]`` rows where the feature is active.
        act_sum: ``[d_sae]`` sum of activations over all rows.
        image_act_sum: ``[d_sae]`` sum of activations over image rows only.
        image_fire_count: ``[d_sae]`` image rows where the feature is active.
        n_rows / n_image_rows: Rows scanned (total and image).
    """

    point: str
    sample_max: Tensor
    fire_count: Tensor
    act_sum: Tensor
    image_act_sum: Tensor
    image_fire_count: Tensor
    n_rows: int
    n_image_rows: int
    info: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return self.sample_max.shape[0]

    @property
    def d_sae(self) -> int:
        return self.sample_max.shape[1]

    # -- derived statistics -----------------------------------------------------------
    @property
    def frequency(self) -> Tensor:
        """Fraction of rows (tokens / patches) on which each feature fires."""
        return self.fire_count.double() / max(self.n_rows, 1)

    @property
    def sample_frequency(self) -> Tensor:
        """Fraction of samples in which each feature fires at least once."""
        return (self.sample_max > 0).double().mean(0)

    @property
    def mean_active(self) -> Tensor:
        """Mean activation over the rows where the feature fires (NaN if it never does)."""
        return self.act_sum.double() / self.fire_count.double()

    @property
    def image_fraction(self) -> Tensor:
        """Share of each feature's activation mass that falls on image rows."""
        return self.image_act_sum.double() / self.act_sum.double()

    @property
    def dead(self) -> Tensor:
        return self.fire_count == 0

    def top_samples(self, feature: int, n: int = 10) -> tuple[Tensor, Tensor]:
        """``(values, sample indices)`` of the ``n`` samples where ``feature`` peaks."""
        values, idx = self.sample_max[:, feature].float().topk(min(n, self.n_samples))
        keep = values > 0
        return values[keep], idx[keep]

    def inactive_samples(self, feature: int) -> Tensor:
        return (self.sample_max[:, feature] <= 0).nonzero(as_tuple=True)[0]

    # -- persistence ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {k: getattr(self, k) for k in _TENSORS}
        tensors["sample_max"] = tensors["sample_max"].half()
        save_file(
            {k: v.contiguous().cpu() for k, v in tensors.items()}, str(path / "scan.safetensors")
        )
        meta = {
            "point": self.point,
            "n_rows": self.n_rows,
            "n_image_rows": self.n_image_rows,
            "info": self.info,
        }
        (path / "scan.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureScan:
        path = Path(path)
        meta = json.loads((path / "scan.json").read_text())
        tensors = load_file(str(path / "scan.safetensors"))
        tensors["sample_max"] = tensors["sample_max"].float()
        return cls(**tensors, **meta)


@torch.no_grad()
def scan_features(
    store: ActivationStore | str | Path,
    sae: SAE,
    point: str | None = None,
    *,
    rows: RowFilter = "all",
    batch_rows: int = 8192,
) -> FeatureScan:
    """Encode every row of ``point`` with ``sae`` and aggregate per sample and feature.

    ``point`` defaults to the point recorded in ``sae.cfg.metadata`` by ``train_sae``.
    ``rows`` restricts the scan to image or text rows (match how the SAE was trained).
    """
    if not isinstance(store, ActivationStore):
        store = ActivationStore(store)
    point = point or sae.cfg.metadata.get("point")
    if point is None:
        raise ValueError("Pass point= (the SAE metadata does not record one)")
    if store.d(point) != sae.d_in:
        raise ValueError(f"SAE d_in={sae.d_in} but {point!r} has width {store.d(point)}")

    device = sae.W_enc.device
    n_samples = store.manifest["n_samples"]
    d_sae = sae.d_sae
    sample_max = torch.zeros(n_samples, d_sae, device=device)
    fire_count = torch.zeros(d_sae, dtype=torch.long, device=device)
    image_fire = torch.zeros(d_sae, dtype=torch.long, device=device)
    act_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)
    image_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)
    n_rows = n_image = 0

    for shard in store.iter_shards(point):
        if rows != "all":
            shard = shard.select(shard.is_image if rows == "image" else ~shard.is_image)
        for start in range(0, len(shard), batch_rows):
            block = shard.select(slice(start, start + batch_rows))
            latents = sae.encode(block.acts.to(device))
            sample_idx = block.sample_idx.to(device)
            is_image = block.is_image.to(device)
            index = sample_idx[:, None].expand_as(latents)
            sample_max.scatter_reduce_(0, index, latents, "amax", include_self=True)
            active = latents > 0
            fire_count += active.sum(0)
            act_sum += latents.sum(0, dtype=torch.float64)
            if is_image.any():
                image_fire += active[is_image].sum(0)
                image_sum += latents[is_image].sum(0, dtype=torch.float64)
            n_rows += len(block)
            n_image += int(is_image.sum())

    return FeatureScan(
        point=point,
        sample_max=sample_max.cpu(),
        fire_count=fire_count.cpu(),
        act_sum=act_sum.cpu(),
        image_act_sum=image_sum.cpu(),
        image_fire_count=image_fire.cpu(),
        n_rows=n_rows,
        n_image_rows=n_image,
        info={
            "store": str(store.path),
            "rows": rows,
            "sae": dict(sae.cfg.metadata),
            "d_sae": d_sae,
            **{k: store.info.get(k) for k in ("model", "adapter") if k in store.info},
        },
    )
