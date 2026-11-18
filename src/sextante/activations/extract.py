"""Batched activation extraction.

A batch is a mapping with the model inputs (``input_ids``, ``attention_mask``,
``pixel_values``, ...) plus two optional keys that sextante consumes itself:

``meta``                list of per-sample dicts (image id, report, language, ...), stored
                        in ``samples.jsonl``
``image_sample_index``  ``[n_images]`` tensor mapping each image to its sample. Only needed
                        when a sample has more than one image (or none).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ..adapters import ModelAdapter, ModelMap, resolve
from .hooks import run_with_cache
from .store import ActivationRows, ActivationStore, ActivationWriter

logger = logging.getLogger(__name__)

TokenSelection = Literal["all", "image", "text", "last"]
_RESERVED_KEYS = ("meta", "image_sample_index")


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters())
    return param.device, param.dtype


def _prepare_inputs(
    batch: Mapping[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    inputs = {}
    for key, value in batch.items():
        if key in _RESERVED_KEYS:
            continue
        if isinstance(value, Tensor):
            value = value.to(device, dtype=dtype if value.is_floating_point() else None)
        inputs[key] = value
    return inputs


def _batch_size(batch: Mapping[str, Any]) -> int:
    for key in ("input_ids", "inputs_embeds", "attention_mask"):
        if isinstance(batch.get(key), Tensor):
            return batch[key].shape[0]
    raise ValueError("Batch must contain input_ids, inputs_embeds or attention_mask")


class ActivationExtractor:
    """Turns batches into per-point :class:`ActivationRows`.

    Args:
        model: A Hugging Face VLM (or a :class:`ModelMap`).
        points: Hook point names to extract, e.g. ``["projector", "llm.12"]``.
        adapter: Adapter or adapter name; auto-detected if omitted.
        tokens: Which sequence positions to keep for ``llm.*`` and ``module:*`` points:
            ``"all"`` non-padding tokens, only ``"image"`` tokens, only ``"text"`` tokens,
            or the ``"last"`` non-padding token. Vision and projector points always keep
            every patch.
        stop_early: Abort the forward pass after the deepest requested point.
    """

    def __init__(
        self,
        model: nn.Module | ModelMap,
        points: Sequence[str],
        *,
        adapter: ModelAdapter | str | None = None,
        tokens: TokenSelection = "all",
        stop_early: bool = True,
    ) -> None:
        if tokens not in ("all", "image", "text", "last"):
            raise ValueError(f"Invalid tokens={tokens!r}")
        self.model_map = resolve(model, adapter)
        self.hook_points = self.model_map.points(points)
        self.points = [p.name for p in self.hook_points]
        self.tokens = tokens
        self.stop_early = stop_early
        self.image_token_id = self.model_map.image_token_id

    @torch.inference_mode()
    def extract_batch(
        self, batch: Mapping[str, Any], sample_offset: int = 0
    ) -> dict[str, ActivationRows]:
        """Run one batch and return ``{point: rows}`` with global sample indices."""
        model = self.model_map.model
        device, dtype = _model_device_dtype(model)
        inputs = _prepare_inputs(batch, device, dtype)
        n_samples = _batch_size(inputs)

        _, cache = run_with_cache(self.model_map, inputs, self.points, stop_early=self.stop_early)

        seq = self._sequence_info(inputs, n_samples, device)
        out: dict[str, ActivationRows] = {}
        for point in self.hook_points:
            act = cache[point.name]
            if point.kind in ("vision", "projector"):
                rows = self._patch_rows(act, batch.get("image_sample_index"), n_samples)
            else:
                rows = self._token_rows(act, *seq)
            rows.sample_idx += sample_offset
            out[point.name] = rows
        return out

    # -- row builders -----------------------------------------------------------------
    def _sequence_info(
        self, inputs: Mapping[str, Any], n_samples: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        input_ids = inputs.get("input_ids")
        mask = inputs.get("attention_mask")
        if mask is None:
            seq_len = (input_ids if input_ids is not None else inputs["inputs_embeds"]).shape[1]
            mask = torch.ones(n_samples, seq_len, dtype=torch.bool, device=device)
        mask = mask.bool()
        if input_ids is not None and self.image_token_id is not None:
            is_image = input_ids == self.image_token_id
        else:
            is_image = torch.zeros_like(mask)
        token_ids = (
            input_ids if input_ids is not None else torch.full_like(mask, -1, dtype=torch.long)
        )
        return mask, is_image, token_ids

    def _token_rows(
        self, act: Tensor, mask: Tensor, is_image: Tensor, token_ids: Tensor
    ) -> ActivationRows:
        if act.dim() != 3 or act.shape[:2] != mask.shape:
            raise ValueError(
                f"Expected token activations of shape {tuple(mask.shape)} + (d,), got "
                f"{tuple(act.shape)}"
            )
        position = mask.long().cumsum(dim=1) - 1  # position among non-padding tokens
        if self.tokens == "all":
            select = mask
        elif self.tokens == "image":
            select = mask & is_image
        elif self.tokens == "text":
            select = mask & ~is_image
        else:  # last non-padding token
            last = position.masked_fill(~mask, -1).argmax(dim=1)
            select = torch.zeros_like(mask)
            select[torch.arange(mask.shape[0], device=mask.device), last] = True
        b_idx, t_idx = select.nonzero(as_tuple=True)
        return ActivationRows(
            acts=act[b_idx, t_idx],
            sample_idx=b_idx.cpu(),
            position=position[b_idx, t_idx].cpu(),
            is_image=is_image[b_idx, t_idx].cpu(),
            token_id=token_ids[b_idx, t_idx].cpu(),
        )

    @staticmethod
    def _patch_rows(
        act: Tensor, image_sample_index: Tensor | None, n_samples: int
    ) -> ActivationRows:
        if act.dim() == 2:  # [n_patches_total, d] -> single image
            act = act.unsqueeze(0)
        if act.dim() != 3:
            raise ValueError(
                f"Expected patch activations [n_images, n_patches, d], got {tuple(act.shape)}"
            )
        n_images, n_patches, _ = act.shape
        if image_sample_index is None:
            if n_images != n_samples:
                raise ValueError(
                    f"Got {n_images} images for {n_samples} samples; pass "
                    "'image_sample_index' in the batch to map images to samples."
                )
            image_sample_index = torch.arange(n_samples)
        image_sample_index = torch.as_tensor(image_sample_index, dtype=torch.long).cpu()
        if image_sample_index.shape != (n_images,):
            raise ValueError(f"image_sample_index must have shape ({n_images},)")
        n = n_images * n_patches
        return ActivationRows(
            acts=act.reshape(n, act.shape[-1]),
            sample_idx=image_sample_index.repeat_interleave(n_patches),
            position=torch.arange(n_patches).repeat(n_images),
            is_image=torch.ones(n, dtype=torch.bool),
            token_id=torch.full((n,), -1, dtype=torch.long),
        )


def extract(
    model: nn.Module | ModelMap,
    batches: Iterable[Mapping[str, Any]],
    out_dir: str | Path,
    points: Sequence[str],
    *,
    adapter: ModelAdapter | str | None = None,
    tokens: TokenSelection = "all",
    dtype: torch.dtype = torch.float16,
    shard_rows: int = 65_536,
    overwrite: bool = False,
    stop_early: bool = True,
    info: Mapping[str, Any] | None = None,
    log_every: int = 50,
) -> ActivationStore:
    """Extract ``points`` for every batch and write them to an :class:`ActivationStore`.

    ``info`` is free-form run metadata saved in the manifest (dataset name, split,
    prompt template, ...). Model and adapter names are added automatically.
    """
    extractor = ActivationExtractor(
        model, points, adapter=adapter, tokens=tokens, stop_early=stop_early
    )
    mm = extractor.model_map
    config = getattr(mm.model, "config", None)
    run_info = {
        "model": getattr(config, "_name_or_path", None) or type(mm.model).__name__,
        "model_class": type(mm.model).__name__,
        "adapter": mm.adapter.name,
        "tokens": tokens,
        "image_token_id": extractor.image_token_id,
        **dict(info or {}),
    }
    t0 = time.perf_counter()
    with ActivationWriter(
        out_dir, dtype=dtype, shard_rows=shard_rows, overwrite=overwrite, info=run_info
    ) as writer:
        for i, batch in enumerate(batches):
            n = _batch_size(batch)
            metas = batch.get("meta") or [{} for _ in range(n)]
            if len(metas) != n:
                raise ValueError(f"Batch {i}: {len(metas)} meta entries for {n} samples")
            offset = writer.add_samples(metas)
            rows_by_point = extractor.extract_batch(batch, offset)
            for hp in extractor.hook_points:
                writer.add(hp.name, rows_by_point[hp.name], kind=hp.kind, layer=hp.layer)
            if log_every and (i + 1) % log_every == 0:
                logger.info(
                    "batch %d, %d samples, %.1fs", i + 1, offset + n, time.perf_counter() - t0
                )
    return ActivationStore(out_dir)
