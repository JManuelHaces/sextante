"""Per-sample views of a feature: token activations over the report, patch grids."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..activations import ActivationStore
from ..sae import SAE

Tokenizer = Any  # anything with convert_ids_to_tokens(), or a callable id -> str

_SPACE_MARKERS = {"▁": " ", "Ġ": " ", "Ċ": "\n"}  # ▁ (SentencePiece), Ġ Ċ (BPE)


@dataclass
class SampleActivations:
    """Activations of one or more features over the rows of one sample.

    ``acts`` is ``[n_rows]`` for a single feature, ``[n_rows, n_features]`` otherwise.
    """

    sample: int
    features: list[int]
    acts: Tensor
    position: Tensor
    token_id: Tensor
    is_image: Tensor
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def max(self) -> float:
        return float(self.acts.max()) if self.acts.numel() else 0.0

    def for_feature(self, feature: int) -> Tensor:
        if self.acts.dim() == 1:
            return self.acts
        return self.acts[:, self.features.index(feature)]


@torch.no_grad()
def sample_activations(
    store: ActivationStore,
    sae: SAE,
    point: str,
    sample: int,
    features: int | Sequence[int],
) -> SampleActivations:
    """Encode the rows of ``sample`` and keep the columns of ``features``."""
    feats = [features] if isinstance(features, int) else list(features)
    rows = store.sample_rows(point, sample)
    latents = sae.encode(rows.acts.to(sae.W_enc.device)).cpu()
    acts = latents[:, feats]
    return SampleActivations(
        sample=sample,
        features=feats,
        acts=acts[:, 0] if isinstance(features, int) else acts,
        position=rows.position,
        token_id=rows.token_id,
        is_image=rows.is_image,
        meta=store.samples[sample] if sample < len(store.samples) else {},
    )


def _token_text(tokenizer: Tokenizer | None) -> Callable[[list[int]], list[str]]:
    if tokenizer is None:
        return lambda ids: [f"<{i}>" for i in ids]
    if hasattr(tokenizer, "convert_ids_to_tokens"):

        def convert(ids: list[int]) -> list[str]:
            pieces = tokenizer.convert_ids_to_tokens(ids)
            out = []
            for piece in pieces:
                for marker, repl in _SPACE_MARKERS.items():
                    piece = piece.replace(marker, repl)
                out.append(piece)
            return out

        return convert
    return lambda ids: [tokenizer(i) for i in ids]


def token_activations(
    sa: SampleActivations, tokenizer: Tokenizer | None = None, *, feature: int | None = None
) -> list[tuple[str, float]]:
    """``[(token text, activation), ...]`` over the text rows of the sample, in order."""
    acts = sa.for_feature(feature if feature is not None else sa.features[0])
    text = ~sa.is_image & (sa.token_id >= 0)
    order = sa.position[text].argsort()
    ids = sa.token_id[text][order].tolist()
    values = acts[text][order].tolist()
    return list(zip(_token_text(tokenizer)(ids), values, strict=True))


def _square_side(n: int) -> int | None:
    side = math.isqrt(n)
    return side if side * side == n else None


def patch_grid(sa: SampleActivations, *, feature: int | None = None, image: int = 0) -> Tensor:
    """Activations over the image rows of the sample as a square ``[h, w]`` grid.

    Image rows are split into images where ``position`` restarts at 0 (vision points) or
    by consecutive blocks of image tokens (LLM points). A leading CLS row (``n = 1 + s^2``)
    is dropped.
    """
    acts = sa.for_feature(feature if feature is not None else sa.features[0])
    img = sa.is_image
    if not img.any():
        raise ValueError("Sample has no image rows at this point")
    values, positions = acts[img], sa.position[img]
    if (sa.token_id[img] < 0).all():  # vision/projector point: positions are patch indices
        starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
    else:  # LLM point: a new image starts wherever image-token positions are not contiguous
        gaps = (positions[1:] - positions[:-1]) != 1
        starts = [0] + (gaps.nonzero(as_tuple=True)[0] + 1).tolist()
    if image >= len(starts):
        raise IndexError(f"Sample has {len(starts)} image(s), asked for image {image}")
    end = starts[image + 1] if image + 1 < len(starts) else len(values)
    block = values[starts[image] : end]
    side = _square_side(len(block))
    if side is None and _square_side(len(block) - 1) is not None:
        block, side = block[1:], _square_side(len(block) - 1)
    if side is None:
        raise ValueError(f"{len(block)} image rows do not form a square grid")
    return block.reshape(side, side)
