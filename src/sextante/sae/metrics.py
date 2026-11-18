"""SAE evaluation: reconstruction quality, sparsity, dead latents and spliced model loss."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..activations import hooks
from ..activations.extract import _model_device_dtype, _prepare_inputs
from ..adapters import ModelMap, resolve
from .model import SAE

SplicePositions = Literal["all", "image", "text"]


@torch.no_grad()
def evaluate(
    sae: SAE, batches: Iterable[Tensor], *, max_batches: int | None = None
) -> dict[str, float]:
    """Reconstruction metrics over raw activation batches (``[n, d_in]`` tensors).

    Returns mean squared error (raw units), explained variance, cosine similarity
    between ``x`` and ``x_hat``, mean L0, and the fraction of latents that never fired.
    """
    device = sae.W_enc.device
    sse = total_sq = cos_sum = l0_sum = 0.0
    n = 0
    sum_x = sum_x2 = None
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    for i, x in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, torch.float32)
        out = sae(x)
        sse += (out.x_hat - x).pow(2).sum().item()
        cos_sum += F.cosine_similarity(out.x_hat, x, dim=-1).sum().item()
        l0_sum += (out.latents > 0).sum().item()
        fired |= (out.latents > 0).any(0)
        sum_x = x.sum(0) if sum_x is None else sum_x + x.sum(0)
        sum_x2 = x.pow(2).sum(0) if sum_x2 is None else sum_x2 + x.pow(2).sum(0)
        n += x.shape[0]
    if n == 0:
        raise ValueError("No batches to evaluate")
    total_sq = (sum_x2 - sum_x.pow(2) / n).sum().item()  # n * total variance
    return {
        "mse": sse / (n * sae.d_in),
        "explained_variance": 1 - sse / max(total_sq, 1e-12),
        "cosine": cos_sum / n,
        "l0": l0_sum / n,
        "dead_fraction": 1 - fired.float().mean().item(),
        "n_rows": n,
    }


def _next_token_loss(
    logits: Tensor, input_ids: Tensor, attention_mask: Tensor | None, ignore_id: int | None
) -> tuple[Tensor, Tensor]:
    """Summed next-token cross-entropy and token count, skipping padding and image tokens."""
    logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    valid = torch.ones_like(targets, dtype=torch.bool)
    if attention_mask is not None:
        mask = attention_mask.bool()
        valid &= mask[:, 1:] & mask[:, :-1]
    if ignore_id is not None:
        valid &= targets != ignore_id
    loss = F.cross_entropy(logits[valid], targets[valid], reduction="sum")
    return loss, valid.sum()


def _splice_fn(sae: SAE, select: Tensor | None, *, zero: bool):
    """Hook that swaps the activation for its SAE reconstruction (or zeros)."""

    def fn(act: Tensor, _point) -> Tensor:
        replacement = torch.zeros_like(act) if zero else sae.reconstruct(act)
        if select is None or act.shape[:2] != select.shape:
            return replacement
        return torch.where(select.unsqueeze(-1), replacement, act)

    return fn


@torch.no_grad()
def spliced_loss(
    model: nn.Module | ModelMap,
    sae: SAE,
    point: str,
    batches: Iterable[Mapping[str, Any]],
    *,
    positions: SplicePositions = "all",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Language-model loss when ``point`` is replaced by the SAE reconstruction.

    Runs every batch three times: clean, with the SAE spliced in, and with the point
    zero-ablated. ``loss_recovered = (zero - sae) / (zero - clean)``: 1 means the SAE
    preserves everything the model needs, 0 means it is as bad as deleting the layer.

    The loss is next-token cross-entropy over text tokens (padding and image placeholder
    tokens are excluded as targets). ``positions`` restricts the splice on ``llm.*``
    points to image or text tokens, to match an SAE trained only on those rows.
    """
    mm = resolve(model)
    hp = mm.point(point)
    device, dtype = _model_device_dtype(mm.model)
    image_id = mm.image_token_id
    totals = {"clean": 0.0, "sae": 0.0, "zero": 0.0}
    count = 0

    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        inputs = _prepare_inputs(batch, device, dtype)
        input_ids = inputs["input_ids"]
        select = None
        if hp.kind in ("llm", "module") and positions != "all":
            if image_id is None:
                raise ValueError("positions='image'/'text' needs an image token id")
            is_image = input_ids == image_id
            select = is_image if positions == "image" else ~is_image

        losses = {}
        for mode in ("clean", "sae", "zero"):
            if mode == "clean":
                out = mm.model(**inputs)
            else:
                fn = _splice_fn(sae, select, zero=mode == "zero")
                with hooks(mm, {hp.name: fn}):
                    out = mm.model(**inputs)
            losses[mode], n = _next_token_loss(
                out.logits, input_ids, inputs.get("attention_mask"), image_id
            )
        clean, with_sae, zero = losses["clean"], losses["sae"], losses["zero"]
        totals["clean"] += clean.item()
        totals["sae"] += with_sae.item()
        totals["zero"] += zero.item()
        count += int(n)

    if count == 0:
        raise ValueError("No target tokens to score")
    ce = {k: v / count for k, v in totals.items()}
    denom = ce["zero"] - ce["clean"]
    return {
        "ce_clean": ce["clean"],
        "ce_sae": ce["sae"],
        "ce_zero": ce["zero"],
        "ce_delta": ce["sae"] - ce["clean"],
        "loss_recovered": (ce["zero"] - ce["sae"]) / denom if abs(denom) > 1e-12 else float("nan"),
        "n_tokens": count,
    }
