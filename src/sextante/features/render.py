"""Rendering helpers: heatmaps over images and highlighted report tokens (HTML)."""

from __future__ import annotations

import html
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# viridis, 5 stops
_STOPS = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float32
)


def colormap(values: np.ndarray) -> np.ndarray:
    """Map values in ``[0, 1]`` to RGB uint8 with a 5-stop viridis ramp."""
    v = np.clip(values, 0, 1) * (len(_STOPS) - 1)
    lo = np.floor(v).astype(int).clip(0, len(_STOPS) - 2)
    t = (v - lo)[..., None]
    return ((1 - t) * _STOPS[lo] + t * _STOPS[lo + 1]).astype(np.uint8)


def heatmap_overlay(
    image: Image.Image,
    grid: Tensor | np.ndarray,
    *,
    alpha: float = 0.6,
    vmax: float | None = None,
) -> Image.Image:
    """Blend a patch grid over ``image``; zero activation leaves the image untouched.

    The grid is stretched over the whole image, which matches processors that resize
    the image to a square (Gemma 3 / MedGemma, LLaVA). ``vmax`` fixes the colour scale
    (use the feature's max over its top examples so images are comparable).
    """
    g = grid.float().cpu().numpy() if isinstance(grid, Tensor) else np.asarray(grid, np.float32)
    scale = vmax if vmax is not None else float(g.max())
    norm = g / scale if scale > 0 else np.zeros_like(g)
    base = image.convert("RGB")
    up = Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8)).resize(
        base.size, Image.Resampling.BILINEAR
    )
    v = np.asarray(up, dtype=np.float32) / 255.0
    colour = colormap(v).astype(np.float32)
    weight = (alpha * v)[..., None]
    out = (1 - weight) * np.asarray(base, dtype=np.float32) + weight * colour
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def tokens_html(
    tokens: Sequence[tuple[str, float]], *, vmax: float | None = None, color: str = "255,140,0"
) -> str:
    """Report tokens as HTML spans whose background opacity follows the activation."""
    values = [v for _, v in tokens]
    scale = vmax if vmax is not None else max(values, default=0.0)
    spans = []
    for text, value in tokens:
        a = 0.0 if scale <= 0 else max(0.0, min(1.0, value / scale))
        spans.append(
            f'<span title="{value:.3g}" style="background: rgba({color},{a:.2f})">'
            f"{html.escape(text).replace(chr(10), '<br>')}</span>"
        )
    return '<span style="white-space: pre-wrap">' + "".join(spans) + "</span>"


def grid_to_image(grid: Tensor, *, vmax: float | None = None, size: int = 224) -> Image.Image:
    """A standalone coloured heatmap (no background image)."""
    g = grid.float().cpu()
    scale = vmax if vmax is not None else float(g.max())
    norm = (g / scale).clamp(0, 1).numpy() if scale > 0 else torch.zeros_like(g).numpy()
    return Image.fromarray(colormap(norm)).resize((size, size), Image.Resampling.NEAREST)
