"""Forward hooks on named hook points, in pure PyTorch.

Two primitives:

* :func:`hooks` registers arbitrary functions on hook points for the duration of a
  ``with`` block. A function may return a tensor to *replace* the point's output, which
  is what steering and ablation build on.
* :func:`capture` records the outputs of a set of points into an :class:`ActivationCache`.

:func:`run_with_cache` combines ``capture`` with a forward pass and, by default, stops the
forward as soon as the deepest requested point has fired, so extracting layer 3 of a
34-layer model only pays for 4 layers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from typing import Any

import torch
from torch import Tensor, nn

from ..adapters import HookPoint, ModelMap, resolve

#: ``fn(activation, point) -> replacement or None``
HookFn = Callable[[Tensor, HookPoint], "Tensor | None"]


class StopForward(Exception):  # noqa: N818 - control flow, not an error
    """Raised inside a hook to abort the forward pass once nothing else is needed."""


def _primary(output: Any) -> tuple[Tensor, Callable[[Tensor], Any]]:
    """Split a module output into its main tensor and a function to rebuild it."""
    if isinstance(output, Tensor):
        return output, lambda t: t
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        rest = output[1:]
        return output[0], lambda t: (t, *rest)
    if isinstance(output, list) and output and isinstance(output[0], Tensor):
        rest = output[1:]
        return output[0], lambda t: [t, *rest]
    if isinstance(output, Mapping) and output:  # transformers ModelOutput
        key = next(iter(output.keys()))
        value = output[key]
        if isinstance(value, Tensor):

            def rebuild(t: Tensor) -> Any:
                output[key] = t
                return output

            return value, rebuild
    raise TypeError(f"Cannot find a tensor in module output of type {type(output).__name__}")


def _forward_hook(point: HookPoint, fn: HookFn):
    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        tensor, rebuild = _primary(output)
        replacement = fn(tensor, point)
        if replacement is None:
            return None
        if replacement.shape != tensor.shape:
            raise ValueError(
                f"Hook on {point.name!r} returned shape {tuple(replacement.shape)}, "
                f"expected {tuple(tensor.shape)}"
            )
        return rebuild(replacement)

    return hook


@contextmanager
def hooks(
    model: nn.Module | ModelMap, fns: Mapping[str, HookFn | Sequence[HookFn]]
) -> Iterator[ModelMap]:
    """Register ``fns`` (point name -> function or list of functions) during the block.

    Hooks are always removed on exit, even if the forward raises.
    """
    model_map = resolve(model)
    handles = []
    try:
        for name, fn_or_fns in fns.items():
            point = model_map.point(name)
            fn_list = [fn_or_fns] if callable(fn_or_fns) else list(fn_or_fns)
            for fn in fn_list:
                handles.append(point.module.register_forward_hook(_forward_hook(point, fn)))
        yield model_map
    finally:
        for handle in handles:
            handle.remove()


class ActivationCache(dict):
    """``{point name: tensor}``; with ``accumulate=True``, ``{point name: [tensor, ...]}``."""

    def stacked(self, name: str, dim: int = 1) -> Tensor:
        """Concatenate accumulated calls (e.g. generation steps) along ``dim``."""
        value = self[name]
        return torch.cat(value, dim=dim) if isinstance(value, list) else value


@contextmanager
def capture(
    model: nn.Module | ModelMap,
    points: Sequence[str],
    *,
    detach: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    accumulate: bool = False,
    stop_after_last: bool = False,
) -> Iterator[ActivationCache]:
    """Record the outputs of ``points`` into a cache during the block.

    Args:
        points: Hook point names (``"llm.12"``, ``"projector"``, ...).
        detach: Detach recorded tensors from the autograd graph.
        device: Move recorded tensors here (e.g. ``"cpu"`` to save GPU memory).
        dtype: Cast recorded tensors to this dtype.
        accumulate: Keep every call (useful during ``generate``) instead of the last one.
        stop_after_last: Raise :class:`StopForward` once the deepest point has fired.
            Only valid when every point has a known forward order (not ``module:``).
    """
    model_map = resolve(model)
    resolved = model_map.points(points)
    cache = ActivationCache()

    last_name = None
    if stop_after_last:
        if any(p.order is None for p in resolved):
            raise ValueError("stop_after_last is not supported with 'module:' hook points")
        last_name = max(resolved, key=lambda p: p.order).name  # type: ignore[arg-type, return-value]

    def record(tensor: Tensor, point: HookPoint) -> None:
        value = tensor.detach() if detach else tensor
        if device is not None or dtype is not None:
            value = value.to(device=device, dtype=dtype)
        if accumulate:
            cache.setdefault(point.name, []).append(value)
        else:
            cache[point.name] = value
        if point.name == last_name:
            raise StopForward

    with hooks(model_map, {p.name: record for p in resolved}):
        yield cache


def run_with_cache(
    model: nn.Module | ModelMap,
    inputs: Mapping[str, Any],
    points: Sequence[str],
    *,
    stop_early: bool = True,
    grad: bool = False,
    **capture_kwargs: Any,
) -> tuple[Any, ActivationCache]:
    """Run ``model(**inputs)`` and return ``(output, cache)``.

    With ``stop_early`` (default) the forward pass is aborted after the deepest requested
    point, and ``output`` is ``None``.
    """
    model_map = resolve(model)
    stop = stop_early and all(model_map.point(p).order is not None for p in points)
    output = None
    with (
        torch.set_grad_enabled(grad),
        capture(model_map, points, stop_after_last=stop, **capture_kwargs) as cache,
    ):
        with suppress(StopForward):
            output = model_map.model(**inputs)
    missing = [model_map.point(p).name for p in points if model_map.point(p).name not in cache]
    if missing:
        raise RuntimeError(
            f"Hook points {missing} never fired. Did the inputs include pixel_values for "
            "vision/projector points?"
        )
    return output, cache
