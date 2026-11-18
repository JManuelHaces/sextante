"""Adapter registry and auto-detection."""

from __future__ import annotations

from torch import nn

from .base import ModelAdapter, ModelMap

_REGISTRY: dict[str, ModelAdapter] = {}


def register_adapter(adapter: ModelAdapter, *, overwrite: bool = False) -> ModelAdapter:
    """Add an adapter to the registry so :func:`resolve` can detect it automatically."""
    if adapter.name in _REGISTRY and not overwrite:
        raise ValueError(f"Adapter {adapter.name!r} already registered (pass overwrite=True)")
    _REGISTRY[adapter.name] = adapter
    return adapter


def unregister_adapter(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_adapter(name: str) -> ModelAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown adapter {name!r}. Registered: {list_adapters()}") from None


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def detect_adapter(model: nn.Module) -> ModelAdapter:
    """Return the registered adapter that matches ``model``.

    Adapters registered later take precedence, so user adapters can override built-ins.
    """
    for adapter in reversed(list(_REGISTRY.values())):
        if adapter.matches(model):
            return adapter
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    raise ValueError(
        f"No adapter matches {type(model).__name__} (model_type={model_type!r}). "
        f"Registered adapters: {list_adapters()}. Write a ModelAdapter and pass it "
        "explicitly or call sextante.register_adapter()."
    )


def resolve(model: nn.Module | ModelMap, adapter: ModelAdapter | str | None = None) -> ModelMap:
    """Build a :class:`ModelMap` for ``model``, detecting the adapter if not given."""
    if isinstance(model, ModelMap):
        return model
    if isinstance(adapter, str):
        adapter = get_adapter(adapter)
    if adapter is None:
        adapter = detect_adapter(model)
    return adapter.resolve(model)
