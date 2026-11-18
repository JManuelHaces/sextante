"""Architecture adapters.

An adapter is a small, declarative description of *where* the interesting parts of a
vision-language model live: the vision encoder layers, the multimodal projector and the
decoder layers of the language model. Everything else in sextante (hooks, extraction,
SAEs, interventions) talks to models only through the :class:`ModelMap` an adapter
produces, so adding a new architecture means writing one adapter and nothing else.

Hook point names
----------------
``vision.{i}``   output of vision-encoder layer ``i`` (residual stream of the encoder)
``projector``    output of the multimodal projector (image tokens in LLM space)
``llm.{i}``      output of decoder layer ``i`` (residual stream after the layer)
``module:{path}`` any submodule by its ``named_modules`` path (escape hatch)

Layer indices accept negative values (``llm.-1`` is the last decoder layer).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from torch import nn

PointKind = Literal["vision", "projector", "llm", "module"]


def get_submodule(root: nn.Module, path: str) -> nn.Module | None:
    """Return ``root.<path>`` or ``None`` if any attribute along the path is missing."""
    if path == "":
        return root
    module: object = root
    for part in path.split("."):
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and part.lstrip("-").isdigit():
            idx = int(part)
            if not -len(module) <= idx < len(module):
                return None
            module = module[idx]
        else:
            module = getattr(module, part, None)
        if module is None:
            return None
    return module if isinstance(module, nn.Module) else None


def _first_match(root: nn.Module, candidates: Sequence[str]) -> tuple[str, nn.Module] | None:
    for path in candidates:
        found = get_submodule(root, path)
        if found is not None:
            return path, found
    return None


@dataclass(frozen=True)
class HookPoint:
    """A resolved hook point: its canonical name, kind, layer index and module."""

    name: str
    kind: PointKind
    module: nn.Module
    path: str
    layer: int | None = None

    @property
    def order(self) -> tuple[int, int] | None:
        """Position in the forward pass, used for early stopping. ``None`` if unknown."""
        rank = {"vision": 0, "projector": 1, "llm": 2}.get(self.kind)
        if rank is None:
            return None
        return (rank, self.layer if self.layer is not None else 0)


@dataclass(frozen=True)
class ModelAdapter:
    """Declarative description of a VLM architecture.

    Each ``*_paths`` field lists candidate attribute paths, tried in order; the first one
    that exists on the model wins. Listing several candidates lets one adapter cover the
    module layouts of different ``transformers`` versions.

    Args:
        name: Registry key, e.g. ``"gemma3"``.
        model_types: ``config.model_type`` values this adapter handles (used for detection).
        vision_layers_paths: Paths to the ``ModuleList`` of vision-encoder layers.
        projector_paths: Paths to the multimodal projector module.
        llm_layers_paths: Paths to the ``ModuleList`` of language-model decoder layers.
        image_token_attrs: Config attributes holding the id of the image placeholder
            token in ``input_ids``.
        class_names: Optional model class names that also trigger detection.
        matcher: Optional custom predicate ``model -> bool`` for detection.
    """

    name: str
    model_types: tuple[str, ...] = ()
    vision_layers_paths: tuple[str, ...] = ()
    projector_paths: tuple[str, ...] = ()
    llm_layers_paths: tuple[str, ...] = ()
    image_token_attrs: tuple[str, ...] = ("image_token_id", "image_token_index")
    class_names: tuple[str, ...] = ()
    matcher: Callable[[nn.Module], bool] | None = field(default=None, compare=False)

    def matches(self, model: nn.Module) -> bool:
        if self.matcher is not None and self.matcher(model):
            return True
        if type(model).__name__ in self.class_names:
            return True
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        return model_type in self.model_types

    def resolve(self, model: nn.Module) -> ModelMap:
        """Locate the components on ``model`` and return a :class:`ModelMap`."""
        missing = []
        resolved: dict[str, tuple[str, nn.Module] | None] = {}
        for key, candidates in (
            ("vision_layers", self.vision_layers_paths),
            ("projector", self.projector_paths),
            ("llm_layers", self.llm_layers_paths),
        ):
            match = _first_match(model, candidates) if candidates else None
            if candidates and match is None:
                missing.append(f"{key} (tried: {', '.join(candidates)})")
            resolved[key] = match
        if resolved["llm_layers"] is None:
            raise ValueError(
                f"Adapter {self.name!r} could not locate required components on "
                f"{type(model).__name__}: {'; '.join(missing)}"
            )
        return ModelMap(model=model, adapter=self, **resolved)

    def image_token_id(self, model: nn.Module) -> int | None:
        config = getattr(model, "config", None)
        for attr in self.image_token_attrs:
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value
        return None


class ModelMap:
    """A model together with the locations of its components, as found by an adapter."""

    def __init__(
        self,
        model: nn.Module,
        adapter: ModelAdapter,
        vision_layers: tuple[str, nn.Module] | None,
        projector: tuple[str, nn.Module] | None,
        llm_layers: tuple[str, nn.Module] | None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self._vision = vision_layers
        self._projector = projector
        self._llm = llm_layers

    # -- component access -------------------------------------------------------------
    @property
    def vision_layers(self) -> nn.ModuleList | None:
        return self._vision[1] if self._vision else None  # type: ignore[return-value]

    @property
    def projector(self) -> nn.Module | None:
        return self._projector[1] if self._projector else None

    @property
    def llm_layers(self) -> nn.ModuleList:
        assert self._llm is not None
        return self._llm[1]  # type: ignore[return-value]

    @property
    def n_vision_layers(self) -> int:
        return len(self.vision_layers) if self.vision_layers is not None else 0

    @property
    def n_llm_layers(self) -> int:
        return len(self.llm_layers)

    @property
    def image_token_id(self) -> int | None:
        return self.adapter.image_token_id(self.model)

    # -- hook points ------------------------------------------------------------------
    def point_names(self) -> list[str]:
        """All canonical hook point names available on this model, in forward order."""
        names = [f"vision.{i}" for i in range(self.n_vision_layers)]
        if self.projector is not None:
            names.append("projector")
        names += [f"llm.{i}" for i in range(self.n_llm_layers)]
        return names

    def point(self, name: str) -> HookPoint:
        """Resolve a hook point name (see module docstring) to a :class:`HookPoint`."""
        if name.startswith("module:"):
            path = name[len("module:") :]
            module = get_submodule(self.model, path)
            if module is None:
                raise KeyError(f"No submodule {path!r} on {type(self.model).__name__}")
            return HookPoint(name=name, kind="module", module=module, path=path)

        if name == "projector":
            if self._projector is None:
                raise KeyError(f"Adapter {self.adapter.name!r} defines no projector")
            path, module = self._projector
            return HookPoint(name="projector", kind="projector", module=module, path=path)

        family, _, idx_str = name.partition(".")
        if family not in ("vision", "llm") or not idx_str.lstrip("-").isdigit():
            raise KeyError(
                f"Invalid hook point {name!r}. Use 'vision.<i>', 'projector', 'llm.<i>' "
                "or 'module:<path>'."
            )
        located = self._vision if family == "vision" else self._llm
        if located is None:
            raise KeyError(f"Adapter {self.adapter.name!r} defines no {family} layers")
        base_path, layers = located
        n = len(layers)  # type: ignore[arg-type]
        idx = int(idx_str)
        if not -n <= idx < n:
            raise KeyError(f"{name!r} out of range: model has {n} {family} layers")
        idx %= n
        return HookPoint(
            name=f"{family}.{idx}",
            kind=family,  # type: ignore[arg-type]
            module=layers[idx],  # type: ignore[index]
            path=f"{base_path}.{idx}",
            layer=idx,
        )

    def points(self, names: Sequence[str]) -> list[HookPoint]:
        resolved = [self.point(n) for n in names]
        seen: set[str] = set()
        for p in resolved:
            if p.name in seen:
                raise ValueError(f"Hook point {p.name!r} requested twice")
            seen.add(p.name)
        return resolved

    def __repr__(self) -> str:
        return (
            f"ModelMap(adapter={self.adapter.name!r}, vision_layers={self.n_vision_layers}, "
            f"projector={self.projector is not None}, llm_layers={self.n_llm_layers})"
        )
