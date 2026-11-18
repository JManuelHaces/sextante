"""Architecture adapters: where the vision tower, projector and LLM layers live."""

from .base import HookPoint, ModelAdapter, ModelMap, get_submodule
from .builtin import GEMMA3, LLAVA
from .registry import (
    detect_adapter,
    get_adapter,
    list_adapters,
    register_adapter,
    resolve,
    unregister_adapter,
)

__all__ = [
    "GEMMA3",
    "LLAVA",
    "HookPoint",
    "ModelAdapter",
    "ModelMap",
    "detect_adapter",
    "get_adapter",
    "get_submodule",
    "list_adapters",
    "register_adapter",
    "resolve",
    "unregister_adapter",
]
