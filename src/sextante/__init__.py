"""sextante: mechanistic interpretability for vision-language models, in pure PyTorch."""

from .activations import (
    ActivationCache,
    ActivationExtractor,
    ActivationRows,
    ActivationStore,
    ActivationWriter,
    capture,
    extract,
    hooks,
    run_with_cache,
)
from .adapters import (
    ModelAdapter,
    ModelMap,
    detect_adapter,
    get_adapter,
    list_adapters,
    register_adapter,
    resolve,
)
from .sae import SAE, SAEConfig, TrainConfig, train_sae
from .utils import get_logger, seed_everything

__version__ = "0.1.0.dev0"

__all__ = [
    "ActivationCache",
    "ActivationExtractor",
    "ActivationRows",
    "ActivationStore",
    "ActivationWriter",
    "ModelAdapter",
    "ModelMap",
    "SAE",
    "SAEConfig",
    "TrainConfig",
    "__version__",
    "capture",
    "detect_adapter",
    "extract",
    "get_adapter",
    "get_logger",
    "hooks",
    "list_adapters",
    "register_adapter",
    "resolve",
    "run_with_cache",
    "seed_everything",
    "train_sae",
]
