"""Feature analysis: top examples, heatmaps, report tokens, corpus statistics, autointerp."""

from .autointerp import (
    Example,
    build_examples,
    claude_llm,
    detection_score,
    explain,
    explain_prompt,
    interpret_feature,
)
from .examples import SampleActivations, patch_grid, sample_activations, token_activations
from .render import grid_to_image, heatmap_overlay, tokens_html
from .scan import FeatureScan, scan_features
from .stats import Association, auroc, correlation_ratio, feature_table, metadata_association

__all__ = [
    "Association",
    "Example",
    "FeatureScan",
    "SampleActivations",
    "auroc",
    "build_examples",
    "claude_llm",
    "correlation_ratio",
    "detection_score",
    "explain",
    "explain_prompt",
    "feature_table",
    "grid_to_image",
    "heatmap_overlay",
    "interpret_feature",
    "metadata_association",
    "patch_grid",
    "sample_activations",
    "scan_features",
    "token_activations",
    "tokens_html",
]
