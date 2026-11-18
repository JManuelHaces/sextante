"""Automatic interpretation of features with an LLM.

Two steps, following Bills et al. (2023) and the detection scoring of later SAE work:

1. :func:`explain` shows the LLM the top-activating examples of a feature (report text
   with the active tokens marked, plus chosen metadata) and asks for a short explanation.
2. :func:`detection_score` checks the explanation: the LLM sees a shuffled mix of
   activating and non-activating examples *without* activations and must pick the ones
   the explanation describes. Balanced accuracy near 0.5 means the explanation does not
   predict the feature.

The LLM is any ``Callable[[str], str]``, so any provider works. :func:`claude_llm` builds
one for the Claude API (``pip install 'sextante[claude]'``).

Scope: examples are textual (report tokens and metadata). For features that live on
image patches, put image-derived labels (findings, view) in ``meta_keys``; showing the
images themselves to a multimodal LLM is not implemented yet.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..activations import ActivationStore
from ..sae import SAE
from .examples import Tokenizer, sample_activations, token_activations
from .scan import FeatureScan

LLM = Callable[[str], str]


@dataclass
class Example:
    """One sample prepared for autointerp."""

    sample: int
    max_act: float
    image_max: float
    tokens: list[tuple[str, float]]
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self, *, marked: bool, scale: float) -> str:
        """The report; with ``marked``, active tokens become ``«token»(0-10)``."""
        parts = []
        for tok, act in self.tokens:
            level = round(10 * act / scale) if scale > 0 else 0
            if marked and level > 0:
                parts.append(
                    f"«{tok.strip() or tok}»({level})" + (" " if tok.endswith(" ") else "")
                )
                if tok.startswith(" "):
                    parts[-1] = " " + parts[-1]
            else:
                parts.append(tok)
        return "".join(parts).strip()


def build_examples(
    store: ActivationStore,
    sae: SAE,
    scan: FeatureScan,
    feature: int,
    samples: Sequence[int],
    *,
    tokenizer: Tokenizer | None = None,
    meta_keys: Sequence[str] = (),
) -> list[Example]:
    """Turn sample indices into :class:`Example` objects for ``feature``."""
    out = []
    for s in samples:
        sa = sample_activations(store, sae, scan.point, int(s), feature)
        image_acts = sa.acts[sa.is_image]
        out.append(
            Example(
                sample=int(s),
                max_act=sa.max,
                image_max=float(image_acts.max()) if image_acts.numel() else 0.0,
                tokens=token_activations(sa, tokenizer),
                meta={k: sa.meta.get(k) for k in meta_keys if k in sa.meta},
            )
        )
    return out


def _format_meta(meta: dict[str, Any]) -> str:
    return "; ".join(f"{k}={v}" for k, v in meta.items())


EXPLAIN_TEMPLATE = """\
You are interpreting one latent feature of a sparse autoencoder trained on the internal \
activations of a medical vision-language model that reads {modality} images and writes \
radiology-style reports.

Below are the samples where the feature is most active. In each report, tokens where the \
feature fires are marked as «token»(level), with level from 1 to 10 relative to the \
strongest activation seen. "image activation" is the feature's peak over the image \
patches, on the same 0-10 scale{meta_note}.

{examples}

What does this feature respond to? Consider clinical findings, anatomy, report phrasing \
or structure, language, and properties of the images or the dataset. If the pattern looks \
like an artifact of the corpus (a template phrase, a hospital-specific convention, a \
position in the text), say so.

Answer in {language} with one line that starts with "Explanation:" and contains a single \
concise phrase, no longer than 25 words."""


def explain_prompt(
    examples: Sequence[Example], *, modality: str = "medical", language: str = "English"
) -> str:
    scale = max((max(e.max_act, e.image_max) for e in examples), default=0.0)
    blocks = []
    for i, e in enumerate(examples, 1):
        image_level = round(10 * e.image_max / scale) if scale > 0 else 0
        lines = [f"Example {i} (image activation: {image_level})"]
        if e.meta:
            lines.append(f"Metadata: {_format_meta(e.meta)}")
        report = e.text(marked=True, scale=scale)
        lines.append(f"Report: {report}" if report else "Report: (no text tokens)")
        blocks.append("\n".join(lines))
    meta_note = (
        "; metadata lists known labels for the sample" if any(e.meta for e in examples) else ""
    )
    return EXPLAIN_TEMPLATE.format(
        modality=modality, language=language, meta_note=meta_note, examples="\n\n".join(blocks)
    )


def parse_explanation(response: str) -> str:
    match = re.search(r"Explanation:\s*(.+)", response)
    return (match.group(1) if match else response).strip().splitlines()[0].strip()


def explain(
    examples: Sequence[Example],
    llm: LLM,
    *,
    modality: str = "medical",
    language: str = "English",
) -> str:
    """Ask ``llm`` for a one-phrase explanation of the feature behind ``examples``."""
    if not examples:
        raise ValueError("Need at least one activating example")
    return parse_explanation(llm(explain_prompt(examples, modality=modality, language=language)))


DETECT_TEMPLATE = """\
A latent feature of a medical vision-language model has this explanation:

"{explanation}"

Below are {n} samples. For each one, decide whether the feature described above would be \
active on it. Some are active and some are not.

{examples}

Reply with a JSON array of the numbers of the samples where the feature is active, for \
example [1, 4]. Reply with [] if none are."""


def detection_prompt(explanation: str, examples: Sequence[Example]) -> str:
    blocks = []
    for i, e in enumerate(examples, 1):
        lines = [f"Sample {i}"]
        if e.meta:
            lines.append(f"Metadata: {_format_meta(e.meta)}")
        report = e.text(marked=False, scale=1.0)
        lines.append(f"Report: {report}" if report else "Report: (no text tokens)")
        blocks.append("\n".join(lines))
    return DETECT_TEMPLATE.format(
        explanation=explanation, n=len(examples), examples="\n\n".join(blocks)
    )


def parse_indices(response: str, n: int) -> set[int]:
    """1-based indices from the first JSON array in ``response`` (invalid ones dropped)."""
    match = re.search(r"\[[^\[\]]*\]", response)
    if not match:
        return set()
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    return {int(v) for v in values if isinstance(v, (int, float)) and 1 <= int(v) <= n}


def detection_score(
    explanation: str,
    positives: Sequence[Example],
    negatives: Sequence[Example],
    llm: LLM,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Balanced accuracy of ``llm`` picking ``positives`` out of a shuffled mix."""
    if not positives or not negatives:
        raise ValueError("Need positive and negative examples")
    mixed = [(e, True) for e in positives] + [(e, False) for e in negatives]
    random.Random(seed).shuffle(mixed)
    picked = parse_indices(llm(detection_prompt(explanation, [e for e, _ in mixed])), len(mixed))
    predictions = [(i + 1) in picked for i in range(len(mixed))]
    truth = [label for _, label in mixed]
    tp = sum(p and t for p, t in zip(predictions, truth, strict=True))
    tn = sum((not p) and (not t) for p, t in zip(predictions, truth, strict=True))
    tpr, tnr = tp / len(positives), tn / len(negatives)
    return {
        "balanced_accuracy": (tpr + tnr) / 2,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "samples": [e.sample for e, _ in mixed],
        "truth": truth,
        "predictions": predictions,
    }


def interpret_feature(
    store: ActivationStore,
    sae: SAE,
    scan: FeatureScan,
    feature: int,
    llm: LLM,
    *,
    tokenizer: Tokenizer | None = None,
    meta_keys: Sequence[str] = (),
    n_explain: int = 10,
    n_score: int = 5,
    modality: str = "medical",
    language: str = "English",
    seed: int = 0,
) -> dict[str, Any]:
    """Explain ``feature`` from its top samples and score it on held-out ones.

    The top ``n_explain`` samples are used to write the explanation; the next
    ``n_score`` activating samples and ``n_score`` random non-activating samples score it.
    """
    _, top = scan.top_samples(feature, n_explain + n_score)
    top = top.tolist()
    if not top:
        return {"feature": feature, "explanation": None, "reason": "never fires"}
    explain_ids, score_ids = top[:n_explain], top[n_explain:]
    kwargs = {"tokenizer": tokenizer, "meta_keys": meta_keys}
    explanation = explain(
        build_examples(store, sae, scan, feature, explain_ids, **kwargs),
        llm,
        modality=modality,
        language=language,
    )
    result: dict[str, Any] = {
        "feature": feature,
        "explanation": explanation,
        "explain_samples": explain_ids,
    }
    inactive = scan.inactive_samples(feature).tolist()
    if score_ids and inactive:
        rng = random.Random(seed)
        negative_ids = rng.sample(inactive, min(len(score_ids), len(inactive)))
        result["detection"] = detection_score(
            explanation,
            build_examples(store, sae, scan, feature, score_ids, **kwargs),
            build_examples(store, sae, scan, feature, negative_ids, **kwargs),
            llm,
            seed=seed,
        )
    return result


def claude_llm(
    model: str = "claude-opus-5-5",
    *,
    effort: str = "medium",
    max_tokens: int = 16000,
    client: Any = None,
) -> LLM:
    """An :data:`LLM` backed by the Claude API (needs ``pip install 'sextante[claude]'``).

    Credentials come from the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login``
    profile). Server-side refusal fallback is enabled, so a request declined by the safety
    classifiers is retried on a fallback model inside the same call. ``effort`` trades
    depth for cost; ``"low"`` is usually enough for explanations.
    """
    if client is None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("claude_llm needs: pip install 'sextante[claude]'") from exc
        client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request (request id {response._request_id})")
        return "".join(block.text for block in response.content if block.type == "text")

    return call
