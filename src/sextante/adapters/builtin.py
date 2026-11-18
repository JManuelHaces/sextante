"""Built-in adapters.

Paths are listed newest-``transformers``-layout first. The v5 layout nests everything
under ``model.``; the v4 layout kept ``vision_tower`` and ``language_model`` at the top.
"""

from __future__ import annotations

from .base import ModelAdapter
from .registry import register_adapter

#: Gemma 3 multimodal (``Gemma3ForConditionalGeneration``). Covers MedGemma 4B/27B
#: multimodal, which share the architecture (SigLIP encoder + Gemma 3 decoder).
GEMMA3 = register_adapter(
    ModelAdapter(
        name="gemma3",
        model_types=("gemma3",),
        vision_layers_paths=(
            "model.vision_tower.encoder.layers",
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ),
        projector_paths=("model.multi_modal_projector", "multi_modal_projector"),
        llm_layers_paths=(
            "model.language_model.layers",
            "language_model.model.layers",
            "language_model.layers",
        ),
    )
)

#: LLaVA in Hugging Face format (``LlavaForConditionalGeneration``): CLIP encoder,
#: MLP projector and a Llama/Mistral/Vicuna decoder. LLaVA-Med must be in HF format;
#: the original LLaVA-repo checkpoints (``llava_mistral``) need custom code and are not
#: supported by this adapter.
LLAVA = register_adapter(
    ModelAdapter(
        name="llava",
        model_types=("llava",),
        vision_layers_paths=(
            "model.vision_tower.encoder.layers",
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ),
        projector_paths=("model.multi_modal_projector", "multi_modal_projector"),
        llm_layers_paths=(
            "model.language_model.layers",
            "language_model.model.layers",
            "language_model.layers",
        ),
    )
)
