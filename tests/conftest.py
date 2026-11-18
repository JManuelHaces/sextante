"""Tiny, randomly initialised VLMs so the whole suite runs on a laptop CPU in seconds."""

from __future__ import annotations

import warnings

import pytest
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_TOKEN = 100
BOI, EOI, PAD = 101, 102, 0


@pytest.fixture(scope="session")
def tiny_gemma3():
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        SiglipVisionConfig,
    )

    torch.manual_seed(0)
    text = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=16,
        max_position_embeddings=128,
        pad_token_id=PAD,
    )
    vision = SiglipVisionConfig(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
    )
    config = Gemma3Config(
        text_config=text,
        vision_config=vision,
        mm_tokens_per_image=4,
        image_token_index=IMAGE_TOKEN,
        boi_token_index=BOI,
        eoi_token_index=EOI,
    )
    return Gemma3ForConditionalGeneration(config).eval()


@pytest.fixture(scope="session")
def tiny_llava():
    from transformers import (
        CLIPVisionConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        MistralConfig,
    )

    torch.manual_seed(0)
    config = LlavaConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            patch_size=8,
            projection_dim=16,
        ),
        text_config=MistralConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            pad_token_id=PAD,
        ),
        image_token_index=IMAGE_TOKEN,
        vision_feature_layer=-2,
    )
    return LlavaForConditionalGeneration(config).eval()


def gemma3_batch(texts: list[list[int]], *, n_image_tokens: int = 4, seed: int = 0) -> dict:
    """Left-padded batch: [BOS, BOI, <img>*n, EOI, *text]. One image per sample."""
    g = torch.Generator().manual_seed(seed)
    seqs = [[2, BOI, *([IMAGE_TOKEN] * n_image_tokens), EOI, *t] for t in texts]
    return _pad(seqs, pixel_values=torch.randn(len(seqs), 3, 32, 32, generator=g))


def llava_batch(texts: list[list[int]], *, n_image_tokens: int = 16, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    seqs = [[1, *([IMAGE_TOKEN] * n_image_tokens), *t] for t in texts]
    return _pad(seqs, pixel_values=torch.randn(len(seqs), 3, 32, 32, generator=g))


def _pad(seqs: list[list[int]], **extra) -> dict:
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(s)
        mask[i, width - len(s) :] = 1
    return {"input_ids": ids, "attention_mask": mask, **extra}
