"""Extract MedGemma activations for an image-report dataset (run on a GPU box).

Input: a JSONL file, one sample per line::

    {"id": "cxr-0001", "image": "images/cxr-0001.png", "report": "...", "lang": "es"}

Each sample is fed as a chat turn (image + instruction) followed by the reference report
as the assistant answer (teacher forcing), so the stored activations cover the image
tokens and every token of the report.

Usage::

    uv run python examples/extract_medgemma.py data/cxr.jsonl out/medgemma-cxr \
        --points projector llm.8 llm.16 llm.24 --batch-size 8

The default model is ``google/medgemma-4b-it``, which is gated: accept its terms on the
Hugging Face Hub and run ``hf auth login`` first.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import sextante

DEFAULT_PROMPT = "Describe the findings in this image."


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def batches(
    samples: list[dict], processor, root: Path, batch_size: int, prompt: str
) -> Iterator[dict]:
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        conversations = []
        for s in chunk:
            image = Image.open(root / s["image"]).convert("RGB")
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": s.get("prompt", prompt)},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": s["report"]}]},
                ]
            )
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs["meta"] = [
            {k: v for k, v in s.items() if k != "prompt"} | {"prompt": s.get("prompt", prompt)}
            for s in chunk
        ]
        yield dict(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data", type=Path, help="JSONL with id, image, report, lang")
    parser.add_argument("out", type=Path, help="output store directory")
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--points", nargs="+", default=["projector", "llm.17"])
    parser.add_argument("--tokens", default="all", choices=["all", "image", "text", "last"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sextante.seed_everything(0)
    sextante.get_logger()
    samples = read_jsonl(args.data)[: args.limit]
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(sextante.resolve(model))

    image_root = args.image_root or args.data.parent
    store = sextante.extract(
        model,
        batches(samples, processor, image_root, args.batch_size, args.prompt),
        args.out,
        args.points,
        tokens=args.tokens,
        overwrite=args.overwrite,
        info={"dataset": str(args.data), "prompt": args.prompt, "n_samples": len(samples)},
    )
    print(store)


if __name__ == "__main__":
    main()
