# sextante

Mechanistic interpretability for vision-language models, in pure PyTorch.

A sextant measures angles to tell you where you are. `sextante` measures a VLM's
internals (vision encoder, multimodal projector, language model) so you can see what it
represents and why it writes what it writes. It is built for medical VLMs (MedGemma,
LLaVA-Med) but works with any Hugging Face VLM once you describe it with a small adapter.

> Status: pre-alpha. `adapters` and `activations` work today; `sae`, `features`,
> `validation`, `intervene` and `transfer` are planned (see [docs/plan.md](docs/plan.md)).

## Install

```bash
uv add sextante        # once released; for now:
git clone <repo> && cd sextante && uv sync
```

## Quick start

```python
import sextante
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("google/medgemma-4b-it", device_map="cuda")

mm = sextante.resolve(model)  # auto-detects the "gemma3" adapter
print(mm.point_names()[:3], "...")  # ['vision.0', 'vision.1', 'vision.2'] ...

# One forward, cache two points. Stops after llm.12, so later layers are not computed.
_, cache = sextante.run_with_cache(model, inputs, ["projector", "llm.12"])

# Batched extraction to disk, with per-sample metadata
store = sextante.extract(model, batches, "out/cxr", ["projector", "llm.12"], tokens="all")
rows = store.load("llm.12")  # rows.acts [N, d], rows.sample_idx, rows.position, ...
store.samples[rows.sample_idx[0]]  # {"id": ..., "report": ..., "lang": "es"}
```

Each batch is the processor output plus an optional `meta` list (one dict per sample).
See [`examples/extract_medgemma.py`](examples/extract_medgemma.py) for a full script.

### Hook points

| Name | What it is |
| --- | --- |
| `vision.{i}` | Output of vision-encoder layer `i` |
| `projector` | Output of the multimodal projector (image tokens in LLM space) |
| `llm.{i}` | Residual stream after decoder layer `i` (negative indices allowed) |
| `module:{path}` | Any submodule, by its `named_modules()` path |

### Editing activations

`sextante.hooks` registers functions that can return a replacement tensor. This is the
basis for ablation and steering:

```python
with sextante.hooks(model, {"llm.12": lambda act, point: act + 4.0 * direction}):
    out = model.generate(**inputs)
```

### Adding an architecture

```python
sextante.register_adapter(
    sextante.ModelAdapter(
        name="myvlm",
        model_types=("myvlm",),  # config.model_type
        vision_layers_paths=("model.vision.blocks",),
        projector_paths=("model.connector",),
        llm_layers_paths=("model.decoder.layers",),
    )
)
```

Built-in adapters: `gemma3` (Gemma 3, MedGemma) and `llava` (LLaVA in HF format,
including LLaVA-Med conversions).

## Store format

```
out/cxr/
  manifest.json          model, adapter, per-point width, row counts and shards
  samples.jsonl          one metadata object per sample
  acts/llm.12/shard_00000.safetensors   acts, sample_idx, position, is_image, token_id
```

Shards are memory-mapped on load. If a run is interrupted, `manifest.json` keeps
`"complete": false` and every shard written so far stays readable.

## Development

```bash
uv sync
uv run pytest          # tiny random VLMs, runs on CPU in seconds
uv run ruff check . && uv run ruff format .
```

## License

MIT
