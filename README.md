# sextante

Mechanistic interpretability for vision-language models, in pure PyTorch.

A sextant measures angles to tell you where you are. `sextante` measures a VLM's
internals (vision encoder, multimodal projector, language model) so you can see what it
represents and why it writes what it writes. It is built for medical VLMs (MedGemma,
LLaVA-Med) but works with any Hugging Face VLM once you describe it with a small adapter.

> Status: pre-alpha. `adapters`, `activations` and `sae` work today; `features`,
> `validation`, `intervene` and `transfer` are planned (see [docs/plan.md](docs/plan.md)).

## Install

```bash
uv add sextante        # once released; for now:
git clone https://github.com/JManuelHaces/sextante && cd sextante && uv sync
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

## Sparse autoencoders

TopK (Gao et al., 2024) and JumpReLU (Rajamanoharan et al., 2024, as in GemmaScope),
trained straight from a store:

```python
from sextante.sae import SAEConfig, TrainConfig, train_sae, evaluate, spliced_loss

result = train_sae(
    "out/cxr",
    "llm.12",
    SAEConfig(d_in=2560, d_sae=16384, activation="topk", k=64),
    TrainConfig(total_tokens=50_000_000, rows="all"),  # or rows="image" / "text"
    save_dir="out/sae-llm12",
)
sae = result.sae
latents = sae.encode(acts)  # [n, 16384], at most 64 non-zero per row

# Does the model still work with the layer replaced by the SAE reconstruction?
spliced_loss(model, sae, "llm.12", batches)  # ce_clean, ce_sae, ce_zero, loss_recovered
```

- Inputs are normalised to mean norm `sqrt(d_in)`; the scale is stored in the SAE, so
  `encode` and `reconstruct` take raw activations.
- TopK uses the auxiliary loss to revive dead latents (no ReLU on the dead
  pre-activations, so latents that never fire still get gradient). JumpReLU uses
  straight-through estimators for its thresholds and an L0 penalty. `l0_coef` depends
  on the data and needs a sweep.
- Saved SAEs are `cfg.json` + `sae_weights.safetensors` with the tensor names used by
  SAELens and GemmaScope (`W_enc [d_in, d_sae]`, `W_dec [d_sae, d_in]`, `b_enc`,
  `b_dec`, `threshold`). `SAE.load_weights` reads those names. For GemmaScope, whose
  encoder does not subtract `b_dec`, use `load_weights(params, encoder_subtracts_b_dec=False)`.

See [`examples/train_sae.py`](examples/train_sae.py).

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
