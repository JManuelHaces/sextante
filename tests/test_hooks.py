import pytest
import torch

from sextante import capture, hooks, resolve, run_with_cache
from tests.conftest import gemma3_batch, llava_batch


def hook_ids(model):
    """Forward hooks per module. transformers installs its own, so compare, don't assume 0."""
    return {n: set(m._forward_hooks) for n, m in model.named_modules()}


@pytest.fixture(params=["gemma3", "llava"])
def model_and_batch(request, tiny_gemma3, tiny_llava):
    if request.param == "gemma3":
        return tiny_gemma3, gemma3_batch([[5, 6, 7], [8, 9]])
    return tiny_llava, llava_batch([[5, 6, 7], [8, 9]])


@torch.no_grad()
def test_llm_points_match_hidden_states(model_and_batch):
    """Decoder layer i output == hidden_states[i+1] (excluding the last, which is normed)."""
    model, batch = model_and_batch
    mm = resolve(model)
    points = [f"llm.{i}" for i in range(mm.n_llm_layers)]
    with capture(mm, points) as cache:
        out = model(**batch, output_hidden_states=True)
    for i in range(mm.n_llm_layers - 1):
        torch.testing.assert_close(cache[f"llm.{i}"], out.hidden_states[i + 1])


@torch.no_grad()
def test_projector_shape(tiny_gemma3, tiny_llava):
    _, cache = run_with_cache(tiny_gemma3, gemma3_batch([[5], [6]]), ["projector"])
    assert cache["projector"].shape == (2, 4, 32)  # mm_tokens_per_image=4, d_text=32
    _, cache = run_with_cache(tiny_llava, llava_batch([[5], [6]]), ["projector", "vision.1"])
    assert cache["projector"].shape == (2, 16, 32)
    assert cache["vision.1"].shape == (2, 17, 24)  # 16 patches + CLS


def test_stop_early_matches_full_forward(model_and_batch):
    model, batch = model_and_batch
    out, early = run_with_cache(model, batch, ["vision.0", "llm.0"], stop_early=True)
    assert out is None
    out, full = run_with_cache(model, batch, ["vision.0", "llm.0"], stop_early=False)
    assert out is not None
    for k in early:
        torch.testing.assert_close(early[k], full[k])


def test_stop_early_skips_later_layers(tiny_gemma3):
    mm = resolve(tiny_gemma3)
    fired = []
    with hooks(mm, {"llm.2": lambda t, p: fired.append(p.name)}):
        run_with_cache(mm, gemma3_batch([[5]]), ["llm.0"])
    assert fired == []


@torch.no_grad()
def test_replacement_changes_output_and_hooks_are_removed(tiny_gemma3):
    batch = gemma3_batch([[5, 6, 7]])
    before = hook_ids(tiny_gemma3)
    base = tiny_gemma3(**batch).logits
    with hooks(tiny_gemma3, {"llm.1": lambda t, p: torch.zeros_like(t)}):
        ablated = tiny_gemma3(**batch).logits
    assert not torch.allclose(base, ablated)
    torch.testing.assert_close(tiny_gemma3(**batch).logits, base)
    assert hook_ids(tiny_gemma3) == before


def test_replacement_shape_checked(tiny_gemma3):
    with (
        hooks(tiny_gemma3, {"llm.0": lambda t, p: t[:, :1]}),
        pytest.raises(ValueError, match="returned shape"),
    ):
        tiny_gemma3(**gemma3_batch([[5, 6]]))


def test_hooks_removed_after_exception(tiny_gemma3):
    before = hook_ids(tiny_gemma3)

    def boom(t, p):
        raise RuntimeError("boom")

    with hooks(tiny_gemma3, {"llm.0": boom}), pytest.raises(RuntimeError):
        tiny_gemma3(**gemma3_batch([[5]]))
    assert hook_ids(tiny_gemma3) == before


def test_capture_options(tiny_gemma3):
    batch = gemma3_batch([[5, 6]])
    with capture(tiny_gemma3, ["llm.0"], dtype=torch.float16, device="cpu", accumulate=True) as c:
        with torch.no_grad():
            tiny_gemma3(**batch)
            tiny_gemma3(**batch)
    assert len(c["llm.0"]) == 2 and c["llm.0"][0].dtype == torch.float16
    assert c.stacked("llm.0").shape[1] == 2 * batch["input_ids"].shape[1]


def test_never_fired_raises(tiny_gemma3):
    batch = gemma3_batch([[5]])
    del batch["pixel_values"]
    batch["input_ids"][batch["input_ids"] == 100] = 7  # no image tokens left
    with pytest.raises(RuntimeError, match="never fired"):
        run_with_cache(tiny_gemma3, batch, ["projector", "llm.0"])


def test_module_point_disables_stop_after_last(tiny_gemma3):
    with pytest.raises(ValueError, match="module"):
        with capture(tiny_gemma3, ["module:lm_head"], stop_after_last=True):
            pass
    out, cache = run_with_cache(tiny_gemma3, gemma3_batch([[5]]), ["module:lm_head"])
    assert out is not None and cache["module:lm_head"].shape[-1] == 128
