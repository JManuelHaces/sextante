import pytest
import torch
from torch import nn

import sextante
from sextante.adapters import ModelAdapter, get_submodule, resolve, unregister_adapter


def test_detects_gemma3(tiny_gemma3):
    mm = resolve(tiny_gemma3)
    assert mm.adapter.name == "gemma3"
    assert mm.n_vision_layers == 2
    assert mm.n_llm_layers == 3
    assert mm.projector is not None
    assert mm.image_token_id == 100


def test_detects_llava(tiny_llava):
    mm = resolve(tiny_llava)
    assert mm.adapter.name == "llava"
    assert (mm.n_vision_layers, mm.n_llm_layers) == (2, 2)
    assert mm.image_token_id == 100


def test_point_names_in_forward_order(tiny_gemma3):
    assert resolve(tiny_gemma3).point_names() == [
        "vision.0",
        "vision.1",
        "projector",
        "llm.0",
        "llm.1",
        "llm.2",
    ]


def test_negative_index_and_canonical_name(tiny_gemma3):
    mm = resolve(tiny_gemma3)
    p = mm.point("llm.-1")
    assert p.name == "llm.2" and p.layer == 2
    assert p.module is mm.llm_layers[2]
    assert p.path == "model.language_model.layers.2"


@pytest.mark.parametrize("bad", ["llm.3", "llm.-4", "vision.x", "foo", "projector.0"])
def test_invalid_points(tiny_gemma3, bad):
    with pytest.raises(KeyError):
        resolve(tiny_gemma3).point(bad)


def test_module_escape_hatch(tiny_gemma3):
    mm = resolve(tiny_gemma3)
    p = mm.point("module:model.language_model.layers.1.mlp")
    assert p.kind == "module" and p.order is None
    assert p.module is mm.llm_layers[1].mlp
    with pytest.raises(KeyError):
        mm.point("module:does.not.exist")


def test_duplicate_points_rejected(tiny_gemma3):
    with pytest.raises(ValueError, match="twice"):
        resolve(tiny_gemma3).points(["llm.2", "llm.-1"])


def test_get_submodule():
    seq = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    root = nn.Module()
    root.block = seq
    assert get_submodule(root, "block.0") is seq[0]
    assert get_submodule(root, "block.-1") is seq[1]
    assert get_submodule(root, "block.5") is None
    assert get_submodule(root, "missing") is None


class ToyVLM(nn.Module):
    """A non-HF architecture with its own naming, to show a custom adapter."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.bridge = nn.Linear(4, 6)
        self.decoder = nn.ModuleList([nn.Linear(6, 6) for _ in range(3)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.bridge(x)
        for layer in self.decoder:
            x = layer(x)
        return x


def test_custom_adapter_registration():
    adapter = ModelAdapter(
        name="toy",
        class_names=("ToyVLM",),
        vision_layers_paths=("encoder",),
        projector_paths=("bridge",),
        llm_layers_paths=("decoder",),
    )
    model = ToyVLM()
    with pytest.raises(ValueError, match="No adapter matches"):
        resolve(model)
    sextante.register_adapter(adapter)
    try:
        mm = resolve(model)
        assert mm.adapter is adapter
        assert mm.point_names()[-1] == "llm.2"
        with pytest.raises(ValueError, match="already registered"):
            sextante.register_adapter(adapter)
    finally:
        unregister_adapter("toy")


def test_explicit_adapter_and_missing_components():
    adapter = ModelAdapter(name="broken", llm_layers_paths=("nope",))
    with pytest.raises(ValueError, match="could not locate"):
        resolve(ToyVLM(), adapter)
    mm = resolve(ToyVLM(), ModelAdapter(name="only-llm", llm_layers_paths=("decoder",)))
    assert mm.point_names() == ["llm.0", "llm.1", "llm.2"]
    with pytest.raises(KeyError, match="no projector"):
        mm.point("projector")


def test_list_adapters():
    assert {"gemma3", "llava"} <= set(sextante.list_adapters())
    assert sextante.get_adapter("gemma3") is sextante.adapters.GEMMA3
    with pytest.raises(KeyError):
        sextante.get_adapter("nope")
    _ = torch  # keep import used
