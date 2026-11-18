import json

import pytest
import torch

from sextante import ActivationRows, ActivationStore, ActivationWriter


def rows(n: int, d: int = 4, start: int = 0) -> ActivationRows:
    return ActivationRows(
        acts=torch.arange(start * d, (start + n) * d, dtype=torch.float32).reshape(n, d),
        sample_idx=torch.arange(start, start + n),
        position=torch.zeros(n, dtype=torch.long),
        is_image=torch.zeros(n, dtype=torch.bool),
        token_id=torch.full((n,), 7),
    )


def test_roundtrip_with_sharding(tmp_path):
    path = tmp_path / "store"
    with ActivationWriter(path, shard_rows=4, dtype=torch.float32, info={"run": "t"}) as w:
        w.add_samples([{"image_id": "a", "lang": "es"}, {"image_id": "b", "lang": "en"}])
        w.add("llm.0", rows(3), kind="llm", layer=0)
        w.add("llm.0", rows(7, start=3))  # crosses two shard boundaries
    store = ActivationStore(path)
    assert store.complete and store.info == {"run": "t"}
    assert store.n_rows("llm.0") == 10 and store.d("llm.0") == 4
    assert [s["n_rows"] for s in store.point_info("llm.0")["shards"]] == [4, 4, 2]
    loaded = store.load("llm.0")
    torch.testing.assert_close(loaded.acts, rows(10).acts)
    assert loaded.sample_idx.tolist() == list(range(10))
    assert store.samples[1] == {"image_id": "b", "lang": "en"}
    assert store.point_info("llm.0")["layer"] == 0


def test_dtype_cast(tmp_path):
    with ActivationWriter(tmp_path / "s") as w:
        w.add("p", rows(2))
    assert ActivationStore(tmp_path / "s").load("p").acts.dtype == torch.float16


def test_refuses_to_overwrite(tmp_path):
    with ActivationWriter(tmp_path / "s") as w:
        w.add("p", rows(1))
    with pytest.raises(FileExistsError):
        ActivationWriter(tmp_path / "s")
    with ActivationWriter(tmp_path / "s", overwrite=True) as w:
        w.add("q", rows(1))
    assert ActivationStore(tmp_path / "s").points == ["q"]


def test_interrupted_run_is_readable_and_marked_incomplete(tmp_path):
    path = tmp_path / "s"
    with pytest.raises(RuntimeError), ActivationWriter(path, shard_rows=2) as w:
        w.add_samples([{}])
        w.add("p", rows(5))  # flushes two full shards, keeps one row buffered
        raise RuntimeError("crash")
    store = ActivationStore(path)
    assert not store.complete
    assert store.n_rows("p") == 4


def test_width_mismatch(tmp_path):
    with ActivationWriter(tmp_path / "s") as w:
        w.add("p", rows(1, d=4))
        with pytest.raises(ValueError, match="width"):
            w.add("p", rows(1, d=5))


def test_rows_validation():
    with pytest.raises(ValueError):
        ActivationRows(
            acts=torch.zeros(3, 2),
            sample_idx=torch.zeros(2),
            position=torch.zeros(3),
            is_image=torch.zeros(3),
            token_id=torch.zeros(3),
        )


def test_manifest_is_valid_json(tmp_path):
    with ActivationWriter(tmp_path / "s") as w:
        w.add_samples([{"report": "Sin hallazgos. Corazón normal."}])
        w.add("p", rows(1))
    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert manifest["format_version"] == 1 and manifest["n_samples"] == 1
    assert ActivationStore(tmp_path / "s").samples[0]["report"].startswith("Sin hallazgos")
