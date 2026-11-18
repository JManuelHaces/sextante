import pytest
import torch

from sextante import ActivationExtractor, extract, resolve, run_with_cache
from tests.conftest import IMAGE_TOKEN, gemma3_batch, llava_batch

TEXTS = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]


def batches(texts=TEXTS, bs=2, maker=gemma3_batch):
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        b = maker(chunk, seed=i)
        b["meta"] = [{"image_id": f"img{i + j}", "lang": "es"} for j in range(len(chunk))]
        out.append(b)
    return out


def test_extract_all_tokens(tmp_path, tiny_gemma3):
    store = extract(
        tiny_gemma3, batches(), tmp_path / "s", ["projector", "llm.1"], dtype=torch.float32
    )
    assert store.complete
    assert store.info["adapter"] == "gemma3" and store.info["tokens"] == "all"
    assert [s["image_id"] for s in store.samples] == ["img0", "img1", "img2"]

    llm = store.load("llm.1")
    lengths = [2 + 4 + 1 + len(t) for t in TEXTS]  # BOS, BOI, 4 image, EOI, text
    assert len(llm) == sum(lengths)  # padding excluded
    for s, n in enumerate(lengths):
        sel = llm.sample_idx == s
        assert llm.position[sel].tolist() == list(range(n))
        assert llm.is_image[sel].sum() == 4
        assert (llm.token_id[sel][llm.is_image[sel]] == IMAGE_TOKEN).all()
        assert llm.token_id[sel][-len(TEXTS[s]) :].tolist() == TEXTS[s]

    proj = store.load("projector")
    assert len(proj) == 3 * 4 and proj.d == 32
    assert proj.sample_idx.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert proj.position.tolist() == list(range(4)) * 3
    assert (proj.token_id == -1).all() and proj.is_image.all()
    info = store.point_info("projector")
    assert info["kind"] == "projector" and info["layer"] is None


def test_extracted_values_match_direct_forward(tmp_path, tiny_gemma3):
    b = batches()[0]
    store = extract(tiny_gemma3, [b], tmp_path / "s", ["llm.2"], dtype=torch.float32)
    _, cache = run_with_cache(tiny_gemma3, b, ["llm.2"])
    mask = b["attention_mask"].bool()
    torch.testing.assert_close(store.load("llm.2").acts, cache["llm.2"][mask])


@pytest.mark.parametrize(
    "tokens, expected",
    [("image", [4, 4, 4]), ("text", [3 + len(t) for t in TEXTS]), ("last", [1, 1, 1])],
)
def test_token_selection(tmp_path, tiny_gemma3, tokens, expected):
    store = extract(tiny_gemma3, batches(), tmp_path / "s", ["llm.0"], tokens=tokens)
    rows = store.load("llm.0")
    assert torch.bincount(rows.sample_idx).tolist() == expected
    if tokens == "image":
        assert rows.is_image.all()
    if tokens == "last":
        assert rows.token_id.tolist() == [t[-1] for t in TEXTS]


def test_padding_side_does_not_matter(tiny_gemma3):
    ex = ActivationExtractor(tiny_gemma3, ["llm.1"])
    left = gemma3_batch(TEXTS[:2])
    right = {k: v.clone() for k, v in left.items()}
    for i in range(2):  # move padding to the right
        n = int(left["attention_mask"][i].sum())
        for k in ("input_ids", "attention_mask"):
            right[k][i] = torch.roll(left[k][i], -(left[k].shape[1] - n))
    a, b = ex.extract_batch(left)["llm.1"], ex.extract_batch(right)["llm.1"]
    assert a.position.tolist() == b.position.tolist()
    assert a.token_id.tolist() == b.token_id.tolist()
    torch.testing.assert_close(a.acts, b.acts, atol=1e-4, rtol=1e-4)


def test_vision_points_llava(tmp_path, tiny_llava):
    store = extract(
        tiny_llava, batches(maker=llava_batch), tmp_path / "s", ["vision.0", "projector", "llm.-1"]
    )
    assert store.points == ["vision.0", "projector", "llm.1"]
    vis = store.load("vision.0")
    assert len(vis) == 3 * 17 and vis.d == 24
    assert torch.bincount(vis.sample_idx).tolist() == [17, 17, 17]
    assert store.info["adapter"] == "llava"


def test_image_sample_index_required_for_multi_image(tiny_gemma3):
    ex = ActivationExtractor(tiny_gemma3, ["projector"])
    b = gemma3_batch([[5], [6]])
    b["pixel_values"] = b["pixel_values"][:1]  # 1 image for 2 samples
    b["input_ids"][1][b["input_ids"][1] == IMAGE_TOKEN] = 7
    with pytest.raises(ValueError, match="image_sample_index"):
        ex.extract_batch(b)
    b["image_sample_index"] = torch.tensor([0])
    rows = ex.extract_batch(b, sample_offset=10)["projector"]
    assert rows.sample_idx.tolist() == [10] * 4


def test_meta_length_checked(tmp_path, tiny_gemma3):
    b = batches()[0]
    b["meta"] = b["meta"][:1]
    with pytest.raises(ValueError, match="meta"):
        extract(tiny_gemma3, [b], tmp_path / "s", ["llm.0"])


def test_extractor_accepts_model_map(tiny_gemma3):
    ex = ActivationExtractor(resolve(tiny_gemma3), ["llm.0"], tokens="last")
    assert ex.points == ["llm.0"]
    with pytest.raises(ValueError):
        ActivationExtractor(tiny_gemma3, ["llm.0"], tokens="bogus")
