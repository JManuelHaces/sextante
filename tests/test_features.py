from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sextante import ActivationStore, extract
from sextante.features import (
    Example,
    FeatureScan,
    auroc,
    claude_llm,
    correlation_ratio,
    detection_score,
    explain,
    explain_prompt,
    feature_table,
    heatmap_overlay,
    interpret_feature,
    metadata_association,
    patch_grid,
    sample_activations,
    scan_features,
    token_activations,
    tokens_html,
)
from sextante.features.autointerp import parse_indices
from sextante.features.render import colormap
from sextante.sae import SAE, SAEConfig
from tests.conftest import IMAGE_TOKEN, gemma3_batch, llava_batch

N_SAMPLES = 12


def texts(i: int) -> list[int]:
    return [5 + (i % 7), 20 + (i % 3), 40 + i, 60][: 2 + i % 3]


@pytest.fixture(scope="module")
def gemma_store(tmp_path_factory, tiny_gemma3):
    batches = []
    for start in range(0, N_SAMPLES, 4):
        ids = range(start, start + 4)
        b = gemma3_batch([texts(i) for i in ids], seed=start)
        b["meta"] = [
            {"id": f"s{i}", "hospital": "A" if i % 2 else "B", "finding": i % 3 == 0} for i in ids
        ]
        batches.append(b)
    path = tmp_path_factory.mktemp("feat") / "acts"
    return extract(
        tiny_gemma3, batches, path, ["projector", "llm.1"], dtype=torch.float32, shard_rows=17
    )


@pytest.fixture(scope="module")
def sae32():
    torch.manual_seed(0)
    sae = SAE(SAEConfig(d_in=32, d_sae=48, k=6, metadata={"point": "llm.1"}))
    with torch.no_grad():
        sae.b_enc.normal_(0, 0.5)
    return sae.eval()


# -- store random access ----------------------------------------------------------------
def test_sample_rows_match_full_load(gemma_store):
    full = gemma_store.load("llm.1")
    assert len(gemma_store.point_info("llm.1")["shards"]) > 3  # rows span many shards
    for s in range(N_SAMPLES):
        rows = gemma_store.sample_rows("llm.1", s)
        torch.testing.assert_close(rows.acts, full.acts[full.sample_idx == s])
        assert rows.position.tolist() == list(range(len(rows)))
    torch.testing.assert_close(gemma_store.rows("llm.1", 10, 40).acts, full.acts[10:40])
    assert len(gemma_store.rows("llm.1", 10**6, 10**6 + 5)) == 0
    with pytest.raises(IndexError):
        gemma_store.sample_rows("llm.1", N_SAMPLES)


def test_column(gemma_store):
    assert torch.equal(gemma_store.column("llm.1", "token_id"), gemma_store.load("llm.1").token_id)
    with pytest.raises(KeyError):
        gemma_store.column("llm.1", "nope")


# -- scan -------------------------------------------------------------------------------
def test_scan_matches_brute_force(gemma_store, sae32):
    scan = scan_features(gemma_store, sae32, batch_rows=10)
    assert scan.point == "llm.1" and scan.n_samples == N_SAMPLES
    rows = gemma_store.load("llm.1")
    with torch.no_grad():
        latents = sae32.encode(rows.acts)
    for s in range(N_SAMPLES):
        torch.testing.assert_close(scan.sample_max[s], latents[rows.sample_idx == s].max(0).values)
    assert torch.equal(scan.fire_count, (latents > 0).sum(0))
    torch.testing.assert_close(scan.act_sum.float(), latents.sum(0), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        scan.image_act_sum.float(), latents[rows.is_image].sum(0), atol=1e-4, rtol=1e-5
    )
    assert scan.n_rows == len(rows) and scan.n_image_rows == int(rows.is_image.sum())
    alive = ~scan.dead
    assert ((scan.image_fraction[alive] >= 0) & (scan.image_fraction[alive] <= 1 + 1e-9)).all()


def test_scan_text_rows_and_errors(gemma_store, sae32):
    text = scan_features(gemma_store, sae32, rows="text")
    assert text.n_image_rows == 0 and (text.image_act_sum == 0).all()
    with pytest.raises(ValueError, match="width"):
        scan_features(gemma_store, SAE(SAEConfig(d_in=8, d_sae=16, k=2)), "llm.1")
    with pytest.raises(ValueError, match="point"):
        scan_features(gemma_store, SAE(SAEConfig(d_in=32, d_sae=16, k=2)))


def test_scan_save_load_and_top_samples(tmp_path, gemma_store, sae32):
    scan = scan_features(gemma_store, sae32)
    scan.save(tmp_path / "scan")
    loaded = FeatureScan.load(tmp_path / "scan")
    torch.testing.assert_close(loaded.sample_max, scan.sample_max, atol=1e-2, rtol=1e-3)
    assert torch.equal(loaded.fire_count, scan.fire_count)
    f = int(scan.fire_count.argmax())
    values, idx = scan.top_samples(f, 5)
    assert (values[:-1] >= values[1:]).all() and (values > 0).all()
    assert idx[0] == scan.sample_max[:, f].argmax()
    inactive = scan.inactive_samples(f)
    assert (scan.sample_max[inactive, f] == 0).all()


# -- per-sample views ---------------------------------------------------------------
class FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return [f"▁t{i}" for i in ids]


def test_sample_activations_and_tokens(gemma_store, sae32):
    sa = sample_activations(gemma_store, sae32, "llm.1", 5, [0, 3])
    rows = gemma_store.sample_rows("llm.1", 5)
    with torch.no_grad():
        torch.testing.assert_close(sa.acts, sae32.encode(rows.acts)[:, [0, 3]])
    assert sa.meta["id"] == "s5"
    tokens = token_activations(sa, FakeTokenizer(), feature=3)
    text_ids = rows.token_id[~rows.is_image].tolist()
    assert [t for t, _ in tokens] == [f" t{i}" for i in text_ids]
    assert text_ids[-len(texts(5)) :] == texts(5)
    assert [v for _, v in tokens] == sa.acts[~rows.is_image, 1].tolist()
    assert token_activations(sa)[0][0].startswith("<")


def test_patch_grid_llm_and_projector(gemma_store, sae32):
    sa = sample_activations(gemma_store, sae32, "llm.1", 0, 2)
    grid = patch_grid(sa)
    assert grid.shape == (2, 2)  # 4 image tokens
    assert torch.equal(grid.flatten(), sa.acts[sa.token_id == IMAGE_TOKEN])
    sae_proj = SAE(SAEConfig(d_in=32, d_sae=16, k=4))
    assert patch_grid(sample_activations(gemma_store, sae_proj, "projector", 3, 0)).shape == (2, 2)


def test_patch_grid_drops_cls(tmp_path, tiny_llava):
    b = llava_batch([[5, 6], [7]])
    store = extract(tiny_llava, [b], tmp_path / "s", ["vision.0"])
    sa = sample_activations(store, SAE(SAEConfig(d_in=24, d_sae=16, k=4)), "vision.0", 1, 0)
    assert len(sa.acts) == 17
    assert torch.equal(patch_grid(sa).flatten(), sa.acts[1:])


def test_patch_grid_errors(gemma_store, sae32):
    sa = sample_activations(gemma_store, sae32, "llm.1", 0, 2)
    with pytest.raises(IndexError):
        patch_grid(sa, image=1)
    sa.is_image[:] = False
    with pytest.raises(ValueError, match="no image rows"):
        patch_grid(sa)


# -- rendering --------------------------------------------------------------------------
def test_heatmap_overlay():
    image = Image.fromarray(np.full((40, 40, 3), 128, dtype=np.uint8))
    same = heatmap_overlay(image, torch.zeros(4, 4))
    assert np.array_equal(np.asarray(same), np.asarray(image))
    grid = torch.zeros(4, 4)
    grid[0, 0] = 3.0
    out = np.asarray(heatmap_overlay(image, grid, alpha=1.0)).astype(int)
    assert out.shape == (40, 40, 3)
    assert np.abs(out[0, 0] - 128).sum() > 50 and np.array_equal(out[-1, -1], [128] * 3)


def test_colormap_endpoints():
    assert colormap(np.array([0.0])).tolist() == [[68, 1, 84]]
    assert colormap(np.array([1.0])).tolist() == [[253, 231, 37]]


def test_tokens_html_escapes():
    out = tokens_html([("<b>", 1.0), (" ok", 0.0)])
    assert "&lt;b&gt;" in out and "rgba(255,140,0,1.00)" in out and "rgba(255,140,0,0.00)" in out


# -- statistics -------------------------------------------------------------------------
def test_correlation_ratio():
    values = torch.tensor([[0.0, 1.0], [0.0, 2.0], [5.0, 1.0], [5.0, 2.0]])
    groups = torch.tensor([0, 0, 1, 1])
    eta = correlation_ratio(values, groups)
    assert eta[0] == pytest.approx(1.0) and eta[1] == pytest.approx(0.0)
    assert correlation_ratio(torch.ones(4, 1), groups)[0] == 0


def test_auroc_matches_pair_counting():
    g = torch.Generator().manual_seed(0)
    values = torch.randint(0, 4, (30, 3), generator=g).float()  # many ties
    labels = torch.rand(30, generator=g) > 0.4
    got = auroc(values, labels)
    pos, neg = values[labels], values[~labels]
    for j in range(3):
        diff = pos[:, j, None] - neg[None, :, j]
        expected = ((diff > 0).float() + 0.5 * (diff == 0).float()).mean()
        assert got[j].item() == pytest.approx(expected.item())


def synthetic_scan(n: int = 200) -> tuple[FeatureScan, list[dict]]:
    g = torch.Generator().manual_seed(0)
    hospital = torch.randint(0, 2, (n,), generator=g)
    finding = torch.randint(0, 2, (n,), generator=g)
    sample_max = torch.zeros(n, 5)
    sample_max[:, 0] = hospital * (1 + torch.rand(n, generator=g))  # follows the site
    sample_max[:, 1] = finding * (1 + torch.rand(n, generator=g))  # follows the finding
    sample_max[:, 2] = 1 + torch.rand(n, generator=g)  # always on
    sample_max[:, 4] = (torch.rand(n, generator=g) > 0.8).float() * torch.rand(n, generator=g)
    fire = (sample_max > 0).sum(0)
    scan = FeatureScan(
        point="llm.1",
        sample_max=sample_max,
        fire_count=fire,
        act_sum=sample_max.sum(0).double(),
        image_act_sum=torch.zeros(5, dtype=torch.float64),
        image_fire_count=torch.zeros(5),
        n_rows=n,
        n_image_rows=0,
        info={},
    )
    samples = [
        {"hospital": "AB"[h], "finding": bool(f)}
        for h, f in zip(hospital.tolist(), finding.tolist(), strict=True)
    ]
    return scan, samples


def test_metadata_association():
    scan, samples = synthetic_scan()
    hosp = metadata_association(scan, samples, "hospital")
    assert hosp.categories == ["A", "B"] and sum(hosp.counts) == len(samples)
    assert hosp.eta2[0] > 0.5 and hosp.eta2[1] < 0.05
    assert hosp.top_category()[0] == "B"
    finding = metadata_association(scan, samples, "finding")
    assert finding.categories == [False, True]  # positive class second
    assert finding.auroc[1] > 0.95 and abs(finding.auroc[0] - 0.5) < 0.15


def test_metadata_association_skips_missing_and_rare():
    scan, samples = synthetic_scan()
    samples = [dict(s) for s in samples]
    samples[0].pop("hospital")
    samples[1]["hospital"] = "C"  # rare category
    a = metadata_association(scan, samples, "hospital", min_count=5)
    assert a.categories == ["A", "B"] and a.n_samples == len(samples) - 2
    with pytest.raises(ValueError, match="categories"):
        metadata_association(scan, [{"x": 1}] * scan.n_samples, "x")


def test_feature_table_labels():
    scan, samples = synthetic_scan()
    table = feature_table(scan, samples, nuisance=["hospital"], targets=["finding"])
    labels = [row["label"] for row in table]
    assert labels[:5] == ["corpus-artifact", "domain", "dense", "dead", ""]
    assert table[0]["strongest"] == "hospital" and table[1]["strongest"] == "finding"
    assert table[3]["strongest"] is None  # dead: no association at all
    assert "auroc:finding" in table[1] and table[1]["top:finding"] is True


# -- autointerp -------------------------------------------------------------------------
def make_examples(word: str, n: int, start: int = 0) -> list[Example]:
    return [
        Example(
            sample=start + i,
            max_act=2.0,
            image_max=0.5,
            tokens=[("There", 0.0), (" is", 0.0), (f" {word}", 2.0), (".", 0.0)],
            meta={"view": "PA"},
        )
        for i in range(n)
    ]


def test_explain_prompt_marks_tokens():
    prompt = explain_prompt(make_examples("effusion", 2), modality="chest X-ray")
    assert "«effusion»(10)" in prompt and "chest X-ray" in prompt
    assert "image activation: 2" in prompt and "view=PA" in prompt


def test_explain_parses_answer():
    llm = lambda prompt: "Thinking aloud...\nExplanation: pleural effusion mentions\nextra"  # noqa: E731
    assert explain(make_examples("effusion", 3), llm) == "pleural effusion mentions"
    with pytest.raises(ValueError):
        explain([], llm)


def oracle(word: str):
    """Fake LLM that marks every sample whose report contains ``word``."""

    def llm(prompt: str) -> str:
        chunks = prompt.split("Sample ")[1:]
        return str([i + 1 for i, c in enumerate(chunks) if word in c])

    return llm


def test_detection_score():
    pos, neg = make_examples("effusion", 4), make_examples("normal", 4, start=10)
    perfect = detection_score("effusion", pos, neg, oracle("effusion"))
    assert perfect["balanced_accuracy"] == 1.0
    assert "«" not in "".join(e.text(marked=False, scale=1) for e in pos)
    none = detection_score("effusion", pos, neg, lambda p: "[]")
    assert none["balanced_accuracy"] == 0.5 and none["true_negative_rate"] == 1.0


def test_parse_indices():
    assert parse_indices("I think [1, 3, 9] fit", 5) == {1, 3}
    assert parse_indices("no list", 5) == set()
    assert parse_indices('[1, "x"]', 5) == {1}
    assert parse_indices("[oops", 5) == set()


def test_interpret_feature_end_to_end(gemma_store, sae32):
    scan = scan_features(gemma_store, sae32)
    feature = int(((scan.sample_frequency > 0.2) & (scan.sample_frequency < 0.8)).nonzero()[0])
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "Explanation: something" if len(prompts) == 1 else "[1]"

    result = interpret_feature(
        gemma_store, sae32, scan, feature, llm, n_explain=3, n_score=2, meta_keys=["hospital"]
    )
    assert result["explanation"] == "something" and len(result["explain_samples"]) == 3
    assert "hospital=" in prompts[0]
    assert len(prompts) == 2 and 0.0 <= result["detection"]["balanced_accuracy"] <= 1.0
    dead = int(scan.dead.nonzero()[0]) if scan.dead.any() else None
    if dead is not None:
        assert interpret_feature(gemma_store, sae32, scan, dead, llm)["explanation"] is None


class FakeClient:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            _request_id="req_test",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Explanation: x"),
            ],
        )


def test_claude_llm_request_shape():
    client = FakeClient()
    llm = claude_llm(client=client, effort="low")
    assert llm("hello") == "Explanation: x"
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"}
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    with pytest.raises(RuntimeError, match="declined"):
        claude_llm(client=FakeClient("refusal"))("x")


def test_store_reopen_keeps_working(gemma_store):
    reopened = ActivationStore(gemma_store.path)
    assert len(reopened.sample_rows("projector", 2)) == 4
