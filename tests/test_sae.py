import json

import pytest
import torch

from sextante import ActivationRows, ActivationStore, ActivationWriter, extract
from sextante.sae import (
    SAE,
    ActivationLoader,
    SAEConfig,
    TrainConfig,
    aux_k_loss,
    estimate_norm_scale,
    evaluate,
    spliced_loss,
    train_sae,
)
from tests.conftest import gemma3_batch

D_IN, N_TRUE = 16, 48


def synthetic(n: int, *, seed: int = 0, active: int = 3, scale: float = 7.0):
    """Rows that are sparse combinations of a hidden unit-norm dictionary, plus an offset."""
    g = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(N_TRUE, D_IN, generator=g)
    dictionary /= dictionary.norm(dim=1, keepdim=True)
    coeffs = torch.zeros(n, N_TRUE)
    idx = torch.stack([torch.randperm(N_TRUE, generator=g)[:active] for _ in range(n)])
    coeffs.scatter_(1, idx, 1 + 2 * torch.rand(n, active, generator=g))
    offset = torch.randn(D_IN, generator=g)
    return scale * (coeffs @ dictionary + offset)


def make_store(path, acts: torch.Tensor, *, shard_rows: int = 1000, is_image=None):
    n = len(acts)
    with ActivationWriter(path, shard_rows=shard_rows, dtype=torch.float32) as w:
        w.add_samples([{} for _ in range(n)])
        w.add(
            "llm.0",
            ActivationRows(
                acts=acts,
                sample_idx=torch.arange(n),
                position=torch.zeros(n, dtype=torch.long),
                is_image=is_image if is_image is not None else torch.zeros(n, dtype=torch.bool),
                token_id=torch.zeros(n, dtype=torch.long),
            ),
            kind="llm",
            layer=0,
        )
    return ActivationStore(path)


# -- model ------------------------------------------------------------------------------
def test_topk_has_exactly_k_active():
    sae = SAE(SAEConfig(d_in=8, d_sae=32, k=5))
    out = sae(torch.randn(10, 8))
    assert ((out.latents > 0).sum(-1) <= 5).all()
    assert ((out.latents != 0).sum(-1) <= 5).all()
    assert out.x_hat.shape == (10, 8)


def test_init_is_tied_and_unit_norm():
    sae = SAE(SAEConfig(d_in=8, d_sae=32, k=4))
    torch.testing.assert_close(sae.W_dec.norm(dim=1), torch.ones(32))
    torch.testing.assert_close(sae.W_enc, sae.W_dec.T)


def test_jumprelu_threshold_gets_gradient():
    sae = SAE(
        SAEConfig(
            d_in=8,
            d_sae=16,
            activation="jumprelu",
            jumprelu_init_threshold=0.5,
            jumprelu_bandwidth=2.0,
        )
    )
    out = sae(torch.randn(64, 8))
    assert ((out.latents == 0) | (out.pre > sae.threshold)).all()
    (out.x_hat_scaled.pow(2).sum() + sae.l0_surrogate(out.pre).sum()).backward()
    assert sae.log_threshold.grad is not None and sae.log_threshold.grad.abs().sum() > 0


def test_l0_surrogate_counts_active():
    sae = SAE(SAEConfig(d_in=8, d_sae=16, activation="jumprelu"))
    out = sae(torch.randn(20, 8))
    torch.testing.assert_close(sae.l0_surrogate(out.pre), (out.latents > 0).float().sum(-1))


def test_decoder_constraint():
    sae = SAE(SAEConfig(d_in=8, d_sae=16, k=4))
    sae(torch.randn(4, 8)).x_hat.sum().backward()
    sae.remove_parallel_decoder_grad_()
    assert (sae.W_dec.grad * sae.W_dec).sum(1).abs().max() < 1e-5
    with torch.no_grad():
        sae.W_dec.mul_(3)
    sae.normalize_decoder_()
    torch.testing.assert_close(sae.W_dec.norm(dim=1), torch.ones(16))


@pytest.mark.parametrize("activation", ["topk", "jumprelu"])
def test_save_load_roundtrip(tmp_path, activation):
    sae = SAE(SAEConfig(d_in=8, d_sae=16, k=3, activation=activation, metadata={"point": "llm.3"}))
    with torch.no_grad():
        sae.b_enc.uniform_()
        sae.input_scale.fill_(2.5)
    sae.save(tmp_path / "sae")
    cfg = json.loads((tmp_path / "sae" / "cfg.json").read_text())
    assert cfg["architecture"] == activation and cfg["metadata"]["point"] == "llm.3"
    loaded = SAE.load(tmp_path / "sae")
    x = torch.randn(5, 8)
    torch.testing.assert_close(loaded(x).x_hat, sae(x).x_hat)
    assert loaded.input_scale.item() == 2.5


def test_load_weights_from_encoder_without_b_dec():
    """GemmaScope-style weights: pre = x @ W_enc + b_enc, JumpReLU, x_hat = f @ W_dec + b_dec."""
    g = torch.Generator().manual_seed(0)
    w = {
        "W_enc": torch.randn(8, 16, generator=g),
        "W_dec": torch.randn(16, 8, generator=g),
        "b_enc": torch.randn(16, generator=g),
        "b_dec": torch.randn(8, generator=g),
        "threshold": torch.rand(16, generator=g),
    }
    x = torch.randn(10, 8, generator=g)
    pre = x @ w["W_enc"] + w["b_enc"]
    expected = (pre * (pre > w["threshold"])) @ w["W_dec"] + w["b_dec"]
    sae = SAE(SAEConfig(d_in=8, d_sae=16, activation="jumprelu"))
    sae.load_weights(w, encoder_subtracts_b_dec=False)
    torch.testing.assert_close(sae(x).x_hat, expected, atol=1e-5, rtol=1e-5)


def test_fold_input_scale_preserves_function():
    sae = SAE(SAEConfig(d_in=8, d_sae=16, activation="jumprelu"))
    with torch.no_grad():
        sae.input_scale.fill_(0.3)
        sae.b_dec.normal_()
    x = 5 * torch.randn(6, 8)
    before = sae(x)
    sae.fold_input_scale()
    after = sae(x)
    assert sae.input_scale.item() == 1.0
    torch.testing.assert_close(after.x_hat, before.x_hat, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(after.latents, before.latents, atol=1e-5, rtol=1e-5)


def test_reconstruct_keeps_shape_and_dtype():
    sae = SAE(SAEConfig(d_in=8, d_sae=16, k=4))
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    y = sae.reconstruct(x)
    assert y.shape == x.shape and y.dtype == torch.bfloat16


def test_aux_k_loss_only_moves_dead_latents():
    sae = SAE(SAEConfig(d_in=8, d_sae=16, k=2))
    x, residual = torch.randn(4, 8), torch.randn(4, 8)
    assert aux_k_loss(sae, x, residual, torch.zeros(16, dtype=torch.bool), 4).item() == 0
    dead = torch.zeros(16, dtype=torch.bool)
    dead[:3] = True
    aux_k_loss(sae, x, residual, dead, 8).backward()
    assert sae.W_enc.grad[:, 3:].abs().sum() == 0 and sae.W_enc.grad[:, :3].abs().sum() > 0
    assert sae.W_dec.grad[3:].abs().sum() == 0
    assert sae.b_dec.grad is None or sae.b_dec.grad.abs().sum() == 0


def test_aux_k_loss_reaches_latents_with_negative_preactivations():
    """A truly dead latent never has positive pre-activations; it must still get gradient."""
    sae = SAE(SAEConfig(d_in=8, d_sae=16, k=2))
    with torch.no_grad():
        sae.b_enc[:4] = -100.0
    dead = torch.zeros(16, dtype=torch.bool)
    dead[:4] = True
    aux_k_loss(sae, torch.randn(32, 8), torch.randn(32, 8), dead, 2).backward()
    assert sae.b_enc.grad[:4].abs().sum() > 0


def test_invalid_configs():
    with pytest.raises(ValueError):
        SAEConfig(d_in=8, d_sae=16, k=17)
    with pytest.raises(ValueError):
        SAEConfig(d_in=8, d_sae=16, activation="relu")


# -- data -------------------------------------------------------------------------------
def test_loader_visits_every_row_once(tmp_path):
    acts = torch.arange(2500, dtype=torch.float32)[:, None].repeat(1, 4)
    store = make_store(tmp_path / "s", acts, shard_rows=300)
    loader = ActivationLoader(store, "llm.0", 100, buffer_shards=3, drop_last=False)
    seen = torch.cat([b[:, 0] for b in loader])
    assert sorted(seen.long().tolist()) == list(range(2500))
    assert not torch.equal(seen, torch.arange(2500.0))  # shuffled
    epoch2 = torch.cat([b[:, 0] for b in loader])
    assert not torch.equal(seen, epoch2)  # reshuffled
    assert len(loader) == 25


def test_loader_row_filter(tmp_path):
    is_image = torch.arange(100) % 4 == 0
    store = make_store(tmp_path / "s", torch.randn(100, 4), shard_rows=30, is_image=is_image)
    n_img = sum(len(b) for b in ActivationLoader(store, "llm.0", 7, rows="image", drop_last=False))
    n_txt = sum(len(b) for b in ActivationLoader(store, "llm.0", 7, rows="text", drop_last=False))
    assert (n_img, n_txt) == (25, 75)


def test_estimate_norm_scale(tmp_path):
    store = make_store(tmp_path / "s", 10 * torch.randn(5000, D_IN))
    s = estimate_norm_scale(store, "llm.0")
    x = store.load("llm.0").acts
    assert abs((s * x).norm(dim=-1).mean().item() - D_IN**0.5) < 1e-3


# -- training ---------------------------------------------------------------------------
FAST = dict(batch_size=256, lr=3e-3, warmup_steps=20, log_every=50, device="cpu")


def test_topk_training_recovers_sparse_structure(tmp_path):
    store = make_store(tmp_path / "s", synthetic(20_000))
    result = train_sae(
        store,
        "llm.0",
        SAEConfig(d_in=D_IN, d_sae=64, k=4),
        TrainConfig(total_tokens=256 * 600, dead_after_tokens=256 * 50, **FAST),
        save_dir=tmp_path / "sae",
    )
    first, last = result.history[0], result.history[-1]
    assert last["mse"] < 0.3 * first["mse"]
    metrics = evaluate(result.sae, ActivationLoader(store, "llm.0", 1000))
    # 48 features in 16 dims interfere heavily; a linear encoder plateaus around 0.85.
    # PCA with the same budget of 4 directions explains ~0.46.
    x = store.load("llm.0").acts
    s = torch.linalg.svdvals(x - x.mean(0))
    assert metrics["explained_variance"] > 0.8, metrics
    assert metrics["explained_variance"] > (s[:4] ** 2).sum() / (s**2).sum() + 0.3
    assert metrics["l0"] <= 4
    assert result.sae.cfg.metadata["point"] == "llm.0"
    assert (tmp_path / "sae" / "history.jsonl").exists()
    reloaded = SAE.load(tmp_path / "sae")
    x = store.load("llm.0").acts[:10]
    torch.testing.assert_close(reloaded.reconstruct(x), result.sae.reconstruct(x))


def test_jumprelu_training_becomes_sparse(tmp_path):
    store = make_store(tmp_path / "s", synthetic(20_000))
    result = train_sae(
        store,
        "llm.0",
        SAEConfig(
            d_in=D_IN,
            d_sae=64,
            activation="jumprelu",
            jumprelu_bandwidth=0.05,
            jumprelu_init_threshold=0.05,
        ),
        TrainConfig(total_tokens=256 * 600, l0_coef=0.05, sparsity_warmup_steps=100, **FAST),
    )
    first, last = result.history[0], result.history[-1]
    assert last["l0"] < 0.5 * first["l0"]
    metrics = evaluate(result.sae, ActivationLoader(store, "llm.0", 1000))
    assert metrics["explained_variance"] > 0.8, metrics
    assert result.sae.threshold.min() > 0


def test_aux_loss_revives_dead_latents(tmp_path):
    x = synthetic(10_000)
    store = make_store(tmp_path / "s", x)

    def half_dead() -> SAE:
        torch.manual_seed(1)
        sae = SAE(SAEConfig(d_in=D_IN, d_sae=64, k=4))
        with torch.no_grad():
            sae.input_scale.fill_(estimate_norm_scale(store, "llm.0"))
            sae.b_dec.copy_((x * sae.input_scale).mean(0))
            sae.b_enc[32:] = -20.0  # these never reach the top k on any input
        return sae

    def train(coef: float) -> float:
        cfg = TrainConfig(
            total_tokens=256 * 1500, dead_after_tokens=256 * 20, aux_k_coef=coef, **FAST
        )
        result = train_sae(store, "llm.0", train_cfg=cfg, sae=half_dead())
        return evaluate(result.sae, ActivationLoader(store, "llm.0", 1000))["dead_fraction"]

    assert train(0.0) == pytest.approx(0.5)
    assert train(1 / 32) < 0.25


def test_train_requires_matching_width(tmp_path):
    store = make_store(tmp_path / "s", torch.randn(100, 4))
    with pytest.raises(ValueError, match="width"):
        train_sae(store, "llm.0", SAEConfig(d_in=8, d_sae=16, k=4))
    with pytest.raises(ValueError, match="d_sae"):
        train_sae(store, "llm.0")


def test_lr_schedule():
    from sextante.sae.train import _lr_factor

    cfg = TrainConfig(total_tokens=100, batch_size=1, warmup_steps=10, decay_fraction=0.2)
    factors = [_lr_factor(s, cfg) for s in range(100)]
    assert factors[0] == pytest.approx(0.1) and factors[9] == 1.0
    assert factors[50] == 1.0 and factors[99] == pytest.approx(1 / 20)


# -- splice ---------------------------------------------------------------------------
def identity_sae(d: int) -> SAE:
    """relu(x) - relu(-x) = x: an SAE that reconstructs perfectly."""
    sae = SAE(SAEConfig(d_in=d, d_sae=2 * d, k=2 * d, normalize_decoder=False))
    eye = torch.eye(d)
    with torch.no_grad():
        sae.W_enc.copy_(torch.cat([eye, -eye], dim=1))
        sae.W_dec.copy_(torch.cat([eye, -eye], dim=0))
    return sae


def test_spliced_loss_identity_sae(tiny_gemma3):
    batches = [gemma3_batch([[5, 6, 7, 8], [9, 10, 11]])]
    m = spliced_loss(tiny_gemma3, identity_sae(32), "llm.1", batches)
    assert m["ce_sae"] == pytest.approx(m["ce_clean"], abs=1e-4)
    assert m["loss_recovered"] == pytest.approx(1.0, abs=1e-3)
    assert m["ce_zero"] != pytest.approx(m["ce_clean"])
    # text tokens of two samples: [EOI->t0 ... ] excludes image-token targets
    assert m["n_tokens"] > 0


def test_spliced_loss_random_sae_is_worse(tiny_gemma3):
    batches = [gemma3_batch([[5, 6, 7, 8]])]
    torch.manual_seed(0)
    m = spliced_loss(tiny_gemma3, SAE(SAEConfig(d_in=32, d_sae=64, k=2)), "llm.1", batches)
    assert m["ce_sae"] > m["ce_clean"]


def test_spliced_loss_positions(tiny_gemma3):
    batches = [gemma3_batch([[5, 6, 7, 8]])]
    torch.manual_seed(0)
    bad = SAE(SAEConfig(d_in=32, d_sae=64, k=2))
    text = spliced_loss(tiny_gemma3, bad, "llm.1", batches, positions="text")
    image = spliced_loss(tiny_gemma3, bad, "llm.1", batches, positions="image")
    assert text["ce_clean"] == pytest.approx(image["ce_clean"])
    assert text["ce_sae"] != pytest.approx(image["ce_sae"])


def test_end_to_end_extract_then_train(tmp_path, tiny_gemma3):
    batches = [gemma3_batch([[5, 6, 7, 8]] * 8, seed=i) for i in range(4)]
    store = extract(tiny_gemma3, batches, tmp_path / "acts", ["llm.1"], dtype=torch.float32)
    result = train_sae(
        store,
        "llm.1",
        SAEConfig(d_in=32, d_sae=64, k=8),
        TrainConfig(total_tokens=64 * 100, batch_size=64, lr=1e-3, warmup_steps=5, device="cpu"),
    )
    assert result.sae.cfg.metadata["adapter"] == "gemma3"
    m = spliced_loss(tiny_gemma3, result.sae, "llm.1", batches[:1])
    assert m["ce_sae"] < m["ce_zero"]
