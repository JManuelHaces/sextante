"""Training loop for TopK and JumpReLU SAEs.

Losses (computed on normalised inputs ``s * x``, summed over features, averaged over rows):

* TopK:     ``mse + aux_k_coef * aux_k``. The auxiliary loss (Gao et al., 2024) asks the
  ``aux_k`` largest *dead* latents to reconstruct the remaining error, which revives them.
* JumpReLU: ``mse + l0_coef * L0``, with ``L0`` made differentiable by a straight-through
  estimator and ``l0_coef`` warmed up linearly from 0.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..activations import ActivationStore
from ..utils import seed_everything
from .data import ActivationLoader, RowFilter, estimate_norm_scale
from .model import SAE, SAEConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Optimisation settings.

    Args:
        total_tokens: Rows to train on (``steps = total_tokens // batch_size``).
        rows: Train on ``"all"`` rows, only ``"image"`` or only ``"text"`` rows.
        normalize_inputs: Estimate ``input_scale`` so inputs have mean norm ``sqrt(d_in)``.
        aux_k: Dead latents used by the TopK auxiliary loss (default ``d_in // 2``).
        dead_after_tokens: A latent that has not fired for this many rows is dead.
        l0_coef: JumpReLU sparsity coefficient. Depends on data; sweep it.
        sparsity_warmup_steps: Steps to ramp ``l0_coef`` from 0.
        decay_fraction: Final fraction of steps over which the LR decays linearly to 0.
    """

    total_tokens: int = 50_000_000
    batch_size: int = 4096
    lr: float = 2e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 1000
    decay_fraction: float = 0.2
    max_grad_norm: float | None = 1.0
    rows: RowFilter = "all"
    normalize_inputs: bool = True
    init_b_dec_to_mean: bool = True
    aux_k: int | None = None
    aux_k_coef: float = 1 / 32
    dead_after_tokens: int = 10_000_000
    l0_coef: float = 1.0
    sparsity_warmup_steps: int = 1000
    buffer_shards: int = 8
    device: str = "auto"
    seed: int = 0
    log_every: int = 100

    @property
    def steps(self) -> int:
        return max(1, self.total_tokens // self.batch_size)


@dataclass
class TrainResult:
    sae: SAE
    history: list[dict[str, float]] = field(default_factory=list)
    tokens_since_fired: Tensor | None = None

    @property
    def dead_fraction(self) -> float:
        return float(self.history[-1]["dead_fraction"]) if self.history else float("nan")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    decay_start = int(cfg.steps * (1 - cfg.decay_fraction))
    if cfg.decay_fraction > 0 and step >= decay_start:
        return max(0.0, (cfg.steps - step) / max(1, cfg.steps - decay_start))
    return 1.0


def _explained_variance(x: Tensor, x_hat: Tensor) -> Tensor:
    resid = (x - x_hat).pow(2).sum()
    total = (x - x.mean(0)).pow(2).sum().clamp_min(1e-12)
    return 1 - resid / total


def aux_k_loss(sae: SAE, x_scaled: Tensor, residual: Tensor, dead: Tensor, aux_k: int) -> Tensor:
    """Normalised MSE of reconstructing ``residual`` with the top ``aux_k`` dead latents.

    Two details matter for actually reviving latents:

    * No ReLU on the selected pre-activations. A truly dead latent has negative
      pre-activations on every input; with a ReLU it would get no gradient at all.
    * ``b_dec`` is detached, so the auxiliary objective only moves the dead latents'
      own encoder/decoder weights and not the bias shared by every latent.

    Only the dead columns of ``W_enc`` are used, so the cost scales with the number of
    dead latents rather than ``d_sae``.
    """
    dead_idx = dead.nonzero(as_tuple=True)[0]
    if dead_idx.numel() == 0:
        return x_scaled.new_zeros(())
    pre = (x_scaled - sae.b_dec.detach()) @ sae.W_enc[:, dead_idx] + sae.b_enc[dead_idx]
    values, local = pre.topk(min(aux_k, dead_idx.numel()), dim=-1)
    latents = torch.zeros_like(pre).scatter(-1, local, values)  # [n, n_dead]
    recon = latents @ sae.W_dec[dead_idx]
    num = (recon - residual).pow(2).sum(-1).mean()
    den = residual.pow(2).sum(-1).mean().clamp_min(1e-12)
    return num / den


def train_sae(
    store: ActivationStore | str | Path,
    point: str,
    sae_cfg: SAEConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    d_sae: int | None = None,
    sae: SAE | None = None,
    save_dir: str | Path | None = None,
    callback: Callable[[int, dict[str, float], SAE], None] | None = None,
) -> TrainResult:
    """Train an SAE on hook point ``point`` of ``store``.

    Either pass a full ``sae_cfg``, just ``d_sae`` (TopK with default ``k``), or an
    existing ``sae`` to keep training (its ``input_scale`` and ``b_dec`` are kept as they
    are). The store's provenance (model, adapter, point) is copied into the metadata.
    """
    train_cfg = train_cfg or TrainConfig()
    if not isinstance(store, ActivationStore):
        store = ActivationStore(store)
    d_in = store.d(point)
    resume = sae is not None
    if resume:
        sae_cfg = sae.cfg
    elif sae_cfg is None:
        if d_sae is None:
            raise ValueError("Pass sae_cfg or d_sae")
        sae_cfg = SAEConfig(d_in=d_in, d_sae=d_sae)
    if sae_cfg.d_in != d_in:
        raise ValueError(f"sae_cfg.d_in={sae_cfg.d_in} but {point!r} has width {d_in}")
    sae_cfg.metadata = {
        "point": point,
        "store": str(store.path),
        "rows": train_cfg.rows,
        **{k: store.info.get(k) for k in ("model", "adapter", "tokens") if k in store.info},
        **sae_cfg.metadata,
    }

    seed_everything(train_cfg.seed)
    device = _resolve_device(train_cfg.device)
    sae = (sae if resume else SAE(sae_cfg)).to(device)
    sae.train()
    loader = ActivationLoader(
        store,
        point,
        train_cfg.batch_size,
        rows=train_cfg.rows,
        buffer_shards=train_cfg.buffer_shards,
        device=device,
        seed=train_cfg.seed,
    )
    batches: Iterator[Tensor] = loader.forever()

    if not resume:
        with torch.no_grad():
            if train_cfg.normalize_inputs:
                sae.input_scale.fill_(estimate_norm_scale(store, point, rows=train_cfg.rows))
            if train_cfg.init_b_dec_to_mean:
                sae.b_dec.copy_((next(batches) * sae.input_scale).mean(0))

    optimizer = torch.optim.Adam(sae.parameters(), lr=train_cfg.lr, betas=train_cfg.adam_betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_factor(s, train_cfg))
    aux_k = train_cfg.aux_k or max(1, d_in // 2)
    tokens_since_fired = torch.zeros(sae_cfg.d_sae, dtype=torch.long, device=device)
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for step in range(train_cfg.steps):
        x = next(batches)
        out = sae(x)
        mse = (out.x_hat_scaled - out.x_scaled).pow(2).sum(-1).mean()
        dead = tokens_since_fired >= train_cfg.dead_after_tokens

        if sae_cfg.activation == "topk":
            residual = (out.x_scaled - out.x_hat_scaled).detach()
            aux = aux_k_loss(sae, out.x_scaled.detach(), residual, dead, aux_k)
            loss = mse + train_cfg.aux_k_coef * aux
            sparsity = aux
        else:
            warm = min(1.0, (step + 1) / max(1, train_cfg.sparsity_warmup_steps))
            l0_term = sae.l0_surrogate(out.pre).mean()
            loss = mse + train_cfg.l0_coef * warm * l0_term
            sparsity = l0_term

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if sae_cfg.normalize_decoder:
            sae.remove_parallel_decoder_grad_()
        if train_cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if sae_cfg.normalize_decoder:
            sae.normalize_decoder_()

        with torch.no_grad():
            fired = (out.latents > 0).any(0)
            tokens_since_fired += x.shape[0]
            tokens_since_fired[fired] = 0

        last = step == train_cfg.steps - 1
        if (train_cfg.log_every and step % train_cfg.log_every == 0) or last:
            with torch.no_grad():
                record = {
                    "step": step,
                    "tokens": (step + 1) * train_cfg.batch_size,
                    "loss": loss.item(),
                    "mse": mse.item(),
                    "explained_variance": _explained_variance(
                        out.x_scaled, out.x_hat_scaled
                    ).item(),
                    "l0": (out.latents > 0).float().sum(-1).mean().item(),
                    "dead_fraction": dead.float().mean().item(),
                    ("aux_loss" if sae_cfg.activation == "topk" else "l0_loss"): sparsity.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.perf_counter() - t0,
                }
            history.append(record)
            logger.info(
                "step %d/%d  mse %.4g  ev %.3f  l0 %.1f  dead %.3f",
                step,
                train_cfg.steps,
                record["mse"],
                record["explained_variance"],
                record["l0"],
                record["dead_fraction"],
            )
            if callback is not None:
                callback(step, record, sae)

    sae.eval()
    result = TrainResult(sae=sae, history=history, tokens_since_fired=tokens_since_fired.cpu())
    if save_dir is not None:
        save_training(result, train_cfg, save_dir)
    return result


def save_training(result: TrainResult, train_cfg: TrainConfig, path: str | Path) -> Path:
    """Save the SAE plus ``train_cfg.json`` and ``history.jsonl`` into ``path``."""
    path = result.sae.save(path)
    (path / "train_cfg.json").write_text(json.dumps(asdict(train_cfg), indent=2))
    with open(path / "history.jsonl", "w") as fh:
        for record in result.history:
            fh.write(json.dumps({k: _jsonable(v) for k, v in record.items()}) + "\n")
    return path


def _jsonable(v: Any) -> Any:
    return None if isinstance(v, float) and not math.isfinite(v) else v
