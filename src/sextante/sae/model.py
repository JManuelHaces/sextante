"""Sparse autoencoders: TopK and JumpReLU.

Both share one parametrisation (names and shapes follow SAELens/GemmaScope)::

    pre    = (s * x - b_dec) @ W_enc + b_enc        # W_enc: [d_in, d_sae]
    f      = activation(pre)                         # TopK or JumpReLU
    x_hat  = (f @ W_dec + b_dec) / s                 # W_dec: [d_sae, d_in]

``s`` is ``input_scale``, a fixed scalar chosen before training so that inputs have mean
norm ``sqrt(d_in)``. It lives in the SAE so the model is applied to raw activations;
:meth:`SAE.fold_input_scale` folds it into the weights if you need ``s = 1``.

TopK keeps the ``k`` largest pre-activations (Gao et al., 2024). JumpReLU zeroes
pre-activations below a learned per-latent threshold and is trained with straight-through
estimators for the threshold (Rajamanoharan et al., 2024, as used in GemmaScope).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

Activation = Literal["topk", "jumprelu"]
WEIGHTS_FILE = "sae_weights.safetensors"
CONFIG_FILE = "cfg.json"


@dataclass
class SAEConfig:
    """Architecture of a sparse autoencoder.

    Args:
        d_in: Width of the activations (e.g. 2560 for MedGemma 4B's LLM).
        d_sae: Number of latents (dictionary size).
        activation: ``"topk"`` or ``"jumprelu"``.
        k: Active latents per row for TopK.
        jumprelu_init_threshold: Initial threshold for JumpReLU (in normalised units).
        jumprelu_bandwidth: Width of the rectangle kernel of the straight-through
            estimator for JumpReLU thresholds.
        normalize_decoder: Keep decoder rows at unit norm during training.
        metadata: Free-form provenance (model, hook point, store, ...), saved in cfg.json.
    """

    d_in: int
    d_sae: int
    activation: Activation = "topk"
    k: int = 64
    jumprelu_init_threshold: float = 0.001
    jumprelu_bandwidth: float = 0.001
    normalize_decoder: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.activation not in ("topk", "jumprelu"):
            raise ValueError(f"activation must be 'topk' or 'jumprelu', got {self.activation!r}")
        if self.activation == "topk" and not 0 < self.k <= self.d_sae:
            raise ValueError(f"k must be in [1, d_sae={self.d_sae}], got {self.k}")


# -- straight-through estimators for JumpReLU -------------------------------------------
def _rectangle(x: Tensor) -> Tensor:
    return (x.abs() < 0.5).to(x.dtype)


def _sum_to_threshold(grad: Tensor, threshold: Tensor) -> Tensor:
    return grad.reshape(-1, threshold.shape[0]).sum(0)


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, threshold = ctx.saved_tensors
        eps = ctx.bandwidth
        grad_x = grad * (x > threshold).to(x.dtype)
        grad_t = -(threshold / eps) * _rectangle((x - threshold) / eps) * grad
        return grad_x, _sum_to_threshold(grad_t, threshold), None


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, threshold = ctx.saved_tensors
        eps = ctx.bandwidth
        grad_t = -(1.0 / eps) * _rectangle((x - threshold) / eps) * grad
        return torch.zeros_like(x), _sum_to_threshold(grad_t, threshold), None


@dataclass
class SAEOutput:
    x_hat: Tensor  # reconstruction, in the input's (raw) units
    latents: Tensor  # [..., d_sae]
    pre: Tensor  # pre-activations, [..., d_sae]
    x_scaled: Tensor  # input times input_scale
    x_hat_scaled: Tensor  # reconstruction before dividing by input_scale


class SAE(nn.Module):
    """A sparse autoencoder over activations of width ``d_in``."""

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_in, d_sae = cfg.d_in, cfg.d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if cfg.activation == "jumprelu":
            self.log_threshold = nn.Parameter(
                torch.full((d_sae,), math.log(cfg.jumprelu_init_threshold))
            )
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Random unit-norm decoder rows, encoder tied to the decoder (W_enc = W_dec.T)."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_dec)
            self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)
            self.W_enc.copy_(self.W_dec.T)
            self.b_enc.zero_()
            self.b_dec.zero_()

    # -- properties -------------------------------------------------------------------
    @property
    def d_in(self) -> int:
        return self.cfg.d_in

    @property
    def d_sae(self) -> int:
        return self.cfg.d_sae

    @property
    def threshold(self) -> Tensor:
        if self.cfg.activation != "jumprelu":
            raise AttributeError("Only JumpReLU SAEs have a threshold")
        return self.log_threshold.exp()

    # -- forward ----------------------------------------------------------------------
    def pre_activations(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(pre, x_scaled)`` for raw input ``x``."""
        x_scaled = x.to(self.W_enc.device, self.W_enc.dtype) * self.input_scale
        return (x_scaled - self.b_dec) @ self.W_enc + self.b_enc, x_scaled

    def activate(self, pre: Tensor) -> Tensor:
        if self.cfg.activation == "topk":
            values, indices = pre.topk(self.cfg.k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, indices, values.relu())
        return _JumpReLU.apply(pre, self.threshold, self.cfg.jumprelu_bandwidth)

    def l0_surrogate(self, pre: Tensor) -> Tensor:
        """Differentiable (via straight-through) count of active latents per row."""
        return _Step.apply(pre, self.threshold, self.cfg.jumprelu_bandwidth).sum(-1)

    def encode(self, x: Tensor) -> Tensor:
        return self.activate(self.pre_activations(x)[0])

    def decode(self, latents: Tensor) -> Tensor:
        return (latents @ self.W_dec + self.b_dec) / self.input_scale

    def forward(self, x: Tensor) -> SAEOutput:
        pre, x_scaled = self.pre_activations(x)
        latents = self.activate(pre)
        x_hat_scaled = latents @ self.W_dec + self.b_dec
        return SAEOutput(
            x_hat=x_hat_scaled / self.input_scale,
            latents=latents,
            pre=pre,
            x_scaled=x_scaled,
            x_hat_scaled=x_hat_scaled,
        )

    def reconstruct(self, x: Tensor) -> Tensor:
        """``x_hat`` with the same shape, dtype and device as ``x`` (any leading dims)."""
        flat = x.reshape(-1, x.shape[-1])
        return self(flat).x_hat.to(x.device, x.dtype).reshape(x.shape)

    # -- decoder norm constraint ------------------------------------------------------
    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_decoder_grad_(self) -> None:
        """Project out the gradient component that would change decoder row norms."""
        if self.W_dec.grad is None:
            return
        rows = self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        parallel = (self.W_dec.grad * rows).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel * rows

    @torch.no_grad()
    def fold_input_scale(self) -> None:
        """Fold ``input_scale`` into the weights so the SAE computes the same function
        with ``input_scale = 1`` (decoder rows then have norm ``1 / s``)."""
        s = self.input_scale.item()
        self.W_enc *= s
        self.b_dec /= s
        self.W_dec /= s
        self.input_scale.fill_(1.0)

    # -- persistence ------------------------------------------------------------------
    def state_for_saving(self) -> dict[str, Tensor]:
        tensors = {
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "W_dec": self.W_dec,
            "b_dec": self.b_dec,
            "input_scale": self.input_scale,
        }
        if self.cfg.activation == "jumprelu":
            tensors["threshold"] = self.threshold
        return {k: v.detach().float().cpu().contiguous() for k, v in tensors.items()}

    def save(self, path: str | Path) -> Path:
        """Write ``cfg.json`` and ``sae_weights.safetensors`` into directory ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(self.state_for_saving(), str(path / WEIGHTS_FILE))
        cfg = asdict(self.cfg) | {"architecture": self.cfg.activation, "format": "sextante-sae/1"}
        (path / CONFIG_FILE).write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> SAE:
        path = Path(path)
        raw = json.loads((path / CONFIG_FILE).read_text())
        fields = SAEConfig.__dataclass_fields__
        sae = cls(SAEConfig(**{k: v for k, v in raw.items() if k in fields}))
        sae.load_weights(load_file(str(path / WEIGHTS_FILE)))
        return sae.to(device)

    @torch.no_grad()
    def load_weights(
        self, tensors: dict[str, Tensor], *, encoder_subtracts_b_dec: bool = True
    ) -> None:
        """Load ``W_enc, b_enc, W_dec, b_dec`` (+ ``threshold``, ``input_scale``).

        Uses the tensor names of SAELens and of GemmaScope's ``params.npz``. sextante's
        encoder computes ``(x - b_dec) @ W_enc + b_enc``. For weights trained with an
        encoder of the form ``x @ W_enc + b_enc`` (GemmaScope, or SAELens with
        ``apply_b_dec_to_input=False``), pass ``encoder_subtracts_b_dec=False`` and
        ``b_enc`` is adjusted so the SAE computes the same function.
        """
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            value = torch.as_tensor(tensors[name]).to(getattr(self, name).device)
            getattr(self, name).copy_(value)
        if not encoder_subtracts_b_dec:
            self.b_enc += self.b_dec @ self.W_enc
        if "input_scale" in tensors:
            self.input_scale.copy_(torch.as_tensor(tensors["input_scale"]).reshape(()))
        if self.cfg.activation == "jumprelu":
            if "threshold" not in tensors:
                raise KeyError("JumpReLU weights need a 'threshold' tensor")
            threshold = torch.as_tensor(tensors["threshold"]).to(self.log_threshold.device)
            self.log_threshold.copy_(threshold.clamp_min(1e-12).log())

    def __repr__(self) -> str:
        extra = f"k={self.cfg.k}" if self.cfg.activation == "topk" else "jumprelu"
        return f"SAE(d_in={self.d_in}, d_sae={self.d_sae}, {extra})"
