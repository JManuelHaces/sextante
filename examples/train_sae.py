"""Train an SAE on one hook point of an activation store (run on a GPU box).

Usage::

    uv run python examples/train_sae.py out/medgemma-cxr llm.17 out/sae-llm17 \
        --d-sae 16384 --k 64 --tokens 50_000_000

    # JumpReLU instead of TopK (sweep --l0-coef; it depends on the data)
    uv run python examples/train_sae.py out/medgemma-cxr llm.17 out/sae-llm17-jr \
        --activation jumprelu --l0-coef 1.0 --lr 7e-5

Prints reconstruction metrics on a held-out slice of the store at the end.
"""

from __future__ import annotations

import argparse
import json

import sextante
from sextante.sae import ActivationLoader, SAEConfig, TrainConfig, evaluate, train_sae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("store")
    parser.add_argument("point")
    parser.add_argument("out")
    parser.add_argument("--d-sae", type=int, default=16384)
    parser.add_argument("--activation", choices=["topk", "jumprelu"], default="topk")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--l0-coef", type=float, default=1.0)
    parser.add_argument("--tokens", type=lambda s: int(s.replace("_", "")), default=50_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rows", choices=["all", "image", "text"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sextante.get_logger()
    store = sextante.ActivationStore(args.store)
    print(store)
    result = train_sae(
        store,
        args.point,
        SAEConfig(d_in=store.d(args.point), d_sae=args.d_sae, activation=args.activation, k=args.k),
        TrainConfig(
            total_tokens=args.tokens,
            batch_size=args.batch_size,
            lr=args.lr,
            rows=args.rows,
            l0_coef=args.l0_coef,
            seed=args.seed,
            log_every=500,
        ),
        save_dir=args.out,
    )
    loader = ActivationLoader(
        store, args.point, 8192, rows=args.rows, device=result.sae.W_enc.device, seed=1
    )
    metrics = evaluate(result.sae, loader, max_batches=50)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
