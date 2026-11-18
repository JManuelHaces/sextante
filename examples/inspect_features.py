"""Scan an SAE over a store, rank features and write an HTML page of top examples.

Usage::

    uv run python examples/inspect_features.py out/medgemma-cxr out/sae-llm17 out/inspect \
        --tokenizer google/medgemma-4b-it --image-root data \
        --nuisance hospital view --targets effusion cardiomegaly --top 40

    # add LLM explanations + detection scores (Claude API, costs money)
    uv run --extra claude python examples/inspect_features.py ... --autointerp

Writes ``scan/`` (reusable), ``features.csv`` (one row per feature) and ``index.html``.
Samples need ``image`` (path relative to --image-root) in their metadata for heatmaps.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image

import sextante
from sextante.features import (
    claude_llm,
    feature_table,
    heatmap_overlay,
    interpret_feature,
    patch_grid,
    sample_activations,
    scan_features,
    token_activations,
    tokens_html,
)
from sextante.sae import SAE


def png_data_uri(image: Image.Image, size: int = 256) -> str:
    image = image.copy()
    image.thumbnail((size, size))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("store")
    parser.add_argument("sae")
    parser.add_argument("out", type=Path)
    parser.add_argument("--tokenizer", default=None, help="HF id/path to decode token ids")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--nuisance", nargs="*", default=[])
    parser.add_argument("--targets", nargs="*", default=[])
    parser.add_argument("--top", type=int, default=30, help="features on the HTML page")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--rank-by", default="frequency", help="feature_table column to sort by")
    parser.add_argument("--autointerp", action="store_true")
    parser.add_argument("--modality", default="chest X-ray")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sextante.get_logger()
    store = sextante.ActivationStore(args.store)
    sae = SAE.load(args.sae, device=args.device).eval()
    point = sae.cfg.metadata["point"]
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    args.out.mkdir(parents=True, exist_ok=True)
    scan = scan_features(store, sae, rows=sae.cfg.metadata.get("rows", "all"))
    scan.save(args.out / "scan")
    table = feature_table(scan, store.samples, nuisance=args.nuisance, targets=args.targets)
    with open(args.out / "features.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    alive = [row for row in table if row["label"] != "dead"]
    chosen = sorted(alive, key=lambda r: -float(r[args.rank_by]))[: args.top]
    llm = claude_llm(effort="low") if args.autointerp else None
    meta_keys = [*args.nuisance, *args.targets]

    sections = []
    for row in chosen:
        f = row["feature"]
        values, samples = scan.top_samples(f, args.examples)
        vmax = float(values[0]) if len(values) else 1.0
        head = (
            f"<h2>Feature {f} <small>{html.escape(row['label'])}</small></h2>"
            f"<p>freq {row['frequency']:.2e} · samples {row['sample_frequency']:.1%} · "
            f"image share {row['image_fraction']:.0%}"
            + (f" · strongest: {html.escape(str(row['strongest']))}" if row["strongest"] else "")
            + "</p>"
        )
        if llm is not None:
            result = interpret_feature(
                store,
                sae,
                scan,
                f,
                llm,
                tokenizer=tokenizer,
                meta_keys=meta_keys,
                modality=args.modality,
            )
            score = result.get("detection", {}).get("balanced_accuracy")
            head += f"<p><b>{html.escape(result['explanation'] or '')}</b>"
            head += f" (detection {score:.2f})</p>" if score is not None else "</p>"
            row["explanation"] = result["explanation"]
        cards = []
        for value, s in zip(values.tolist(), samples.tolist(), strict=True):
            sa = sample_activations(store, sae, point, s, f)
            img_html = ""
            image_path = sa.meta.get("image")
            if args.image_root and image_path and sa.is_image.any():
                image = Image.open(args.image_root / image_path)
                overlay = heatmap_overlay(image, patch_grid(sa), vmax=vmax)
                img_html = f'<img src="{png_data_uri(overlay)}">'
            tokens = tokens_html(token_activations(sa, tokenizer), vmax=vmax)
            label = html.escape(json.dumps({k: sa.meta.get(k) for k in meta_keys}, default=str))
            cards.append(
                f'<div class="card">{img_html}<div><b>sample {s}</b> max {value:.3g}'
                f"<br><small>{label}</small><p>{tokens}</p></div></div>"
            )
        sections.append(head + "".join(cards))

    page = (
        "<!doctype html><meta charset='utf-8'><title>sextante features</title><style>"
        "body{font-family:system-ui;margin:24px;max-width:1100px}"
        ".card{display:flex;gap:12px;margin:8px 0;padding:8px;border:1px solid #ddd}"
        ".card img{width:256px;height:auto}</style>"
        f"<h1>{html.escape(point)} · {scan.d_sae} features · {scan.n_samples} samples</h1>"
        + "".join(sections)
    )
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
