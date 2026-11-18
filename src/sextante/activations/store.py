"""On-disk activation store.

Layout::

    store/
      manifest.json             format version, model info, per-point shapes and shards
      samples.jsonl             one JSON object per sample (image id, report, language, ...)
      acts/<point>/shard_00000.safetensors

Every shard holds row-aligned tensors:

``acts``        ``[N, d]`` activations (stored in ``dtype``, float16 by default)
``sample_idx``  ``[N]`` int64, index into ``samples.jsonl``
``position``    ``[N]`` int64, token position (ignoring padding) or image patch index
``is_image``    ``[N]`` bool, whether the row is an image token / patch
``token_id``    ``[N]`` int64, input token id for LLM points, ``-1`` for vision/projector

Shards are loaded with safetensors, which memory-maps the file, so reading a single point
of a large store does not pull everything into RAM.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

FORMAT_VERSION = 1
_COLUMNS = ("acts", "sample_idx", "position", "is_image", "token_id")


@dataclass
class ActivationRows:
    """A block of activation rows with their per-row metadata."""

    acts: Tensor
    sample_idx: Tensor
    position: Tensor
    is_image: Tensor
    token_id: Tensor

    def __post_init__(self) -> None:
        n = self.acts.shape[0]
        for name in _COLUMNS[1:]:
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {getattr(self, name).shape}")

    def __len__(self) -> int:
        return self.acts.shape[0]

    @property
    def d(self) -> int:
        return self.acts.shape[1]

    def to_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).contiguous() for name in _COLUMNS}

    def select(self, mask_or_index: Tensor) -> ActivationRows:
        return ActivationRows(**{k: v[mask_or_index] for k, v in self.to_dict().items()})

    @classmethod
    def cat(cls, blocks: Iterable[ActivationRows]) -> ActivationRows:
        blocks = list(blocks)
        if not blocks:
            raise ValueError("Cannot concatenate zero blocks")
        return cls(**{k: torch.cat([getattr(b, k) for b in blocks]) for k in _COLUMNS})


def _point_dir(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


class ActivationWriter:
    """Streams activation rows to a store, one shard file per ``shard_rows`` rows.

    Use as a context manager; the manifest is rewritten after each shard so an
    interrupted run leaves a readable (``complete: false``) store behind.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dtype: torch.dtype = torch.float16,
        shard_rows: int = 65_536,
        overwrite: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.path} exists and is not empty (overwrite=False)")
            shutil.rmtree(self.path)
        (self.path / "acts").mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.shard_rows = shard_rows
        self._samples_fh = open(self.path / "samples.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        self._n_samples = 0
        self._buffers: dict[str, list[ActivationRows]] = {}
        self._buffered: dict[str, int] = {}
        self._manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "complete": False,
            "dtype": str(dtype).removeprefix("torch."),
            "n_samples": 0,
            "info": dict(info or {}),
            "points": {},
        }
        self._closed = False

    # -- samples ----------------------------------------------------------------------
    def add_samples(self, metas: Iterable[Mapping[str, Any]]) -> int:
        """Append per-sample metadata. Returns the global index of the first sample."""
        start = self._n_samples
        for meta in metas:
            self._samples_fh.write(json.dumps(dict(meta), ensure_ascii=False, default=str) + "\n")
            self._n_samples += 1
        return start

    # -- rows -------------------------------------------------------------------------
    def add(
        self, point: str, rows: ActivationRows, *, kind: str | None = None, layer: int | None = None
    ) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed")
        entry = self._manifest["points"].get(point)
        if entry is None:
            entry = {
                "kind": kind,
                "layer": layer,
                "d": rows.d,
                "n_rows": 0,
                "dir": f"acts/{_point_dir(point)}",
                "shards": [],
            }
            (self.path / entry["dir"]).mkdir(parents=True, exist_ok=True)
            self._manifest["points"][point] = entry
        elif rows.d != entry["d"]:
            raise ValueError(f"{point}: width {rows.d} != {entry['d']} seen before")
        if len(rows) == 0:
            return
        rows = ActivationRows(
            acts=rows.acts.detach().to("cpu", self.dtype),
            sample_idx=rows.sample_idx.to("cpu", torch.int64),
            position=rows.position.to("cpu", torch.int64),
            is_image=rows.is_image.to("cpu", torch.bool),
            token_id=rows.token_id.to("cpu", torch.int64),
        )
        self._buffers.setdefault(point, []).append(rows)
        self._buffered[point] = self._buffered.get(point, 0) + len(rows)
        while self._buffered[point] >= self.shard_rows:
            self._flush(point, self.shard_rows)

    def _flush(self, point: str, n: int | None = None) -> None:
        blocks = self._buffers.get(point)
        if not blocks:
            return
        merged = ActivationRows.cat(blocks)
        n = len(merged) if n is None else n
        head, tail = merged.select(slice(0, n)), merged.select(slice(n, None))
        entry = self._manifest["points"][point]
        shard_name = f"shard_{len(entry['shards']):05d}.safetensors"
        save_file(head.to_dict(), str(self.path / entry["dir"] / shard_name))
        entry["shards"].append({"file": shard_name, "n_rows": len(head)})
        entry["n_rows"] += len(head)
        self._buffers[point] = [tail] if len(tail) else []
        self._buffered[point] = len(tail)
        self._write_manifest()

    def _write_manifest(self) -> None:
        self._samples_fh.flush()
        self._manifest["n_samples"] = self._n_samples
        tmp = self.path / "manifest.json.tmp"
        tmp.write_text(json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path / "manifest.json")

    def close(self) -> None:
        if self._closed:
            return
        for point in list(self._buffers):
            self._flush(point)
        self._manifest["complete"] = True
        self._write_manifest()
        self._samples_fh.close()
        self._closed = True

    def __enter__(self) -> ActivationWriter:
        return self

    def __exit__(self, exc_type, *_: Any) -> None:
        if exc_type is None:
            self.close()
        else:  # keep what was flushed, mark incomplete
            self._write_manifest()
            self._samples_fh.close()
            self._closed = True


class ActivationStore:
    """Read-only view of a store written by :class:`ActivationWriter`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {self.path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported store format {self.manifest.get('format_version')}")
        self._samples: list[dict[str, Any]] | None = None

    @property
    def complete(self) -> bool:
        return bool(self.manifest["complete"])

    @property
    def info(self) -> dict[str, Any]:
        return self.manifest["info"]

    @property
    def points(self) -> list[str]:
        return list(self.manifest["points"])

    @property
    def samples(self) -> list[dict[str, Any]]:
        if self._samples is None:
            with open(self.path / "samples.jsonl", encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
            self._samples = lines[: self.manifest["n_samples"]]
        return self._samples

    def point_info(self, point: str) -> dict[str, Any]:
        try:
            return self.manifest["points"][point]
        except KeyError:
            raise KeyError(f"{point!r} not in store. Available: {self.points}") from None

    def n_rows(self, point: str) -> int:
        return self.point_info(point)["n_rows"]

    def d(self, point: str) -> int:
        return self.point_info(point)["d"]

    def shard_paths(self, point: str) -> list[Path]:
        entry = self.point_info(point)
        return [self.path / entry["dir"] / s["file"] for s in entry["shards"]]

    def iter_shards(self, point: str) -> Iterator[ActivationRows]:
        for shard in self.shard_paths(point):
            yield ActivationRows(**load_file(str(shard)))

    def load(self, point: str) -> ActivationRows:
        """Load every row of ``point`` into memory."""
        shards = list(self.iter_shards(point))
        if not shards:
            raise ValueError(f"{point!r} has no rows")
        return ActivationRows.cat(shards)

    def __repr__(self) -> str:
        pts = ", ".join(f"{p}[{self.n_rows(p)}x{self.d(p)}]" for p in self.points)
        return f"ActivationStore({str(self.path)!r}, samples={self.manifest['n_samples']}, {pts})"
