"""Corpus statistics: is a feature about the model's domain, or about the dataset?

Each sample has one value per feature (its max activation, from a :class:`FeatureScan`).
We measure how much of that value is explained by each metadata variable:

* **Correlation ratio** ``eta^2`` for categorical variables: between-group variance over
  total variance, in ``[0, 1]``.
* **AUROC** for binary variables: probability that a random positive sample activates
  more than a random negative one (0.5 = no association).

Splitting metadata into *nuisance* variables (hospital, scanner, view, acquisition year,
prompt template) and *target* variables (clinical findings, report labels) gives a
simple screen: a feature explained mostly by nuisance variables is a candidate corpus
artifact; one explained by targets is a candidate domain feature. It is a screen for
what to inspect, not a test, and confounded datasets (findings concentrated in one
hospital) will blur it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .scan import FeatureScan


def correlation_ratio(values: Tensor, groups: Tensor) -> Tensor:
    """``eta^2`` of each column of ``values [n, d]`` with integer ``groups [n]``."""
    values = values.double()
    n_groups = int(groups.max()) + 1
    counts = torch.bincount(groups, minlength=n_groups).double()
    sums = torch.zeros(n_groups, values.shape[1], dtype=torch.float64).index_add_(0, groups, values)
    means = sums / counts.clamp_min(1)[:, None]
    grand = values.mean(0)
    between = (counts[:, None] * (means - grand) ** 2).sum(0)
    total = ((values - grand) ** 2).sum(0)
    return torch.where(total > 0, between / total.clamp_min(1e-30), torch.zeros_like(total))


def auroc(values: Tensor, labels: Tensor) -> Tensor:
    """AUROC of each column of ``values [n, d]`` for boolean ``labels [n]`` (ties averaged)."""
    labels = labels.bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both positive and negative samples")
    v = values.double()
    order = v.argsort(dim=0)
    sorted_v = torch.gather(v, 0, order)
    ranks = torch.empty_like(v)
    base = torch.arange(1, v.shape[0] + 1, dtype=torch.float64)[:, None].expand_as(v)
    ranks.scatter_(0, order, base.contiguous())
    # average ranks over ties, column by column where ties exist
    for j in torch.nonzero((sorted_v[1:] == sorted_v[:-1]).any(0)).flatten().tolist():
        col = v[:, j]
        uniq, inverse = torch.unique(col, return_inverse=True)
        mean_rank = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inverse, ranks[:, j])
        mean_rank /= torch.bincount(inverse, minlength=len(uniq)).double()
        ranks[:, j] = mean_rank[inverse]
    rank_sum = ranks[labels].sum(0)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass
class Association:
    """Association of every feature with one metadata variable."""

    key: str
    categories: list[Any]
    counts: list[int]
    means: Tensor  # [n_categories, d_sae] mean per-sample max activation
    eta2: Tensor  # [d_sae]
    auroc: Tensor | None  # [d_sae] for binary variables (positive = categories[1])
    n_samples: int

    def top_category(self) -> list[Any]:
        """Category with the highest mean activation, per feature."""
        return [self.categories[i] for i in self.means.argmax(0).tolist()]


def _bool_like(categories: list[Any]) -> bool:
    return set(categories) <= {
        True,
        False,
        "0",
        "1",
        "true",
        "false",
        "yes",
        "no",
    }  # 0/1 == False/True


def metadata_association(
    scan: FeatureScan,
    samples: Sequence[Mapping[str, Any]],
    key: str,
    *,
    min_count: int = 5,
) -> Association:
    """``eta^2`` (and AUROC if binary) between each feature and ``samples[i][key]``.

    Samples missing the key and categories with fewer than ``min_count`` samples are
    ignored.
    """
    if len(samples) != scan.n_samples:
        raise ValueError(f"{len(samples)} samples but the scan has {scan.n_samples}")
    raw = [(i, s.get(key)) for i, s in enumerate(samples)]
    raw = [(i, v) for i, v in raw if v is not None and v == v]  # drop None / NaN
    counts: dict[Any, int] = {}
    for _, v in raw:
        counts[v] = counts.get(v, 0) + 1
    categories = sorted((c for c, n in counts.items() if n >= min_count), key=str)
    if len(categories) < 2:
        raise ValueError(f"{key!r} has fewer than 2 categories with >= {min_count} samples")
    index = {c: j for j, c in enumerate(categories)}
    kept = [(i, index[v]) for i, v in raw if v in index]
    rows = torch.tensor([i for i, _ in kept])
    groups = torch.tensor([g for _, g in kept])
    values = scan.sample_max[rows].double()

    n_cat = len(categories)
    sums = torch.zeros(n_cat, values.shape[1], dtype=torch.float64).index_add_(0, groups, values)
    group_counts = torch.bincount(groups, minlength=n_cat)
    means = (sums / group_counts[:, None]).float()

    binary_auroc = None
    if n_cat == 2:
        if _bool_like(categories):
            truthy = {True, "1", "true", "yes"}  # 1 == True
            positive = next(j for j, c in enumerate(categories) if c in truthy)
        else:
            positive = 1
        binary_auroc = auroc(values, groups == positive).float()
        if positive == 0:
            categories = categories[::-1]
            means = means.flip(0)
            group_counts = group_counts.flip(0)
    return Association(
        key=key,
        categories=categories,
        counts=group_counts.tolist(),
        means=means,
        eta2=correlation_ratio(values, groups).float(),
        auroc=binary_auroc,
        n_samples=len(kept),
    )


def feature_table(
    scan: FeatureScan,
    samples: Sequence[Mapping[str, Any]],
    *,
    nuisance: Sequence[str] = (),
    targets: Sequence[str] = (),
    min_eta2: float = 0.1,
    dense_threshold: float = 0.9,
    min_count: int = 5,
) -> list[dict[str, Any]]:
    """One row per feature with firing statistics, associations and a screening label.

    ``label`` is the first that applies:

    ``dead``             never fires on this corpus
    ``corpus-artifact``  strongest association is a nuisance variable, with
                         ``eta^2 >= min_eta2``
    ``domain``           strongest association is a target variable, ``eta^2 >= min_eta2``
    ``dense``            no association, but fires in more than ``dense_threshold`` of
                         samples (often position, template or bias features)
    ``""``               none of the above
    """
    assoc = {
        k: metadata_association(scan, samples, k, min_count=min_count)
        for k in (*nuisance, *targets)
    }
    freq, sfreq = scan.frequency, scan.sample_frequency
    mean_active, image_frac = scan.mean_active, scan.image_fraction
    top_cat = {k: a.top_category() for k, a in assoc.items()}
    table = []
    for f in range(scan.d_sae):
        row: dict[str, Any] = {
            "feature": f,
            "frequency": float(freq[f]),
            "sample_frequency": float(sfreq[f]),
            "mean_active": float(mean_active[f]),
            "image_fraction": float(image_frac[f]),
        }
        for k, a in assoc.items():
            row[f"eta2:{k}"] = float(a.eta2[f])
            row[f"top:{k}"] = top_cat[k][f]
            if a.auroc is not None:
                row[f"auroc:{k}"] = float(a.auroc[f])
        best_n = max(((float(assoc[k].eta2[f]), k) for k in nuisance), default=(0.0, None))
        best_t = max(((float(assoc[k].eta2[f]), k) for k in targets), default=(0.0, None))
        if scan.fire_count[f] == 0:
            label = "dead"
        elif best_n[0] >= min_eta2 and best_n[0] > best_t[0]:
            label = "corpus-artifact"
        elif best_t[0] >= min_eta2:
            label = "domain"
        elif sfreq[f] > dense_threshold:
            label = "dense"
        else:
            label = ""
        row["label"] = label
        best = best_n if best_n[0] > best_t[0] else best_t
        row["strongest"] = best[1] if best[0] > 0 else None
        table.append(row)
    return table
