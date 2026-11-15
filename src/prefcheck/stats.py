"""Small statistics helpers. Nothing here needs numpy."""

from __future__ import annotations

import math
from collections.abc import Sequence


def round3(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile_nearest_rank(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile: the value at rank ceil(pct/100 * n) of the sorted data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def chosen_longer_ratio(chosen_lens: Sequence[int], rejected_lens: Sequence[int]) -> float | None:
    """Share of pairs whose chosen response is strictly longer than the rejected one."""
    if not chosen_lens:
        return None
    longer = sum(1 for c, r in zip(chosen_lens, rejected_lens) if c > r)
    return longer / len(chosen_lens)


def point_biserial_r(chosen_lens: Sequence[int], rejected_lens: Sequence[int]) -> float | None:
    """Point-biserial correlation between the label (1 = chosen, 0 = rejected) and response length.

    Both responses of every pair are pooled (2n samples). Returns None when it is
    undefined (no data, or all lengths equal).
    """
    xs = [*chosen_lens, *rejected_lens]
    ys = [1] * len(chosen_lens) + [0] * len(rejected_lens)
    n = len(xs)
    if n == 0 or not chosen_lens:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def length_ratio_quantiles(chosen_lens: Sequence[int], rejected_lens: Sequence[int]) -> dict[str, float | None]:
    """p10/p50/p90 of chosen/rejected token-length ratios (pairs with an empty side are skipped)."""
    ratios = [c / r for c, r in zip(chosen_lens, rejected_lens) if r > 0]
    return {f"length_ratio_p{p}": round3(percentile_nearest_rank(ratios, p)) for p in (10, 50, 90)}


def length_bias_metrics(chosen_lens: Sequence[int], rejected_lens: Sequence[int]) -> dict[str, float | None]:
    """Length-bias metrics reported in ``Report.stats``."""
    out: dict[str, float | None] = {
        "chosen_longer_ratio": round3(chosen_longer_ratio(chosen_lens, rejected_lens)),
        "point_biserial_r": round3(point_biserial_r(chosen_lens, rejected_lens)),
    }
    out.update(length_ratio_quantiles(chosen_lens, rejected_lens))
    return out
