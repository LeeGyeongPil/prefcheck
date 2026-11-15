"""Length-bias metrics (AC-5) checked against hand computation."""

from __future__ import annotations

from prefcheck import CheckConfig, check_rows
from prefcheck.stats import length_bias_metrics, percentile_nearest_rank, point_biserial_r

from conftest import conv_row

# Pairs (chosen, rejected) token lengths: (4,2) (6,2) (5,3) (3,3)
# pooled x = [4,6,5,3,2,2,3,3], y = [1,1,1,1,0,0,0,0]; mean x = 3.5, mean y = 0.5
# Sxy = 4, Sxx = 14, Syy = 2  ->  r = 4 / sqrt(28) = 0.7559
# chosen longer in 3 of 4 pairs; ratios sorted = [1, 1.667, 2, 3] -> p10 = 1 (rank 1), p50 = 1.667 (rank 2), p90 = 3 (rank 4)
CHOSEN, REJECTED = [4, 6, 5, 3], [2, 2, 3, 3]


def test_length_bias_metrics():
    m = length_bias_metrics(CHOSEN, REJECTED)
    assert m["chosen_longer_ratio"] == 0.75
    assert m["point_biserial_r"] == 0.756
    assert (m["length_ratio_p10"], m["length_ratio_p50"], m["length_ratio_p90"]) == (1.0, 1.667, 3.0)

    # end to end: the approximate tokenizer counts words + 1 (the <|im_end|> token) for conversational rows
    words = lambda n: " ".join(["w"] * n)  # noqa: E731
    rows = [conv_row("question here", words(c - 1), words(r - 1)) for c, r in zip(CHOSEN, REJECTED)]
    report = check_rows(rows, CheckConfig(max_length=None))
    assert report.stats["chosen_longer_ratio"] == 0.75
    assert report.stats["point_biserial_r"] == 0.756
    bias = [f for f in report.findings if f.rule == "length_bias"]
    assert len(bias) == 1 and bias[0].row is None and bias[0].severity == "warn"

    # no bias when lengths are balanced
    assert length_bias_metrics([3, 5], [5, 3])["chosen_longer_ratio"] == 0.5
    balanced = [conv_row("question here", words(c - 1), words(r - 1)) for c, r in [(4, 6), (6, 4), (5, 3), (3, 5)]]
    assert not [f for f in check_rows(balanced, CheckConfig(max_length=None)).findings if f.rule == "length_bias"]


def test_degenerate_inputs():
    assert point_biserial_r([], []) is None
    assert point_biserial_r([3, 3], [3, 3]) is None  # no variance
    assert percentile_nearest_rank([], 50) is None
    assert percentile_nearest_rank([5], 90) == 5
    assert length_bias_metrics([2], [0])["length_ratio_p50"] is None  # empty rejected side skipped
