#!/usr/bin/env python3
"""Recompute act-2 statistics from the web-mirrored JSON receipt.

Run from the post directory:

    uv run --with scipy python tools/recompute_act2_statistics.py \
      --out receipts/act2_statistics.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest, fisher_exact, wilcoxon


HERE = Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - radius, center + radius]


def paired_result(j: np.ndarray, control: np.ndarray) -> dict:
    """Exact two-sided paired rank and binary-hit summaries."""
    difference = control - j  # positive means J-lens has the lower (better) rank
    nonzero = difference[difference != 0]
    wins = int((difference > 0).sum())
    losses = int((difference < 0).sum())
    ties = int((difference == 0).sum())

    j_hit = j <= 20
    c_hit = control <= 20
    both = int((j_hit & c_hit).sum())
    j_only = int((j_hit & ~c_hit).sum())
    control_only = int((~j_hit & c_hit).sum())
    neither = int((~j_hit & ~c_hit).sum())
    discordant = j_only + control_only

    return {
        "rank_difference_definition": "control_best_rank - jlens_best_rank",
        "wins_losses_ties": {"jlens_wins": wins, "control_wins": losses, "ties": ties},
        "sign_test": {
            "method": "exact binomial, two-sided",
            "p_value": binomtest(
                wins, len(nonzero), p=0.5, alternative="two-sided"
            ).pvalue,
        },
        "wilcoxon_signed_rank": {
            "method": "exact, two-sided",
            "p_value": wilcoxon(
                difference, alternative="two-sided", method="exact"
            ).pvalue,
        },
        "paired_hit_table": {
            "both_hit": both,
            "jlens_only_hit": j_only,
            "control_only_hit": control_only,
            "neither_hit": neither,
        },
        "mcnemar_exact": {
            "method": "exact binomial on discordant pairs, two-sided",
            "p_value": (
                binomtest(
                    j_only, discordant, p=0.5, alternative="two-sided"
                ).pvalue
                if discordant
                else 1.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=HERE / "receipts" / "demo_qwen35-397b.json",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = json.loads(args.receipt.read_text())
    items = data["act2"]["items"]
    ranks = {
        name: np.asarray([row[name]["best_rank"] for row in items], dtype=np.int64)
        for name in ("jlens", "logit_lens", "random_J")
    }
    hits = {name: int((values <= 20).sum()) for name, values in ranks.items()}
    n = len(items)

    result = {
        "source_receipt": args.receipt.name,
        "scoring": "best rank over the preregistered layer band; hit iff rank <= 20",
        "n_items": n,
        "software": {"scipy": scipy.__version__},
        "hit_rates": {
            name: {
                "hits": count,
                "n": n,
                "rate": count / n,
                "wilson_95": wilson(count, n),
            }
            for name, count in hits.items()
        },
        "paired": {
            "jlens_vs_identity": paired_result(ranks["jlens"], ranks["logit_lens"]),
            "jlens_vs_random_J": paired_result(ranks["jlens"], ranks["random_J"]),
        },
        "unpaired_hit_sensitivity": {
            "jlens_vs_identity": {
                "table": [[hits["jlens"], n - hits["jlens"]], [hits["logit_lens"], n - hits["logit_lens"]]],
                "fisher_exact_one_sided_p": fisher_exact(
                    [[hits["jlens"], n - hits["jlens"]], [hits["logit_lens"], n - hits["logit_lens"]]],
                    alternative="greater",
                ).pvalue,
                "fisher_exact_two_sided_p": fisher_exact(
                    [[hits["jlens"], n - hits["jlens"]], [hits["logit_lens"], n - hits["logit_lens"]]],
                    alternative="two-sided",
                ).pvalue,
            },
            "jlens_vs_random_J": {
                "table": [[hits["jlens"], n - hits["jlens"]], [hits["random_J"], n - hits["random_J"]]],
                "fisher_exact_one_sided_p": fisher_exact(
                    [[hits["jlens"], n - hits["jlens"]], [hits["random_J"], n - hits["random_J"]]],
                    alternative="greater",
                ).pvalue,
                "fisher_exact_two_sided_p": fisher_exact(
                    [[hits["jlens"], n - hits["jlens"]], [hits["random_J"], n - hits["random_J"]]],
                    alternative="two-sided",
                ).pvalue,
            },
        },
        "items": [
            {
                "target": row["target"],
                "jlens": int(ranks["jlens"][i]),
                "identity": int(ranks["logit_lens"][i]),
                "random_J": int(ranks["random_J"][i]),
            }
            for i, row in enumerate(items)
        ],
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
