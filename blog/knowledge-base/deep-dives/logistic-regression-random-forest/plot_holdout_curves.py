#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib==3.10.3",
#   "numpy==2.3.2",
#   "scikit-learn==1.7.1",
# ]
# ///
"""Draw one single-model HTRU2 holdout curve figure per fitted model.

The four head-to-head figures in this Deep Dive compare three models at once.
This generator produces the single-model views the article needs earlier, while
each model is being introduced on its own: the precision-recall and receiver
operating characteristic curves on the fixed HTRU2 held-out rows, with the
prevalence-only control drawn as the floor and no rival model in the frame.

Nothing is refitted here. The curves come from the per-row held-out
probabilities already committed by `reproduce.py` in
`receipts/test-predictions.csv`. Every summary value drawn on a figure is
re-derived from those rows and then checked against the independently committed
`receipts/test-metrics.csv`, so a drifted receipt fails loudly instead of
producing a plausible picture.

Run from this bundle:

    uv run plot_holdout_curves.py --generate
    uv run plot_holdout_curves.py --verify

Byte verification is scoped to the environment recorded in each receipt, on the
same terms as `reproduce.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BUNDLE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).name
STUDY_ID = "logistic-regression-random-forest"
DATASET = "htru2"
CONTROL = "dummy_prior"

PREDICTIONS = BUNDLE / "receipts" / "test-predictions.csv"
METRICS = BUNDLE / "receipts" / "test-metrics.csv"
ANALYSIS_RECEIPT = BUNDLE / "receipts" / "analysis.receipt.json"
PROVENANCE = BUNDLE / "provenance.json"

PAPER = "#F7F4F0"
INK = "#2C2924"
MUTED = "#5A544C"
BLUE = "#4B6787"
GREEN = "#6F8D5E"
CLAY = "#A67C52"
GRID = "#D9D0C4"

TOLERANCE = 1e-9

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "path.simplify": False,
        "axes.unicode_minus": False,
    }
)


@dataclass(frozen=True)
class ModelPlot:
    """One single-model figure: which model, what it is called, how it looks."""

    key: str
    label: str
    heading: str
    color: str
    stem: str

    @property
    def figure(self) -> Path:
        return BUNDLE / f"{self.stem}.png"

    @property
    def receipt(self) -> Path:
        return BUNDLE / f"{self.stem}.receipt.json"


PLOTS = (
    ModelPlot(
        key="logistic_regression",
        label="Logistic regression",
        heading="Model one on the fixed HTRU2 holdout: one global score, two views",
        color=BLUE,
        stem="fig-logistic-htru2-curves",
    ),
    ModelPlot(
        key="random_forest",
        label="Random forest",
        heading="Model two on the fixed HTRU2 holdout: many averaged trees, same two views",
        color=CLAY,
        stem="fig-forest-htru2-curves",
    ),
)


class GeneratorError(RuntimeError):
    """Raised when an input, an invariant, or a byte check fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def rounded(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def verify_inputs() -> dict[str, str]:
    """Hash the committed inputs and confirm the analysis receipt agrees."""
    artifacts = json.loads(ANALYSIS_RECEIPT.read_text(encoding="utf-8"))["artifacts"]
    hashes: dict[str, str] = {}
    for path in (PREDICTIONS, METRICS):
        relative = path.relative_to(BUNDLE).as_posix()
        actual = sha256_file(path)
        expected = artifacts.get(relative)
        if expected is None:
            raise GeneratorError(f"{relative} is not listed in the analysis receipt")
        if actual != expected:
            raise GeneratorError(
                f"{relative} SHA-256 mismatch (receipt {expected}, found {actual})"
            )
        hashes[relative] = actual
    hashes[ANALYSIS_RECEIPT.relative_to(BUNDLE).as_posix()] = sha256_file(
        ANALYSIS_RECEIPT
    )
    return hashes


def load_holdout(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, model probabilities, and control probabilities."""
    labels: list[int] = []
    scores: list[float] = []
    control: list[float] = []
    with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != DATASET:
                continue
            labels.append(int(row["target"]))
            scores.append(float(row[f"{model}__probability"]))
            control.append(float(row[f"{CONTROL}__probability"]))
    if not labels:
        raise GeneratorError(f"no {DATASET} rows found in {PREDICTIONS.name}")
    return np.asarray(labels), np.asarray(scores), np.asarray(control)


def load_metrics() -> dict[str, dict[str, str]]:
    with METRICS.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == DATASET]
    return {row["model"]: row for row in rows}


def check_close(name: str, computed: float, recorded: float) -> None:
    if abs(computed - recorded) > TOLERANCE:
        raise GeneratorError(
            f"{name} disagrees with the committed metrics receipt "
            f"(recomputed {computed!r}, receipt {recorded!r})"
        )


def build_values(model: str) -> dict[str, Any]:
    """Recompute every plotted quantity and cross-check the metrics receipt."""
    labels, scores, control = load_holdout(model)
    metrics = load_metrics()

    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    prevalence = positives / labels.size

    average_precision = float(average_precision_score(labels, scores))
    roc_area = float(roc_auc_score(labels, scores))
    check_close(
        "average precision",
        average_precision,
        float(metrics[model]["average_precision"]),
    )
    check_close("ROC AUC", roc_area, float(metrics[model]["roc_auc"]))
    check_close(
        "control average precision",
        float(average_precision_score(labels, control)),
        float(metrics[CONTROL]["average_precision"]),
    )
    check_close(
        "control positive rate",
        prevalence,
        float(metrics[CONTROL]["average_precision"]),
    )

    predicted = scores >= 0.5
    true_positives = int(np.sum(predicted & (labels == 1)))
    false_positives = int(np.sum(predicted & (labels == 0)))
    false_negatives = int(np.sum(~predicted & (labels == 1)))
    true_negatives = int(np.sum(~predicted & (labels == 0)))
    for name, computed, recorded in (
        ("tp", true_positives, int(metrics[model]["tp"])),
        ("fp", false_positives, int(metrics[model]["fp"])),
        ("fn", false_negatives, int(metrics[model]["fn"])),
        ("tn", true_negatives, int(metrics[model]["tn"])),
    ):
        if computed != recorded:
            raise GeneratorError(
                f"threshold-0.5 {name} disagrees with the metrics receipt "
                f"(recomputed {computed}, receipt {recorded})"
            )

    operating_recall = true_positives / positives
    operating_precision = true_positives / (true_positives + false_positives)
    operating_fpr = false_positives / negatives
    check_close(
        "precision at 0.5",
        operating_precision,
        float(metrics[model]["precision_at_0_5"]),
    )
    check_close(
        "recall at 0.5", operating_recall, float(metrics[model]["recall_at_0_5"])
    )

    precision, recall, _ = precision_recall_curve(labels, scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)

    return {
        "rows": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "prevalence": rounded(prevalence),
        "average_precision": rounded(average_precision),
        "roc_auc": rounded(roc_area),
        "operating_point": {
            "threshold": 0.5,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": rounded(operating_precision),
            "recall": rounded(operating_recall),
            "false_positive_rate": rounded(operating_fpr),
        },
        "curves": {
            "precision_recall_points": int(precision.size),
            "roc_points": int(false_positive_rate.size),
        },
        "_precision": precision,
        "_recall": recall,
        "_fpr": false_positive_rate,
        "_tpr": true_positive_rate,
    }


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=10)
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.05)


def draw(plot: ModelPlot, values: dict[str, Any], path: Path) -> None:
    point = values["operating_point"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    figure.patch.set_facecolor(PAPER)

    left, right = axes
    style_axis(left)
    left.step(
        values["_recall"],
        values["_precision"],
        where="post",
        color=plot.color,
        linewidth=2.4,
        label=f"{plot.label} (AP {values['average_precision']:.4f})",
    )
    left.axhline(
        values["prevalence"],
        color=GREEN,
        linewidth=1.8,
        linestyle="--",
        label=f"Prevalence-only control (AP {values['prevalence']:.4f})",
    )
    left.scatter(
        [point["recall"]],
        [point["precision"]],
        s=90,
        color=INK,
        zorder=4,
        label="Threshold 0.5",
    )
    left.annotate(
        f"threshold 0.5\nrecall {point['recall']:.4f}, precision {point['precision']:.4f}",
        xy=(point["recall"], point["precision"]),
        xytext=(0.10, 0.36),
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.2},
    )
    left.set_title(
        "Precision-recall view", fontsize=13, fontweight="bold", color=INK, loc="left"
    )
    left.set_xlabel(
        f"Recall (share of the {values['positives']:,} labeled positives recovered)",
        color=INK,
        fontsize=10,
    )
    left.set_ylabel("Precision", color=INK, fontsize=10)
    left.legend(loc="lower left", fontsize=9, framealpha=0.95, facecolor="#FBF9F6")

    style_axis(right)
    right.plot(
        values["_fpr"],
        values["_tpr"],
        color=plot.color,
        linewidth=2.4,
        label=f"{plot.label} (ROC AUC {values['roc_auc']:.4f})",
    )
    right.plot(
        [0, 1],
        [0, 1],
        color=MUTED,
        linewidth=1.4,
        linestyle="--",
        label="Chance line (ROC AUC 0.5000)",
    )
    right.scatter(
        [point["false_positive_rate"]],
        [point["recall"]],
        s=90,
        color=INK,
        zorder=4,
        label="Threshold 0.5",
    )
    right.annotate(
        (
            f"threshold 0.5\n{point['true_positives']} of "
            f"{values['positives']} found\n"
            f"{point['false_positives']} false alarms"
        ),
        xy=(point["false_positive_rate"], point["recall"]),
        xytext=(0.06, 0.60),
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.2},
    )
    right.set_title(
        "Receiver operating characteristic view",
        fontsize=13,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    right.set_xlabel(
        f"False-positive rate (share of the {values['negatives']:,} labeled negatives)",
        color=INK,
        fontsize=10,
    )
    right.set_ylabel("True-positive rate (recall)", color=INK, fontsize=10)
    right.legend(loc="lower right", fontsize=9, framealpha=0.95, facecolor="#FBF9F6")

    figure.suptitle(
        plot.heading,
        fontsize=16,
        fontweight="bold",
        color=INK,
        x=0.02,
        ha="left",
        y=0.98,
    )
    figure.text(
        0.02,
        0.925,
        f"{values['rows']:,} rows held out from scaling, tuning, and refitting in the"
        " published workflow. Only this model and the prevalence-only control are plotted.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(
        path,
        dpi=200,
        facecolor=PAPER,
        edgecolor="none",
        metadata={"Software": f"praxagent {STUDY_ID}"},
    )
    plt.close(figure)


def build_receipt(
    plot: ModelPlot, values: dict[str, Any], inputs: dict[str, str]
) -> dict[str, Any]:
    plotted = {key: value for key, value in values.items() if not key.startswith("_")}
    return {
        "study_id": STUDY_ID,
        "figure": plot.figure.name,
        "dataset": DATASET,
        "model": plot.key,
        "control": CONTROL,
        "claim": (
            "Held-out precision-recall and receiver operating characteristic curves "
            f"for the fitted {plot.label.lower()} on the fixed HTRU2 test rows, with "
            "the prevalence-only control as the floor. Single fixed split; no "
            "refitting; no rival model in the frame."
        ),
        "derived_from": (
            "Per-row held-out probabilities committed by reproduce.py; every summary "
            "value is re-derived here and cross-checked against "
            "receipts/test-metrics.csv."
        ),
        "inputs": inputs,
        "generator": {
            "path": SCRIPT,
            "sha256": sha256_file(BUNDLE / SCRIPT),
            "verify": f"uv run {SCRIPT} --verify",
        },
        "plotted_values": plotted,
        "environment": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "verification_scope": {
            "claim": "byte identity in the recorded reference environment only",
            "cross_platform_note": (
                "operating systems, numerical libraries, FreeType builds, and font "
                "files can change output bytes"
            ),
        },
        "schema_version": 1,
        "provenance": {"outputs": {plot.figure.name: sha256_file(plot.figure)}},
    }


def update_manifest() -> None:
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    figures = manifest["figures"]
    receipts = manifest["receipts"]
    for plot in reversed(PLOTS):
        if plot.figure.name not in figures:
            figures.insert(0, plot.figure.name)
        receipts[plot.receipt.name] = sha256_file(plot.receipt)
    receipts.pop("plot_logistic_holdout.py", None)
    receipts[SCRIPT] = sha256_file(BUNDLE / SCRIPT)
    manifest["receipts"] = dict(sorted(receipts.items()))
    PROVENANCE.write_bytes(canonical_json_bytes(manifest))


def generate() -> None:
    inputs = verify_inputs()
    for plot in PLOTS:
        values = build_values(plot.key)
        draw(plot, values, plot.figure)
        plot.receipt.write_bytes(
            canonical_json_bytes(build_receipt(plot, values, inputs))
        )
        print(f"wrote {plot.figure.name} ({sha256_file(plot.figure)})")
        print(f"wrote {plot.receipt.name}")
    update_manifest()
    print(f"updated {PROVENANCE.name}")


def verify() -> None:
    verify_inputs()
    for plot in PLOTS:
        recorded = json.loads(plot.receipt.read_text(encoding="utf-8"))
        expected = recorded["provenance"]["outputs"][plot.figure.name]
        committed = sha256_file(plot.figure)
        if committed != expected:
            raise GeneratorError(
                f"committed {plot.figure.name} does not match its receipt "
                f"(receipt {expected}, found {committed})"
            )
        values = build_values(plot.key)
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / plot.figure.name
            draw(plot, values, candidate)
            regenerated = sha256_file(candidate)
        if regenerated != expected:
            raise GeneratorError(
                "regenerated figure bytes differ from the receipt in this "
                f"environment (receipt {expected}, regenerated {regenerated})"
            )
        print(f"verified {plot.figure.name} and every plotted value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate", action="store_true", help="write figures and receipts"
    )
    group.add_argument("--verify", action="store_true", help="re-derive and byte-check")
    arguments = parser.parse_args()
    if arguments.generate:
        generate()
    else:
        verify()


if __name__ == "__main__":
    main()
