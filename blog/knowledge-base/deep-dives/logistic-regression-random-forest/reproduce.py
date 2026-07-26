#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib==3.10.3",
#   "numpy==2.3.2",
#   "scikit-learn==1.7.1",
# ]
# ///
"""Generate and byte-verify the logistic-regression/random-forest study.

The primary example is UCI HTRU2 (rare pulsar candidates). The transfer
example is UCI Rice (Cammeo and Osmancik). Both datasets are CC BY 4.0. The
canonical evaluation uses a fixed stratified 80/20 split, tunes only on the
training portion with identical stratified folds, and evaluates the fixed test
portion outside fitting and tuning. Dataset, model, and story selection followed
exploratory screening, so the results are descriptive rather than prospective
confirmation. Neither primary model uses class weighting or oversampling.

Run from this bundle:

    uv run --frozen reproduce.py --generate
    uv run --frozen reproduce.py --verify

Byte verification is scoped to the environment recorded in the generated
receipt.  A different operating system, numerical stack, FreeType build, or
font file can legitimately produce different bytes; canonical release
artifacts should therefore be generated and verified in the same pinned Linux
environment used by publication CI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import struct
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

# Keep low-level numerical kernels single-threaded. Grid-search candidates run
# in separate processes; each random forest itself remains single-threaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib import ft2font, font_manager
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RECEIPTS_DIRNAME = "receipts"
LOCK_FILE = "reproduce.py.lock"
ATTRIBUTION_FILE = "ATTRIBUTION.txt"
SOURCE_MANIFEST = "source-manifest.json"
ANALYSIS_RECEIPT = "receipts/analysis.receipt.json"
METRICS_CSV = "receipts/test-metrics.csv"
PREDICTIONS_CSV = "receipts/test-predictions.csv"
TUNING_CSV = "receipts/training-cv-tuning.csv"
CALIBRATION_CSV = "receipts/calibration-bins.csv"
BOOTSTRAP_CSV = "receipts/paired-bootstrap.csv"
IMPORTANCE_CSV = "receipts/permutation-importance.csv"
NOTEBOOK = "forest-in-the-sky.ipynb"
PROVENANCE = "provenance.json"

PERFORMANCE_FIGURE = "fig-held-out-ranking-metrics.png"
CALIBRATION_FIGURE = "fig-held-out-calibration.png"
CONFUSION_FIGURE = "fig-held-out-confusion-counts.png"
IMPORTANCE_FIGURE = "fig-held-out-permutation-importance.png"
OG_CARD = "og-card.png"

PERFORMANCE_RECEIPT = "fig-held-out-ranking-metrics.receipt.json"
CALIBRATION_RECEIPT = "fig-held-out-calibration.receipt.json"
CONFUSION_RECEIPT = "fig-held-out-confusion-counts.receipt.json"
IMPORTANCE_RECEIPT = "fig-held-out-permutation-importance.receipt.json"
OG_RECEIPT = "fig-og-card.receipt.json"

# Single-model holdout curves are drawn by the companion generator below, which
# owns their pixels and their receipts. They are listed here so the post-wide
# manifest this script writes covers the whole bundle.
SINGLE_MODEL_GENERATOR = "plot_holdout_curves.py"
SINGLE_MODEL_FIGURES = (
    "fig-forest-htru2-curves.png",
    "fig-logistic-htru2-curves.png",
)
SINGLE_MODEL_RECEIPTS = (
    "fig-forest-htru2-curves.receipt.json",
    "fig-logistic-htru2-curves.receipt.json",
)

STUDY_ID = "uci-htru2-rice-logistic-random-forest-v1"
GLOBAL_SEED = 20260721
TEST_SIZE = 0.20
CV_SPLITS = 5
BOOTSTRAP_REPLICATES = 2_000
PERMUTATION_REPEATS = 30
THRESHOLD = 0.5
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)

HTRU_FILE = "data/HTRU_2.csv"
HTRU_README = "data/Readme.txt"
RICE_FILE = "data/Rice_Cammeo_Osmancik.arff"
RICE_CITATION = "data/Citation_Request.txt"

EXPECTED_SOURCE_HASHES = {
    HTRU_FILE: "b13b4d8929e96ecd196e464c1c8a454c3ac2ffa631015f6388957531a9923f59",
    HTRU_README: "691efe1b5b910401959a9b4f74ed0959dcd205d69c73cf524646e6f63a3eb86b",
    RICE_FILE: "1af97883100c89de2ea2972f7a28d428f4f1c14711a61defc0b0569e9eb65665",
    RICE_CITATION: "7184131b2ca0a456f619ede4cec7cf84607f8d9b32e6f5aab5be6e59f52dfc48",
}

SOURCE_ARCHIVES = {
    "htru2": {
        "url": "https://archive.ics.uci.edu/static/public/372/htru2.zip",
        "sha256": "ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379",
    },
    "rice": {
        "url": (
            "https://archive.ics.uci.edu/static/public/545/"
            "rice+cammeo+and+osmancik.zip"
        ),
        "sha256": "fe94e42046b829de21b92b0ffb6a22774fde021328cae799faa802a14e8dbed9",
    },
}

HTRU_FEATURES = (
    "profile_mean",
    "profile_standard_deviation",
    "profile_excess_kurtosis",
    "profile_skewness",
    "dm_snr_mean",
    "dm_snr_standard_deviation",
    "dm_snr_excess_kurtosis",
    "dm_snr_skewness",
)
HTRU_FEATURE_LABELS = {
    "profile_mean": "Profile mean",
    "profile_standard_deviation": "Profile standard deviation",
    "profile_excess_kurtosis": "Profile excess kurtosis",
    "profile_skewness": "Profile skewness",
    "dm_snr_mean": "Dispersion-measure signal-to-noise ratio mean",
    "dm_snr_standard_deviation": "Dispersion-measure signal-to-noise ratio standard deviation",
    "dm_snr_excess_kurtosis": "Dispersion-measure signal-to-noise ratio excess kurtosis",
    "dm_snr_skewness": "Dispersion-measure signal-to-noise ratio skewness",
}
RICE_FEATURES = (
    "area",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "convex_area",
    "extent",
)
RICE_FEATURE_LABELS = {
    "area": "Area",
    "perimeter": "Perimeter",
    "major_axis_length": "Major axis length",
    "minor_axis_length": "Minor axis length",
    "eccentricity": "Eccentricity",
    "convex_area": "Convex area",
    "extent": "Extent",
}

DATASET_ORDER = ("htru2", "rice")
MODEL_ORDER = ("dummy_prior", "logistic_regression", "random_forest")
MODEL_LABELS = {
    "dummy_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}
METRIC_ORDER = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "precision_at_0_5",
    "recall_at_0_5",
    "f1_at_0_5",
    "log_loss",
    "brier_score",
)
METRIC_LABELS = {
    "average_precision": "Average precision",
    "roc_auc": "ROC AUC",
    "balanced_accuracy": "Balanced accuracy",
    "precision_at_0_5": "Precision at 0.5",
    "recall_at_0_5": "Recall at 0.5",
    "f1_at_0_5": "F1 at 0.5",
    "log_loss": "Log loss",
    "brier_score": "Brier score",
}

PAPER = "#F7F4F0"
INK = "#2C2924"
MUTED = "#5A544C"
BLUE = "#4B6787"
GREEN = "#6F8D5E"
CLAY = "#A67C52"
GRID = "#D9D0C4"
MODEL_COLORS = {
    "dummy_prior": MUTED,
    "logistic_regression": BLUE,
    "random_forest": CLAY,
}
MODEL_MARKERS = {
    "dummy_prior": "D",
    "logistic_regression": "o",
    "random_forest": "s",
}

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "path.simplify": False,
        "axes.unicode_minus": False,
    }
)


class ReproductionError(RuntimeError):
    """Raised when an invariant or byte-level verification fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def rounded(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def metric_text(value: float) -> str:
    return f"{float(value):.4f}"


def csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.12g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ReproductionError(f"{path} is not a PNG")
    return struct.unpack(">II", header[16:24])


def current_environment() -> dict[str, str]:
    font_path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "matplotlib_version": matplotlib.__version__,
        "freetype_version": ft2font.__freetype_version__,
        "figure_font": font_path.name,
        "figure_font_sha256": sha256_file(font_path),
        "matplotlib_backend": str(matplotlib.get_backend()),
    }


def validate_static_inputs() -> None:
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = HERE / relative
        if not path.is_file():
            raise ReproductionError(f"missing source file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ReproductionError(
                f"source hash mismatch for {relative}: expected {expected}, found {actual}"
            )
    for relative in (LOCK_FILE, ATTRIBUTION_FILE):
        if not (HERE / relative).is_file():
            raise ReproductionError(f"missing required bundle file: {relative}")


def audit_matrix(X: np.ndarray, y: np.ndarray) -> dict[str, int]:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ReproductionError("feature/target arrays have inconsistent shapes")
    if not np.isfinite(X).all():
        raise ReproductionError("source feature matrix contains non-finite values")
    full = np.column_stack((X, y))
    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "missing_or_nonfinite_feature_cells": int(np.size(X) - np.isfinite(X).sum()),
        "exact_full_row_duplicates": int(len(full) - len(np.unique(full, axis=0))),
        "exact_feature_row_duplicates": int(len(X) - len(np.unique(X, axis=0))),
    }


def load_htru2() -> dict[str, Any]:
    raw = np.loadtxt(HERE / HTRU_FILE, delimiter=",", dtype=float)
    if raw.shape != (17_898, 9):
        raise ReproductionError(f"HTRU2 expected shape (17898, 9), found {raw.shape}")
    X = raw[:, :8]
    y = raw[:, 8].astype(np.int64)
    labels, counts = np.unique(y, return_counts=True)
    if labels.tolist() != [0, 1] or counts.tolist() != [16_259, 1_639]:
        raise ReproductionError("HTRU2 class counts differ from the UCI record")
    audit = audit_matrix(X, y)
    if audit["exact_feature_row_duplicates"] != 0:
        raise ReproductionError("HTRU2 unexpectedly contains exact duplicate features")
    return {
        "key": "htru2",
        "display_name": "HTRU2",
        "X": X,
        "y": y,
        "features": HTRU_FEATURES,
        "feature_labels": HTRU_FEATURE_LABELS,
        "negative_label": "non-pulsar candidate",
        "positive_label": "pulsar candidate",
        "source_path": HTRU_FILE,
        "row_definition": "one human-annotated pulsar candidate feature row",
        "audit": audit,
        "seed_offset": 372,
    }


def load_rice() -> dict[str, Any]:
    rows: list[list[str]] = []
    in_data = False
    with (HERE / RICE_FILE).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                continue
            parsed = next(csv.reader([line]))
            if len(parsed) != 8:
                raise ReproductionError("Rice ARFF row does not have eight fields")
            rows.append(parsed)
    if len(rows) != 3_810:
        raise ReproductionError(f"Rice expected 3810 rows, found {len(rows)}")
    X = np.asarray([[float(value) for value in row[:7]] for row in rows], dtype=float)
    # Cammeo is the minority class and is used as positive solely to define
    # binary probability metrics. It carries no value judgment.
    class_names = np.asarray([row[7].strip() for row in rows])
    if set(class_names.tolist()) != {"Cammeo", "Osmancik"}:
        raise ReproductionError("Rice source has unexpected class names")
    y = (class_names == "Cammeo").astype(np.int64)
    labels, counts = np.unique(y, return_counts=True)
    if labels.tolist() != [0, 1] or counts.tolist() != [2_180, 1_630]:
        raise ReproductionError("Rice class counts differ from the UCI record")
    audit = audit_matrix(X, y)
    if audit["exact_feature_row_duplicates"] != 0:
        raise ReproductionError("Rice unexpectedly contains exact duplicate features")
    return {
        "key": "rice",
        "display_name": "Rice",
        "X": X,
        "y": y,
        "features": RICE_FEATURES,
        "feature_labels": RICE_FEATURE_LABELS,
        "negative_label": "Osmancik",
        "positive_label": "Cammeo",
        "source_path": RICE_FILE,
        "row_definition": "one segmented rice-grain image feature row",
        "audit": audit,
        "seed_offset": 545,
    }


def split_hash(indices: np.ndarray) -> str:
    return sha256_text("\n".join(str(int(value)) for value in indices) + "\n")


def model_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= THRESHOLD).astype(np.int64)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def calibration_rows(
    dataset: str,
    model: str,
    y: np.ndarray,
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    # np.digitize with the interior edges gives bins [0,.1), ..., [.9,1].
    membership = np.digitize(probability, CALIBRATION_EDGES[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for bin_index in range(10):
        selected = membership == bin_index
        count = int(selected.sum())
        positive_count = int(y[selected].sum()) if count else 0
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "bin_index": bin_index,
                "lower_bound_inclusive": float(CALIBRATION_EDGES[bin_index]),
                "upper_bound": float(CALIBRATION_EDGES[bin_index + 1]),
                "upper_bound_inclusive": bin_index == 9,
                "count": count,
                "positive_count": positive_count,
                "mean_predicted_probability": (
                    float(probability[selected].mean()) if count else None
                ),
                "observed_positive_fraction": (
                    float(y[selected].mean()) if count else None
                ),
            }
        )
    if sum(row["count"] for row in rows) != len(y):
        raise ReproductionError("calibration bins do not cover the test rows")
    return rows


def tuning_rows(
    dataset: str,
    model: str,
    search: GridSearchCV,
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "candidate_index": index,
                "mean_validation_average_precision": float(results["mean_test_score"][index]),
                "std_validation_average_precision": float(results["std_test_score"][index]),
                "rank_validation_average_precision": int(results["rank_test_score"][index]),
                "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                "selected": index == int(search.best_index_),
            }
        )
    return rows


def paired_bootstrap(
    dataset: str,
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in METRIC_ORDER} for model in MODEL_ORDER
    }
    deltas: dict[str, list[float]] = {metric: [] for metric in METRIC_ORDER}
    rows: list[dict[str, Any]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        y_sample = y[sampled]
        row: dict[str, Any] = {"dataset": dataset, "replicate": replicate}
        replicate_metrics: dict[str, dict[str, Any]] = {}
        for model in MODEL_ORDER:
            observed = model_metrics(y_sample, probabilities[model][sampled])
            replicate_metrics[model] = observed
            for metric in METRIC_ORDER:
                value = float(observed[metric])
                values[model][metric].append(value)
                row[f"{model}__{metric}"] = value
        for metric in METRIC_ORDER:
            delta = float(
                replicate_metrics["random_forest"][metric]
                - replicate_metrics["logistic_regression"][metric]
            )
            deltas[metric].append(delta)
            row[f"random_forest_minus_logistic__{metric}"] = delta
        rows.append(row)

    def interval(samples: Sequence[float]) -> dict[str, float]:
        array = np.asarray(samples, dtype=float)
        lower, upper = np.quantile(array, [0.025, 0.975], method="linear")
        return {
            "lower": rounded(lower),
            "median": rounded(np.median(array)),
            "upper": rounded(upper),
        }

    summary = {
        "method": "paired stratified nonparametric bootstrap of the fixed held-out rows",
        "replicates": BOOTSTRAP_REPLICATES,
        "random_state": seed,
        "confidence_level_label": "95% bootstrap stability interval",
        "interval_endpoints": {
            "lower_percentile": 2.5,
            "upper_percentile": 97.5,
            "quantile_method": "numpy.quantile(method='linear')",
        },
        "resampling": (
            "positive and negative held-out row indices are sampled with replacement "
            "within class; the same sampled indices are used for every model"
        ),
        "not_a_population_confidence_interval": True,
        "limitations": [
            "conditions on the single fixed test split and its observed class counts",
            "does not include training-split, tuning, refitting, or forest-seed variation",
            "assumes held-out rows are exchangeable within class",
            "does not repair unrecorded grouping or dependence in either public dataset",
            "does not establish performance in a new survey, field, instrument, or population",
        ],
        "models": {
            model: {metric: interval(values[model][metric]) for metric in METRIC_ORDER}
            for model in MODEL_ORDER
        },
        "random_forest_minus_logistic_regression": {
            "delta_definition": "random forest metric minus logistic regression metric",
            "interpretation": (
                "positive favors random forest for higher-is-better metrics; negative "
                "favors random forest for lower-is-better log loss and Brier score"
            ),
            "metrics": {metric: interval(deltas[metric]) for metric in METRIC_ORDER},
        },
    }
    return rows, summary


def fit_and_evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    key = dataset["key"]
    X = dataset["X"]
    y = dataset["y"]
    all_indices = np.arange(len(y), dtype=np.int64)
    split_seed = GLOBAL_SEED + dataset["seed_offset"]
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=split_seed,
        shuffle=True,
        stratify=y,
    )
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    if set(train_indices).intersection(test_indices):
        raise ReproductionError(f"{key} split overlap")
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    cv_seed = GLOBAL_SEED + 10_000 + dataset["seed_offset"]
    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=cv_seed)
    cv_splits = list(splitter.split(X_train, y_train))
    cv_record = []
    for fold, (fold_train, fold_validation) in enumerate(cv_splits):
        cv_record.append(
            {
                "fold": fold,
                "training_rows": int(len(fold_train)),
                "validation_rows": int(len(fold_validation)),
                "validation_positives": int(y_train[fold_validation].sum()),
                "training_original_index_sha256": split_hash(train_indices[fold_train]),
                "validation_original_index_sha256": split_hash(
                    train_indices[fold_validation]
                ),
            }
        )

    logistic_seed = GLOBAL_SEED + 20_000 + dataset["seed_offset"]
    forest_seed = GLOBAL_SEED + 30_000 + dataset["seed_offset"]
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    class_weight=None,
                    max_iter=5_000,
                    random_state=logistic_seed,
                ),
            ),
        ]
    )
    logistic_search = GridSearchCV(
        logistic,
        param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="average_precision",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    forest = RandomForestClassifier(
        n_estimators=250,
        criterion="gini",
        bootstrap=True,
        class_weight=None,
        random_state=forest_seed,
        n_jobs=1,
    )
    forest_search = GridSearchCV(
        forest,
        param_grid={
            "max_depth": [6, 12, None],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", 0.75],
        },
        scoring="average_precision",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    print(f"[{key}] fitting logistic-regression training-fold grid", flush=True)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        logistic_search.fit(X_train, y_train)
    logistic_elapsed = time.perf_counter() - started
    best_logistic = logistic_search.best_estimator_.named_steps["classifier"]
    if np.any(best_logistic.n_iter_ >= best_logistic.max_iter):
        raise ReproductionError(f"{key} selected logistic regression did not converge")
    print(f"[{key}] fitting random-forest training-fold grid", flush=True)
    started = time.perf_counter()
    forest_search.fit(X_train, y_train)
    forest_elapsed = time.perf_counter() - started
    dummy = DummyClassifier(strategy="prior", random_state=split_seed)
    dummy.fit(X_train, y_train)

    estimators: dict[str, BaseEstimator] = {
        "dummy_prior": dummy,
        "logistic_regression": logistic_search.best_estimator_,
        "random_forest": forest_search.best_estimator_,
    }
    for model, estimator in estimators.items():
        if np.asarray(estimator.classes_).tolist() != [0, 1]:
            raise ReproductionError(f"{key} {model} has unexpected probability class order")
    probabilities = {
        model: np.asarray(estimator.predict_proba(X_test)[:, 1], dtype=float)
        for model, estimator in estimators.items()
    }
    metrics = {model: model_metrics(y_test, probability) for model, probability in probabilities.items()}

    calibration: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        calibration.extend(calibration_rows(key, model, y_test, probabilities[model]))

    prediction_rows: list[dict[str, Any]] = []
    for position, original_index in enumerate(test_indices):
        row: dict[str, Any] = {
            "dataset": key,
            "test_position": position,
            "original_row_index_zero_based": int(original_index),
            "target": int(y_test[position]),
        }
        for model in MODEL_ORDER:
            probability = float(probabilities[model][position])
            row[f"{model}__probability"] = probability
            row[f"{model}__prediction_at_0_5"] = int(probability >= THRESHOLD)
        prediction_rows.append(row)

    bootstrap_rows, bootstrap_summary = paired_bootstrap(
        key,
        y_test,
        probabilities,
        seed=GLOBAL_SEED + 40_000 + dataset["seed_offset"],
    )

    importance_rows: list[dict[str, Any]] = []
    importance_summary: dict[str, list[dict[str, Any]]] = {}
    importance_seeds: dict[str, int] = {}
    for model_index, model in enumerate(("logistic_regression", "random_forest")):
        importance_seed = (
            GLOBAL_SEED + 50_000 + dataset["seed_offset"] + model_index
        )
        importance_seeds[model] = importance_seed
        observed = permutation_importance(
            estimators[model],
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS,
            random_state=importance_seed,
            n_jobs=1,
        )
        records: list[dict[str, Any]] = []
        for feature_index, feature in enumerate(dataset["features"]):
            repeats = observed.importances[feature_index]
            record = {
                "feature": feature,
                "feature_label": dataset["feature_labels"][feature],
                "mean_average_precision_decrease": rounded(repeats.mean()),
                "standard_deviation_across_permutations": rounded(
                    repeats.std(ddof=0)
                ),
            }
            records.append(record)
            for repeat, value in enumerate(repeats):
                importance_rows.append(
                    {
                        "dataset": key,
                        "model": model,
                        "feature": feature,
                        "feature_label": dataset["feature_labels"][feature],
                        "repeat": repeat,
                        "average_precision_decrease": float(value),
                    }
                )
        importance_summary[model] = sorted(
            records,
            key=lambda item: item["mean_average_precision_decrease"],
            reverse=True,
        )

    tuning = tuning_rows(key, "logistic_regression", logistic_search)
    tuning.extend(tuning_rows(key, "random_forest", forest_search))

    return {
        "dataset": {
            "key": key,
            "display_name": dataset["display_name"],
            "source_path": dataset["source_path"],
            "source_sha256": sha256_file(HERE / dataset["source_path"]),
            "row_definition": dataset["row_definition"],
            "features": list(dataset["features"]),
            "feature_labels": dataset["feature_labels"],
            "negative_label": dataset["negative_label"],
            "positive_label": dataset["positive_label"],
            "audit": dataset["audit"],
            "class_counts": {
                "negative": int(np.sum(y == 0)),
                "positive": int(np.sum(y == 1)),
            },
            "positive_prevalence": rounded(np.mean(y)),
        },
        "split": {
            "method": "fixed stratified random holdout",
            "test_fraction": TEST_SIZE,
            "random_state": split_seed,
            "training_rows": int(len(train_indices)),
            "training_positives": int(y_train.sum()),
            "test_rows": int(len(test_indices)),
            "test_positives": int(y_test.sum()),
            "training_original_index_sha256": split_hash(train_indices),
            "test_original_index_sha256": split_hash(test_indices),
        },
        "training_only_tuning": {
            "objective": "average_precision",
            "folds": CV_SPLITS,
            "random_state": cv_seed,
            "selection_tie_behavior": (
                "GridSearchCV selects the first candidate in the declared ParameterGrid "
                "order among exactly tied mean validation scores"
            ),
            "shared_fold_records": cv_record,
            "logistic_regression": {
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "fixed_parameters": {
                    "solver": "lbfgs",
                    "penalty": "l2",
                    "max_iter": 5_000,
                    "random_state": logistic_seed,
                },
                "class_weight": None,
                "oversampling": None,
                "grid": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
                "best_params": logistic_search.best_params_,
                "converged": True,
                "selected_fit_iterations": [int(value) for value in best_logistic.n_iter_],
                "best_mean_validation_average_precision": rounded(
                    logistic_search.best_score_
                ),
                "elapsed_seconds_diagnostic_only": rounded(logistic_elapsed, 3),
            },
            "random_forest": {
                "n_estimators": 250,
                "fixed_parameters": {
                    "criterion": "gini",
                    "bootstrap": True,
                    "random_state": forest_seed,
                    "n_jobs": 1,
                },
                "class_weight": None,
                "oversampling": None,
                "grid": {
                    "max_depth": [6, 12, None],
                    "min_samples_leaf": [1, 5],
                    "max_features": ["sqrt", 0.75],
                },
                "best_params": forest_search.best_params_,
                "best_mean_validation_average_precision": rounded(
                    forest_search.best_score_
                ),
                "elapsed_seconds_diagnostic_only": rounded(forest_elapsed, 3),
            },
            "test_set_role": (
                "kept outside fitting and hyperparameter tuning within the published "
                "workflow; used for descriptive evaluation and post-evaluation diagnostics"
            ),
        },
        "test_metrics": metrics,
        "calibration": calibration,
        "paired_bootstrap": bootstrap_summary,
        "permutation_importance": {
            "scoring": "average_precision",
            "repeats": PERMUTATION_REPEATS,
            "random_states": importance_seeds,
            "computed_on": "fixed held-out test rows after model selection",
            "not_used_for_tuning": True,
            "limitations": [
                "repeat spread is not a confidence interval",
                "correlated features can share or mask importance",
                "permutation can create feature combinations not common in the source table",
            ],
            "models": importance_summary,
        },
        "rows": {
            "predictions": prediction_rows,
            "tuning": tuning,
            "calibration": calibration,
            "bootstrap": bootstrap_rows,
            "importance": importance_rows,
        },
        "scope_limits": [
            "the public table provides no grouping identifiers for dependence-aware splitting",
            "a fixed row-level holdout does not establish performance in a new acquisition context",
            "test-set interpretation does not turn association into causation",
            "model comparison is conditional on the declared grids, split, threshold, and metrics",
        ],
    }


def source_manifest(datasets: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "license": "CC BY 4.0",
        "attribution": ATTRIBUTION_FILE,
        "attribution_sha256": sha256_file(HERE / ATTRIBUTION_FILE),
        "datasets": {
            "htru2": {
                "title": "HTRU2",
                "creator": "Robert Lyon",
                "year": 2015,
                "repository": "UCI Machine Learning Repository",
                "doi": "10.24432/C5DK6R",
                "dataset_page": "https://archive.ics.uci.edu/dataset/372/htru2",
                "archive": SOURCE_ARCHIVES["htru2"],
                "included_files": {
                    path: EXPECTED_SOURCE_HASHES[path]
                    for path in (HTRU_FILE, HTRU_README)
                },
                "rows": datasets["htru2"]["dataset"]["audit"]["rows"],
                "features": list(HTRU_FEATURES),
                "target_encoding": {"0": "non-pulsar candidate", "1": "pulsar candidate"},
                "positive_class": "pulsar candidate",
                "missing_values": 0,
                "known_metadata_limit": (
                    "the release contains no positional or other astronomical details "
                    "and no grouping identifier for dependence-aware splitting"
                ),
            },
            "rice": {
                "title": "Rice (Cammeo and Osmancik)",
                "creators": ["Ilkay Cinar", "Murat Koklu"],
                "year": 2019,
                "repository": "UCI Machine Learning Repository",
                "doi": "10.24432/C5MW4Z",
                "dataset_page": (
                    "https://archive.ics.uci.edu/dataset/545/"
                    "rice+cammeo+and+osmancik"
                ),
                "archive": SOURCE_ARCHIVES["rice"],
                "included_files": {
                    path: EXPECTED_SOURCE_HASHES[path]
                    for path in (RICE_FILE, RICE_CITATION)
                },
                "rows": datasets["rice"]["dataset"]["audit"]["rows"],
                "features": list(RICE_FEATURES),
                "target_encoding": {"0": "Osmancik", "1": "Cammeo"},
                "positive_class": "Cammeo",
                "positive_class_note": (
                    "chosen to define binary metrics; it is not a value judgment"
                ),
                "missing_values": 0,
                "known_metadata_limit": (
                    "the release provides no field, batch, camera-session, or other "
                    "grouping identifier for dependence-aware splitting"
                ),
            },
        },
    }


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=10)


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(
        path,
        dpi=dpi,
        facecolor=PAPER,
        edgecolor="none",
        metadata={"Software": f"praxagent {STUDY_ID}"},
    )
    plt.close(figure)


def plot_ranking_metrics(path: Path, results: dict[str, Any]) -> None:
    metrics = ("average_precision", "roc_auc")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    figure.patch.set_facecolor(PAPER)
    for row_index, key in enumerate(DATASET_ORDER):
        result = results[key]
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            style_axis(axis)
            axis.set_xlim(0.0, 1.02)
            axis.set_xticks(np.linspace(0, 1, 6))
            axis.set_yticks(
                range(len(MODEL_ORDER)),
                [MODEL_LABELS[model] for model in MODEL_ORDER],
            )
            axis.invert_yaxis()
            for y_position, model in enumerate(MODEL_ORDER):
                point = result["test_metrics"][model][metric]
                interval = result["paired_bootstrap"]["models"][model][metric]
                # A percentile-bootstrap interval need not contain the original
                # sample statistic.  Draw the interval endpoints and the point
                # independently instead of forcing Matplotlib's ``xerr`` API to
                # treat the point as the interval's centre.
                axis.hlines(
                    y_position,
                    interval["lower"],
                    interval["upper"],
                    color=MODEL_COLORS[model],
                    linewidth=1.8,
                )
                axis.vlines(
                    [interval["lower"], interval["upper"]],
                    y_position - 0.08,
                    y_position + 0.08,
                    color=MODEL_COLORS[model],
                    linewidth=1.8,
                )
                axis.plot(
                    point,
                    y_position,
                    marker=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model],
                    markersize=8,
                    linestyle="none",
                )
                label_to_left = point > 0.96
                axis.annotate(
                    metric_text(point),
                    (point, y_position),
                    xytext=(-9 if label_to_left else 9, -4),
                    textcoords="offset points",
                    ha="right" if label_to_left else "left",
                    va="center",
                    color=INK,
                    fontsize=9,
                )
            delta = result["paired_bootstrap"][
                "random_forest_minus_logistic_regression"
            ]["metrics"][metric]
            point_delta = (
                result["test_metrics"]["random_forest"][metric]
                - result["test_metrics"]["logistic_regression"][metric]
            )
            metric_short = "AP" if metric == "average_precision" else "ROC AUC"
            axis.set_title(
                f"{result['dataset']['display_name']}: {metric_short}\n"
                f"Forest - logistic = {point_delta:+.4f} "
                f"[{delta['lower']:+.4f}, {delta['upper']:+.4f}]",
                color=INK,
                fontsize=11,
                fontweight="bold",
            )
            axis.set_xlabel(f"Held-out {metric_short} (higher is better)", color=INK)
    figure.suptitle(
        "Held-out ranking metrics under the same evaluation discipline",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    figure.text(
        0.5,
        0.009,
        "Fixed 20% test sets; tuning used training folds only in the published workflow.\n"
        "Intervals describe resampling stability on each fixed test set, not population confidence intervals; compare models within a dataset.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.02, 0.075, 0.98, 0.95))
    save_figure(figure, path, dpi=200)


def plot_calibration(path: Path, results: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    figure.patch.set_facecolor(PAPER)
    for axis, key in zip(axes, DATASET_ORDER):
        result = results[key]
        style_axis(axis)
        axis.grid(axis="both", color=GRID, linewidth=0.8, alpha=0.8)
        axis.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1.2, label="Ideal reference")
        for model in MODEL_ORDER:
            bins = [row for row in result["calibration"] if row["model"] == model and row["count"]]
            x = [row["mean_predicted_probability"] for row in bins]
            y = [row["observed_positive_fraction"] for row in bins]
            brier = result["test_metrics"][model]["brier_score"]
            axis.plot(
                x,
                y,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.5,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} (Brier {brier:.4f})",
            )
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(result["dataset"]["display_name"], fontsize=12, fontweight="bold", color=INK)
        axis.set_xlabel("Mean predicted probability in fixed-width bin", color=INK)
        axis.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Observed positive fraction", color=INK)
    figure.suptitle(
        "Held-out probability calibration",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "Ten fixed probability bins; empty bins are omitted. Bin counts and exact values are in the figure receipt. These two test sets do not establish population calibration.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.92))
    save_figure(figure, path, dpi=200)


def plot_confusions(path: Path, results: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    figure.patch.set_facecolor(PAPER)
    for row_index, key in enumerate(DATASET_ORDER):
        result = results[key]
        labels = [result["dataset"]["negative_label"], result["dataset"]["positive_label"]]
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            confusion = result["test_metrics"][model]["confusion"]
            matrix = np.asarray(
                [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]],
                dtype=float,
            )
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(float(matrix.max()), 1.0))
            for true_index in range(2):
                for predicted_index in range(2):
                    value = int(matrix[true_index, predicted_index])
                    text_color = "white" if value > matrix.max() * 0.55 else INK
                    axis.text(predicted_index, true_index, f"{value:,}", ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)
            axis.set_xticks([0, 1], labels, rotation=18, ha="right", fontsize=8)
            axis.set_yticks([0, 1], labels, fontsize=8)
            axis.set_xlabel("Predicted at threshold 0.5", color=INK, fontsize=9)
            if column_index == 0:
                axis.set_ylabel(f"{result['dataset']['display_name']}\nTrue class", color=INK, fontsize=9)
            axis.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold", color=INK)
            for spine in axis.spines.values():
                spine.set_color(GRID)
    figure.suptitle(
        "Held-out confusion counts use the same 0.5 probability threshold",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    figure.text(
        0.5,
        0.01,
        "Counts describe these fixed test rows. The HTRU2 positive class is rare, so the dummy-prior baseline predicts no positives at threshold 0.5.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.03, 0.05, 0.98, 0.94))
    save_figure(figure, path, dpi=200)


def plot_importance(path: Path, results: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    figure.patch.set_facecolor(PAPER)
    for row_index, key in enumerate(DATASET_ORDER):
        result = results[key]
        all_records = [
            record
            for model in ("logistic_regression", "random_forest")
            for record in result["permutation_importance"]["models"][model]
        ]
        lower_bound = min(
            0.0,
            min(
                record["mean_average_precision_decrease"]
                - record["standard_deviation_across_permutations"]
                for record in all_records
            ),
        )
        upper_bound = max(
            0.0,
            max(
                record["mean_average_precision_decrease"]
                + record["standard_deviation_across_permutations"]
                for record in all_records
            ),
        )
        span = max(upper_bound - lower_bound, 0.01)
        x_limits = (lower_bound - 0.08 * span, upper_bound + 0.16 * span)
        for column_index, model in enumerate(("logistic_regression", "random_forest")):
            axis = axes[row_index, column_index]
            style_axis(axis)
            records = list(reversed(result["permutation_importance"]["models"][model]))
            positions = np.arange(len(records))
            means = np.asarray([item["mean_average_precision_decrease"] for item in records])
            spread = np.asarray([item["standard_deviation_across_permutations"] for item in records])
            axis.errorbar(
                means,
                positions,
                xerr=spread,
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markersize=6,
                capsize=3,
                linewidth=1.3,
            )
            axis.axvline(0, color=MUTED, linewidth=1.0)
            axis.set_xlim(*x_limits)
            axis.set_yticks(positions, [item["feature_label"] for item in records], fontsize=8)
            axis.set_title(
                f"{result['dataset']['display_name']}: {MODEL_LABELS[model]}",
                fontsize=11,
                fontweight="bold",
                color=INK,
            )
            axis.set_xlabel("Average-precision decrease after shuffling", fontsize=9, color=INK)
    figure.suptitle(
        "Held-out permutation importance is model-specific",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        "Points are means over 30 shuffles; whiskers are one repeat standard deviation, not confidence intervals. Correlated features can share or mask importance.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.98, 0.95))
    save_figure(figure, path, dpi=200)


def plot_og_card(path: Path, results: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.3), sharex=True)
    figure.patch.set_facecolor(PAPER)
    for axis, key in zip(axes, DATASET_ORDER):
        result = results[key]
        style_axis(axis)
        axis.set_xlim(0, 1.02)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticks(range(3), [MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
        axis.invert_yaxis()
        for position, model in enumerate(MODEL_ORDER):
            value = result["test_metrics"][model]["average_precision"]
            axis.scatter(value, position, s=85, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model], zorder=3)
            label_to_left = value > 0.96
            axis.annotate(
                metric_text(value),
                (value, position),
                xytext=(-8 if label_to_left else 8, -4),
                textcoords="offset points",
                ha="right" if label_to_left else "left",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold",
            )
        axis.set_title(result["dataset"]["display_name"], fontsize=14, fontweight="bold", color=INK)
        axis.set_xlabel("Held-out average precision", color=INK, fontsize=10)
    figure.text(0.05, 0.935, "Forest in the Sky!", fontsize=27, fontweight="bold", color=INK, ha="left")
    figure.text(
        0.05,
        0.885,
        "Logistic regression and random forest on pulsar candidates and rice grains.",
        fontsize=14,
        color=MUTED,
        ha="left",
    )
    figure.text(
        0.5,
        0.035,
        "Fixed held-out results from these two datasets; model tuning used training folds only.",
        fontsize=10,
        color=MUTED,
        ha="center",
    )
    figure.tight_layout(rect=(0.04, 0.08, 0.97, 0.81))
    save_figure(figure, path, dpi=100)
    if png_dimensions(path) != (1200, 630):
        raise ReproductionError("og-card.png must be exactly 1200 by 630 pixels")


def make_notebook(generator_hash: str, lock_hash: str) -> dict[str, Any]:
    base = (
        "https://raw.githubusercontent.com/praxagent/praxagent-ai/main/"
        "blog-source/content/knowledge-base/deep-dives/"
        "logistic-regression-random-forest/"
    )
    required = {
        "reproduce.py": generator_hash,
        LOCK_FILE: lock_hash,
        ATTRIBUTION_FILE: sha256_file(HERE / ATTRIBUTION_FILE),
        **EXPECTED_SOURCE_HASHES,
    }
    download_code = """from pathlib import Path
from urllib.request import urlopen
import hashlib
import subprocess
import sys

BASE = %s
FILES = %s

for relative, expected_sha256 in FILES.items():
    destination = Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = urlopen(BASE + relative, timeout=60).read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f\"hash mismatch for {relative}: {observed}\")
    destination.write_bytes(payload)

subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"uv==0.11.2\"], check=True)
subprocess.run([\"uv\", \"run\", \"--frozen\", \"reproduce.py\", \"--generate\"], check=True)
print(\"Analysis artifacts regenerated in this Colab runtime.\")
""" % (repr(base), repr(required))
    display_code = """from IPython.display import display, Image

for figure in [
    \"fig-held-out-ranking-metrics.png\",
    \"fig-held-out-calibration.png\",
    \"fig-held-out-confusion-counts.png\",
    \"fig-held-out-permutation-importance.png\",
]:
    display(Image(filename=figure))
"""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Forest in the Sky: reproducible analysis companion\n",
                    "\n",
                    "Downloads hash-pinned public bundle files, runs the locked analysis, and displays the generated figures.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in download_code.rstrip().splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in display_code.rstrip().splitlines()],
            },
        ],
        "metadata": {
            "colab": {"name": NOTEBOOK, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def figure_receipt(
    *,
    figure_id: str,
    title: str,
    description: str,
    alt_text: str,
    caption: str,
    output_path: str,
    output_dir: Path,
    generator_hash: str,
    analysis_hash: str,
    source_fields: list[str],
    plotted_data: Any,
    uncertainty: dict[str, Any],
    exclusions: list[str],
    redundant_channels: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "figure_id": figure_id,
        "title": title,
        "description": description,
        "alt_text": alt_text,
        "caption_suggestion": caption,
        "data_source": {
            "artifact": ANALYSIS_RECEIPT,
            "artifact_sha256": analysis_hash,
            "source_fields": source_fields,
            "row_selection": "the fixed held-out rows declared in the analysis receipt",
            "transformation": "computed by reproduce.py without manual figure values",
        },
        "plotted_data": plotted_data,
        "uncertainty": uncertainty,
        "claim_scope_exclusions": exclusions,
        "accessibility": {
            "color_is_not_the_only_channel": True,
            "redundant_channels": redundant_channels,
            "full_text_equivalent": "plotted_data and caption_suggestion in this receipt",
            "alt_text": alt_text,
        },
        "provenance": {
            "study_id": STUDY_ID,
            "generator": "reproduce.py",
            "generator_sha256": generator_hash,
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "verification_command": "uv run --frozen reproduce.py --verify",
            "outputs": {output_path: sha256_file(output_dir / output_path)},
        },
    }


def build_figure_receipts(
    output_dir: Path,
    results: dict[str, Any],
    generator_hash: str,
    analysis_hash: str,
) -> dict[str, dict[str, Any]]:
    ranking_data = {}
    calibration_data = {}
    confusion_data = {}
    importance_data = {}
    for key in DATASET_ORDER:
        ranking_data[key] = {}
        for metric in ("average_precision", "roc_auc"):
            ranking_data[key][metric] = {
                "point_estimates": {
                    model: rounded(results[key]["test_metrics"][model][metric])
                    for model in MODEL_ORDER
                },
                "model_stability_intervals": {
                    model: results[key]["paired_bootstrap"]["models"][model][metric]
                    for model in MODEL_ORDER
                },
                "random_forest_minus_logistic_regression": {
                    "point": rounded(
                        results[key]["test_metrics"]["random_forest"][metric]
                        - results[key]["test_metrics"]["logistic_regression"][metric]
                    ),
                    "interval": results[key]["paired_bootstrap"][
                        "random_forest_minus_logistic_regression"
                    ]["metrics"][metric],
                },
            }
        calibration_data[key] = {
            "bins": results[key]["calibration"],
            "probability_score_point_estimates": {
                model: {
                    metric: rounded(results[key]["test_metrics"][model][metric])
                    for metric in ("brier_score", "log_loss")
                }
                for model in MODEL_ORDER
            },
        }
        confusion_data[key] = {
            model: results[key]["test_metrics"][model]["confusion"]
            for model in MODEL_ORDER
        }
        importance_data[key] = results[key]["permutation_importance"]

    htru_ap = ranking_data["htru2"]["average_precision"]["point_estimates"]
    rice_ap = ranking_data["rice"]["average_precision"]["point_estimates"]
    htru_roc = ranking_data["htru2"]["roc_auc"]["point_estimates"]
    rice_roc = ranking_data["rice"]["roc_auc"]["point_estimates"]

    def ranking_difference(key: str, metric: str) -> dict[str, Any]:
        return ranking_data[key][metric]["random_forest_minus_logistic_regression"]

    def calibration_bin(key: str, model: str, index: int) -> dict[str, Any]:
        return next(
            row
            for row in calibration_data[key]["bins"]
            if row["model"] == model and row["bin_index"] == index
        )

    def middle_bin_count_range(key: str) -> tuple[int, int]:
        counts = [
            row["count"]
            for row in calibration_data[key]["bins"]
            if row["model"] in {"logistic_regression", "random_forest"}
            and 1 <= row["bin_index"] <= 8
            and row["count"] > 0
        ]
        return min(counts), max(counts)

    def importance_record(key: str, model: str, feature: str) -> dict[str, Any]:
        return next(
            row
            for row in importance_data[key]["models"][model]
            if row["feature"] == feature
        )

    htru_ap_difference = ranking_difference("htru2", "average_precision")
    htru_roc_difference = ranking_difference("htru2", "roc_auc")
    rice_ap_difference = ranking_difference("rice", "average_precision")
    rice_roc_difference = ranking_difference("rice", "roc_auc")
    htru_logistic_low = calibration_bin("htru2", "logistic_regression", 0)
    htru_forest_low = calibration_bin("htru2", "random_forest", 0)
    htru_logistic_high = calibration_bin("htru2", "logistic_regression", 9)
    htru_forest_high = calibration_bin("htru2", "random_forest", 9)
    rice_logistic_low = calibration_bin("rice", "logistic_regression", 0)
    rice_forest_low = calibration_bin("rice", "random_forest", 0)
    rice_logistic_high = calibration_bin("rice", "logistic_regression", 9)
    rice_forest_high = calibration_bin("rice", "random_forest", 9)
    htru_middle_min, htru_middle_max = middle_bin_count_range("htru2")
    rice_middle_min, rice_middle_max = middle_bin_count_range("rice")
    htru_logistic_kurtosis = importance_record(
        "htru2", "logistic_regression", "profile_excess_kurtosis"
    )
    htru_forest_kurtosis = importance_record(
        "htru2", "random_forest", "profile_excess_kurtosis"
    )
    htru_logistic_dm_spread = importance_record(
        "htru2", "logistic_regression", "dm_snr_standard_deviation"
    )
    htru_forest_dm_spread = importance_record(
        "htru2", "random_forest", "dm_snr_standard_deviation"
    )
    rice_logistic_convex = importance_record(
        "rice", "logistic_regression", "convex_area"
    )
    rice_logistic_major = importance_record(
        "rice", "logistic_regression", "major_axis_length"
    )
    rice_forest_major = importance_record(
        "rice", "random_forest", "major_axis_length"
    )
    rice_forest_perimeter = importance_record(
        "rice", "random_forest", "perimeter"
    )
    interval_note = {
        "kind": "95% paired stratified bootstrap stability interval",
        "replicates": BOOTSTRAP_REPLICATES,
        "not_a_population_confidence_interval": True,
        "by_dataset": {
            key: {
                "random_state": results[key]["paired_bootstrap"]["random_state"],
                "interval_endpoints": results[key]["paired_bootstrap"][
                    "interval_endpoints"
                ],
                "limitations": results[key]["paired_bootstrap"]["limitations"],
            }
            for key in DATASET_ORDER
        },
    }
    shared_exclusions = [
        "no population-wide performance claim",
        "no causal claim",
        "no independence claim beyond the released row tables",
        "no claim that either selected model is universally superior",
    ]
    receipts = {
        PERFORMANCE_RECEIPT: figure_receipt(
            figure_id="held-out-ranking-metrics",
            title="Held-out ranking metrics",
            description="Dummy, logistic-regression, and random-forest AP and ROC AUC on each fixed test set with fixed-model bootstrap stability intervals.",
            alt_text=(
                f"Held-out HTRU2 average precision is {htru_ap['dummy_prior']:.4f} for the dummy prior, "
                f"{htru_ap['logistic_regression']:.4f} for logistic regression, and {htru_ap['random_forest']:.4f} for random forest; "
                f"the forest-minus-logistic difference is {htru_ap_difference['point']:+.4f} with fixed-test stability interval "
                f"[{htru_ap_difference['interval']['lower']:+.4f}, {htru_ap_difference['interval']['upper']:+.4f}]. "
                f"HTRU2 ROC AUC is {htru_roc['dummy_prior']:.4f}, {htru_roc['logistic_regression']:.4f}, and {htru_roc['random_forest']:.4f}, respectively; "
                f"the paired difference is {htru_roc_difference['point']:+.4f} "
                f"[{htru_roc_difference['interval']['lower']:+.4f}, {htru_roc_difference['interval']['upper']:+.4f}]. "
                f"Held-out Rice average precision is {rice_ap['dummy_prior']:.4f}, {rice_ap['logistic_regression']:.4f}, and {rice_ap['random_forest']:.4f}; "
                f"the paired difference is {rice_ap_difference['point']:+.4f} "
                f"[{rice_ap_difference['interval']['lower']:+.4f}, {rice_ap_difference['interval']['upper']:+.4f}]. "
                f"Rice ROC AUC is {rice_roc['dummy_prior']:.4f}, {rice_roc['logistic_regression']:.4f}, and {rice_roc['random_forest']:.4f}; "
                f"the paired difference is {rice_roc_difference['point']:+.4f} "
                f"[{rice_roc_difference['interval']['lower']:+.4f}, {rice_roc_difference['interval']['upper']:+.4f}]. "
                "Intervals resample each fixed test set within class and are not population confidence intervals."
            ),
            caption=(
                "Finding: under one shared training-only tuning and fixed-holdout protocol, the two learned models both exceed the dummy-prior baseline on average precision and ROC AUC, while their point-estimate ordering can differ by metric and dataset. "
                "The plotted intervals are paired fixed-test bootstrap stability intervals, not population confidence intervals; they omit split, tuning, refitting, grouping, and acquisition uncertainty. Compare model values within a dataset because average precision depends on prevalence."
            ),
            output_path=PERFORMANCE_FIGURE,
            output_dir=output_dir,
            generator_hash=generator_hash,
            analysis_hash=analysis_hash,
            source_fields=["test_metrics.*.average_precision", "test_metrics.*.roc_auc", "paired_bootstrap.models.*", "paired_bootstrap.random_forest_minus_logistic_regression.metrics"],
            plotted_data=ranking_data,
            uncertainty=interval_note,
            exclusions=shared_exclusions,
            redundant_channels=["direct labels", "marker shapes", "position", "exact values in receipt"],
        ),
        CALIBRATION_RECEIPT: figure_receipt(
            figure_id="held-out-calibration",
            title="Held-out probability calibration",
            description="Ten fixed-width probability bins for each model and dataset, with exact bin counts and Brier scores.",
            alt_text=(
                f"On HTRU2, Brier score and log loss are "
                f"{results['htru2']['test_metrics']['dummy_prior']['brier_score']:.4f} and {results['htru2']['test_metrics']['dummy_prior']['log_loss']:.4f} for the dummy prior, "
                f"{results['htru2']['test_metrics']['logistic_regression']['brier_score']:.4f} and {results['htru2']['test_metrics']['logistic_regression']['log_loss']:.4f} for logistic regression, and "
                f"{results['htru2']['test_metrics']['random_forest']['brier_score']:.4f} and {results['htru2']['test_metrics']['random_forest']['log_loss']:.4f} for random forest. "
                f"In the 0.0 to 0.1 reliability bin, logistic regression has {htru_logistic_low['count']:,} rows with mean prediction {htru_logistic_low['mean_predicted_probability']:.4f} and observed fraction {htru_logistic_low['observed_positive_fraction']:.4f}; "
                f"random forest has {htru_forest_low['count']:,} rows at {htru_forest_low['mean_predicted_probability']:.4f} and {htru_forest_low['observed_positive_fraction']:.4f}. "
                f"In the 0.9 to 1.0 bin, the corresponding counts and values are {htru_logistic_high['count']:,} at {htru_logistic_high['mean_predicted_probability']:.4f} and {htru_logistic_high['observed_positive_fraction']:.4f}, and "
                f"{htru_forest_high['count']:,} at {htru_forest_high['mean_predicted_probability']:.4f} and {htru_forest_high['observed_positive_fraction']:.4f}. "
                f"HTRU2 middle bins contain only {htru_middle_min} to {htru_middle_max} logistic or forest rows. "
                f"On Rice, Brier score and log loss are {results['rice']['test_metrics']['dummy_prior']['brier_score']:.4f} and {results['rice']['test_metrics']['dummy_prior']['log_loss']:.4f} for the dummy prior, "
                f"{results['rice']['test_metrics']['logistic_regression']['brier_score']:.4f} and {results['rice']['test_metrics']['logistic_regression']['log_loss']:.4f} for logistic regression, and "
                f"{results['rice']['test_metrics']['random_forest']['brier_score']:.4f} and {results['rice']['test_metrics']['random_forest']['log_loss']:.4f} for random forest. "
                f"The Rice low and high bins contain {rice_logistic_low['count']} and {rice_logistic_high['count']} logistic rows and {rice_forest_low['count']} and {rice_forest_high['count']} forest rows; "
                f"all middle bins contain {rice_middle_min} to {rice_middle_max} rows. Sparse bins make local departures from the diagonal noisy."
            ),
            caption=(
                "Finding: probability quality differs by model and dataset even when discrimination metrics are close. Each point summarizes one nonempty, fixed-width probability bin, and the legend reports the held-out Brier score. Sparse middle-probability bins can move sharply; these fixed test sets do not establish population calibration."
            ),
            output_path=CALIBRATION_FIGURE,
            output_dir=output_dir,
            generator_hash=generator_hash,
            analysis_hash=analysis_hash,
            source_fields=["calibration", "test_metrics.*.brier_score"],
            plotted_data=calibration_data,
            uncertainty={"kind": "none plotted", "bin_counts_are_reported": True},
            exclusions=shared_exclusions + ["no calibrated-probability guarantee outside these test rows"],
            redundant_channels=["direct labels", "marker shapes", "line styles", "position", "exact values in receipt"],
        ),
        CONFUSION_RECEIPT: figure_receipt(
            figure_id="held-out-confusion-counts",
            title="Held-out confusion counts",
            description="True-negative, false-positive, false-negative, and true-positive counts at the fixed 0.5 threshold.",
            alt_text=(
                f"At threshold 0.5 on HTRU2, the dummy prior has {confusion_data['htru2']['dummy_prior']['tn']:,} true negatives, "
                f"{confusion_data['htru2']['dummy_prior']['fp']} false positives, {confusion_data['htru2']['dummy_prior']['fn']} false negatives, and {confusion_data['htru2']['dummy_prior']['tp']} true positives; "
                f"logistic regression has {confusion_data['htru2']['logistic_regression']['tn']:,}, {confusion_data['htru2']['logistic_regression']['fp']}, {confusion_data['htru2']['logistic_regression']['fn']}, and {confusion_data['htru2']['logistic_regression']['tp']}; "
                f"random forest has {confusion_data['htru2']['random_forest']['tn']:,}, {confusion_data['htru2']['random_forest']['fp']}, {confusion_data['htru2']['random_forest']['fn']}, and {confusion_data['htru2']['random_forest']['tp']}. "
                f"On Rice, the dummy prior has {confusion_data['rice']['dummy_prior']['tn']} true negatives, {confusion_data['rice']['dummy_prior']['fp']} false positives, "
                f"{confusion_data['rice']['dummy_prior']['fn']} false negatives, and {confusion_data['rice']['dummy_prior']['tp']} true positives; "
                f"logistic regression has {confusion_data['rice']['logistic_regression']['tn']}, {confusion_data['rice']['logistic_regression']['fp']}, {confusion_data['rice']['logistic_regression']['fn']}, and {confusion_data['rice']['logistic_regression']['tp']}; "
                f"random forest has {confusion_data['rice']['random_forest']['tn']}, {confusion_data['rice']['random_forest']['fp']}, {confusion_data['rice']['random_forest']['fn']}, and {confusion_data['rice']['random_forest']['tp']}."
            ),
            caption=(
                "Finding: the 0.5 threshold exposes which errors each model makes, rather than hiding them inside one score. Counts describe only the fixed test rows. The dummy baseline is shown under the same threshold; class imbalance makes its HTRU2 accuracy potentially look acceptable while its positive recall is zero."
            ),
            output_path=CONFUSION_FIGURE,
            output_dir=output_dir,
            generator_hash=generator_hash,
            analysis_hash=analysis_hash,
            source_fields=["test_metrics.*.confusion"],
            plotted_data=confusion_data,
            uncertainty={"kind": "none plotted", "threshold": THRESHOLD},
            exclusions=shared_exclusions,
            redundant_channels=["direct labels", "cell position", "exact values in receipt"],
        ),
        IMPORTANCE_RECEIPT: figure_receipt(
            figure_id="held-out-permutation-importance",
            title="Held-out permutation importance",
            description="Mean AP decrease and repeat spread when each feature is shuffled on the fixed test set.",
            alt_text=(
                f"Four panels show mean held-out AP decrease over {PERMUTATION_REPEATS} feature shuffles, with repeat standard deviations. "
                f"In HTRU2, profile excess kurtosis has the largest decrease for logistic regression, {htru_logistic_kurtosis['mean_average_precision_decrease']:.4f}, and random forest, {htru_forest_kurtosis['mean_average_precision_decrease']:.4f}; "
                f"the next values are {htru_logistic_dm_spread['mean_average_precision_decrease']:.4f} and {htru_forest_dm_spread['mean_average_precision_decrease']:.4f} for dispersion-measure signal-to-noise-ratio standard deviation. "
                f"In Rice, logistic regression's largest decreases are {rice_logistic_convex['mean_average_precision_decrease']:.4f} for convex area and {rice_logistic_major['mean_average_precision_decrease']:.4f} for major-axis length; "
                f"random forest's are {rice_forest_major['mean_average_precision_decrease']:.4f} for major-axis length and {rice_forest_perimeter['mean_average_precision_decrease']:.4f} for perimeter. "
                "Shuffling also breaks relationships among predictors, so correlated features can share or mask importance. The whiskers are permutation-repeat spread, not confidence intervals, and the results are not causal."
            ),
            caption=(
                "Finding: feature importance is a property of a fitted model and scoring procedure, not an intrinsic ranking of measurements. Points are mean held-out AP decreases over 30 shuffles and whiskers are one repeat standard deviation, not confidence intervals. Correlated inputs can share or mask importance, and this post-evaluation diagnostic was not used to tune either model."
            ),
            output_path=IMPORTANCE_FIGURE,
            output_dir=output_dir,
            generator_hash=generator_hash,
            analysis_hash=analysis_hash,
            source_fields=["permutation_importance.models.*"],
            plotted_data=importance_data,
            uncertainty={
                "kind": "permutation repeat standard deviation",
                "repeats": PERMUTATION_REPEATS,
                "random_states": {
                    key: results[key]["permutation_importance"]["random_states"]
                    for key in DATASET_ORDER
                },
                "not_a_confidence_interval": True,
            },
            exclusions=shared_exclusions + ["no model-independent feature ranking"],
            redundant_channels=["direct labels", "marker shapes", "position", "exact values in receipt"],
        ),
        OG_RECEIPT: figure_receipt(
            figure_id="forest-in-the-sky-card",
            title="Forest in the Sky!",
            description="Social card with the three held-out AP point estimates for each dataset.",
            alt_text=(
                f"Forest in the Sky. Held-out average precision for dummy prior, logistic regression, and random forest is {htru_ap['dummy_prior']:.4f}, {htru_ap['logistic_regression']:.4f}, and {htru_ap['random_forest']:.4f} on HTRU2, and {rice_ap['dummy_prior']:.4f}, {rice_ap['logistic_regression']:.4f}, and {rice_ap['random_forest']:.4f} on Rice. Results are limited to these fixed held-out sets."
            ),
            caption=(
                "The card summarizes held-out average precision under the same training-only tuning and fixed-test protocol. It reports point estimates only and makes no population or universal model-ranking claim."
            ),
            output_path=OG_CARD,
            output_dir=output_dir,
            generator_hash=generator_hash,
            analysis_hash=analysis_hash,
            source_fields=["test_metrics.*.average_precision"],
            plotted_data={
                key: ranking_data[key]["average_precision"]["point_estimates"]
                for key in DATASET_ORDER
            },
            uncertainty={"kind": "none plotted on summary card", "intervals_available_in": PERFORMANCE_RECEIPT},
            exclusions=shared_exclusions,
            redundant_channels=["direct labels", "position", "exact values in receipt"],
        ),
    }
    return receipts


def provenance_numbers(
    output_dir: Path,
    receipt_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    """Build the post-wide numerical ledger from committed receipt payloads.

    Individual headline bindings make prose drift easy to detect. The complete
    dataset and per-figure bindings enumerate the remaining evidentiary values,
    including every plotted coordinate, interval, count, and importance value.
    """
    analysis = json.loads((output_dir / ANALYSIS_RECEIPT).read_text(encoding="utf-8"))
    datasets = analysis["datasets"]

    def entry(
        *,
        identifier: str,
        appears_as: str,
        value: Any,
        receipt: str,
        fragment: str,
        derivation: str,
    ) -> dict[str, Any]:
        return {
            "id": identifier,
            "value": value,
            "appears_as": appears_as,
            "receipt": receipt,
            "receipt_sha256": receipt_hashes[receipt],
            "source": f"{receipt}#{fragment}",
            "computation": derivation,
        }

    numbers: list[dict[str, Any]] = [
        entry(
            identifier="htru2-row-count",
            value=datasets["htru2"]["dataset"]["audit"]["rows"],
            appears_as="HTRU2 has 17,898 candidate rows",
            receipt=ANALYSIS_RECEIPT,
            fragment="datasets.htru2.dataset.audit.rows",
            derivation="Direct audited row count from the parsed HTRU2 source table.",
        ),
        entry(
            identifier="htru2-positive-count",
            value=datasets["htru2"]["dataset"]["class_counts"]["positive"],
            appears_as="HTRU2 includes 1,639 positive-labeled candidates",
            receipt=ANALYSIS_RECEIPT,
            fragment="datasets.htru2.dataset.class_counts.positive",
            derivation="Direct count of HTRU2 rows whose parsed class label is positive.",
        ),
        entry(
            identifier="rice-row-count",
            value=datasets["rice"]["dataset"]["audit"]["rows"],
            appears_as="Rice has 3,810 grain rows",
            receipt=ANALYSIS_RECEIPT,
            fragment="datasets.rice.dataset.audit.rows",
            derivation="Direct audited row count from the parsed Rice source table.",
        ),
        entry(
            identifier="rice-positive-count",
            value=datasets["rice"]["dataset"]["class_counts"]["positive"],
            appears_as="Rice includes 1,630 Cammeo grains",
            receipt=ANALYSIS_RECEIPT,
            fragment="datasets.rice.dataset.class_counts.positive",
            derivation="Direct count of Rice rows whose parsed class label is Cammeo.",
        ),
    ]
    for key in DATASET_ORDER:
        result = datasets[key]
        display = result["dataset"]["display_name"]
        for model in MODEL_ORDER:
            model_text = MODEL_LABELS[model].lower()
            for metric, short in (("average_precision", "AP"), ("roc_auc", "ROC AUC")):
                value = result["test_metrics"][model][metric]
                numbers.append(
                    entry(
                        identifier=f"{key}-{model}-{metric}",
                        value=rounded(value),
                        appears_as=(
                            f"{display} {model_text} held-out {short} "
                            f"{metric_text(value)}"
                        ),
                        receipt=ANALYSIS_RECEIPT,
                        fragment=f"datasets.{key}.test_metrics.{model}.{metric}",
                        derivation=f"Direct {metric} value computed on the fixed held-out {display} rows.",
                    )
                )
        point = (
            result["test_metrics"]["random_forest"]["average_precision"]
            - result["test_metrics"]["logistic_regression"]["average_precision"]
        )
        interval = result["paired_bootstrap"][
            "random_forest_minus_logistic_regression"
        ]["metrics"]["average_precision"]
        appears = (
            f"{display} random-forest-minus-logistic-regression held-out AP "
            f"{point:+.4f} [{interval['lower']:+.4f}, {interval['upper']:+.4f}]"
        )
        numbers.append(
            entry(
                identifier=f"{key}-paired-ap-difference",
                value={
                    "point": rounded(point),
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                },
                appears_as=appears,
                receipt=PERFORMANCE_RECEIPT,
                fragment=(
                    f"plotted_data.{key}.average_precision."
                    "random_forest_minus_logistic_regression"
                ),
                derivation=(
                    "Random-forest average precision minus logistic-regression average "
                    "precision, with the paired class-stratified bootstrap endpoints."
                ),
            )
        )

    for key in DATASET_ORDER:
        display = datasets[key]["dataset"]["display_name"]
        result = datasets[key]
        logistic_confusion = result["test_metrics"]["logistic_regression"]["confusion"]
        forest_confusion = result["test_metrics"]["random_forest"]["confusion"]
        middle_counts = [
            row["count"]
            for row in result["calibration"]
            if row["model"] in {"logistic_regression", "random_forest"}
            and 1 <= row["bin_index"] <= 8
            and row["count"] > 0
        ]
        derived_article_values = {
            "positive_prevalence_percent": rounded(
                100.0
                * result["dataset"]["class_counts"]["positive"]
                / result["dataset"]["audit"]["rows"],
                2,
            ),
            "test_class_counts": {
                "rows": result["split"]["test_rows"],
                "positive": result["split"]["test_positives"],
                "negative": result["split"]["test_rows"]
                - result["split"]["test_positives"],
            },
            "predicted_positive_at_0_5": {
                "logistic_regression": logistic_confusion["tp"]
                + logistic_confusion["fp"],
                "random_forest": forest_confusion["tp"] + forest_confusion["fp"],
            },
            "random_forest_minus_logistic_regression_workload": {
                "predicted_positive": (
                    forest_confusion["tp"]
                    + forest_confusion["fp"]
                    - logistic_confusion["tp"]
                    - logistic_confusion["fp"]
                ),
                "true_positive": forest_confusion["tp"] - logistic_confusion["tp"],
                "false_positive": forest_confusion["fp"] - logistic_confusion["fp"],
            },
            "nonempty_middle_reliability_bin_count_range": {
                "minimum": min(middle_counts),
                "maximum": max(middle_counts),
            },
            "selected_hyperparameters": {
                model: result["training_only_tuning"][model]["best_params"]
                for model in ("logistic_regression", "random_forest")
            },
        }
        numbers.append(
            entry(
                identifier=f"{key}-complete-held-out-result-values",
                appears_as=f"{display} complete held-out result values",
                value={
                    "receipt_dataset": result,
                    "derived_article_values": derived_article_values,
                },
                receipt=ANALYSIS_RECEIPT,
                fragment=f"datasets.{key}",
                derivation=(
                    "Complete generated dataset audit, split, tuning, test-metric, "
                    "paired-bootstrap, calibration, and permutation summaries. "
                    "Derived article values use positive / rows for prevalence, "
                    "tp + fp for predicted-positive workload, forest minus logistic "
                    "for paired workload deltas, test rows minus test positives for "
                    "test negatives, and min/max over nonempty middle-bin counts."
                ),
            )
        )

    figure_coverage = (
        (PERFORMANCE_RECEIPT, "held-out ranking figure values"),
        (CALIBRATION_RECEIPT, "held-out probability-score and reliability-bin figure values"),
        (CONFUSION_RECEIPT, "held-out confusion-count and workload figure values"),
        (IMPORTANCE_RECEIPT, "held-out permutation-importance figure values"),
    )
    for receipt, label in figure_coverage:
        payload = json.loads((output_dir / receipt).read_text(encoding="utf-8"))
        for key in DATASET_ORDER:
            display = datasets[key]["dataset"]["display_name"]
            numbers.append(
                entry(
                    identifier=f"{key}-{payload['figure_id']}-values",
                    appears_as=f"{display} {label}",
                    value=payload["plotted_data"][key],
                    receipt=receipt,
                    fragment=f"plotted_data.{key}",
                    derivation=(
                        "Complete machine-readable values plotted for this dataset, "
                        "including coordinates, counts, intervals, and spreads where applicable."
                    ),
                )
            )

    card = json.loads((output_dir / OG_RECEIPT).read_text(encoding="utf-8"))
    numbers.append(
        entry(
            identifier="social-card-held-out-average-precision-values",
            appears_as="Social-card held-out average-precision values",
            value=card["plotted_data"],
            receipt=OG_RECEIPT,
            fragment="plotted_data",
            derivation="All held-out average-precision point estimates printed on the social card.",
        )
    )
    return numbers


def build_outputs(output_dir: Path, reference_environment: dict[str, str]) -> None:
    validate_static_inputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RECEIPTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    generator_hash = sha256_file(HERE / "reproduce.py")
    lock_hash = sha256_file(HERE / LOCK_FILE)

    loaded = {"htru2": load_htru2(), "rice": load_rice()}
    results = {key: fit_and_evaluate(loaded[key]) for key in DATASET_ORDER}

    write_json(output_dir / SOURCE_MANIFEST, source_manifest(results))

    prediction_fields = [
        "dataset",
        "test_position",
        "original_row_index_zero_based",
        "target",
        *[
            field
            for model in MODEL_ORDER
            for field in (
                f"{model}__probability",
                f"{model}__prediction_at_0_5",
            )
        ],
    ]
    write_csv(
        output_dir / PREDICTIONS_CSV,
        prediction_fields,
        (row for key in DATASET_ORDER for row in results[key]["rows"]["predictions"]),
    )
    tuning_fields = [
        "dataset",
        "model",
        "candidate_index",
        "mean_validation_average_precision",
        "std_validation_average_precision",
        "rank_validation_average_precision",
        "params_json",
        "selected",
    ]
    write_csv(
        output_dir / TUNING_CSV,
        tuning_fields,
        (row for key in DATASET_ORDER for row in results[key]["rows"]["tuning"]),
    )
    calibration_fields = [
        "dataset",
        "model",
        "bin_index",
        "lower_bound_inclusive",
        "upper_bound",
        "upper_bound_inclusive",
        "count",
        "positive_count",
        "mean_predicted_probability",
        "observed_positive_fraction",
    ]
    write_csv(
        output_dir / CALIBRATION_CSV,
        calibration_fields,
        (row for key in DATASET_ORDER for row in results[key]["rows"]["calibration"]),
    )
    bootstrap_fields = [
        "dataset",
        "replicate",
        *[f"{model}__{metric}" for model in MODEL_ORDER for metric in METRIC_ORDER],
        *[f"random_forest_minus_logistic__{metric}" for metric in METRIC_ORDER],
    ]
    write_csv(
        output_dir / BOOTSTRAP_CSV,
        bootstrap_fields,
        (row for key in DATASET_ORDER for row in results[key]["rows"]["bootstrap"]),
    )
    importance_fields = [
        "dataset",
        "model",
        "feature",
        "feature_label",
        "repeat",
        "average_precision_decrease",
    ]
    write_csv(
        output_dir / IMPORTANCE_CSV,
        importance_fields,
        (row for key in DATASET_ORDER for row in results[key]["rows"]["importance"]),
    )
    metrics_rows = []
    for key in DATASET_ORDER:
        for model in MODEL_ORDER:
            row = {"dataset": key, "model": model, "threshold": THRESHOLD}
            row.update({metric: results[key]["test_metrics"][model][metric] for metric in METRIC_ORDER})
            row.update(results[key]["test_metrics"][model]["confusion"])
            metrics_rows.append(row)
    metrics_fields = ["dataset", "model", "threshold", *METRIC_ORDER, "tn", "fp", "fn", "tp"]
    write_csv(output_dir / METRICS_CSV, metrics_fields, metrics_rows)

    plot_ranking_metrics(output_dir / PERFORMANCE_FIGURE, results)
    plot_calibration(output_dir / CALIBRATION_FIGURE, results)
    plot_confusions(output_dir / CONFUSION_FIGURE, results)
    plot_importance(output_dir / IMPORTANCE_FIGURE, results)
    plot_og_card(output_dir / OG_CARD, results)

    write_json(output_dir / NOTEBOOK, make_notebook(generator_hash, lock_hash))

    analysis_payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "analysis_scope": (
            "Held-out comparison of unweighted, non-oversampled logistic regression "
            "and random forest, with a dummy-prior baseline, on UCI HTRU2 and UCI Rice"
        ),
        "protocol": {
            "split": "fixed stratified 80/20 holdout created before learned preprocessing",
            "primary_metric": "average_precision",
            "tuning": "five shared stratified training-only folds per dataset",
            "threshold_metrics": "fixed probability threshold 0.5",
            "test_rows_outside_fitting_and_tuning_in_published_workflow": True,
            "class_weighting": None,
            "oversampling": None,
            "model_comparison": "same source rows, split, training folds, and test metrics within each dataset",
            "selection_history": (
                "dataset, model, and narrative choices followed exploratory screening of "
                "these public datasets; held-out estimates are descriptive, not a "
                "prospective confirmation"
            ),
        },
        "datasets": {
            key: {name: value for name, value in results[key].items() if name != "rows"}
            for key in DATASET_ORDER
        },
        "artifacts": {
            relative: sha256_file(output_dir / relative)
            for relative in (
                SOURCE_MANIFEST,
                METRICS_CSV,
                PREDICTIONS_CSV,
                TUNING_CSV,
                CALIBRATION_CSV,
                BOOTSTRAP_CSV,
                IMPORTANCE_CSV,
                PERFORMANCE_FIGURE,
                CALIBRATION_FIGURE,
                CONFUSION_FIGURE,
                IMPORTANCE_FIGURE,
                OG_CARD,
                NOTEBOOK,
            )
        },
        "provenance": {
            "generator": "reproduce.py",
            "generator_sha256": generator_hash,
            "lockfile": LOCK_FILE,
            "lockfile_sha256": lock_hash,
            "attribution": ATTRIBUTION_FILE,
            "attribution_sha256": sha256_file(HERE / ATTRIBUTION_FILE),
            "source_hashes": EXPECTED_SOURCE_HASHES,
            "reference_environment": reference_environment,
        },
        "runtime_note": (
            "elapsed_seconds fields in training diagnostics are informational and are "
            "not publication artifacts because wall time is not byte-reproducible"
        ),
        "verification_scope": {
            "claim": "byte identity in the recorded reference environment only",
            "environment_recorded": True,
            "cross_platform_note": (
                "operating systems, numerical libraries, FreeType builds, and font "
                "files can change output bytes; canonical release artifacts should "
                "be generated and verified in the same pinned Linux environment"
            ),
        },
    }
    # Remove wall-clock diagnostics from the byte-reproduced receipt. They are useful
    # during a run but cannot be part of an exact deterministic artifact.
    for key in DATASET_ORDER:
        tuning = analysis_payload["datasets"][key]["training_only_tuning"]
        tuning["logistic_regression"].pop("elapsed_seconds_diagnostic_only", None)
        tuning["random_forest"].pop("elapsed_seconds_diagnostic_only", None)
    write_json(output_dir / ANALYSIS_RECEIPT, analysis_payload)
    analysis_hash = sha256_file(output_dir / ANALYSIS_RECEIPT)

    figure_receipts = build_figure_receipts(
        output_dir,
        results,
        generator_hash,
        analysis_hash,
    )
    for relative, payload in figure_receipts.items():
        write_json(output_dir / relative, payload)

    receipt_paths = [
        ANALYSIS_RECEIPT,
        METRICS_CSV,
        PREDICTIONS_CSV,
        TUNING_CSV,
        CALIBRATION_CSV,
        BOOTSTRAP_CSV,
        IMPORTANCE_CSV,
        SOURCE_MANIFEST,
        PERFORMANCE_RECEIPT,
        CALIBRATION_RECEIPT,
        CONFUSION_RECEIPT,
        IMPORTANCE_RECEIPT,
        OG_RECEIPT,
        *SINGLE_MODEL_RECEIPTS,
        SINGLE_MODEL_GENERATOR,
        LOCK_FILE,
        ATTRIBUTION_FILE,
        HTRU_FILE,
        HTRU_README,
        RICE_FILE,
        RICE_CITATION,
    ]
    receipts = {}
    for relative in receipt_paths:
        source = (output_dir / relative) if (output_dir / relative).is_file() else (HERE / relative)
        receipts[relative] = sha256_file(source)

    manifest = {
        "schema_version": 1,
        "local_bundle": True,
        "study_id": STUDY_ID,
        "generator": {
            "path": "reproduce.py",
            "sha256": generator_hash,
            "verify": "uv run --frozen reproduce.py --verify",
        },
        "notebook": {"path": NOTEBOOK, "sha256": sha256_file(output_dir / NOTEBOOK)},
        "receipts": receipts,
        "figures": [
            *SINGLE_MODEL_FIGURES,
            PERFORMANCE_FIGURE,
            CALIBRATION_FIGURE,
            CONFUSION_FIGURE,
            IMPORTANCE_FIGURE,
            OG_CARD,
        ],
        "numbers": provenance_numbers(output_dir, receipts),
        "scope": {
            "primary_dataset": "UCI HTRU2",
            "transfer_dataset": "UCI Rice (Cammeo and Osmancik)",
            "no_population_inference": True,
            "bootstrap_intervals_are_not_population_confidence_intervals": True,
        },
        "reference_environment": reference_environment,
        "verification_scope": {
            "claim": "byte identity in the recorded reference environment only",
            "current_artifact_environment": "the reference_environment object above",
            "recommended_release_environment": (
                "a pinned Linux publication environment; these artifacts must be "
                "regenerated there before they may be described as Linux-generated"
            ),
            "cross_platform_byte_identity_claimed": False,
        },
    }
    write_json(output_dir / PROVENANCE, manifest)


def generated_names() -> tuple[str, ...]:
    return (
        SOURCE_MANIFEST,
        ANALYSIS_RECEIPT,
        METRICS_CSV,
        PREDICTIONS_CSV,
        TUNING_CSV,
        CALIBRATION_CSV,
        BOOTSTRAP_CSV,
        IMPORTANCE_CSV,
        PERFORMANCE_FIGURE,
        CALIBRATION_FIGURE,
        CONFUSION_FIGURE,
        IMPORTANCE_FIGURE,
        OG_CARD,
        PERFORMANCE_RECEIPT,
        CALIBRATION_RECEIPT,
        CONFUSION_RECEIPT,
        IMPORTANCE_RECEIPT,
        OG_RECEIPT,
        NOTEBOOK,
        PROVENANCE,
    )


def generate() -> None:
    started = time.perf_counter()
    build_outputs(HERE, current_environment())
    elapsed = time.perf_counter() - started
    print(
        f"generated {STUDY_ID} artifacts in {elapsed:.1f}s; runtime is diagnostic only"
    )


def verify() -> None:
    reference = json.loads((HERE / ANALYSIS_RECEIPT).read_text(encoding="utf-8"))[
        "provenance"
    ]["reference_environment"]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="verify-logistic-forest-") as temp:
        candidate = Path(temp)
        build_outputs(candidate, reference)
        differences: list[str] = []
        for relative in generated_names():
            committed = HERE / relative
            rebuilt = candidate / relative
            if not committed.is_file():
                differences.append(f"missing committed artifact: {relative}")
            elif not rebuilt.is_file():
                differences.append(f"missing rebuilt artifact: {relative}")
            elif committed.read_bytes() != rebuilt.read_bytes():
                differences.append(f"byte mismatch: {relative}")
        if differences:
            raise ReproductionError(
                "verification failed:\n"
                + "\n".join(differences)
                + "\nreference environment:\n"
                + json.dumps(reference, indent=2, sort_keys=True)
                + "\ncurrent environment:\n"
                + json.dumps(current_environment(), indent=2, sort_keys=True)
            )
    elapsed = time.perf_counter() - started
    print(
        f"verified {len(generated_names())} generated artifacts byte for byte in {elapsed:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    else:
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
