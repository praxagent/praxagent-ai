#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "einops==0.6.1",
#   "numpy==1.26.4",
#   "torch==2.4.0",
#   "transformers==4.43.4",
# ]
# ///
"""Reproduce an auditable SciFact retrieval comparison and its artifacts.

The expensive path downloads the checksum-pinned SciFact benchmark and the
revision-pinned Jina embedding model, then evaluates three document indexes:

* naive: fixed 256-token chunks encoded independently;
* late: the same spans pooled after full-document contextualization; and
* whole_document: one pooled vector for each complete document.

All three arms use the same corpus, queries, qrels, tokenizer, model weights,
query encoder, document text, and cosine document ranking. Chunked document
scores are the maximum score over that document's chunks.

Run the model evaluation (uses the PEP 723 lock beside this file):

    uv run --frozen reproduce.py --run --device cpu --batch-size 32 --threads 8

Verify committed metrics, rankings, figures, receipts, and hashes offline:

    python3 reproduce.py --verify

The verification path uses only the Python standard library. Model weights,
the benchmark archive, embeddings, and caches are deliberately not committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
RECEIPTS_DIR = HERE / "receipts"

DATASET_NAME = "SciFact BEIR retrieval benchmark"
DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
    "datasets/scifact.zip"
)
DATASET_ARCHIVE_SHA256 = (
    "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
)
DATASET_COMPONENT_SHA256 = {
    "corpus.jsonl": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
    "queries.jsonl": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
    "qrels/test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
}
EXPECTED_CORPUS_COUNT = 5_183
EXPECTED_TEST_QUERY_COUNT = 300
EXPECTED_TEST_QREL_COUNT = 339

MODEL_ID = "jinaai/jina-embeddings-v2-small-en"
MODEL_REVISION = "44e7d1d6caec8c883c2d4b207588504d519788d0"
MODEL_CODE_ID = "jinaai/jina-bert-implementation"
MODEL_CODE_REVISION = "f3ec4cf7de7e561007f27c9efc7148b0bd713f81"
MODEL_SNAPSHOT_SHA256 = {
    "config.json": "940582c6d428408d92d628b94541092e7ca9cbee6c3f147de54e9d4ddfe30ace",
    "model.safetensors": "c9a9a7ec012d01efd780474fbb65e25917f3a2aebdff84b5f87daa00f7e90b27",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "e9f999ac74497843ed9f4303246a8f43d9f100ee8aab8e133667903f447ceb48",
    "tokenizer_config.json": "25cbd867af916b5a5718e80e9a702cf72c61bf890a1bc141c6fbce6d74c99632",
    "vocab.txt": "109753d618dbb576a35112f9c20ef35cf3517d46106175bcf010c986a4bef1df",
}
MODEL_CODE_SNAPSHOT_SHA256 = {
    "configuration_bert.py": "3866dabde9264ef5057df1d0d7eefb4dbdf79a49e5bce6c699e7889792eaf650",
    "modeling_bert.py": "db5a054f8650e3dafc9f43dbfb82b63ea4f7abe8528d8169a593c4957b4f9902",
}
OFFICIAL_IMPLEMENTATION_COMMIT = (
    "1d3bb02bf091becd0771455e4e7959463935e26c"
)

CHUNK_SIZE = 256
TOP_K = 10
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_REPLICATES = 20_000
FLOAT_TOLERANCE = 1e-9
AGGREGATE_FLOAT_TOLERANCE = 1e-12
STUDY_ID = "late-chunking-scifact-matched-content-token-256-v1"
EXPECTED_LOCK_SHA256 = (
    "5a6ad830aee2d307b4d845c70e3b23f72f30b27938dcd342881e9d364dd06344"
)
CANONICAL_COMMAND = (
    "uv run --frozen reproduce.py --run --device cpu --batch-size 32 --threads 8"
)
CANONICAL_PACKAGE_VERSIONS = {
    "einops": "0.6.1",
    "huggingface-hub": "0.36.2",
    "numpy": "1.26.4",
    "safetensors": "0.8.0",
    "tokenizers": "0.19.1",
    "torch": "2.4.0",
    "transformers": "4.43.4",
}

CONDITIONS = ("naive", "late", "whole_document")
CONDITION_LABELS = {
    "naive": "Naive 256-token chunks",
    "late": "Late 256-token chunks",
    "whole_document": "Whole document",
}
METRIC_KEYS = ("ndcg_at_10", "recall_at_10", "mrr_at_10")

AGGREGATE_PATH = RECEIPTS_DIR / "aggregate.json"
PER_QUERY_PATH = RECEIPTS_DIR / "per-query.csv"
RANKINGS_PATH = RECEIPTS_DIR / "top-10-rankings.jsonl"
QRELS_PATH = RECEIPTS_DIR / "scifact-test-qrels.tsv"
RUN_RECEIPT_PATH = RECEIPTS_DIR / "run.receipt.json"
LOCK_PATH = HERE / "reproduce.py.lock"
ATTRIBUTION_PATH = HERE / "ATTRIBUTION.md"
QUALITY_FIGURE_PATH = HERE / "fig-scifact-retrieval.svg"
DELTA_FIGURE_PATH = HERE / "fig-query-deltas.svg"
QUALITY_RECEIPT_PATH = HERE / "fig-scifact-retrieval.receipt.json"
DELTA_RECEIPT_PATH = HERE / "fig-query-deltas.receipt.json"
PROVENANCE_PATH = HERE / "provenance.json"


class ReproductionError(RuntimeError):
    """Raised when a frozen input or derived artifact fails validation."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_token_sequences(sequences: Sequence[Sequence[int]]) -> str:
    """Hash token IDs with explicit sequence boundaries."""
    digest = hashlib.sha256(b"praxagent-token-sequences-v1\0")
    for sequence in sequences:
        digest.update(len(sequence).to_bytes(8, "big", signed=False))
        for token_id in sequence:
            digest.update(int(token_id).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def validate_snapshot(
    snapshot_root: Path, expected_hashes: dict[str, str], label: str
) -> None:
    """Validate every executable or model input before loading it."""
    for relative, expected in expected_hashes.items():
        path = snapshot_root / relative
        if not path.is_file():
            raise ReproductionError(f"{label} is missing {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ReproductionError(
                f"{label} {relative} has SHA-256 {actual}, expected {expected}"
            )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_bundle_path(path: Path) -> str:
    return path.relative_to(HERE).as_posix()


def download_checked(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "praxagent-late-chunking-reproduction/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)

    actual = sha256_file(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ReproductionError(
            f"dataset archive SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual}"
        )
    os.replace(partial, destination)


def extract_checked(archive: Path, destination: Path) -> Path:
    dataset_root = destination / "scifact"
    required = (
        dataset_root / "corpus.jsonl",
        dataset_root / "queries.jsonl",
        dataset_root / "qrels/test.tsv",
    )
    if not all(path.is_file() for path in required):
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            root = destination.resolve()
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ReproductionError(
                        f"unsafe path in dataset archive: {member.filename}"
                    )
            bundle.extractall(destination)

    if not all(path.is_file() for path in required):
        raise ReproductionError("dataset archive is missing required SciFact files")
    for relative, expected in DATASET_COMPONENT_SHA256.items():
        actual = sha256_file(dataset_root / relative)
        if actual != expected:
            raise ReproductionError(
                f"dataset component {relative} has SHA-256 {actual}, expected {expected}"
            )
    return dataset_root


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReproductionError(
                        f"{path.name}:{line_number}: invalid JSON"
                    ) from exc


def load_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_columns = {"query-id", "corpus-id", "score"}
        if set(reader.fieldnames or ()) != expected_columns:
            raise ReproductionError(
                f"unexpected qrels columns in {path.name}: {reader.fieldnames!r}"
            )
        for row in reader:
            score = int(row["score"])
            if score > 0:
                qrels[row["query-id"]][row["corpus-id"]] = score
    return dict(qrels)


def load_scifact(
    dataset_root: Path,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    dict[str, dict[str, int]],
]:
    corpus_rows = list(iter_jsonl(dataset_root / "corpus.jsonl"))
    query_rows = list(iter_jsonl(dataset_root / "queries.jsonl"))

    qrels = load_qrels_tsv(dataset_root / "qrels/test.tsv")

    corpus_by_id = {str(row["_id"]): row for row in corpus_rows}
    queries_by_id = {str(row["_id"]): str(row["text"]) for row in query_rows}
    query_ids = sorted(qrels, key=lambda value: int(value))
    corpus_ids = [str(row["_id"]) for row in corpus_rows]
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ReproductionError("the corpus contains duplicate document IDs")

    if len(corpus_ids) != EXPECTED_CORPUS_COUNT:
        raise ReproductionError(
            f"expected {EXPECTED_CORPUS_COUNT} corpus documents, "
            f"found {len(corpus_ids)}"
        )
    if len(query_ids) != EXPECTED_TEST_QUERY_COUNT:
        raise ReproductionError(
            f"expected {EXPECTED_TEST_QUERY_COUNT} test queries, "
            f"found {len(query_ids)}"
        )
    qrel_count = sum(len(documents) for documents in qrels.values())
    if qrel_count != EXPECTED_TEST_QREL_COUNT:
        raise ReproductionError(
            f"expected {EXPECTED_TEST_QREL_COUNT} positive test qrels, "
            f"found {qrel_count}"
        )
    if set(qrels) - set(queries_by_id):
        raise ReproductionError("a test qrel refers to a missing query")
    if any(doc_id not in corpus_by_id for docs in qrels.values() for doc_id in docs):
        raise ReproductionError("a test qrel refers to a missing corpus document")

    corpus_texts = []
    for doc_id in corpus_ids:
        row = corpus_by_id[doc_id]
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        corpus_texts.append(f"{title} {text}".strip())
    query_texts = [queries_by_id[query_id] for query_id in query_ids]
    return corpus_ids, corpus_texts, query_ids, query_texts, dict(qrels)


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ReproductionError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ReproductionError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_rows(np: Any, values: Any) -> Any:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ReproductionError("embeddings must be a two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ReproductionError("embedding model produced a non-finite value")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ReproductionError("embedding model produced a non-finite or zero norm")

    normalized = values / norms
    if not np.isfinite(normalized).all():
        raise ReproductionError("embedding normalization produced a non-finite value")
    return normalized


def encode_mean_pooled(
    *,
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    torch: Any,
    np: Any,
    device: str,
    batch_size: int,
) -> Any:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            if encoded["input_ids"].shape[1] > 8192:
                raise ReproductionError(
                    "an encoder input exceeds the pinned model's 8192-token limit"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            states = model(**encoded)[0].float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu().numpy())
    return normalize_rows(np, np.vstack(outputs))


def encode_token_id_sequences(
    *,
    token_id_sequences: Sequence[Sequence[int]],
    tokenizer: Any,
    model: Any,
    torch: Any,
    np: Any,
    device: str,
    batch_size: int,
) -> Any:
    """Encode frozen content-token sequences without decoding and retokenizing."""
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(token_id_sequences), batch_size):
            encoded = prepare_token_id_batch(
                token_id_sequences[start : start + batch_size], tokenizer
            )
            if encoded["input_ids"].shape[1] > 8192:
                raise ReproductionError(
                    "an encoder input exceeds the pinned model's 8192-token limit"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            states = model(**encoded)[0].float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu().numpy())
    return normalize_rows(np, np.vstack(outputs))


def prepare_token_id_batch(
    token_id_sequences: Sequence[Sequence[int]], tokenizer: Any
) -> dict[str, Any]:
    """Add special tokens and padding without retokenizing frozen content IDs."""
    prepared = [
        tokenizer.prepare_for_model(
            list(token_ids),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        for token_ids in token_id_sequences
    ]
    return tokenizer.pad(prepared, padding=True, return_tensors="pt")


def raw_chunk_spans(token_count: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + CHUNK_SIZE, token_count))
        for start in range(0, token_count, CHUNK_SIZE)
    ]


def extend_spans_for_special_tokens(
    spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Match the pinned implementation's [CLS]/[SEP] ownership policy."""
    extended = []
    for index, (start, stop) in enumerate(spans):
        left = start if index == 0 else start + 1
        right = stop + 1 + int(index == len(spans) - 1)
        extended.append((left, right))
    return extended


def prepare_documents(
    *, corpus_texts: Sequence[str], tokenizer: Any
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[int],
    list[list[tuple[int, int]]],
    list[int],
]:
    document_token_ids: list[list[int]] = []
    naive_chunk_token_ids: list[list[int]] = []
    chunk_doc_indices: list[int] = []
    late_annotations: list[list[tuple[int, int]]] = []
    token_counts: list[int] = []

    for doc_index, text in enumerate(corpus_texts):
        tokenized = tokenizer(text, add_special_tokens=False)
        token_ids = list(tokenized["input_ids"])
        if not token_ids:
            raise ReproductionError(f"document {doc_index} tokenized to no content")
        if len(token_ids) + 2 > 8192:
            raise ReproductionError(
                f"document {doc_index} has {len(token_ids)} content tokens and "
                "does not fit the pinned model's input window"
            )

        spans = raw_chunk_spans(len(token_ids))
        document_token_ids.append(token_ids)
        token_counts.append(len(token_ids))
        late_annotations.append(extend_spans_for_special_tokens(spans))
        reconstructed: list[int] = []
        for start, stop in spans:
            chunk_token_ids = token_ids[start:stop]
            naive_chunk_token_ids.append(chunk_token_ids)
            reconstructed.extend(chunk_token_ids)
            chunk_doc_indices.append(doc_index)
        if reconstructed != token_ids:
            raise ReproductionError(
                f"document {doc_index}: chunk spans do not preserve content tokens"
            )

    return (
        document_token_ids,
        naive_chunk_token_ids,
        chunk_doc_indices,
        late_annotations,
        token_counts,
    )


def encode_late_chunks(
    *,
    document_token_ids: Sequence[Sequence[int]],
    annotations: Sequence[Sequence[tuple[int, int]]],
    tokenizer: Any,
    model: Any,
    torch: Any,
    np: Any,
    device: str,
    batch_size: int,
) -> tuple[Any, list[int], Any]:
    chunk_vectors = []
    chunk_doc_indices = []
    whole_document_vectors = []
    with torch.inference_mode():
        for batch_start in range(0, len(document_token_ids), batch_size):
            batch_token_ids = document_token_ids[
                batch_start : batch_start + batch_size
            ]
            batch_annotations = annotations[
                batch_start : batch_start + batch_size
            ]
            encoded = prepare_token_id_batch(
                batch_token_ids,
                tokenizer,
            )
            if encoded["input_ids"].shape[1] > 8192:
                raise ReproductionError(
                    "a late-chunking encoder input exceeds 8192 tokens"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            states = model(**encoded)[0].float()
            lengths = encoded["attention_mask"].sum(dim=1).tolist()

            for local_index, spans in enumerate(batch_annotations):
                sequence_length = int(lengths[local_index])
                whole_document_vectors.append(
                    states[local_index, :sequence_length].mean(dim=0).cpu().numpy()
                )
                for left, right in spans:
                    if not 0 <= left < right <= sequence_length:
                        raise ReproductionError(
                            f"late span {(left, right)} exceeds encoded length "
                            f"{sequence_length}"
                        )
                    vector = states[local_index, left:right].mean(dim=0)
                    chunk_vectors.append(vector.cpu().numpy())
                    chunk_doc_indices.append(batch_start + local_index)

    return (
        normalize_rows(np, np.vstack(chunk_vectors)),
        chunk_doc_indices,
        normalize_rows(np, np.vstack(whole_document_vectors)),
    )


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    return sum(
        ((2**relevance) - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances[:k])
    )


def metrics_for_ranking(
    ranked_doc_ids: Sequence[str], relevant: dict[str, int]
) -> dict[str, float]:
    top = list(ranked_doc_ids[:TOP_K])
    observed = [relevant.get(doc_id, 0) for doc_id in top]
    ideal = sorted(relevant.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, TOP_K)
    ndcg = dcg_at_k(observed, TOP_K) / ideal_dcg if ideal_dcg else 0.0
    retrieved_relevant = sum(int(score > 0) for score in observed)
    recall = retrieved_relevant / len(relevant) if relevant else 0.0
    reciprocal_rank = 0.0
    for rank, score in enumerate(observed, 1):
        if score > 0:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "mrr_at_10": reciprocal_rank,
    }


def score_documents(
    *, np: Any, query_embeddings: Any, chunk_embeddings: Any,
    chunk_doc_indices: Sequence[int], document_count: int
) -> Any:
    similarities = np.matmul(query_embeddings, chunk_embeddings.T)
    scores = np.full(
        (query_embeddings.shape[0], document_count),
        -np.inf,
        dtype=np.float32,
    )
    owners = np.asarray(chunk_doc_indices, dtype=np.int64)
    for doc_index in range(document_count):
        positions = np.flatnonzero(owners == doc_index)
        if not len(positions):
            raise ReproductionError(f"document {doc_index} has no embedding")
        scores[:, doc_index] = similarities[:, positions].max(axis=1)
    return scores


def evaluate_condition(
    *,
    np: Any,
    condition: str,
    doc_scores: Any,
    corpus_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    ranking_rows = []
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-doc_scores[query_index], kind="stable")
        top_indices = order[:TOP_K]
        top_doc_ids = [corpus_ids[index] for index in top_indices]
        metrics = metrics_for_ranking(top_doc_ids, qrels[query_id])
        metric_rows.append(
            {
                "query_id": query_id,
                "relevant_count": len(qrels[query_id]),
                **metrics,
            }
        )
        ranking_rows.append(
            {
                "query_id": query_id,
                "condition": condition,
                "top_10": [
                    {
                        "rank": rank,
                        "document_id": corpus_ids[doc_index],
                        "score": round(float(doc_scores[query_index, doc_index]), 9),
                        "relevance": qrels[query_id].get(
                            corpus_ids[doc_index], 0
                        ),
                    }
                    for rank, doc_index in enumerate(top_indices, 1)
                ],
                **metrics,
            }
        )
    return metric_rows, ranking_rows


def percentile(sorted_values: Sequence[float], proportion: float) -> float:
    if not sorted_values:
        raise ReproductionError("cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


class SplitMix64:
    """Small, specified PRNG whose stream is independent of Python versions."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (value ^ (value >> 31)) & self.MASK

    def randbelow(self, bound: int) -> int:
        if bound < 1:
            raise ReproductionError("random bound must be positive")
        modulus = 1 << 64
        limit = modulus - (modulus % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def paired_bootstrap_interval(
    values: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[float, float]:
    rng = SplitMix64(seed)
    count = len(values)
    means = []
    for _ in range(replicates):
        means.append(sum(values[rng.randbelow(count)] for _ in range(count)) / count)
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def aggregate_rows(
    rows_by_condition: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    by_query = {
        condition: {row["query_id"]: row for row in rows}
        for condition, rows in rows_by_condition.items()
    }
    query_ids = sorted(by_query["late"], key=lambda value: int(value))
    methods: dict[str, Any] = {}
    for condition in CONDITIONS:
        method_metrics = {}
        for metric in METRIC_KEYS:
            values = [float(by_query[condition][query_id][metric]) for query_id in query_ids]
            method_metrics[metric] = {"mean": sum(values) / len(values)}
            if metric == "ndcg_at_10":
                lower, upper = paired_bootstrap_interval(
                    values, seed=BOOTSTRAP_SEED + CONDITIONS.index(condition)
                )
                method_metrics[metric][
                    "query_bootstrap_95_percent_interval"
                ] = [lower, upper]
        methods[condition] = method_metrics

    late_minus_naive = [
        float(by_query["late"][query_id]["ndcg_at_10"])
        - float(by_query["naive"][query_id]["ndcg_at_10"])
        for query_id in query_ids
    ]
    lower, upper = paired_bootstrap_interval(late_minus_naive)
    epsilon = 1e-12
    improved = sum(value > epsilon for value in late_minus_naive)
    worse = sum(value < -epsilon for value in late_minus_naive)
    tied = len(late_minus_naive) - improved - worse
    return {
        "schema_version": 1,
        "benchmark": "SciFact test retrieval",
        "query_count": len(query_ids),
        "metric_unit": "nDCG, Recall, and reciprocal rank on a 0-1 scale",
        "methods": methods,
        "paired_late_minus_naive_ndcg_at_10": {
            "mean_difference": sum(late_minus_naive) / len(late_minus_naive),
            "query_bootstrap_95_percent_interval": [lower, upper],
            "improved_queries": improved,
            "tied_queries": tied,
            "worse_queries": worse,
            "uncertainty_semantics": (
                "Percentile interval from 20,000 paired resamples of the 300 "
                "fixed SciFact test queries. It describes query-level stability "
                "for this benchmark and is not a population confidence interval."
            ),
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "query",
            "prng": "SplitMix64 with rejection-sampled bounded integers",
            "method_interval_seeds": {
                condition: BOOTSTRAP_SEED + CONDITIONS.index(condition)
                for condition in CONDITIONS
            },
            "paired_difference_seed": BOOTSTRAP_SEED,
            "percentile_interpolation": (
                "linear interpolation at zero-based position (B - 1) * p"
            ),
        },
    }


def combine_per_query_rows(
    rows_by_condition: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    indexed = {
        condition: {row["query_id"]: row for row in rows}
        for condition, rows in rows_by_condition.items()
    }
    combined = []
    for query_id in sorted(indexed["late"], key=lambda value: int(value)):
        base = indexed["late"][query_id]
        row: dict[str, Any] = {
            "query_id": query_id,
            "relevant_count": base["relevant_count"],
        }
        for condition in CONDITIONS:
            for metric in METRIC_KEYS:
                row[f"{condition}_{metric}"] = indexed[condition][query_id][metric]
        row["late_minus_naive_ndcg_at_10"] = (
            row["late_ndcg_at_10"] - row["naive_ndcg_at_10"]
        )
        combined.append(row)
    return combined


def write_per_query_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, float):
                    serialized[key] = f"{value:.12f}"
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def load_per_query_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {
                "query_id": row["query_id"],
                "relevant_count": int(row["relevant_count"]),
            }
            for key, value in row.items():
                if key not in converted:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def write_rankings(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_rankings(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def snapshot_hashes(snapshot_root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(snapshot_root.rglob("*")):
        if path.is_file():
            hashes[path.relative_to(snapshot_root).as_posix()] = sha256_file(path)
    return hashes


def run_experiment(args: argparse.Namespace) -> None:
    if not LOCK_PATH.is_file():
        raise ReproductionError("reproduce.py.lock is required for a model run")
    actual_lock_hash = sha256_file(LOCK_PATH)
    if actual_lock_hash != EXPECTED_LOCK_SHA256:
        raise ReproductionError(
            f"reproduce.py.lock has SHA-256 {actual_lock_hash}, "
            f"expected {EXPECTED_LOCK_SHA256}"
        )
    if not ATTRIBUTION_PATH.is_file():
        raise ReproductionError("ATTRIBUTION.md is required for a model run")

    try:
        installed_versions = {
            package: importlib.metadata.version(package)
            for package in CANONICAL_PACKAGE_VERSIONS
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise ReproductionError(
            f"canonical runtime package is not installed: {exc.name}"
        ) from exc
    if installed_versions != CANONICAL_PACKAGE_VERSIONS:
        raise ReproductionError(
            "installed material package versions do not match the canonical lock"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as exc:
        raise ReproductionError(
            "the model run requires the locked PEP 723 environment; use "
            "`uv run --frozen reproduce.py --run`"
        ) from exc

    try:
        uv_version = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReproductionError("could not record the uv version") from exc

    cache_dir = Path(args.cache_dir).expanduser().resolve()

    print("[1/6] Validating SciFact inputs", flush=True)
    archive = cache_dir / "scifact.zip"
    dataset_dir = cache_dir / "dataset"
    download_checked(DATASET_URL, archive, DATASET_ARCHIVE_SHA256)
    dataset_root = extract_checked(archive, dataset_dir)
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_scifact(
        dataset_root
    )
    QRELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset_root / "qrels/test.tsv", QRELS_PATH)
    if sha256_file(QRELS_PATH) != DATASET_COMPONENT_SHA256["qrels/test.tsv"]:
        raise ReproductionError("committed qrels copy differs from the pinned input")

    torch.manual_seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_interop_threads(1)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = choose_device(torch, args.device)

    started = datetime.now(timezone.utc)
    phase_times: dict[str, float] = {}
    phase_start = time.perf_counter()
    print("[2/6] Fetching and validating the pinned model snapshots", flush=True)
    model_snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=sorted(MODEL_SNAPSHOT_SHA256),
        )
    )
    code_snapshot = Path(
        snapshot_download(
            MODEL_CODE_ID,
            revision=MODEL_CODE_REVISION,
            allow_patterns=sorted(MODEL_CODE_SNAPSHOT_SHA256),
        )
    )
    validate_snapshot(model_snapshot, MODEL_SNAPSHOT_SHA256, "model snapshot")
    validate_snapshot(
        code_snapshot, MODEL_CODE_SNAPSHOT_SHA256, "remote-code snapshot"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_snapshot),
        local_files_only=True,
    )
    config_class = get_class_from_dynamic_module(
        "configuration_bert.JinaBertConfig",
        str(code_snapshot),
        local_files_only=True,
    )
    config = config_class.from_pretrained(
        str(model_snapshot), local_files_only=True
    )
    model_class = get_class_from_dynamic_module(
        "modeling_bert.JinaBertModel",
        str(code_snapshot),
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        str(model_snapshot),
        config=config,
        local_files_only=True,
    ).to(device)
    model.eval()
    phase_times["load_model_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    print("[3/6] Freezing shared document token spans", flush=True)
    (
        document_token_ids,
        naive_token_ids,
        naive_owners,
        late_spans,
        token_counts,
    ) = prepare_documents(corpus_texts=corpus_texts, tokenizer=tokenizer)
    phase_times["prepare_documents_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    print("[4/6] Encoding queries", flush=True)
    query_embeddings = encode_mean_pooled(
        texts=query_texts,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        np=np,
        device=device,
        batch_size=args.batch_size,
    )
    phase_times["encode_queries_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    print("[5/6] Encoding naive chunks from the frozen token IDs", flush=True)
    naive_embeddings = encode_token_id_sequences(
        token_id_sequences=naive_token_ids,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        np=np,
        device=device,
        batch_size=args.batch_size,
    )
    phase_times["encode_naive_chunks_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    print("[6/6] Encoding documents for late chunks and the control", flush=True)
    late_embeddings, late_owners, whole_embeddings = encode_late_chunks(
        document_token_ids=document_token_ids,
        annotations=late_spans,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        np=np,
        device=device,
        batch_size=args.batch_size,
    )
    phase_times["encode_late_chunks_seconds"] = time.perf_counter() - phase_start
    if late_owners != naive_owners:
        raise ReproductionError("naive and late chunk ownership differ")

    phase_start = time.perf_counter()
    scores_by_condition = {
        "naive": score_documents(
            np=np,
            query_embeddings=query_embeddings,
            chunk_embeddings=naive_embeddings,
            chunk_doc_indices=naive_owners,
            document_count=len(corpus_ids),
        ),
        "late": score_documents(
            np=np,
            query_embeddings=query_embeddings,
            chunk_embeddings=late_embeddings,
            chunk_doc_indices=late_owners,
            document_count=len(corpus_ids),
        ),
        "whole_document": np.matmul(query_embeddings, whole_embeddings.T),
    }
    phase_times["rank_and_score_seconds"] = time.perf_counter() - phase_start

    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    all_rankings = []
    for condition in CONDITIONS:
        metrics, rankings = evaluate_condition(
            np=np,
            condition=condition,
            doc_scores=scores_by_condition[condition],
            corpus_ids=corpus_ids,
            query_ids=query_ids,
            qrels=qrels,
        )
        rows_by_condition[condition] = metrics
        all_rankings.extend(rankings)

    aggregate = aggregate_rows(rows_by_condition)
    per_query = combine_per_query_rows(rows_by_condition)
    write_json(AGGREGATE_PATH, aggregate)
    write_per_query_csv(PER_QUERY_PATH, per_query)
    write_rankings(RANKINGS_PATH, all_rankings)
    component_files = {
        path.relative_to(dataset_root).as_posix(): sha256_file(path)
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
    }
    finished = datetime.now(timezone.utc)
    run_receipt = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at_utc": finished.isoformat(),
        "purpose": (
            "Educational, auditable matched-content-token re-evaluation of naive, "
            "late, and whole-document dense retrieval on the fixed SciFact test split."
        ),
        "canonical_command": CANONICAL_COMMAND,
        "observed_invocation": {
            "executable_name": Path(sys.executable).name,
            "argv": [Path(sys.argv[0]).name, *sys.argv[1:]],
        },
        "generator": {
            "path": "reproduce.py",
            "sha256_at_run": sha256_file(Path(__file__).resolve()),
        },
        "environment_lock": {
            "path": relative_bundle_path(LOCK_PATH),
            "sha256": actual_lock_hash,
            "expected_sha256": EXPECTED_LOCK_SHA256,
            "uv_version": uv_version,
        },
        "attribution": {
            "path": relative_bundle_path(ATTRIBUTION_PATH),
            "sha256": sha256_file(ATTRIBUTION_PATH),
        },
        "dataset": {
            "name": DATASET_NAME,
            "url": DATASET_URL,
            "archive_sha256": DATASET_ARCHIVE_SHA256,
            "component_sha256": component_files,
            "counts": {
                "corpus_documents": len(corpus_ids),
                "test_queries": len(query_ids),
                "positive_test_qrels": sum(len(value) for value in qrels.values()),
            },
            "licenses": {
                "claims_and_evidence_annotations": "CC-BY-4.0",
                "corpus_abstracts": "ODC-By-1.0",
                "license_source": (
                    "https://github.com/allenai/scifact/blob/master/LICENSE.md"
                ),
            },
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "remote_code_id": MODEL_CODE_ID,
            "remote_code_revision": MODEL_CODE_REVISION,
            "model_snapshot_sha256": snapshot_hashes(model_snapshot),
            "remote_code_snapshot_sha256": snapshot_hashes(code_snapshot),
            "context_limit_tokens_including_special_tokens": 8192,
            "pooling": "attention-mask mean pooling followed by L2 normalization",
        },
        "protocol": {
            "document_text": (
                "stripped SciFact title, one ASCII space, then stripped abstract text"
            ),
            "chunk_size_content_tokens": CHUNK_SIZE,
            "chunk_overlap_tokens": 0,
            "naive_special_tokens": (
                "the tokenizer adds [CLS] and [SEP] independently to every chunk"
            ),
            "late_special_token_policy": (
                "[CLS] belongs to the first span and [SEP] to the last span, "
                "matching the pinned official implementation"
            ),
            "official_implementation_commit": OFFICIAL_IMPLEMENTATION_COMMIT,
            "instructions": "no query or document instruction prefix",
            "query_path": (
                "one attention-mask mean-pooled, L2-normalized embedding per query"
            ),
            "document_score": (
                "exhaustive cosine scoring against every corpus chunk, then maximum "
                "chunk score per document before full-corpus document ranking"
            ),
            "whole_document_control": (
                "attention-mask mean pool over the same full-document contextual "
                "states used for late chunks, then L2 normalize"
            ),
            "ranking_ties": "stable original corpus row order",
            "metric": (
                "document-level nDCG@10 with gain 2^relevance-1, Recall@10, and "
                "MRR@10; the frozen SciFact test qrels are binary"
            ),
            "content_token_counts": {
                "minimum": min(token_counts),
                "maximum": max(token_counts),
                "mean": sum(token_counts) / len(token_counts),
            },
            "chunk_count": len(naive_token_ids),
            "matched_content_token_spans_sha256": sha256_token_sequences(
                naive_token_ids
            ),
            "frozen_document_content_token_ids_sha256": sha256_token_sequences(
                document_token_ids
            ),
            "matched_content_tokens": (
                "Naive chunks and late full-document inputs are both constructed "
                "directly from the same frozen document content-token IDs; no "
                "decode-and-retokenize step is used."
            ),
            "special_token_positions_are_matched": False,
            "comparison_scope": (
                "The naive and late arms match content-token slices but not all "
                "pooled positions: each naive chunk has its own [CLS] and [SEP], "
                "while the late protocol assigns the document-level special-token "
                "states only to the first and last spans. This is an end-to-end "
                "protocol comparison, not a context-conditioning-only ablation."
            ),
            "deliberate_differences_from_pinned_paper_helper": [
                (
                    "The naive arm consumes the original content token-ID slices. "
                    "The pinned helper decodes slices before its evaluator retokenizes "
                    "them, which can change boundary tokenization."
                ),
                (
                    "This run scores every corpus chunk and ranks every document "
                    "after max-chunk aggregation. The pinned helper first retrieves a "
                    "candidate chunk list and then collapses it to documents."
                ),
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "device": device,
            "batch_size": args.batch_size,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "random_seed": BOOTSTRAP_SEED,
            "dependency_versions": installed_versions,
            "started_at_utc": started.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "phase_seconds": {
                key: round(value, 6) for key, value in phase_times.items()
            },
            "deterministic_kernel_mode": True,
            "cross_platform_bitwise_identity_claimed": False,
            "canonical_configuration": (
                device == "cpu"
                and args.batch_size == 32
                and torch.get_num_threads() == 8
            ),
        },
        "outputs": {
            relative_bundle_path(path): sha256_file(path)
            for path in (AGGREGATE_PATH, PER_QUERY_PATH, RANKINGS_PATH, QRELS_PATH)
        },
        "scope": [
            "This is one fixed public benchmark and one pinned embedding model.",
            "The query-bootstrap interval describes stability across these 300 queries.",
            "The run is not evidence that late chunking always improves retrieval.",
            "Deterministic-kernel mode does not promise bitwise identity across hardware.",
            "This is not an exact replication of the paper's reported SciFact cell.",
        ],
    }
    write_json(RUN_RECEIPT_PATH, run_receipt)
    render_publication_artifacts()
    verify_artifacts()


def fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def svg_quality(aggregate: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    methods = aggregate["methods"]
    values = {
        condition: float(methods[condition]["ndcg_at_10"]["mean"])
        for condition in CONDITIONS
    }
    intervals = {
        condition: [
            float(item)
            for item in methods[condition]["ndcg_at_10"][
                "query_bootstrap_95_percent_interval"
            ]
        ]
        for condition in CONDITIONS
    }
    alt = (
        "SciFact test document nDCG at 10 on a zero-to-one axis, with "
        "query-bootstrap intervals: naive 256-token chunks "
        f"{fmt_metric(values['naive'])} [{fmt_metric(intervals['naive'][0])}, "
        f"{fmt_metric(intervals['naive'][1])}]; late 256-token chunks "
        f"{fmt_metric(values['late'])} [{fmt_metric(intervals['late'][0])}, "
        f"{fmt_metric(intervals['late'][1])}]; whole-document encoding "
        f"{fmt_metric(values['whole_document'])} "
        f"[{fmt_metric(intervals['whole_document'][0])}, "
        f"{fmt_metric(intervals['whole_document'][1])}]. Higher is better."
    )
    title = "SciFact retrieval quality across three document protocols"
    desc = alt

    x0, x1 = 210.0, 700.0
    y_positions = {"naive": 145, "late": 215, "whole_document": 285}

    def x(value: float) -> float:
        return x0 + value * (x1 - x0)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" '
        'viewBox="0 0 760 390" role="img" '
        'aria-labelledby="scifactQualityTitle scifactQualityDesc">',
        f'  <title id="scifactQualityTitle">{html.escape(title)}</title>',
        f'  <desc id="scifactQualityDesc">{html.escape(desc)}</desc>',
        "  <defs>",
        "    <style>",
        "      .t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}",
        "      .title{font-size:15px;font-weight:700}",
        "      .h{font-size:13px;font-weight:700}",
        "      .s{font-size:12px}",
        "      .m{font-size:11px;fill:#5A544C}",
        "      .grid{stroke:#D9D0C4;stroke-width:1}",
        "      .axis{stroke:#7F786D;stroke-width:1.4}",
        "      .ci{stroke:#2C2924;stroke-width:2}",
        "      .naive{fill:#F3E8E0;stroke:#A67C52;stroke-width:2}",
        "      .late{fill:#E8F0F7;stroke:#4B6787;stroke-width:2}",
        "      .whole{fill:#EDE8E1;stroke:#7F786D;stroke-width:2}",
        "    </style>",
        "  </defs>",
        '  <rect width="760" height="390" fill="#F7F4F0"/>',
        f'  <text x="24" y="29" class="t title">{html.escape(title)}</text>',
        '  <text x="24" y="50" class="t m">Same 5,183 documents, 300 queries, qrels, model, and query path; higher is better.</text>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        xpos = x(value)
        parts.extend(
            [
                f'  <line x1="{xpos:.1f}" y1="91" x2="{xpos:.1f}" y2="312" class="grid"/>',
                f'  <text x="{xpos:.1f}" y="332" text-anchor="middle" class="t m">{value:.1f}</text>',
            ]
        )
    parts.extend(
        [
            f'  <line x1="{x0:.1f}" y1="312" x2="{x1:.1f}" y2="312" class="axis"/>',
            '  <text x="455" y="354" text-anchor="middle" class="t s">Document nDCG@10 (0 to 1; higher is better)</text>',
        ]
    )
    classes = {"naive": "naive", "late": "late", "whole_document": "whole"}
    shapes = {"naive": "square", "late": "circle", "whole_document": "diamond"}
    for condition in CONDITIONS:
        y = y_positions[condition]
        value = values[condition]
        low, high = intervals[condition]
        parts.extend(
            [
                f'  <text x="194" y="{y + 4}" text-anchor="end" class="t h">{html.escape(CONDITION_LABELS[condition])}</text>',
                f'  <line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" class="ci"/>',
                f'  <line x1="{x(low):.1f}" y1="{y - 7}" x2="{x(low):.1f}" y2="{y + 7}" class="ci"/>',
                f'  <line x1="{x(high):.1f}" y1="{y - 7}" x2="{x(high):.1f}" y2="{y + 7}" class="ci"/>',
            ]
        )
        xpos = x(value)
        if shapes[condition] == "circle":
            parts.append(
                f'  <circle cx="{xpos:.1f}" cy="{y}" r="8" class="{classes[condition]}"/>'
            )
        elif shapes[condition] == "square":
            parts.append(
                f'  <rect x="{xpos - 7:.1f}" y="{y - 7}" width="14" height="14" class="{classes[condition]}"/>'
            )
        else:
            parts.append(
                f'  <path d="M{xpos:.1f} {y - 9} L{xpos + 9:.1f} {y} L{xpos:.1f} {y + 9} L{xpos - 9:.1f} {y} Z" class="{classes[condition]}"/>'
            )
        parts.append(
            f'  <text x="{min(x(high) + 12, 706):.1f}" y="{y + 4}" class="t s">{fmt_metric(value)} [{fmt_metric(low)}, {fmt_metric(high)}]</text>'
        )
    parts.extend(
        [
            '  <rect x="24" y="365" width="712" height="21" rx="7" fill="#EAF1E5" stroke="#6F8D5E" stroke-width="1.3"/>',
            '  <text x="380" y="380" text-anchor="middle" class="t m">Intervals resample the fixed test queries; they do not establish performance on another corpus.</text>',
            "</svg>",
            "",
        ]
    )
    plotted = {
        condition: {
            "mean_ndcg_at_10": values[condition],
            "query_bootstrap_95_percent_interval": intervals[condition],
            "marker": shapes[condition],
        }
        for condition in CONDITIONS
    }
    return "\n".join(parts), alt, plotted


def svg_deltas(
    aggregate: dict[str, Any], per_query: Sequence[dict[str, Any]]
) -> tuple[str, str, dict[str, Any]]:
    paired = aggregate["paired_late_minus_naive_ndcg_at_10"]
    deltas = sorted(float(row["late_minus_naive_ndcg_at_10"]) for row in per_query)
    mean = float(paired["mean_difference"])
    low, high = [
        float(value) for value in paired["query_bootstrap_95_percent_interval"]
    ]
    improved = int(paired["improved_queries"])
    tied = int(paired["tied_queries"])
    worse = int(paired["worse_queries"])
    alt = (
        f"Sorted query-level SciFact nDCG at 10 differences for late minus naive "
        f"chunking: {improved} queries improved, {tied} tied, and {worse} worsened. "
        f"The mean difference is {mean:+.4f}, with a paired query-bootstrap "
        f"interval from {low:+.4f} to {high:+.4f}. Points above zero favor late "
        "chunking; points below zero favor naive chunking."
    )
    title = "Late chunking helps some SciFact queries and hurts others"
    desc = alt
    x0, x1 = 82.0, 716.0
    y0, y1 = 100.0, 386.0

    def x(index: int) -> float:
        return x0 + index * (x1 - x0) / max(1, len(deltas) - 1)

    def y(value: float) -> float:
        return y1 - (value + 1.0) * (y1 - y0) / 2.0

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="480" '
        'viewBox="0 0 760 480" role="img" '
        'aria-labelledby="queryDeltaTitle queryDeltaDesc">',
        f'  <title id="queryDeltaTitle">{html.escape(title)}</title>',
        f'  <desc id="queryDeltaDesc">{html.escape(desc)}</desc>',
        "  <defs>",
        "    <style>",
        "      .t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}",
        "      .title{font-size:15px;font-weight:700}",
        "      .h{font-size:13px;font-weight:700}",
        "      .s{font-size:12px}",
        "      .m{font-size:11px;fill:#5A544C}",
        "      .grid{stroke:#D9D0C4;stroke-width:1}",
        "      .zero{stroke:#2C2924;stroke-width:1.8}",
        "      .positive{fill:#E8F0F7;stroke:#4B6787;stroke-width:0.8}",
        "      .negative{fill:#F3E8E0;stroke:#A67C52;stroke-width:0.8}",
        "      .tie{fill:#EDE8E1;stroke:#7F786D;stroke-width:0.8}",
        "    </style>",
        "  </defs>",
        '  <rect width="760" height="480" fill="#F7F4F0"/>',
        f'  <text x="24" y="29" class="t title">{html.escape(title)}</text>',
        '  <text x="24" y="50" class="t m">Each marker is one test query, sorted by late-minus-naive nDCG@10.</text>',
    ]
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        ypos = y(value)
        css_class = "zero" if value == 0 else "grid"
        parts.extend(
            [
                f'  <line x1="{x0:.1f}" y1="{ypos:.1f}" x2="{x1:.1f}" y2="{ypos:.1f}" class="{css_class}"/>',
                f'  <text x="70" y="{ypos + 4:.1f}" text-anchor="end" class="t m">{value:+.1f}</text>',
            ]
        )
    parts.extend(
        [
            '  <text x="18" y="243" transform="rotate(-90 18 243)" text-anchor="middle" class="t s">Late minus naive nDCG@10</text>',
            f'  <text x="{x0:.1f}" y="407" text-anchor="middle" class="t m">1</text>',
            f'  <text x="{x1:.1f}" y="407" text-anchor="middle" class="t m">{len(deltas)}</text>',
            '  <text x="399" y="426" text-anchor="middle" class="t s">Queries sorted from most negative to most positive difference</text>',
        ]
    )
    epsilon = 1e-12
    for index, value in enumerate(deltas):
        xpos, ypos = x(index), y(value)
        if value > epsilon:
            parts.append(
                f'  <circle cx="{xpos:.2f}" cy="{ypos:.2f}" r="2.4" class="positive"/>'
            )
        elif value < -epsilon:
            parts.append(
                f'  <rect x="{xpos - 2.2:.2f}" y="{ypos - 2.2:.2f}" width="4.4" height="4.4" class="negative"/>'
            )
        else:
            parts.append(
                f'  <path d="M{xpos:.2f} {ypos - 3:.2f} L{xpos + 3:.2f} {ypos:.2f} L{xpos:.2f} {ypos + 3:.2f} L{xpos - 3:.2f} {ypos:.2f} Z" class="tie"/>'
            )
    footer = (
        f"Improved {improved} | tied {tied} | worsened {worse} | "
        f"mean {mean:+.4f} [{low:+.4f}, {high:+.4f}]"
    )
    parts.extend(
        [
            '  <rect x="24" y="448" width="712" height="26" rx="8" fill="#EAF1E5" stroke="#6F8D5E" stroke-width="1.3"/>',
            f'  <text x="380" y="466" text-anchor="middle" class="t s">{html.escape(footer)}</text>',
            "</svg>",
            "",
        ]
    )
    plotted = {
        "sorted_late_minus_naive_ndcg_at_10": deltas,
        "mean_difference": mean,
        "query_bootstrap_95_percent_interval": [low, high],
        "improved_queries": improved,
        "tied_queries": tied,
        "worse_queries": worse,
        "markers": {
            "improved": "circle",
            "tied": "diamond",
            "worse": "square",
        },
    }
    return "\n".join(parts), alt, plotted


def figure_receipt(
    *,
    figure_id: str,
    title: str,
    description: str,
    alt_text: str,
    figure_path: Path,
    figure_sha256: str,
    plotted_data: dict[str, Any],
    source_paths: Sequence[Path],
    exclusions: Sequence[str],
    uncertainty: str,
    transformation: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "figure_id": figure_id,
        "title": title,
        "description": description,
        "alt_text": alt_text,
        "data_source": {
            "artifacts": [
                {
                    "path": relative_bundle_path(path),
                    "sha256": sha256_file(path),
                }
                for path in source_paths
            ],
            "selection": "all 300 queries in the fixed SciFact test qrels",
            "aggregation": "query-level metrics aggregated by arithmetic mean",
            "transformation": transformation,
        },
        "provenance": {
            "study_id": STUDY_ID,
            "generator_path": "reproduce.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "plotting_library": "dependency-free Python SVG generator",
            "outputs": {
                relative_bundle_path(figure_path): figure_sha256
            },
            "verification_command": "python3 reproduce.py --verify",
        },
        "plotted_data": plotted_data,
        "uncertainty": uncertainty,
        "claim_scope_exclusions": list(exclusions),
        "accessibility": {
            "svg_role_img": True,
            "direct_title_and_description": True,
            "color_is_not_the_only_channel": True,
            "article_alt_text_check": (
                "scripts/check_site.py compares the article shortcode alt text "
                "with this receipt"
            ),
            "full_text_equivalent": relative_bundle_path(PER_QUERY_PATH),
        },
    }


def build_publication_artifacts() -> dict[Path, bytes]:
    aggregate = load_json(AGGREGATE_PATH)
    per_query = load_per_query_csv(PER_QUERY_PATH)
    quality_svg, quality_alt, quality_data = svg_quality(aggregate)
    delta_svg, delta_alt, delta_data = svg_deltas(aggregate, per_query)

    expected: dict[Path, bytes] = {
        QUALITY_FIGURE_PATH: quality_svg.encode("utf-8"),
        DELTA_FIGURE_PATH: delta_svg.encode("utf-8"),
    }
    quality_receipt = figure_receipt(
            figure_id="late-chunking-scifact-retrieval",
            title="SciFact retrieval quality across three document protocols",
            description=(
                "Mean document-level nDCG@10 and query-bootstrap intervals for "
                "naive chunks, late chunks, and whole-document encoding."
            ),
            alt_text=quality_alt,
            figure_path=QUALITY_FIGURE_PATH,
            figure_sha256=sha256_bytes(expected[QUALITY_FIGURE_PATH]),
            plotted_data=quality_data,
            source_paths=(AGGREGATE_PATH, PER_QUERY_PATH),
            exclusions=(
                "No claim of superiority on another dataset or model.",
                "No claim that the query-bootstrap interval samples a population.",
            ),
            uncertainty=(
                "Each method interval uses 20,000 query resamples of the 300 "
                "fixed test queries. Endpoints are the 2.5th and 97.5th "
                "percentiles with linear interpolation at (B - 1) * p. The "
                "three method intervals use the seeds recorded in aggregate.json; "
                "they are not called paired intervals."
            ),
            transformation=(
                "Arithmetic mean of each arm's 300 query-level nDCG@10 values; "
                "intervals are percentile query bootstraps as specified below."
            ),
        )
    delta_receipt = figure_receipt(
            figure_id="late-chunking-scifact-query-deltas",
            title="Late chunking helps some SciFact queries and hurts others",
            description=(
                "Sorted paired query-level nDCG@10 differences for late minus "
                "naive chunking."
            ),
            alt_text=delta_alt,
            figure_path=DELTA_FIGURE_PATH,
            figure_sha256=sha256_bytes(expected[DELTA_FIGURE_PATH]),
            plotted_data=delta_data,
            source_paths=(AGGREGATE_PATH, PER_QUERY_PATH),
            exclusions=(
                "A positive mean does not imply every query improves.",
                "The paired differences do not establish a universal effect.",
            ),
            uncertainty=(
                "The interval uses 20,000 resamples of the 300 query-level paired "
                "late-minus-naive differences with seed 20260717. Endpoints are "
                "the 2.5th and 97.5th percentiles with linear interpolation at "
                "(B - 1) * p. It describes this fixed benchmark, not a population."
            ),
            transformation=(
                "Subtract naive nDCG@10 from late nDCG@10 within each query, "
                "then sort the 300 paired differences for plotting; compute the "
                "displayed mean and paired percentile-bootstrap interval from "
                "those differences."
            ),
        )

    expected[QUALITY_RECEIPT_PATH] = canonical_json_bytes(quality_receipt)
    expected[DELTA_RECEIPT_PATH] = canonical_json_bytes(delta_receipt)

    paired = aggregate["paired_late_minus_naive_ndcg_at_10"]
    method_values = {
        condition: aggregate["methods"][condition]["ndcg_at_10"]["mean"]
        for condition in CONDITIONS
    }
    receipt_hashes = {
        relative_bundle_path(path): sha256_bytes(expected[path])
        for path in (QUALITY_RECEIPT_PATH, DELTA_RECEIPT_PATH)
    }
    receipt_hashes.update(
        {
            relative_bundle_path(path): sha256_file(path)
            for path in (
                RUN_RECEIPT_PATH,
                AGGREGATE_PATH,
                PER_QUERY_PATH,
                RANKINGS_PATH,
                QRELS_PATH,
                LOCK_PATH,
                ATTRIBUTION_PATH,
            )
        }
    )
    provenance = {
        "schema_version": 1,
        "local_bundle": True,
        "generator": {
            "path": "reproduce.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "verify": "python3 reproduce.py --verify",
        },
        "receipts": receipt_hashes,
        "figures": [
            relative_bundle_path(QUALITY_FIGURE_PATH),
            relative_bundle_path(DELTA_FIGURE_PATH),
        ],
        "numbers": [
            {
                "id": "scifact-query-count",
                "value": aggregate["query_count"],
                "appears_as": "300 test queries",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-naive-ndcg10",
                "value": method_values["naive"],
                "appears_as": f"{method_values['naive']:.4f}",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-late-ndcg10",
                "value": method_values["late"],
                "appears_as": f"{method_values['late']:.4f}",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-whole-document-ndcg10",
                "value": method_values["whole_document"],
                "appears_as": f"{method_values['whole_document']:.4f}",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-late-minus-naive",
                "value": paired["mean_difference"],
                "appears_as": f"{paired['mean_difference']:+.4f}",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-improved-queries",
                "value": paired["improved_queries"],
                "appears_as": f"{paired['improved_queries']} improved",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-tied-queries",
                "value": paired["tied_queries"],
                "appears_as": f"{paired['tied_queries']} tied",
                "source": "receipts/aggregate.json",
            },
            {
                "id": "scifact-worse-queries",
                "value": paired["worse_queries"],
                "appears_as": f"{paired['worse_queries']} worsened",
                "source": "receipts/aggregate.json",
            },
        ],
    }
    expected[PROVENANCE_PATH] = canonical_json_bytes(provenance)
    return expected


def render_publication_artifacts() -> None:
    for path, data in build_publication_artifacts().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
        raise ReproductionError(
            f"{label}: expected {expected:.12f}, found {actual:.12f}"
        )


def assert_nested_matches(actual: Any, expected: Any, label: str) -> None:
    """Compare a complete JSON value, tolerating only tiny float drift."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ReproductionError(f"{label}: object keys differ")
        for key in sorted(actual):
            assert_nested_matches(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ReproductionError(f"{label}: list lengths differ")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            assert_nested_matches(
                actual_item, expected_item, f"{label}[{index}]"
            )
        return
    if isinstance(actual, float) and isinstance(expected, (int, float)):
        if not math.isclose(
            actual,
            float(expected),
            rel_tol=0.0,
            abs_tol=AGGREGATE_FLOAT_TOLERANCE,
        ):
            raise ReproductionError(
                f"{label}: expected {expected!r}, found {actual!r}"
            )
        return
    if actual != expected:
        raise ReproductionError(
            f"{label}: expected {expected!r}, found {actual!r}"
        )


def verify_rankings_and_metrics() -> None:
    aggregate = load_json(AGGREGATE_PATH)
    per_query = load_per_query_csv(PER_QUERY_PATH)
    rankings = load_rankings(RANKINGS_PATH)
    if sha256_file(QRELS_PATH) != DATASET_COMPONENT_SHA256["qrels/test.tsv"]:
        raise ReproductionError("committed SciFact qrels hash does not match the pin")
    qrels = load_qrels_tsv(QRELS_PATH)
    if len(qrels) != EXPECTED_TEST_QUERY_COUNT or sum(
        len(documents) for documents in qrels.values()
    ) != EXPECTED_TEST_QREL_COUNT:
        raise ReproductionError("committed SciFact qrels have unexpected counts")

    if len(per_query) != EXPECTED_TEST_QUERY_COUNT:
        raise ReproductionError(
            f"per-query receipt has {len(per_query)} rows, expected "
            f"{EXPECTED_TEST_QUERY_COUNT}"
        )
    if len(rankings) != EXPECTED_TEST_QUERY_COUNT * len(CONDITIONS):
        raise ReproductionError("top-10 rankings receipt has the wrong row count")

    per_query_ids = [row["query_id"] for row in per_query]
    if len(set(per_query_ids)) != len(per_query_ids):
        raise ReproductionError("per-query receipt contains duplicate query IDs")
    if set(per_query_ids) != set(qrels):
        raise ReproductionError("per-query receipt query IDs differ from frozen qrels")

    ranking_index: dict[tuple[str, str], dict[str, Any]] = {}
    for ranking in rankings:
        condition = ranking.get("condition")
        query_id = ranking.get("query_id")
        if condition not in CONDITIONS or query_id not in qrels:
            raise ReproductionError("ranking has an unknown condition or query ID")
        if "query_text" in ranking or "relevant_documents" in ranking:
            raise ReproductionError("ranking receipt must not redistribute query text or qrels")
        key = (str(condition), str(query_id))
        if key in ranking_index:
            raise ReproductionError(f"duplicate ranking row for {condition} {query_id}")
        ranking_index[key] = ranking

    expected_ranking_keys = {
        (condition, query_id) for condition in CONDITIONS for query_id in qrels
    }
    if set(ranking_index) != expected_ranking_keys:
        raise ReproductionError("ranking condition/query coverage is incomplete")

    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for row in per_query:
        query_id = row["query_id"]
        if row["relevant_count"] != len(qrels[query_id]):
            raise ReproductionError(
                f"query {query_id}: relevant count differs from frozen qrels"
            )
        for condition in CONDITIONS:
            ranking = ranking_index[(condition, query_id)]
            top_10 = ranking.get("top_10")
            if not isinstance(top_10, list) or len(top_10) != TOP_K:
                raise ReproductionError(
                    f"query {query_id} {condition}: top-10 must contain ten rows"
                )
            if any(not isinstance(item, dict) for item in top_10):
                raise ReproductionError(
                    f"query {query_id} {condition}: top-10 rows must be objects"
                )
            if [item["rank"] for item in top_10] != list(range(1, TOP_K + 1)):
                raise ReproductionError(
                    f"query {query_id} {condition}: invalid rank sequence"
                )
            doc_ids = [str(item["document_id"]) for item in top_10]
            if len(set(doc_ids)) != TOP_K:
                raise ReproductionError(
                    f"query {query_id} {condition}: top-10 document IDs repeat"
                )
            previous_score = math.inf
            for item, doc_id in zip(top_10, doc_ids, strict=True):
                score = float(item["score"])
                if not math.isfinite(score) or score > previous_score + 1e-12:
                    raise ReproductionError(
                        f"query {query_id} {condition}: scores are not finite and sorted"
                    )
                previous_score = score
                expected_relevance = qrels[query_id].get(doc_id, 0)
                if item.get("relevance") != expected_relevance:
                    raise ReproductionError(
                        f"query {query_id} {condition}: relevance differs from qrels"
                    )

            recomputed = metrics_for_ranking(
                doc_ids, qrels[query_id]
            )
            metric_row = {
                "query_id": query_id,
                "relevant_count": row["relevant_count"],
                **recomputed,
            }
            rows_by_condition[condition].append(metric_row)
            for metric in METRIC_KEYS:
                assert_close(
                    recomputed[metric],
                    float(ranking[metric]),
                    f"ranking row {query_id} {condition} {metric}",
                )
                assert_close(
                    recomputed[metric],
                    float(row[f"{condition}_{metric}"]),
                    f"query {query_id} {condition} {metric}",
                )

        expected_delta = (
            float(row["late_ndcg_at_10"]) - float(row["naive_ndcg_at_10"])
        )
        assert_close(
            expected_delta,
            float(row["late_minus_naive_ndcg_at_10"]),
            f"query {query_id} stored late-minus-naive delta",
        )

    recomputed_aggregate = aggregate_rows(rows_by_condition)
    csv_deltas = [float(row["late_minus_naive_ndcg_at_10"]) for row in per_query]
    epsilon = 1e-12
    csv_counts = {
        "improved_queries": sum(value > epsilon for value in csv_deltas),
        "tied_queries": sum(abs(value) <= epsilon for value in csv_deltas),
        "worse_queries": sum(value < -epsilon for value in csv_deltas),
    }
    recomputed_paired = recomputed_aggregate[
        "paired_late_minus_naive_ndcg_at_10"
    ]
    for key, count in csv_counts.items():
        if count != recomputed_paired[key]:
            raise ReproductionError(
                f"per-query CSV {key} differs from recomputed rankings"
            )
    assert_close(
        sum(csv_deltas) / len(csv_deltas),
        float(recomputed_paired["mean_difference"]),
        "per-query CSV mean late-minus-naive delta",
    )
    assert_nested_matches(recomputed_aggregate, aggregate, "aggregate")


def verify_run_receipt(run_receipt: dict[str, Any]) -> None:
    """Bind the frozen inputs and recorded outputs to this exact generator."""
    if run_receipt.get("study_id") != STUDY_ID:
        raise ReproductionError("run receipt has the wrong study ID")
    expected_generator_hash = sha256_file(Path(__file__).resolve())
    generator = run_receipt.get("generator", {})
    if generator.get("path") != "reproduce.py" or generator.get(
        "sha256_at_run"
    ) != expected_generator_hash:
        raise ReproductionError(
            "run receipt was not produced by the committed reproduce.py"
        )
    environment_lock = run_receipt.get("environment_lock", {})
    if environment_lock.get("path") != relative_bundle_path(
        LOCK_PATH
    ) or environment_lock.get("sha256") != EXPECTED_LOCK_SHA256:
        raise ReproductionError("run receipt is not bound to reproduce.py.lock")
    if (
        sha256_file(LOCK_PATH) != EXPECTED_LOCK_SHA256
        or environment_lock.get("expected_sha256") != EXPECTED_LOCK_SHA256
    ):
        raise ReproductionError("canonical reproduce.py.lock hash is wrong")
    if run_receipt.get("canonical_command") != CANONICAL_COMMAND:
        raise ReproductionError("run receipt has the wrong canonical command")
    observed = run_receipt.get("observed_invocation", {})
    expected_argv = [
        "reproduce.py",
        "--run",
        "--device",
        "cpu",
        "--batch-size",
        "32",
        "--threads",
        "8",
    ]
    if (
        not isinstance(observed.get("executable_name"), str)
        or observed.get("argv") != expected_argv
    ):
        raise ReproductionError("run receipt has the wrong observed Python argv")
    attribution = run_receipt.get("attribution", {})
    if attribution.get("path") != relative_bundle_path(
        ATTRIBUTION_PATH
    ) or attribution.get("sha256") != sha256_file(ATTRIBUTION_PATH):
        raise ReproductionError("run receipt is not bound to ATTRIBUTION.md")

    dataset = run_receipt.get("dataset", {})
    if dataset.get("archive_sha256") != DATASET_ARCHIVE_SHA256:
        raise ReproductionError("run receipt has the wrong SciFact archive hash")
    component_hashes = dataset.get("component_sha256", {})
    for relative, expected_hash in DATASET_COMPONENT_SHA256.items():
        if component_hashes.get(relative) != expected_hash:
            raise ReproductionError(
                f"run receipt has the wrong SciFact hash for {relative}"
            )
    expected_counts = {
        "corpus_documents": EXPECTED_CORPUS_COUNT,
        "test_queries": EXPECTED_TEST_QUERY_COUNT,
        "positive_test_qrels": EXPECTED_TEST_QREL_COUNT,
    }
    if dataset.get("counts") != expected_counts:
        raise ReproductionError("run receipt has unexpected SciFact counts")

    model = run_receipt.get("model", {})
    for key, expected in {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "remote_code_id": MODEL_CODE_ID,
        "remote_code_revision": MODEL_CODE_REVISION,
    }.items():
        if model.get(key) != expected:
            raise ReproductionError(f"run receipt has the wrong model {key}")
    snapshot_hashes_recorded = model.get("model_snapshot_sha256", {})
    for relative, expected_hash in MODEL_SNAPSHOT_SHA256.items():
        if snapshot_hashes_recorded.get(relative) != expected_hash:
            raise ReproductionError(
                f"run receipt has the wrong model snapshot hash for {relative}"
            )
    code_hashes_recorded = model.get("remote_code_snapshot_sha256", {})
    for relative, expected_hash in MODEL_CODE_SNAPSHOT_SHA256.items():
        if code_hashes_recorded.get(relative) != expected_hash:
            raise ReproductionError(
                f"run receipt has the wrong remote-code hash for {relative}"
            )

    protocol = run_receipt.get("protocol", {})
    for key, expected in {
        "chunk_size_content_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": 0,
        "official_implementation_commit": OFFICIAL_IMPLEMENTATION_COMMIT,
    }.items():
        if protocol.get(key) != expected:
            raise ReproductionError(f"run receipt has the wrong protocol {key}")
    if not isinstance(protocol.get("chunk_count"), int) or protocol["chunk_count"] < 1:
        raise ReproductionError("run receipt has an invalid chunk count")
    if protocol.get("special_token_positions_are_matched") is not False:
        raise ReproductionError("run receipt must disclose unmatched special-token positions")
    token_span_hash = protocol.get("matched_content_token_spans_sha256")
    if not isinstance(token_span_hash, str) or len(token_span_hash) != 64:
        raise ReproductionError("run receipt has an invalid content-token span hash")
    document_token_hash = protocol.get(
        "frozen_document_content_token_ids_sha256"
    )
    if not isinstance(document_token_hash, str) or len(document_token_hash) != 64:
        raise ReproductionError("run receipt has an invalid document-token hash")

    runtime = run_receipt.get("runtime", {})
    if runtime.get("dependency_versions") != CANONICAL_PACKAGE_VERSIONS:
        raise ReproductionError("run receipt package versions are not canonical")
    if (
        runtime.get("device") != "cpu"
        or runtime.get("batch_size") != 32
        or runtime.get("torch_threads") != 8
        or runtime.get("torch_interop_threads") != 1
        or runtime.get("canonical_configuration") is not True
        or runtime.get("deterministic_kernel_mode") is not True
        or runtime.get("cross_platform_bitwise_identity_claimed") is not False
    ):
        raise ReproductionError("committed run receipt is not the canonical CPU run")

    expected_outputs = {
        relative_bundle_path(path)
        for path in (AGGREGATE_PATH, PER_QUERY_PATH, RANKINGS_PATH, QRELS_PATH)
    }
    outputs = run_receipt.get("outputs", {})
    if set(outputs) != expected_outputs:
        raise ReproductionError("run receipt has an unexpected output set")
    for relative, expected_hash in outputs.items():
        path = HERE / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ReproductionError(f"run receipt hash mismatch for {relative}")


def verify_artifacts() -> None:
    required = (
        AGGREGATE_PATH,
        PER_QUERY_PATH,
        RANKINGS_PATH,
        QRELS_PATH,
        RUN_RECEIPT_PATH,
        LOCK_PATH,
        ATTRIBUTION_PATH,
        QUALITY_FIGURE_PATH,
        DELTA_FIGURE_PATH,
        QUALITY_RECEIPT_PATH,
        DELTA_RECEIPT_PATH,
        PROVENANCE_PATH,
    )
    missing = [relative_bundle_path(path) for path in required if not path.is_file()]
    if missing:
        raise ReproductionError(f"missing publication artifacts: {', '.join(missing)}")

    run_receipt = load_json(RUN_RECEIPT_PATH)
    if not isinstance(run_receipt, dict):
        raise ReproductionError("run receipt root must be an object")
    verify_run_receipt(run_receipt)

    verify_rankings_and_metrics()
    expected = build_publication_artifacts()
    for path, expected_bytes in expected.items():
        actual = path.read_bytes()
        if actual != expected_bytes:
            raise ReproductionError(
                f"{relative_bundle_path(path)} is stale or was edited by hand; "
                "run `python3 reproduce.py --render`"
            )

    provenance = load_json(PROVENANCE_PATH)
    for relative, expected_hash in provenance["receipts"].items():
        path = HERE / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ReproductionError(f"provenance hash mismatch for {relative}")

    print(
        "Late Chunking matched-content-token re-evaluation verified: "
        "300 query rankings, derived "
        "metrics, two SVGs, per-figure receipts, and provenance hashes match."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run model inference")
    mode.add_argument(
        "--render",
        action="store_true",
        help="regenerate figures and receipts from committed result rows",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="offline byte- and metric-verification of committed artifacts",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(HERE.parents[4] / ".cache/late-chunking"),
        help="untracked dataset/model scratch directory",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.batch_size < 1:
            raise ReproductionError("--batch-size must be positive")
        if args.threads < 0:
            raise ReproductionError("--threads must be nonnegative")
        if args.run:
            if (args.device, args.batch_size, args.threads) != ("cpu", 32, 8):
                raise ReproductionError(
                    "publication artifacts require the canonical run settings: "
                    "--device cpu --batch-size 32 --threads 8"
                )
            run_experiment(args)
        elif args.render:
            render_publication_artifacts()
            verify_artifacts()
        else:
            verify_artifacts()
    except ReproductionError as exc:
        print(f"reproduction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
