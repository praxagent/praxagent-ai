#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.2",
# ]
# ///
"""Is CUBLAS_WORKSPACE_CONFIG still required for deterministic torch.mm?

This is the exact protocol behind receipts/cublas-determinism.json.

CUBLAS_WORKSPACE_CONFIG is read when CUDA initializes, so a single process
cannot test "variable set" and "variable absent" honestly. This script
therefore acts as its own driver: run it with no arguments and it launches
one fresh copy of itself per condition, each with a controlled environment.

    uv run cublas_determinism_protocol.py

Conditions:

* det_probe  (no config / with config): does torch.mm raise under
  torch.use_deterministic_algorithms(True)?
* det_run    (with config): 20 deterministic-mode trials, all-pairs
  bitwise comparison.
* stress     (no config / with config): nondeterministic algorithms
  allowed, four CUDA streams issuing concurrent GEMMs, 20 trials,
  all-pairs bitwise comparison.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

N_TRIALS = 20
MATRIX = 2048
N_STREAMS = 4


def worker(condition: str) -> None:
    import torch

    out = {
        "condition": condition,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""
        ),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }

    def all_pairs_stats(results: list) -> dict:
        max_div = 0.0
        exact = 0
        total = 0
        for r1, r2 in itertools.combinations(results, 2):
            total += 1
            if torch.equal(r1.view(torch.int32), r2.view(torch.int32)):
                exact += 1
            max_div = max(max_div, (r1 - r2).abs().max().item())
        return {"pairs": total, "exact_pairs": exact, "max_div": max_div}

    generator = torch.Generator(device="cpu").manual_seed(42)
    a = torch.randn(MATRIX, MATRIX, generator=generator).cuda()
    b = torch.randn(MATRIX, MATRIX, generator=generator).cuda()

    if condition == "det_probe":
        torch.use_deterministic_algorithms(True)
        try:
            _ = a @ b
            torch.cuda.synchronize()
            out["probe"] = "completed"
        except RuntimeError as error:
            out["probe"] = "raised"
            out["error_first_line"] = str(error).splitlines()[0]

    elif condition == "det_run":
        torch.use_deterministic_algorithms(True)
        _ = a @ b
        torch.cuda.synchronize()  # warmup
        results = []
        for _ in range(N_TRIALS):
            results.append((a @ b).clone())
            torch.cuda.synchronize()
        out["stats"] = all_pairs_stats(results)

    elif condition == "stress":
        torch.use_deterministic_algorithms(False)
        streams = [torch.cuda.Stream() for _ in range(N_STREAMS)]
        _ = a @ b
        torch.cuda.synchronize()  # warmup
        results = []
        for _ in range(N_TRIALS):
            outputs = [None] * N_STREAMS
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    outputs[i] = a @ b
            for stream in streams:
                stream.synchronize()
            results.append(outputs[0].clone())
        out["stats"] = all_pairs_stats(results)

    print(json.dumps(out))


def launch(condition: str, workspace: str | None) -> dict:
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if workspace:
        env["CUBLAS_WORKSPACE_CONFIG"] = workspace
    env["COND"] = condition
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    receipt = {
        "probe_no_config": launch("det_probe", None),
        "probe_with_config": launch("det_probe", ":4096:8"),
        "det_run_with_config": launch("det_run", ":4096:8"),
        "stress_no_config": launch("stress", None),
        "stress_with_config": launch("stress", ":4096:8"),
    }
    print(json.dumps(receipt, indent=2))
    with open("cublas_determinism_receipt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)


if __name__ == "__main__":
    if "COND" in os.environ:
        worker(os.environ["COND"])
    else:
        main()
