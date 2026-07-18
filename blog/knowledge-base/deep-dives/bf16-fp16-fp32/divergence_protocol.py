#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.2",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///
"""Cross-device run-to-run divergence protocol for grid_sample backward.

This is the exact protocol behind receipts/grid-sample-divergence.json.
Run it on each device under test (it was run on Google Colab CPU, T4, A100,
L4, and RTX PRO 6000 Blackwell instances):

    uv run divergence_protocol.py

It measures whether repeated executions of the CUDA (or CPU) implementation
of torch.nn.functional.grid_sample's backward pass produce bit-identical
input gradients when every input tensor is byte-identical across trials.

Design points (see the Deep Dive for rationale):

* inputs generated once on CPU with a fixed seed, then transferred, so every
  device and arm consumes identical bytes;
* per-geometry warmup before trials;
* all-pairs comparison over 30 trials (435 pairs), not a single baseline;
* full-tensor bitwise comparison (int32 view) plus max elementwise
  difference -- never a scalar summary such as grad.sum(), where positive
  and negative coordinate drift can cancel;
* stream-local synchronization so the background-load arm does not
  serialize the device;
* a deterministic-mode probe that classifies the operation on this stack.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading

import torch


def hash_tensor(t: torch.Tensor) -> str:
    """Return a SHA-256 hash of the tensor's underlying data."""
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()


torch.use_deterministic_algorithms(False)

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

env_info = {
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda if CUDA else "N/A",
    "cudnn_version": torch.backends.cudnn.version() if CUDA else "N/A",
    "compute_capability": (
        f"{torch.cuda.get_device_properties(0).major}."
        f"{torch.cuda.get_device_properties(0).minor}"
        if CUDA
        else "N/A"
    ),
    "device_name": torch.cuda.get_device_name(0) if CUDA else "CPU",
    "dtype": "float32",
}

print("--- Environment Receipt ---")
for key, value in env_info.items():
    print(f"{key:<20}: {value}")
print()


def run_collision_protocol(label: str = "idle") -> dict:
    input_res = 32
    output_resolutions = [16, 32, 64, 128, 256, 512]
    n_trials = 30

    print(f"--- Running Protocol (Condition: {label}) ---")
    generator = torch.Generator(device="cpu").manual_seed(42)
    img_base = torch.randn(
        1, 3, input_res, input_res, generator=generator
    ).to(DEVICE)

    results = {
        "condition": label,
        "trials_per_res": n_trials,
        "input_res": input_res,
        "total_pairs_per_res": (n_trials * (n_trials - 1)) // 2,
        "total_coords": img_base.numel(),
        "input_hashes": {"img_base_sha256": hash_tensor(img_base)},
        "res": [],
        "max_drift_coords": [],
        "max_div": [],
        "pairwise_match_rate": [],
        "all_pairwise_divs": {},
    }

    for out_res in output_resolutions:
        grid = (
            torch.rand(1, out_res, out_res, 2, generator=generator) * 2 - 1
        ).to(DEVICE)
        grad_output = torch.randn(
            1, 3, out_res, out_res, generator=generator
        ).to(DEVICE)

        results["input_hashes"][f"grid_{out_res}_sha256"] = hash_tensor(grid)
        results["input_hashes"][f"grad_out_{out_res}_sha256"] = hash_tensor(
            grad_output
        )

        # Warm up this exact geometry before measuring.
        warmup_img = img_base.clone().requires_grad_()
        warmup_out = torch.nn.functional.grid_sample(
            warmup_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        warmup_out.backward(grad_output)
        if CUDA:
            torch.cuda.current_stream().synchronize()

        grads = []
        for _ in range(n_trials):
            img = img_base.clone().requires_grad_()
            out = torch.nn.functional.grid_sample(
                img,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            out.backward(grad_output)
            if CUDA:
                torch.cuda.current_stream().synchronize()
            grads.append(img.grad.clone())

        max_divergence = 0.0
        max_changed_coords = 0
        exact_pairs = 0
        total_pairs = 0
        pair_divs = []

        for g1, g2 in itertools.combinations(grads, 2):
            total_pairs += 1
            div = (g1 - g2).abs().max().item()
            pair_divs.append(div)
            max_divergence = max(max_divergence, div)

            c1 = g1.contiguous().view(torch.int32)
            c2 = g2.contiguous().view(torch.int32)
            changed = torch.count_nonzero(c1 != c2).item()
            max_changed_coords = max(max_changed_coords, changed)

            if changed == 0:
                exact_pairs += 1

        match_rate = 100.0 * exact_pairs / total_pairs
        total_coords = grads[0].numel()

        print(
            f"Output Res: {out_res:<3}x{out_res:<3} | "
            f"Pair Match: {match_rate:>5.1f}% | "
            f"Max Pair Drift: {max_changed_coords:>4}/{total_coords} | "
            f"Max Pair Div: {max_divergence:.4e}"
        )

        results["res"].append(out_res)
        results["max_drift_coords"].append(max_changed_coords)
        results["max_div"].append(max_divergence)
        results["pairwise_match_rate"].append(match_rate)
        results["all_pairwise_divs"][out_res] = pair_divs

    print()
    return results


idle_results = run_collision_protocol("idle")

stop_event = threading.Event()
load_ready = threading.Event()


def heavy_background_stress() -> None:
    streams = [torch.cuda.Stream() for _ in range(4)]
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    load_ready.set()

    while not stop_event.is_set():
        for stream in streams:
            with torch.cuda.stream(stream):
                _ = a @ b
        # Stream-local sync so the test thread's stream is not stalled.
        for stream in streams:
            stream.synchronize()


if CUDA:
    load_thread = threading.Thread(target=heavy_background_stress)
    load_thread.start()
    load_ready.wait()
    try:
        load_results = run_collision_protocol("background-matmul")
    finally:
        stop_event.set()
        load_thread.join()
else:
    load_results = run_collision_protocol("background-matmul")

experiment_data = {
    "environment": env_info,
    "idle_condition": idle_results,
    "background_matmul_condition": load_results,
}

with open("experiment_results.json", "w", encoding="utf-8") as handle:
    json.dump(experiment_data, handle, indent=2)
print("Saved complete pairwise metrics and hashes to experiment_results.json\n")

print("--- Deterministic-mode classification ---")
torch.use_deterministic_algorithms(True)

generator = torch.Generator(device="cpu").manual_seed(42)
img_base = torch.randn(1, 3, 32, 32, generator=generator).to(DEVICE)
grid = (torch.rand(1, 32, 32, 2, generator=generator) * 2 - 1).to(DEVICE)
grad_output = torch.randn(1, 3, 32, 32, generator=generator).to(DEVICE)

probe_img = img_base.clone().requires_grad_()
try:
    probe_out = torch.nn.functional.grid_sample(
        probe_img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    probe_out.backward(grad_output)
except RuntimeError as error:
    print("PyTorch rejected grid_sample backward:")
    print(str(error).splitlines()[0])
else:
    print(
        "grid_sample backward completed under deterministic mode "
        "on this stack; inspect the version-specific implementation."
    )
finally:
    torch.use_deterministic_algorithms(False)
