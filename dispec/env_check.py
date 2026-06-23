"""Environment & hardware validation for DiSpec.

Run: python -m dispec.env_check

Confirms the box can actually run the engine: CUDA available, bf16 matmul works,
Triton importable, and reports free VRAM so we can size models / KV cache.
"""

from __future__ import annotations

import importlib
import sys


def _check_torch() -> "tuple[object, dict]":
    import torch

    info: dict = {}
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        return torch, info

    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
    info["bf16_supported"] = torch.cuda.is_bf16_supported()

    # Real bf16 matmul on device (not just a flag check).
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    _ = (x @ x).float().sum().item()
    torch.cuda.synchronize()
    info["bf16_matmul"] = "ok"

    free, total = torch.cuda.mem_get_info()
    info["vram_free_gb"] = round(free / 1e9, 2)
    info["vram_total_gb"] = round(total / 1e9, 2)
    return torch, info


def _check_optional(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        return f"MISSING ({e.__class__.__name__})"


def main() -> int:
    _, info = _check_torch()
    print("== DiSpec environment ==")
    for k, v in info.items():
        print(f"  {k:18}: {v}")
    for mod in ("triton", "transformers", "numpy", "fastapi", "prometheus_client"):
        print(f"  {mod:18}: {_check_optional(mod)}")

    if not info.get("cuda_available"):
        print("\nFATAL: CUDA not available.", file=sys.stderr)
        return 1
    if info.get("vram_total_gb", 0) < 6:
        print("\nWARN: <6 GB VRAM; use 0.5B target + int4 fallback.", file=sys.stderr)
    print("\nOK: environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
