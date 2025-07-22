#!/usr/bin/env python3
"""verify_pytorch_device.py – Quick check of PyTorch backend availability.

Usage (from project root)::

    python verify_pytorch_device.py

It reports:
• PyTorch version
• macOS version (if applicable)
• Whether the build was compiled with MPS support and if it is currently usable
• Whether CUDA is available (for completeness)
• Runs a tiny tensor allocation on each available backend as a sanity test.
"""
from __future__ import annotations

import platform
import sys

try:
    import torch
except ImportError as e:
    sys.exit(f"❌ PyTorch not installed: {e}. Run 'pip install torch' first.")


def check_mps() -> None:  # noqa: D401
    built = torch.backends.mps.is_built()
    avail = torch.backends.mps.is_available()
    print(f"MPS built     : {built}")
    print(f"MPS available : {avail}")
    if avail:
        try:
            x = torch.ones(1, device="mps")
            print(f"MPS tensor OK : {x.device}")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Failed to allocate on MPS: {exc}")


def check_cuda() -> None:  # noqa: D401
    avail = torch.cuda.is_available()
    print(f"CUDA available: {avail}")
    if avail:
        try:
            print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
            x = torch.ones(1, device="cuda:0")
            print(f"CUDA tensor OK: {x.device}")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Failed to allocate on CUDA: {exc}")


def main() -> None:  # noqa: D401
    print("PyTorch version:", torch.__version__)
    print("OS             :", platform.platform())
    print("Python         :", platform.python_version())
    print("\n--- Backend Checks ---")
    print("CPU always available – default device\n")
    check_mps()
    check_cuda()


if __name__ == "__main__":
    main() 