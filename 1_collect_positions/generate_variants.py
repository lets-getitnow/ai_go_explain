"""
Variant Generator Utility
=========================
Purpose
-------
Produce _controlled_ variants of KataGo training slices (``.npz`` files) so
that later pipeline stages can probe which convolutional channels are
sensitive to _global_ context versus _spatial_ board shape.

For the **initial** implementation we support a single probe:
"zero-global" – replace the ``globalInputNC`` array with **all-zeros** while
leaving the binary board planes untouched.  This isolates channels that rely
on the 𝐆-input vector (komi, move number, ruleset flags, score estimate, etc.).

High-Level Requirements
-----------------------
• **Zero-fallback mandate** – abort immediately on any unexpected file shape
  or missing key.
• Deterministic – regenerated variants must be identical bit-for-bit across
  runs.
• CLI-driven – script is runnable from project root:

    python 1_collect_positions/generate_variants.py \
        --input-dir selfplay_out/ \
        --output-dir variants/ \
        --mode zero_global

• Mirror directory tree – the variant files maintain the same relative path
  as the originals so downstream code can locate them predictably.
• No copy for baseline – we do **not** duplicate unchanged files; the caller
  should simply point at the original directory for the baseline variant.

Future extensions
-----------------
Once we have reliable komi indices and ko encodings, implement:
    – ``komi_sweep``   (±n points)
    – ``history_shuffle``
    – ``ko_toggle``
    – ``board_holdout``
Each new mode must live in ``VARIANT_FUNCS`` with full validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Variant transformation functions
# Each function takes the full dict of arrays from an npz and mutates it.
# ────────────────────────────────────────────────────────────────────────────

def _zero_global(arrs: Dict[str, np.ndarray]) -> None:
    """Set the ``globalInputNC`` array to zeros (float32) in-place."""
    key = "globalInputNC"
    if key not in arrs:
        raise KeyError(f"Required array '{key}' missing from .npz file")
    g = arrs[key]
    if not isinstance(g, np.ndarray):
        raise TypeError(f"'{key}' is not a numpy array – got {type(g)}")
    arrs[key] = np.zeros_like(g, dtype=np.float32)


def _identity(arrs: Dict[str, np.ndarray]) -> None:
    """No-op transform to copy original inputs verbatim for the 'baseline' variant."""
    return


VARIANT_FUNCS: Dict[str, Callable[[Dict[str, np.ndarray]], None]] = {
    "baseline": _identity,
    "zero_global": _zero_global,
}

# ────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate controlled input variants (e.g. zero_global)."
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory tree containing original .npz files",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory under which to write variant files",
    )
    p.add_argument(
        "--mode",
        required=True,
        help=(
            "Variant mode(s) to generate. "
            "Use comma-separated list (e.g. zero_global,komi_sweep) or "
            "'all' to run every available mode."
        ),
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main processing
# ────────────────────────────────────────────────────────────────────────────

def _process_file(src: Path, dst: Path, transform: Callable[[Dict[str, np.ndarray]], None]) -> None:
    """Load src .npz, apply *transform*, save to dst (parent dirs auto-made)."""
    # Load – we need the full dict so that we can write back exactly.
    with np.load(src, allow_pickle=False) as data:
        arrs: Dict[str, np.ndarray] = {k: v.copy() for k, v in data.items()}

    # Sanity checks
    if "binaryInputNCHWPacked" not in arrs:
        raise KeyError(
            f"{src} missing required 'binaryInputNCHWPacked' array")

    # Apply transformation in-place
    transform(arrs)

    # Ensure parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Save – use exakt same compress defaults (np.savez_compressed)
    np.savez_compressed(dst, **arrs)


def main() -> None:  # noqa: D401
    args = _parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    # Parse modes list
    raw_modes = [m.strip() for m in args.mode.split(',')] if args.mode else []
    if not raw_modes:
        print("[ERROR] --mode must specify at least one variant", file=sys.stderr)
        sys.exit(1)

    if raw_modes == ["all"]:
        modes = list(VARIANT_FUNCS.keys())
    else:
        invalid = [m for m in raw_modes if m not in VARIANT_FUNCS]
        if invalid:
            print(f"[ERROR] Unknown mode(s): {', '.join(invalid)}", file=sys.stderr)
            print(f"        Valid options: {', '.join(VARIANT_FUNCS)}", file=sys.stderr)
            sys.exit(1)
        modes = raw_modes

    if not input_dir.exists():
        print(f"[ERROR] --input-dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Enumerate all .npz once to avoid repeated disk scans
    files = sorted(input_dir.rglob("*.npz"))
    if not files:
        print(f"[ERROR] No .npz files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    for mode in modes:
        transform = VARIANT_FUNCS[mode]
        print(f"[INFO] Generating variant '{mode}' for {len(files)} source files…")

        for src in files:
            rel = src.relative_to(input_dir)
            dst = output_dir / mode / rel
            print(f"[DEBUG] {src} → {dst}")
            _process_file(src, dst, transform)

        print(
            f"✅ Completed variant generation: {mode} ({len(files)} files) → {output_dir/mode}")


if __name__ == "__main__":
    main() 