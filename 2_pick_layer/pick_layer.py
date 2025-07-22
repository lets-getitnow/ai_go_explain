"""
Pick-Layer Helper Script
========================
Purpose
-------
Identify and record an *intermediate* residual block in a **general KataGo network** so all downstream interpretability experiments know exactly which layer to probe. Works purely from the model filename (`*.bin.gz`) – no heavy model loading required.

Capabilities
------------
1. **List layers** – `--list` prints all trunk block layer names (e.g. `trunk_block_0_output … trunk_block_27_output`).
2. **Automatic pick** – if `--choose` is omitted, the script picks the exact *middle* block.
3. **Manual pick** – supply `--choose <layer_name>` to override.
4. Records the decision to `layer_selection.yml` for reproducibility.

Zero-Fallback Guarantee
-----------------------
Any missing or malformed argument halts with an explicit error – no silent defaults beyond those shown in `--help`.
"""

import argparse
import re
import sys
import yaml
from pathlib import Path
from datetime import date


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Choose an intermediate KataGo layer and write layer_selection.yml")
    p.add_argument("--model-path", type=Path, required=True, help="Path to KataGo *.bin.gz model file")
    p.add_argument("--board-size", type=int, default=7, help="Board size (e.g. 7, 9, 19)")
    p.add_argument("--choose", type=str, help="Layer name to select instead of automatic middle block")
    p.add_argument("--list", action="store_true", help="Only list candidate trunk block layer names and exit")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main logic
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: D401
    args = parse_args()

    if not args.model_path.exists():
        sys.exit(f"❌ Model file not found: {args.model_path}")

    match = re.search(r"b(\d+)c(\d+)", args.model_path.name)
    if not match:
        sys.exit("❌ Model filename must contain 'b<blocks>c<channels>', e.g. b28c512…")

    blocks = int(match.group(1))
    channels = int(match.group(2))

    candidate_layers = [f"trunk_block_{i}_output" for i in range(blocks)]

    if args.list:
        print("\n".join(candidate_layers))
        return

    chosen_layer: str
    if args.choose:
        if args.choose not in candidate_layers:
            sys.exit(
                f"❌ '{args.choose}' not among recognised trunk layers. "
                f"Run with --list to see options."
            )
        chosen_layer = args.choose
    else:
        chosen_layer = candidate_layers[blocks // 2]

    meta = {
        "network_file": str(args.model_path),
        "board_size": args.board_size,
        "trunk_blocks": blocks,
        "chosen_layer": chosen_layer,
        "layer_shape": f"{channels}×{args.board_size}×{args.board_size} (approx)",
        "pooling": "spatial mean",
        "date": date.today().isoformat(),
        "rationale": "Middle trunk balances local and global signals. Derived from filename metadata only.",
    }

    out_path = Path("layer_selection.yml")
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True)

    print(f"✅ Recorded {chosen_layer} → {out_path}")


if __name__ == "__main__":
    main()
