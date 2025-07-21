"""
Pick‑Layer Helper Script
========================
Purpose
-------
Identify and record the *middle* residual block in a KataGo 9×9 network so
all downstream interpretability experiments know exactly which layer to
probe. Works directly on KataGo’s native binary model files (`*.bin.gz`)—
no conversion to PyTorch required.

High‑Level Requirements
-----------------------
• A KataGo model file (e.g. `models/kata9x9-b18c384nbt-20231025.bin.gz`).
  The filename must contain the pattern `b<blocks>c<channels>`.
• Python 3 with PyYAML installed (`pip install pyyaml`).

What the Script Does
--------------------
1. Parses the filename to discover how many residual blocks (`b18`) and
   how many channels (`c384`).
2. Chooses the exact middle block (⌈blocks / 2⌉; for 18 blocks that is
   block 9).
3. Writes a reproducible description of that choice to
   `layer_selection.yml` using real Unicode characters (×, ‑, —) via
   `allow_unicode=True`.

Example YAML Output
-------------------
chosen_layer: trunk_block_9_output
date: '2025-07-21'
layer_shape: "384×9×9 (approx)"
network_file: models/kata9x9-b18c384nbt-20231025.bin.gz
pooling: spatial mean
rationale: "Mid‑trunk balances local patterns vs. mixed policy/value signals. Binary model not loaded—layer chosen by filename metadata only."
trunk_blocks: 18
"""

import re
import yaml
from pathlib import Path
from datetime import date

# ── CONFIGURE YOUR MODEL PATH HERE ─────────────────────────────
MODEL_PATH = Path("../models/kata9x9-b18c384nbt-20231025.bin.gz")

# ── 1) Extract block & channel counts from filename ────────────
match = re.search(r"b(\d+)c(\d+)", MODEL_PATH.name)
if not match:
    raise ValueError(
        "Model filename must contain 'b<blocks>c<channels>', e.g. b18c384")

blocks = int(match.group(1))
channels = int(match.group(2))

# ── 2) Pick the middle residual block (0‑indexed) ───────────────
mid_block = blocks // 2  # for 18 → 9, for 17 → 8
chosen_layer = f"trunk_block_{mid_block}_output"

# ── 3) Record the decision to YAML for reproducibility ─────────
meta = {
    "network_file": str(MODEL_PATH),
    "trunk_blocks": blocks,
    "chosen_layer": chosen_layer,
    "layer_shape": f"{channels}×9×9 (approx)",
    "pooling": "spatial mean",
    "date": date.today().isoformat(),
    "rationale": "Mid‑trunk balances local patterns vs. mixed policy/value signals. "
                 "Binary model not loaded—layer chosen by filename metadata only."
}

with open("../layer_selection.yml", "w", encoding="utf-8") as f:
    yaml.safe_dump(meta, f, allow_unicode=True)

print(f"✅ Recorded {chosen_layer} from {blocks} residual blocks → ../layer_selection.yml")
