"""
Activation Extraction Script
===========================
Purpose
-------
Produce the dataset required for all *down-stream* interpretability steps:
"position → pooled activations from the chosen network layer".
It realises Step 3 of the pipeline outlined in the repository README:

3-A Load positions (.npz files)             → source data for inference
3-B Run inference up to chosen layer        → expose network "thought bubble"
3-C Spatial-pool (mean) each channel        → collapse 9×9 grid per channel
3-D Stack all rows                          → big matrix 𝐀  (N_pos × C_ch)
3-E Ensure non-negative & scale             → required by NMF
3-F Persist to disk                         → re-usable by NMF / SAE

High-Level Requirements
-----------------------
• No hard-coded paths – accept CLI flags or environment variables.
• Read the chosen layer from *layer_selection.yml* (written by *pick_layer.py*).
• Fail fast if the layer, model or input data are missing – **zero fallbacks**.
• Never mutate model weights; inference only.
• Output files must live under *activations/*** and include:
    pooled_<layer>.npy         (N_positions, C_channels)
    pooled_meta.json           reproducible metadata
    pos_index_to_npz.txt       mapping row → original .npz file
• Keep memory usage modest – stream positions in mini-batches (CPU friendly).
• Entire script is runnable from the project root, e.g.:
      python extract_pooled_activations.py \
             --positions-dir selfplay_out/ \
             --model-path models/kata9x9-b18c384nbt-20231025.ckpt \
             --batch-size 256

Implementation Notes (concise)
-----------------------------
1. *Model loading*: Uses KataGo's PyTorch training infrastructure to load
   `.ckpt` checkpoint files directly with proper layer naming.
2. *Hooking the layer*: Register a forward-hook on the desired module name
   (e.g. "trunk_block_9_output").
3. *Position decoding*: Each .npz must contain the exact input tensor shape
   expected by the network (C_in × 9 × 9). Provide or adapt
   `decode_position_npz()` accordingly.
4. *Non-negativity*: If any pooled value < 0, shift the entire column so its
   minimum becomes zero, then optionally divide by column max.
5. *Zero-fallback mandate*: any unhandled state raises an explicit error;
   no silent defaults, optional chaining or defensive programming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

# Add KataGo python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "KataGo" / "python"))

# Global board size (will be overridden by --board-size CLI flag)
BOARD_SIZE: int = 7

# ────────────────────────────────────────────────────────────────────────────
# Helper functions (❗ YOU MAY need to adapt these to your environment)
# ────────────────────────────────────────────────────────────────────────────

def load_layer_selection(path: Path) -> Tuple[str, int]:
    """Return (chosen_layer_name, trunk_channels) from layer_selection.yml."""
    with path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    if "chosen_layer" not in meta or "layer_shape" not in meta:
        raise KeyError("layer_selection.yml missing required keys")
    return meta["chosen_layer"], int(meta["layer_shape"].split("×")[0])


def load_katago_pytorch(model_path: Path) -> torch.nn.Module:
    """Load a KataGo PyTorch checkpoint as a `torch.nn.Module`.

    Loads a KataGo training checkpoint (.ckpt file) using KataGo's existing
    PyTorch infrastructure. The model will have proper layer names for hooking
    and accepts input tensors of shape (B, C_in, 9, 9).
    
    Args:
        model_path: Path to KataGo .ckpt checkpoint file
        
    Returns:
        PyTorch model in eval mode with named modules matching KataGo layer names
        
    Raises:
        ImportError: If KataGo python modules cannot be imported
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
    try:
        from katago.train.load_model import load_model
    except ImportError as e:
        raise ImportError(
            f"Failed to import KataGo modules. Ensure KataGo/python is in path: {e}"
        )
    
    try:
        # Load model using KataGo's infrastructure
        # pos_len=9 for 9x9 boards, use_swa=False for standard model, device="cpu" for consistency
        model, swa_model, other_state_dict = load_model(
            checkpoint_file=str(model_path),
            use_swa=False,
            device="cpu",
            pos_len=BOARD_SIZE,
            verbose=False
        )
        
        # Ensure model is in eval mode for inference
        model.eval()
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load KataGo model from {model_path}: {e}")


def decode_position_npz(npz_file: Path) -> np.ndarray:
    """Convert a single-position .npz into a float32 input tensor.

    Expected output shape: (C_in, 9, 9).
    Adapt this decoder to your data schema – fail if required arrays are
    missing.
    """
    with np.load(npz_file) as data:
        if "input" not in data:
            raise KeyError(f"{npz_file} missing 'input' array")
        arr = data["input"].astype(np.float32)
    if arr.shape[-2:] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Unexpected board shape in {npz_file}: {arr.shape}")
    return arr


# ────────────────────────────────────────────────────────────────────────────
# Core extractor
# ────────────────────────────────────────────────────────────────────────────

class ActivationExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Forward hook to capture activations
        self._captured: List[torch.Tensor] = []

        try:
            target_module = dict(self.model.named_modules())[layer_name]
        except KeyError:
            available_layers = [name for name, _ in self.model.named_modules()]
            raise KeyError(
                f"Layer '{layer_name}' not found in model modules. "
                f"Available layers: {available_layers[:10]}... (showing first 10)"
            )

        def _hook(_module, _inp, out):  # noqa: D401, N802
            # No mutation – detach ASAP.
            self._captured.append(out.detach())

        target_module.register_forward_hook(_hook)

    # ── PUBLIC API ──────────────────────────────────────────────────────
    def run(self, position_files: List[Path]) -> Tuple[np.ndarray, List[str]]:
        rows: List[np.ndarray] = []
        index: List[str] = []

        for start in range(0, len(position_files), self.batch_size):
            batch_paths = position_files[start : start + self.batch_size]
            batch_np = np.stack([decode_position_npz(p) for p in batch_paths])
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=self.device)

            # ── INFERENCE ──────────────────────────────────────────────
            self._captured.clear()
            _ = self.model(batch_tensor)
            if not self._captured:
                raise RuntimeError("Forward hook did not capture any activations")
            act = self._captured[0]  # shape: (B, C, 9, 9)
            if act.dim() != 4:
                raise ValueError(f"Expected 4-D activations, got {act.shape}")

            # ── SPATIAL MEAN POOL ─────────────────────────────────────
            pooled = act.mean(dim=(-1, -2)).cpu().numpy()  # (B, C)
            rows.append(pooled)
            index.extend([str(p) for p in batch_paths])

        big_matrix = np.concatenate(rows, axis=0)  # (N, C)
        return big_matrix, index


# ────────────────────────────────────────────────────────────────────────────
# Non-negativity & scaling helpers
# ────────────────────────────────────────────────────────────────────────────

def shift_to_non_negative(matrix: np.ndarray) -> np.ndarray:
    mins = matrix.min(axis=0, keepdims=True)  # (1, C)
    if (mins < 0).any():
        matrix = matrix - mins  # broadcast
    return matrix


def scale_columns(matrix: np.ndarray) -> np.ndarray:
    maxs = matrix.max(axis=0, keepdims=True)
    if (maxs == 0).any():
        raise ValueError("Column with all zeros encountered – cannot scale")
    return matrix / maxs


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Extract pooled activations from KataGo model")
    p.add_argument("--positions-dir", required=True, type=Path, help="Directory containing .npz position files")
    p.add_argument("--model-path", required=True, type=Path, help="Path to KataGo PyTorch checkpoint (.ckpt)")
    p.add_argument("--batch-size", type=int, default=256, help="Positions per inference batch")
    p.add_argument("--output-dir", type=Path, default=Path("activations"), help="Where to write outputs")
    p.add_argument("--board-size", type=int, default=7, help="Board size (e.g. 7, 9, 19)")
    p.add_argument("--device", default="cpu", help="CUDA device ID or 'cpu'")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()

    global BOARD_SIZE  # noqa: PLW0603
    BOARD_SIZE = args.board_size

    chosen_layer, channels = load_layer_selection(Path("2_pick_layer/layer_selection.yml"))

    model = load_katago_pytorch(args.model_path)
    extractor = ActivationExtractor(model, chosen_layer, args.batch_size, args.device)

    # ── Enumerate positions ───────────────────────────────────────────
    position_files = sorted(args.positions_dir.rglob("*.npz"))
    if not position_files:
        raise FileNotFoundError(f"No .npz files found under {args.positions_dir}")

    matrix, index = extractor.run(position_files)

    if matrix.shape[1] != channels:
        raise ValueError(
            f"Channel mismatch: expected {channels}, got {matrix.shape[1]}")

    matrix = shift_to_non_negative(matrix)
    matrix = scale_columns(matrix)

    # ── Persist ───────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = chosen_layer.replace("_output", "")
    np.save(args.output_dir / f"pooled_{layer_tag}.npy", matrix)

    with (args.output_dir / "pos_index_to_npz.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(index))

    meta = {
        "date": date.today().isoformat(),
        "source_model": str(args.model_path),
        "layer": chosen_layer,
        "positions": len(position_files),
        "channels": channels,
        "batch_size": args.batch_size,
        "non_negative_shift": True,
        "column_scaled": True,
    }
    with (args.output_dir / "pooled_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"✅ Extracted {matrix.shape[0]} rows × {matrix.shape[1]} channels → "
        f"{args.output_dir}/pooled_{layer_tag}.npy")


if __name__ == "__main__":
    main() 