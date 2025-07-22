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


def load_katago_pytorch(ckpt_path: Path) -> torch.nn.Module:
    """Load a KataGo PyTorch checkpoint as a `torch.nn.Module`.

    Loads a KataGo training checkpoint (model.ckpt file) using KataGo's existing
    PyTorch infrastructure. The model will have proper layer names for hooking
    and accepts input tensors of shape (B, C_in, 9, 9).
    
    Args:
        ckpt_path: Path to KataGo model.ckpt checkpoint file
        
    Returns:
        PyTorch model in eval mode with named modules matching KataGo layer names
        
    Raises:
        ImportError: If KataGo python modules cannot be imported
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        
    try:
        from katago.train.load_model import load_model
    except ImportError as e:
        raise ImportError(
            f"Failed to import KataGo modules. Ensure KataGo/python is in path: {e}"
        )
    
    try:
        # Temporarily monkey-patch torch.load to handle PyTorch 2.6 compatibility
        # PyTorch 2.6 changed weights_only default from False to True
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        try:
            # Load model using KataGo's infrastructure
            # pos_len=BOARD_SIZE for correct board size, use_swa=False for standard model, device="cpu" for consistency
            model, swa_model, other_state_dict = load_model(
                checkpoint_file=str(ckpt_path),
                use_swa=False,
                device="cpu",
                pos_len=BOARD_SIZE,
                verbose=False
            )
        finally:
            # Always restore original torch.load
            torch.load = original_torch_load
        
        # Ensure model is in eval mode for inference
        model.eval()
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load KataGo model from {ckpt_path}: {e}")


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
        """Initialise extractor using KataGo's ExtraOutputs API.

        Beginning July 2025 the recommended way to obtain intermediate tensors
        from KataGo's PyTorch implementation is to pass an `ExtraOutputs`
        instance into the *forward* call.  This avoids fragile global state
        (registering hooks) and guarantees that the tensor names exactly match
        those reported by KataGo itself.

        Args:
            model:  The KataGo network in *eval* mode.
            layer_name:  Name of the tensor to capture – **must** match one of
                          the strings reported in ``ExtraOutputs.available``.
            batch_size:  Mini-batch size for inference.
            device:  CUDA device string or "cpu".
        """

        self.model = model.to(device).eval()
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Import here to fail fast if the user hasn't cloned KataGo/python.
        try:
            from katago.train.model_pytorch import ExtraOutputs  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Unable to import 'ExtraOutputs' from KataGo. "
                "Ensure the KataGo repository is cloned next to this project "
                "(…/KataGo/python) and PYTHONPATH is set accordingly."
            ) from e

        # Run one dummy forward pass to populate `.available` so we can give a
        # Cannot pre-validate the layer name without knowing input tensor
        # shape.  We therefore defer the check until the first real forward
        # pass where we verify that the requested activation was produced.

        # The real ExtraOutputs instance requesting exactly the layer we want.
        self._extra = ExtraOutputs(requested=[layer_name])
        self._layer_name = layer_name
        # Sanity – detach tensors immediately after capture.
        self._extra.no_grad = True

    # ── PUBLIC API ──────────────────────────────────────────────────────
    def run(self, position_files: List[Path]) -> Tuple[np.ndarray, List[str]]:
        rows: List[np.ndarray] = []
        index: List[str] = []

        for start in range(0, len(position_files), self.batch_size):
            batch_paths = position_files[start : start + self.batch_size]
            batch_np = np.stack([decode_position_npz(p) for p in batch_paths])
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=self.device)

            # ── INFERENCE ──────────────────────────────────────────────
            self._extra.clear()
            _ = self.model(batch_tensor, extra_outputs=self._extra)  # type: ignore[call-arg]

            # After the first forward pass we can validate that the requested
            # layer name actually exists according to KataGo – this is the
            # authoritative list of *all* intermediate tensors the network
            # could provide.  We delay this check until now because we need a
            # real input tensor of the correct shape to populate
            # `ExtraOutputs.available`.
            if not hasattr(self, "_validated"):
                if self._layer_name not in self._extra.available:
                    sample = sorted(self._extra.available)[:15]  # show a subset
                    raise KeyError(
                        f"Layer '{self._layer_name}' not recognised by KataGo.\n"
                        f"Some available names: {', '.join(sample)} …"
                    )
                # Mark so we don't waste time on subsequent batches
                self._validated = True  # type: ignore[attr-defined]

            if self._layer_name not in self._extra.outputs:
                raise RuntimeError(
                    f"ExtraOutputs did not capture '{self._layer_name}'. "
                    "This indicates a mismatch between requested tensor and "
                    "what the network produced."
                )

            act = self._extra.outputs[self._layer_name]
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
    p.add_argument("--ckpt-path", required=True, type=Path, help="Path to KataGo model.ckpt checkpoint file")
    p.add_argument("--batch-size", type=int, default=256, help="Positions per inference batch")
    p.add_argument("--output-dir", type=Path, default=Path("activations"), help="Where to write outputs")
    p.add_argument("--board-size", type=int, default=7, help="Board size (e.g. 7, 9, 19)")
    p.add_argument("--device", default="cpu", help="CUDA device ID or 'cpu'")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()

    global BOARD_SIZE  # noqa: PLW0603
    BOARD_SIZE = args.board_size

    # Find project root (directory containing this script's parent)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    layer_selection_path = project_root / "2_pick_layer" / "layer_selection.yml"
    
    chosen_layer, channels = load_layer_selection(layer_selection_path)

    model = load_katago_pytorch(args.ckpt_path)
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
        "source_model": str(args.ckpt_path),
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