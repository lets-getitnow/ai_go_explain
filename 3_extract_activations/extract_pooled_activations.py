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


def decode_position_npz(npz_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Decode **all** positions contained in a KataGo training `.npz` file.

    Every `.npz` written by KataGo's self-play contains a *batch* of positions.
    The original quick-start demo purposefully consumed only the first sample
    to keep runtime tiny.  For real analysis we need the **entire** batch so
    that subsequent NMF sees thousands of rows instead of just a handful.

    Returns
    -------
    binary_input : np.ndarray
        Shape **(N, C_in, BOARD_SIZE, BOARD_SIZE)** – board feature planes for
        *all* N positions in the file.
    global_input : np.ndarray
        Shape **(N, G_in)** – global feature vectors corresponding to each
        position.
    """
    with np.load(npz_file) as data:
        if "binaryInputNCHWPacked" not in data:
            raise KeyError(f"{npz_file} missing 'binaryInputNCHWPacked' array")
        if "globalInputNC" not in data:
            raise KeyError(f"{npz_file} missing 'globalInputNC' array")
 
        binaryInputNCHWPacked = data["binaryInputNCHWPacked"]
        globalInputNC = data["globalInputNC"]
        
        # Unpack the binary format using KataGo's decoding logic
        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((BOARD_SIZE * BOARD_SIZE + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:BOARD_SIZE*BOARD_SIZE]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], BOARD_SIZE, BOARD_SIZE
        )).astype(np.float32)
        
    # Cast to float32 once at the very end for memory efficiency
    binary_arrs = binaryInputNCHW.astype(np.float32)             # (N, C_in, B, B)
    global_arrs = globalInputNC.astype(np.float32)               # (N, G_in)

    if binary_arrs.shape[0] != global_arrs.shape[0]:
        raise ValueError(
            f"Sample count mismatch in {npz_file}: "
            f"{binary_arrs.shape[0]} vs {global_arrs.shape[0]}"
        )

    if binary_arrs.shape[-2:] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"Unexpected board shape in {npz_file}: {binary_arrs.shape[-2:]}"
        )

    return binary_arrs, global_arrs


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
        """Run inference on **every** position in `position_files`.

        The method streams data in mini-batches of size ``self.batch_size`` to
        keep peak memory usage predictable.  It returns:

        • ``big_matrix`` – shape (N_positions, C_channels)
        • ``index``      – list mapping each row → originating `.npz` file
        """

        print(f"[INFO] Starting extraction over {len(position_files)} .npz files with batch size {self.batch_size}…")

        rows: List[np.ndarray] = []
        index: List[str] = []

        buffer_bin: List[np.ndarray] = []  # (C_in, B, B)
        buffer_glob: List[np.ndarray] = []  # (G_in,)
        buffer_paths: List[str] = []

        def _flush_buffer() -> None:  # capture outer-scope via closure
            """Run one batched forward pass and append pooled outputs."""
            if not buffer_bin:
                return  # nothing to do

            batch_binary = np.stack(buffer_bin)
            batch_global = np.stack(buffer_glob)

            batch_binary_tensor = torch.tensor(
                batch_binary, dtype=torch.float32, device=self.device
            )
            batch_global_tensor = torch.tensor(
                batch_global, dtype=torch.float32, device=self.device
            )

            # ── INFERENCE ────────────────────────────────────────────
            self._extra.returned.clear()
            self._extra.available.clear()
            _ = self.model(
                batch_binary_tensor,
                batch_global_tensor,
                extra_outputs=self._extra,  # type: ignore[call-arg]
            )

            # Validate layer name exactly once
            if not hasattr(self, "_validated"):
                if self._layer_name not in self._extra.available:
                    sample = sorted(self._extra.available)[:15]
                    raise KeyError(
                        f"Layer '{self._layer_name}' not recognised. "
                        f"Some available names: {', '.join(sample)} …"
                    )
                self._validated = True  # type: ignore[attr-defined]

            if self._layer_name not in self._extra.returned:
                raise RuntimeError(
                    f"ExtraOutputs did not capture '{self._layer_name}'. "
                    "This indicates a mismatch between requested tensor and "
                    "what the network produced."
                )

            act = self._extra.returned[self._layer_name]
            if act.dim() != 4:
                raise ValueError(f"Expected 4-D activations, got {act.shape}")

            pooled = act.mean(dim=(-1, -2)).cpu().numpy()  # (B, C_channels)
            rows.append(pooled)
            index.extend(buffer_paths)
            print(f"[PROGRESS] Processed {sum(r.shape[0] for r in rows)} positions so far…")

            # Clear buffers for next mini-batch
            buffer_bin.clear()
            buffer_glob.clear()
            buffer_paths.clear()

        # ── Stream over *.npz files ─────────────────────────────────
        for npz_file in position_files:
            binary_arrs, global_arrs = decode_position_npz(npz_file)

            if binary_arrs.shape[0] != global_arrs.shape[0]:
                raise ValueError(
                    f"Sample count mismatch in {npz_file}: "
                    f"{binary_arrs.shape[0]} vs {global_arrs.shape[0]}"
                )

            for i in range(binary_arrs.shape[0]):
                buffer_bin.append(binary_arrs[i])
                buffer_glob.append(global_arrs[i])
                buffer_paths.append(str(npz_file))  # duplicate path per row

                if len(buffer_bin) == self.batch_size:
                    _flush_buffer()

        # Flush any remaining samples that didn't fill a complete batch
        _flush_buffer()

        big_matrix = np.concatenate(rows, axis=0)  # (N_positions, C_channels)
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
    print(f"[INFO] {len(position_files)} .npz files found. Starting extraction…")

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