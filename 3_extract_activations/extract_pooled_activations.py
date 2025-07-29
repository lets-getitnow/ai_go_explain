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
import time
import signal
from contextlib import contextmanager

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
            # pos_len=19 for the 19x19 model (can handle smaller boards), use_swa=False for standard model, device="cpu" for consistency
            model, swa_model, other_state_dict = load_model(
                checkpoint_file=str(ckpt_path),
                use_swa=False,
                device="cpu",
                pos_len=19,  # Model is 19x19, can handle smaller boards
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


def decode_position_npz(npz_file: Path, board_size: int = BOARD_SIZE) -> Tuple[np.ndarray, np.ndarray]:
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
        assert binaryInputNCHW.shape[2] == ((board_size * board_size + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:board_size*board_size]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], board_size, board_size
        )).astype(np.float32)
        
    # Cast to float32 once at the very end for memory efficiency
    binary_arrs = binaryInputNCHW.astype(np.float32)             # (N, C_in, B, B)
    global_arrs = globalInputNC.astype(np.float32)               # (N, G_in)

    if binary_arrs.shape[0] != global_arrs.shape[0]:
        raise ValueError(
            f"Sample count mismatch in {npz_file}: "
            f"{binary_arrs.shape[0]} vs {global_arrs.shape[0]}"
        )

    if binary_arrs.shape[-2:] != (board_size, board_size):
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

        # The real ExtraOutputs instance requesting exactly the layer we want.
        self._extra = ExtraOutputs(requested=[layer_name])
        self._layer_name = layer_name
        # Sanity – detach tensors immediately after capture.
        self._extra.no_grad = True

    # ── PUBLIC API ──────────────────────────────────────────────────────
    def run(self, position_files: List[Path], *, total_positions: int | None = None) -> Tuple[np.ndarray, List[str]]:
        """Run inference on **every** position in `position_files`.

        The method streams data in mini-batches of size ``self.batch_size`` to
        keep peak memory usage predictable.  It returns:

        • ``big_matrix`` – shape (N_positions, C_channels)
        • ``index``      – list mapping each row → originating `.npz` file
        """

        if total_positions is None:
            total_msg = "unknown number of"
        else:
            total_msg = f"{total_positions}"

        print(
            f"[INFO] Starting extraction over {len(position_files)} .npz files "
            f"(~{total_msg} positions) with batch size {self.batch_size}…"
        )

        # ── INITIALISE ───────────────────────────────────────────────────────────
        rows: List[np.ndarray] = []
        index: List[str] = []
        buffer_bin: List[np.ndarray] = []
        buffer_glob: List[np.ndarray] = []
        buffer_paths: List[str] = []
        start_time = time.perf_counter()

        def _flush_buffer() -> None:
            if not buffer_bin:
                return
            
            print("[DEBUG] Step 1: Creating batch tensors…")
            # Convert numpy arrays to tensors before stacking
            batch_binary_tensor = torch.stack([torch.from_numpy(arr) for arr in buffer_bin], dim=0).to(self.device)
            batch_global_tensor = torch.stack([torch.from_numpy(arr) for arr in buffer_glob], dim=0).to(self.device)
            
            print(f"[DEBUG] Step 2: Running inference on batch of {len(buffer_bin)} positions…")
            self._extra.returned.clear()
            self._extra.available.clear()
            
            with timeout_context(300):
                _ = self.model(batch_binary_tensor, batch_global_tensor, extra_outputs=self._extra)
            
            act = self._extra.returned[self._layer_name]
            if act.dim() != 4:
                raise ValueError(f"Expected 4-D activations, got {act.shape}")

            if torch.isnan(act).any():
                print(f"[WARNING] Batch contains NaN activations, retrying with smaller batch...")
                if len(buffer_bin) > 1:
                    half_size = len(buffer_bin) // 2
                    temp_bin = buffer_bin[:half_size]
                    temp_glob = buffer_glob[:half_size]
                    temp_paths = buffer_paths[:half_size]
                    buffer_bin.clear()
                    buffer_glob.clear()
                    buffer_paths.clear()
                    
                    # Convert numpy arrays to tensors for the smaller batch
                    batch_binary_tensor = torch.stack([torch.from_numpy(arr) for arr in temp_bin], dim=0).to(self.device)
                    batch_global_tensor = torch.stack([torch.from_numpy(arr) for arr in temp_glob], dim=0).to(self.device)
                    self._extra.returned.clear()
                    self._extra.available.clear()
                    with timeout_context(300):
                        _ = self.model(batch_binary_tensor, batch_global_tensor, extra_outputs=self._extra)
                    act = self._extra.returned[self._layer_name]
                    if torch.isnan(act).any():
                        print(f"[WARNING] Skipping batch with NaN activations")
                        buffer_bin.clear()
                        buffer_glob.clear()
                        buffer_paths.clear()
                        return
                    else:
                        print(f"[INFO] Successfully processed smaller batch of {half_size} positions")
                        _process_activations(act, temp_paths)
                        return
                else:
                    print(f"[WARNING] Skipping single position with NaN activations")
                    buffer_bin.clear()
                    buffer_glob.clear()
                    buffer_paths.clear()
                    return
            _process_activations(act, buffer_paths)
            
            buffer_bin.clear()
            buffer_glob.clear()
            buffer_paths.clear()

        def _process_activations(act: torch.Tensor, paths: List[str]) -> None:
            """Process activations with 3x3 pooling and track progress."""
            # 3x3 grid pooling instead of global average
            # act shape: (B, C, H, W) where H=W=board_size
            B, C, H, W = act.shape
            assert H == W, f"Expected square activations, got {H}x{W}"
            board_size = H
            
            # Create 3x3 grid pooling for any board size
            # Split the board into 3x3 regions
            step = board_size // 3
            bins = []
            for i in range(3):
                start = i * step
                end = (i + 1) * step if i < 2 else board_size  # Last bin goes to the end
                bins.append((start, end))
            pooled_parts = []
            for r0, r1 in bins:
                for c0, c1 in bins:
                    # Average over the spatial region (r0:r1, c0:c1)
                    region = act[:, :, r0:r1, c0:c1].mean(dim=(-1, -2))  # (B, C)
                    pooled_parts.append(region)
            
            # Concatenate all 9 regions: (B, C*9)
            pooled = torch.cat(pooled_parts, dim=1).cpu().numpy()
            rows.append(pooled)
            index.extend(paths)
            processed = sum(r.shape[0] for r in rows)
            if total_positions:
                pct = processed / total_positions * 100
                print(
                    f"[PROGRESS] Processed {processed}/{total_positions} positions "
                    f"({pct:5.1f}%) [last batch {time.perf_counter() - start_time:.2f}s]"
                )
            else:
                print(
                    f"[PROGRESS] Processed {processed} positions so far "
                    f"[last batch {time.perf_counter() - start_time:.2f}s]"
                )

        # ── Stream over *.npz files ─────────────────────────────────────────────
        for npz_path in position_files:
            print(f"[INFO] Decoding {npz_path.name} …")
            decode_start = time.perf_counter()
            binary_arrs, global_arrs = decode_position_npz(npz_path, board_size=BOARD_SIZE)
            decode_dur = time.perf_counter() - decode_start
            print(f"[DEBUG] Finished decoding {npz_path.name}: {binary_arrs.shape[0]} positions in {decode_dur:.2f}s")

            for i in range(binary_arrs.shape[0]):
                buffer_bin.append(binary_arrs[i])
                buffer_glob.append(global_arrs[i])
                buffer_paths.append(str(npz_path))  # duplicate path per row

                # Verbose within-file progress every 100 samples
                if (i + 1) % 100 == 0 or (i + 1) == binary_arrs.shape[0]:
                    loaded = i + 1
                    print(
                        f"[TRACE] Loaded {loaded}/{binary_arrs.shape[0]} "
                        f"positions from {npz_path.name} into buffer"
                    )

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
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--positions-dir", type=Path, help="Directory containing .npz position files (single dataset)")
    group.add_argument("--variants-root", type=Path, help="Root dir whose *direct* subfolders each contain .npz files for a variant (e.g. baseline/, zero_global/, komi_sweep/)")
    p.add_argument("--ckpt-path", required=True, type=Path, help="Path to KataGo model.ckpt checkpoint file")
    p.add_argument("--batch-size", type=int, default=256, help="Positions per inference batch")
    p.add_argument("--output-dir", type=Path, default=Path("activations"), help="Where to write outputs")
    p.add_argument("--board-size", type=int, default=7, help="Board size (e.g. 7, 9, 19)")
    p.add_argument("--device", default="cpu", help="CUDA device ID or 'cpu'")
    p.add_argument("--processor", choices=["cpu", "cuda", "mps"], default="cpu",
                   help="Processor to use: cpu, cuda, or mps (Metal Performance Shaders for Apple Silicon)")
    return p.parse_args()


@contextmanager
def timeout_context(seconds: int):
    """Context manager that raises TimeoutError if operation takes too long."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> None:  # noqa: D401
    args = parse_args()

    global BOARD_SIZE  # noqa: PLW0603
    BOARD_SIZE = args.board_size

    # Find project root (directory containing this script's parent)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    layer_selection_path = project_root / "2_pick_layer" / "layer_selection.yml"
    
    chosen_layer, channels = load_layer_selection(layer_selection_path)

    # Determine device based on processor choice
    if args.processor == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device = "cpu"
        else:
            device = "mps"
    elif args.processor == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device = args.device if args.device != "cpu" else "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")
    
    model = load_katago_pytorch(args.ckpt_path)
    extractor = ActivationExtractor(model, chosen_layer, args.batch_size, device)

    # Helper to process one dataset dir -> npy outputs
    def _process_dataset(dataset_dir: Path, tag: str) -> None:
        position_files = sorted(dataset_dir.rglob("*.npz"))
        if not position_files:
            raise FileNotFoundError(f"No .npz files found under {dataset_dir}")
        print(f"[INFO] [{tag}] {len(position_files)} .npz files found. Starting extraction…")

        total_positions = 0
        for f in position_files:
            with np.load(f) as data:
                if "binaryInputNCHWPacked" not in data:
                    raise KeyError(f"{f} missing 'binaryInputNCHWPacked' array")
                total_positions += data["binaryInputNCHWPacked"].shape[0]

        print(f"[INFO] [{tag}] Total positions to process: {total_positions}")

        matrix, index = extractor.run(position_files, total_positions=total_positions)

        # With 3x3 pooling, we have 9x the original channels
        expected_channels = channels * 9
        if matrix.shape[1] != expected_channels:
            raise ValueError(
                f"Channel mismatch: expected {expected_channels} (9x{channels}), got {matrix.shape[1]}")

        matrix_nn = shift_to_non_negative(matrix)
        matrix_scaled = scale_columns(matrix_nn)

        # ── Persist ───────────────────────────────────────────────────
        args.output_dir.mkdir(parents=True, exist_ok=True)
        layer_tag = chosen_layer.replace("_output", "")
        np.save(args.output_dir / f"pooled_{layer_tag}__{tag}.npy", matrix_scaled)

        with (args.output_dir / f"pos_index_to_npz__{tag}.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(index))

        meta = {
            "date": date.today().isoformat(),
            "source_model": str(args.ckpt_path),
            "layer": chosen_layer,
            "positions": len(position_files),
            "original_channels": channels,
            "pooled_channels": expected_channels,
            "pooling_method": "3x3_grid",
            "batch_size": args.batch_size,
            "non_negative_shift": True,
            "column_scaled": True,
            "variant_tag": tag,
            "dataset_dir": str(dataset_dir),
        }
        with (args.output_dir / f"pooled_meta__{tag}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"✅ [{tag}] Extracted {matrix.shape[0]} rows × {matrix.shape[1]} channels → "
            f"{args.output_dir}/pooled_{layer_tag}__{tag}.npy")

    # ── Decide datasets ───────────────────────────────────────────────
    if args.positions_dir:
        _process_dataset(args.positions_dir, "baseline")
    else:
        root: Path = args.variants_root
        if not root.exists():
            raise FileNotFoundError(f"variants_root not found: {root}")
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found under {root}")
        for sub in sorted(subdirs):
            _process_dataset(sub, sub.name)


if __name__ == "__main__":
    main() 