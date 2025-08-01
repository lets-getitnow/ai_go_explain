#!/usr/bin/env python3
"""
Human Games Pipeline Runner
===========================
Purpose
-------
Run the complete activation analysis pipeline (steps 1-5) on human SGF games.
This script automates the entire process from SGF conversion to final HTML reports.

Pipeline Steps:
1. Convert human SGF games to NPZ format
2. Pick a layer for analysis (uses existing layer_selection.yml)
3. Extract activations from the chosen layer
4. Run NMF to find interpretable parts
5. Inspect parts and generate HTML reports

Usage
------
python run_human_games_pipeline.py \
    --input-dir games/go13 \
    --output-dir human_games_analysis \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

def run_command(cmd: List[str], description: str) -> None:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Set up environment with KataGo Python path
    env = os.environ.copy()
    katago_python_path = str(Path(__file__).parent.parent / "KataGo" / "python")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = katago_python_path + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = katago_python_path
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure all required dependencies are installed")
        sys.exit(1)

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete activation analysis pipeline on human games"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing human SGF files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for all analysis results"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to KataGo model checkpoint"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=7,
        help="Board size (default: 7)"
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip SGF to NPZ conversion (use existing NPZ files)"
    )
    parser.add_argument(
        "--skip-layer-pick",
        action="store_true", 
        help="Skip layer selection (use existing layer_selection.yml)"
    )
    parser.add_argument(
        "--processor",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Processor to use for activation extraction (default: cpu)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of SGF files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    if not args.model_path.exists():
        print(f"Error: Model file {args.model_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting human games analysis pipeline")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Board size: {args.board_size}")
    
    # Step 1: Convert SGF to NPZ (if not skipped)
    if not args.skip_conversion:
        npz_dir = args.output_dir / "npz_files"
        convert_cmd = [
            "python3", "1_collect_positions/convert_human_games.py",
            "--input-dir", str(args.input_dir),
            "--output-dir", str(npz_dir),
            "--board-size", str(args.board_size)
        ]
        if args.max_files:
            convert_cmd.extend(["--max-files", str(args.max_files)])
        run_command(convert_cmd, "Convert SGF games to NPZ format")
    else:
        npz_dir = args.output_dir / "npz_files"
        if not npz_dir.exists():
            print(f"Error: NPZ directory {npz_dir} does not exist")
            sys.exit(1)
        print("⏭️  Skipping SGF conversion (using existing NPZ files)")
    
    # Step 2: Pick layer (if not skipped)
    if not args.skip_layer_pick:
        layer_selection_file = Path("2_pick_layer/layer_selection.yml")
        if not layer_selection_file.exists():
            print("⚠️  No existing layer_selection.yml found, running layer selection...")
            run_command([
                "python3", "2_pick_layer/pick_layer.py",
                "--model-path", str(args.model_path),
                "--board-size", str(args.board_size)
            ], "Select layer for analysis")
        else:
            print("⏭️  Using existing layer_selection.yml")
    else:
        print("⏭️  Skipping layer selection")
    
    # Step 3: Extract activations
    activations_dir = args.output_dir / "activations"
    batch_size = "32" if args.processor == "mps" else "64"
    run_command([
        "python3", "3_extract_activations/extract_pooled_activations.py",
        "--positions-dir", str(npz_dir),
        "--ckpt-path", str(args.model_path),
        "--output-dir", str(activations_dir),
        "--batch-size", batch_size,
        "--board-size", str(args.board_size),
        "--processor", args.processor
    ], "Extract activations from chosen layer")
    
    # Step 4: Run NMF analysis
    nmf_dir = args.output_dir / "nmf_parts"
    run_command([
        "python3", "4_nmf_parts/run_nmf.py",
        "--activations-file", str(activations_dir / "pooled_rconv14.out__baseline.npy"),
        "--output-dir", str(nmf_dir),
        "--num-components", "50",
        "--max-iter", "1000"
    ], "Run NMF to find interpretable parts")
    
    # Step 5: Inspect parts and generate HTML reports
    inspect_dir = args.output_dir / "inspect_parts"
    run_command([
        "python3", "5_inspect_parts/inspect_parts_human_games.py",
        "--nmf-dir", str(nmf_dir),
        "--npz-dir", str(npz_dir),
        "--output-dir", str(inspect_dir),
        "--max-positions", "10",
        "--board-size", str(args.board_size)
    ], "Inspect NMF parts and generate analysis")
    
    # Generate HTML reports
    html_dir = args.output_dir / "html_reports"
    run_command([
        "python3", "5_inspect_parts/generate_html_reports.py",
        "--summary-file", str(inspect_dir / "strong_positions_summary.csv"),
        "--output-dir", str(html_dir),
        "--board-size", str(args.board_size)
    ], "Generate HTML visualization reports")
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"Results available in: {args.output_dir}")
    print(f"HTML reports: {html_dir}")
    print(f"Open {html_dir}/index.html to view the results")

if __name__ == "__main__":
    main() 