#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (Human Games Version)

Simplified version for human games data that doesn't require selfplay data.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_nmf_data(nmf_dir: Path):
    """Load NMF components and activations."""
    components_path = nmf_dir / "nmf_components.npy"
    activations_path = nmf_dir / "nmf_activations.npy"
    meta_path = nmf_dir / "nmf_meta.json"
    
    if not components_path.exists():
        raise FileNotFoundError(f"NMF components not found: {components_path}")
    if not activations_path.exists():
        raise FileNotFoundError(f"NMF activations not found: {activations_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"NMF meta not found: {meta_path}")
    
    components = np.load(components_path)
    activations = np.load(activations_path)
    
    with meta_path.open('r') as f:
        meta = json.load(f)
    
    return components, activations, meta

def strongest_indices_for_part(part_idx: int, activ: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k strongest activations for the given component."""
    part_act = activ[:, part_idx]
    top = np.argsort(part_act)[-k:][::-1]
    return top

def analyze_part(part_idx: int, components: np.ndarray, activations: np.ndarray, meta: Dict) -> Dict[str, Any]:
    """Analyze a single NMF part."""
    component = components[part_idx]
    
    # Get strongest activations for this part
    top_indices = strongest_indices_for_part(part_idx, activations, 10)
    top_activations = activations[top_indices, part_idx]
    
    analysis = {
        "part_idx": part_idx,
        "component_shape": component.shape,
        "component_stats": {
            "min": float(np.min(component)),
            "max": float(np.max(component)),
            "mean": float(np.mean(component)),
            "std": float(np.std(component)),
            "sparsity": float(np.mean(component == 0))
        },
        "top_activations": {
            "indices": top_indices.tolist(),
            "values": top_activations.tolist()
        },
        "meta": meta
    }
    
    return analysis

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmf-dir", type=Path, required=True, help="Directory containing NMF results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--max-parts", type=int, default=10, help="Maximum number of parts to analyze")
    
    args = parser.parse_args()
    
    print("=== Step 5 – Inspect Parts (Human Games) ===")
    
    # Load NMF data
    print(f"Loading NMF data from {args.nmf_dir}...")
    components, activations, meta = load_nmf_data(args.nmf_dir)
    
    print(f"Components shape: {components.shape}")
    print(f"Activations shape: {activations.shape}")
    print(f"Meta keys: {list(meta.keys())}")
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Analyze parts
    n_parts = min(args.max_parts, components.shape[0])
    print(f"Analyzing {n_parts} parts...")
    
    analyses = []
    for part_idx in range(n_parts):
        print(f"Analyzing part {part_idx}...")
        analysis = analyze_part(part_idx, components, activations, meta)
        analyses.append(analysis)
    
    # Save analysis
    output_file = args.output_dir / "part_analyses.json"
    with output_file.open('w') as f:
        json.dump(analyses, f, indent=2, cls=NumpyEncoder)
    
    print(f"✅ Analysis saved to {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    for analysis in analyses:
        part_idx = analysis["part_idx"]
        stats = analysis["component_stats"]
        print(f"Part {part_idx}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, sparsity={stats['sparsity']:.2%}")

if __name__ == "__main__":
    main()