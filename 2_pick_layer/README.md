# Step 2: Pick Layer

Identify and record the middle residual block in the KataGo 9×9 network for downstream interpretability experiments.

## What This Does
Analyzes the model filename to determine network architecture and selects the middle trunk block as the layer to probe. For an 18-block network (`b18c384`), this chooses block 9.

## Usage
```bash
cd 2_pick_layer
python pick_layer.py
```

## Output
Creates `layer_selection.yml` with the chosen layer information:
```yaml
chosen_layer: trunk_block_9_output
date: '2025-01-21'
layer_shape: "384×9×9 (approx)"
network_file: models/kata9x9-b18c384nbt-20231025.bin.gz
pooling: spatial mean
rationale: "Mid‑trunk balances local patterns vs. mixed policy/value signals..."
trunk_blocks: 18
```

## Rationale
The middle layer strikes a balance between:
- **Lower layers**: Simple local patterns (edges, shapes)
- **Higher layers**: Complex mixed signals (policy/value predictions)

Middle layers tend to contain the most interpretable "Go concepts" that humans can understand.

## Next Step
→ Go to `3_extract_activations/` to extract the actual neural network activations from this layer. 