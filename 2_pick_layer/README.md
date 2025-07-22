# Step 2: Pick Layer

Identify and record an intermediate residual block (e.g. the middle trunk block) in a *general* KataGo network. This example assumes a 7×7 board, but the same layer names apply to any board size.

## What This Does
Runs a single dummy forward pass with KataGo's PyTorch model and prints every available tensor name via `model.named_modules()`. You can then choose one of those names and write it to `layer_selection.yml` for downstream scripts.

## Usage
```bash
cd 2_pick_layer
python pick_layer.py --model-path ../models/<latest_general_net>.bin.gz --board-size 7 --choose trunk_block_14_output  # example
```

## Output
Creates `layer_selection.yml` with the chosen layer information:
```yaml
chosen_layer: trunk_block_14_output
date: '2025-07-21'
network_file: models/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz
board_size: 7
pooling: spatial mean
rationale: "Middle trunk balances local vs. global signals"
```

## Rationale
Middle trunk layers often strike the best trade-off between low-level pattern detectors and high-level policy/value fusion, making them prime candidates for interpretability studies.

## Next Step
→ Go to `3_extract_activations/` to extract the actual neural network activations from this layer. 