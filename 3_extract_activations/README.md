# Step 3: Extract Activations

Convert KataGo's binary model to PyTorch format and extract pooled activations from the chosen layer.

## Prerequisites
- Completed Step 1 (positions in `selfplay_out/`)
- Completed Step 2 (`layer_selection.yml` exists)
- PyTorch installed

## One-Time Setup: Convert Model

1. **Clone KataGo & its helper:**
   ```bash
   git clone https://github.com/lightvector/KataGo.git
   cd KataGo/python             # contains export_model.py
   pip install -r requirements.txt  # torch, numpy, pyyaml
   ```

2. **Convert the binary weights to PyTorch** (keeps original layer names so the hook works):
   ```bash
   python export_model.py \
     --model  ../../models/kata9x9-b18c384nbt-20231025.bin.gz \
     --output ../../models/kata9x9-b18c384nbt-20231025.pt \
     --no-half          # stay in fp32 to avoid hook size mismatches
   ```

3. **Update `load_katago_pytorch()` function** in `extract_pooled_activations.py`:
   ```python
   from importlib import import_module

   def load_katago_pytorch(path: Path) -> torch.nn.Module:
       kata_mod = import_module("katago_pytorch_model")  # dropped by exporter
       net = kata_mod.KataGoModel()
       net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
       net.eval()
       return net
   ```

## Extract Activations

Run the extractor to produce the pooled activation matrix:

```bash
cd 3_extract_activations
python extract_pooled_activations.py \
  --positions-dir ../selfplay_out \
  --model-path   ../models/kata9x9-b18c384nbt-20231025.pt \
  --batch-size   256 \
  --device       cuda:0   # or cpu
```

## Output

Creates `activations/` directory with:
```text
activations/
  pooled_trunk_block_9.npy      # (N_positions, 384) - the main data
  pooled_meta.json              # metadata about extraction
  pos_index_to_npz.txt          # mapping back to original positions
```

The `pooled_trunk_block_9.npy` file contains the matrix **A** that will be used for NMF decomposition in Step 4.

## Next Step
→ Continue to Step 4 (NMF parts finding) with your extracted activation matrix. 