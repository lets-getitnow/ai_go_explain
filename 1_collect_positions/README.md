# Step 1: Collect Positions

Generate varied 9×9 board positions using KataGo's selfplay engine.

## Prerequisites
- KataGo binary installed
- Model file: `models/kata9x9-b18c384nbt-20231025.bin.gz`
- Config file: `selfplay9.cfg`

## Command
```bash
katago selfplay \
  -config selfplay9.cfg \
  -models-dir models/ \
  -output-dir selfplay_out/ \
  -max-games-total 200
```

## Output
Creates `selfplay_out/` directory with the following structure:
```
selfplay_out/
├── log*.log (execution logs)
└── kata9x9-b18c384nbt-20231025.bin.gz/
    ├── selfplay-*.cfg (generated config)
    ├── sgfs/
    │   └── *.sgfs (game records)
    └── tdata/
        └── *.npz (board positions)
```

The `.npz` files in `tdata/` contain the actual position data that will be used in Step 3 for activation extraction.

## Next Step
→ Go to `2_pick_layer/` to choose which network layer to analyze. 