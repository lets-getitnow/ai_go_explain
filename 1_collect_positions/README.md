# Step 1: Collect Positions

Generate varied 7×7 board positions using KataGo's self-play engine.

## Prerequisites
- KataGo binary installed
- Model file: `models/<latest_general_net>.bin.gz`  <!-- e.g. kata1-b28c512nbt-sXXXXX.bin.gz -->
- Config file: `selfplay7.cfg`

## Command
```bash
katago selfplay \
  -config selfplay7.cfg \
  -models-dir models/ \
  -output-dir selfplay_out/ \
  -max-games-total 200
```

## Output
Creates `selfplay_out/` directory with the following structure:
```
selfplay_out/
├── log*.log (execution logs)
└── <latest_general_net>/
    ├── selfplay-*.cfg (generated config)
    ├── sgfs/
    │   └── *.sgfs (game records)
    └── tdata/
        └── *.npz (board positions)
```

The `.npz` files in `tdata/` contain the actual position data that will be used in Step 3 for activation extraction.

## Next Step
→ Go to `2_pick_layer/` to choose which network layer to analyze. 