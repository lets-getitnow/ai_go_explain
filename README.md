Mission: To explain go ai neural networks for people to learn better from them. 

Tighly integrated with https://github.com/lightvector/KataGo

<img width="50%" height="50%" alt="explaingo" src="https://github.com/user-attachments/assets/a4ef65bd-4251-40f3-9918-376755b91440" />

Currently using SAE Approach: https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf open to best approach since this project is quite early.

**Minimal Steps (n00b version)**

1. **Collect positions:** (see [1_collect_positions/](1_collect_positions/)) Grab a few thousand varied 7×7 board snapshots (early, fights, endgame) using any standard KataGo network.
2. **Pick one layer:** (see [2_pick_layer/](2_pick_layer/)) Choose an intermediate layer of the *general* KataGo network (e.g. middle trunk block).
3. **Extract activations:** (see [3_extract_activations/](3_extract_activations/)) For each 7×7 position, record that layer's output and average it down to one number list (channel-average).
4. **Run simple parts finder (NMF):** (see [4_nmf_parts/](4_nmf_parts/)) Factor those lists into ~50–70 "parts" using Non-negative Matrix Factorization. This creates interpretable parts that represent recurring patterns in the neural activations.
5. **Inspect parts:** (see [5_inspect_parts/](5_inspect_parts/)) For each part, look at the boards where it's strongest; note any clear pattern (e.g. ladders, atari, eyes). Generate HTML reports to visualize the top positions for each part and identify meaningful go concepts.
6. **Add tiny heuristics:** Auto‑flag basics (atari present, ladder path, ko, eye forming) to see which parts match which flags.
7. **(If parts look real) Train sparse autoencoder:** Replace NMF with a small sparse model on the same pooled data to get cleaner, fewer‑on features.
8. **Name good features:** Only name those with a clear, repeatable pattern; leave the messy ones unnamed.
9. **Optional sanity check:** For a named feature, see if positions "feel different" when you imagine that feature absent (qualitative is fine).
10. **Stop and use:** Start using the named features to explain moves; only add complexity (spatial detail, more layers) if you later need finer distinctions.

## Caveat: Global / Contextual Channels

KataGo's inputs include more than the current board arrangement—move history, komi, rules, ko state, and estimated score are all encoded. Therefore some internal channels capture **contextual information rather than spatial shape**. Typical triggers include:

* Move number / player to play
* Komi or handicap stones
* Ko threats or super-ko repetition state
* Current score lead / win-probability trend
* Recent tactical sequences (history-dependent)

### Detecting and controlling for non-spatial channels

1. **Board-holdout tests** – Feed the *same* board position while varying one global input (e.g. komi) and record which channels change.
2. **History shuffling** – Randomize move history that leads to identical final board states.
3. **Komi sweep** – Vary komi ±N points while keeping board identical.
4. **Score perturbation** – Modify estimated score/win-probability inputs.
5. **Ko toggles** – Flip individual ko capture states.

### Implementation Status

✅ **Variant Generation**: `1_collect_positions/generate_variants.py` supports `zero_global` mode  
✅ **Multi-Variant Extraction**: `3_extract_activations/extract_pooled_activations.py` processes multiple datasets  
✅ **Contextual Detection**: `3_extract_activations/contextual_channel_detector.py` analyzes variance  
✅ **Layer Analysis**: Tested `rconv14.out` layer - found **purely spatial** (0 contextual channels)  

### Current Findings

**Layer `rconv14.out` Analysis**:
- **Tested**: Zeroing globalInputNC vs baseline
- **Result**: Identical activations (0 contextual channels)
- **Conclusion**: Layer is purely spatial - no global context sensitivity
- **Action**: Proceed with NMF using all 4608 channels

### Future Work

- **Test earlier layers**: Investigate layers closer to input for global sensitivity
- **Additional variants**: Implement `komi_sweep`, `history_shuffle` modes
- **Multi-layer analysis**: Systematic testing across network depth
- **Direct correlation**: Analyze correlation between globalInputNC values and activations

The contextual channel detection infrastructure is now ready and can be applied to any layer in the network.

