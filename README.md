Mission: To explain go ai neural networks for people to learn better from them. 

SAE Approach:

**Minimal Steps (n00b version)**

1. **Collect positions:** (see `1_collect_positions/`) Grab a few thousand varied 9×9 board snapshots (early, fights, endgame).
2. **Pick one layer:** (see `2_pick_layer/`) Choose a middle layer of the existing 9×9 net.
3. **Extract activations:** (see `3_extract_activations/`) For each position, record that layer's output and average it down to one number list (channel average).
4. **Run simple parts finder (NMF):** Factor those lists into \~50–70 "parts."
5. **Inspect parts:** For each part, look at the boards where it's strongest; note any clear pattern (e.g. ladders, atari, eyes).
6. **Add tiny heuristics:** Auto‑flag basics (atari present, ladder path, ko, eye forming) to see which parts match which flags.
7. **(If parts look real) Train sparse autoencoder:** Replace NMF with a small sparse model on the same pooled data to get cleaner, fewer‑on features.
8. **Name good features:** Only name those with a clear, repeatable pattern; leave the messy ones unnamed.
9. **Optional sanity check:** For a named feature, see if positions "feel different" when you imagine that feature absent (qualitative is fine).
10. **Stop and use:** Start using the named features to explain moves; only add complexity (spatial detail, more layers) if you later need finer distinctions.

