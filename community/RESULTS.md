# Community benchmark cards

Every row is `zdraw bench --card` on someone's Mac: the certified prompt at
1024x1024, 4 steps (50 on the base model), seed 46, one cold process. `hash OK` counts cards whose
output matched the certified hash for that model and profile (a mismatch
usually means a fallback route or an env override - the card says which).
Walls are medians of cold one-shots; warm runs are in the raw cards.
Rendered from `results.jsonl` (0 cards).

| chip | RAM | model | profile | n | wall s (median) | RSS GB (median) | hash OK | routes | zdraw | contributors |
|---|---|---|---|---|---|---|---|---|---|---|
