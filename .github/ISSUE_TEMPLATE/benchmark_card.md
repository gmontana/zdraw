---
name: Benchmark card
about: Add your Mac's `zdraw bench --card` result to the community table
title: "card: <chip>, <RAM> GB, <model>"
labels: benchmark-card
---

Thanks - every card makes the numbers honest on one more machine.

1. Quiet box: no other GPU or heavy CPU work running (six load averages under
   2.0, `tools/perf_when_quiet.sh` if you have the repo).
2. Run, then paste the **last line** of the output (the JSON) here:

   ```sh
   zdraw bench --card --model flux2-klein-4b --weights /path/to/FLUX.2-klein-4B --repeat 3
   ```

   ```json
   PASTE THE JSON LINE HERE
   ```

3. Paste `zdraw doctor --json` too (it says which routes ran):

   ```json
   PASTE DOCTOR JSON HERE
   ```

4. Your GitHub handle for the credit column (optional): @

A maintainer appends the card to `community/results.jsonl` with your handle
and re-renders `community/RESULTS.md`. Cards with `hash_match: false` are
welcome - they usually reveal a fallback route or an override, and the card
says which.
