# Demos

[Generation (20 seconds)](../assets/demo/klein-cli.mp4) ·
[Generation, reroll and save (38 seconds)](../assets/demo/klein-workflow.mp4)

The CLI shows an approximate preview after each denoising step, then the decoded
image. The longer recording rerolls in the same loaded session and saves the result.
Both videos play at **1×**, with no inference cut or accelerated.

## Run the same workflow

After downloading Klein, run from the repository root:

```sh
./zig-out/bin/zdraw session --model flux2-klein-4b \
  --width 1024 --height 1024 --steps 4 --seed 45 --out-dir progressive-demo
```

Enter each line after the previous render finishes:

```text
a red fox sitting in deep snow, golden hour light
reroll
save selected-fox.png
```

Sessions increment the seed before rendering; these images use seeds 46 and 47.
The recording reports 11.80 s and 11.32 s on an M4 Max with 128 GiB RAM, macOS
26.5.2, W16 weights and the 1 October release candidate. Weight pages were cached before the first
render; the reroll also reuses the loaded model and prompt conditioning.
These individual observations are separate from the [benchmark medians](resolutions.md).

[Capture metadata](../assets/demo/capture.json) records timings, hashes and the
terminal recording method. See [terminal display](usage.md#watch-generation-in-the-terminal)
for compatible terminals and preview limits.

## Editing example

| Input | “Add a blue wool scarf around the fox's neck” |
| --- | --- |
| ![Original fox](../assets/demo/editing-before.png) | ![Edited fox wearing a scarf](../assets/demo/editing-after.png) |

```sh
./zig-out/bin/zdraw generate --model flux2-klein-4b \
  --edit assets/demo/editing-before.png \
  --prompt "add a blue wool scarf around the fox's neck" \
  --width 512 --height 512 --steps 4 --seed 46 --out scarf.png
```

Unretouched output from the 1 October release candidate on the same Mac with W16 weights.
Instruction edits can alter other areas; use a [masked edit](usage.md#edit-a-photograph)
for spatial control. This example uses four steps; edits default to two (eight on the base model).
[Metadata](../assets/demo/editing.json) and [render log](../assets/demo/editing-log.txt)
record the command and input/output identities.
