# Using zdraw

## Build and check the installation

Install Zig 0.16.0 and Xcode with its Metal compiler, then complete Xcode's
first-run setup. Command Line Tools alone may not include the Metal compiler.

```sh
xcrun metal --version
zig build
./zig-out/bin/zdraw version
./zig-out/bin/zdraw doctor
```

The build defaults to ReleaseFast. Keep `zig-out/lib/steel.metallib` beside
`zig-out/bin/`; copying only the binary omits a required library. The examples
below run from the repository root.

`zdraw preview --prompt "a red boat" --out preview.png` checks image output
without model weights; it does not run diffusion.

## Download a model

```sh
./zig-out/bin/zdraw fetch flux2-klein-4b
```

| Model | Default directory | Checkpoint + W16 pack |
| --- | --- | ---: |
| `flux2-klein-4b` | `~/.zdraw/models/FLUX.2-klein-4B` | About 24 GB |
| `flux2-klein-base-4b` | `~/.zdraw/models/FLUX.2-klein-base-4B` | About 24 GB |
| `z-image-turbo` | `~/.zdraw/models/Z-Image-Turbo` | About 49 GB |

Downloads resume when rerun. Use `--dir PATH` for another destination and pass
that path to generation with `--weights PATH`; `--no-pack` downloads without
building the W16 pack (rerun without it to pack). `zdraw inspect FILE.safetensors`
lists a checkpoint's tensors, and `zdraw help COMMAND` or `COMMAND --help` prints
each command's options. `HF_TOKEN` supports gated
repositories. Weights retain the upstream terms listed in [NOTICE](../NOTICE).

## Generate an image

```sh
./zig-out/bin/zdraw generate --model flux2-klein-4b \
  --prompt "a red fox in deep snow" --seed 7 --show --out fox.png
```

Use `--model z-image-turbo` for Z-Image; without `--model`, every command
defaults to `z-image-turbo`. Images default to 1024×1024 and seed 42 (sessions:
512×512). Model paths, steps and guidance default automatically: the distilled models sample in four steps at guidance 1;
`flux2-klein-base-4b`, the undistilled checkpoint, samples in 50 steps with
classifier-free guidance 4 (set `--guidance` to change it, or `--guidance 1`
to disable it) and takes about 25 times longer per image. Klein text-to-image
also accepts `--seeds 7,11,23` in place of `--seed` (distinct values; not with
edits); filenames receive a `-s<seed>` suffix. `--repeat N` renders N times with
the weights loaded once and prints each run's time.

### Dimensions

- **Klein:** each side must be divisible by 32, area divisible by 8192 and
  area no larger than 1,048,576 pixels. Square presets 128, 256, 512, 768 and
  1024 work, as does 1024×768. Shapes such as 64×64 and 544×800 are rejected.
- **Z-Image:** each side must be divisible by 16; tested through 2048×2048.

The [resolution report](resolutions.md) includes runtime, memory and sample
images. Tiny outputs have limited detail.

## Watch generation in the terminal

`--show` and interactive sessions display an approximate preview after each
Klein denoising step, then the decoded image. Kitty/iTerm image protocols are
detected automatically; other terminals use ANSI colour cells.

Use `--no-progressive` for final-image display only. `generate` shows images
only with `--show`; in sessions, `--no-show` disables them. Progressive display needs sufficiently large interactive terminals and
is skipped for redirected streams. Z-Image and multi-seed batches show final
images only.

## Edit a photograph

Editing is supported by the Klein models, with one reference image. Give
`--edit` an image and describe the change:

```sh
./zig-out/bin/zdraw generate --model flux2-klein-4b \
  --edit photo.png --prompt "add a red scarf" --out edited.png
```

Edits default to two steps (eight on `flux2-klein-base-4b`). They can change
other parts of the image; [the demo](demo.md#editing-example) shows a four-step
example. Edited and masked outputs are written at the input photo's own
resolution, and unchanged areas keep the original pixels.

For image-to-image variation, use `--init-image photo.png --strength 0.3` instead
of `--edit`. Strength defaults to 0.6 and ranges from zero (VAE round trip only, which can still
change pixels) to one (start from noise).

Add a mask for spatial control. White selects the area to redraw; black
preserves the image outside the feathered boundary. For 512×512 inputs:

```sh
./zig-out/bin/zdraw generate --model flux2-klein-4b \
  --init-image photo.png --mask mask.png --strength 0.75 \
  --prompt "a fox wearing a red scarf in the snow" \
  --width 512 --height 512 --steps 4 --seed 46 --out masked.png
```

Use a mask matching the input's dimensions and describe the desired final
scene. `--edit` and `--init-image` are separate modes and cannot be combined.

## Interactive session

```sh
./zig-out/bin/zdraw session --model flux2-klein-4b --out-dir images
```

Type a description to generate; later lines add to the prompt.

| Command | Effect |
| --- | --- |
| `new TEXT` | Start a new prompt. |
| `reroll` | Generate with another seed. |
| `undo` | Restore the previous prompt and generate again. |
| `save copy.png` | Copy the last result to a named file. |
| `seed 42`, `steps 4`, `size 512x512` | Change generation settings. |
| `model z-image-turbo` | Switch to another downloaded model. |
| `prompt`, `clear` | Show or clear the current prompt. |
| `stats on`, `stats off` | Detailed memory statistics after each render. |
| `help`, `quit` (or `exit`) | List commands or exit. |

[Stanza](https://github.com/gmontana/stanza) provides history and completion.
Ctrl-C clears the line; Ctrl-D exits. Automatic saves use numbered filenames
and skip existing files. `--no-auto-save` keeps results in memory until `save`.
`seed N` sets the base seed and each render adds its image index. Sessions do
not edit images and have no `guidance` command; pass `--guidance` at launch.
`--preview` runs a session without weights to check the terminal display.

## LoRA adapters and weight packs

Adapters are merged when packing; changing their strength requires a new pack.

```sh
zig build kleinpack -- --weights "$HOME/.zdraw/models/FLUX.2-klein-4B" \
  --size 4b --lora adapter.safetensors --lora-scale 0.8 --out adapted.zpack
ZDRAW_KLEIN_ZPACK=adapted.zpack ./zig-out/bin/zdraw generate \
  --model flux2-klein-4b --prompt "a red fox in deep snow" --out adapted.png
```

Pack-tool help and the [environment inventory](env-flags.md) cover quantisation
and other options, which can affect image quality. Klein W6 packs need
`ZDRAW_KLEIN_ACT=f32`.

## Troubleshooting

- **Metal compiler:** check `xcode-select -p` and `xcrun metal --version`.
  If Xcode reports a missing toolchain, run
  `xcodebuild -downloadComponent MetalToolchain`. Review an unaccepted licence
  with `sudo xcodebuild -license`.
- **Incomplete weights:** rerun `fetch`. A packed transformer alone is not a
  complete model.
- **Unexpected speed or output:** check `doctor --json`, remove experimental
  `ZDRAW_*` overrides and report the version, pack and command. `zdraw bench`
  renders the fixed fox prompt at 1024px, seed 46, four steps (50 on the base
  model) to `zdraw-bench-<model>.png` (`--out` overrides) and checks it against
  its certified hash; the card's last stdout line is JSON (`schema_version` 1).
  A mismatch needs investigation.
- **Z-Image strict profile:** at 1024px its reference VAE attention uses the CPU
  and can take minutes. Use the default `product` profile for normal generation.
- **Memory pressure:** published process-memory figures are not minimum RAM
  requirements. Only 128 GiB Macs have been tested.
- **Safety filter:** prompt filtering is enabled by default. Image classification
  needs `zdraw fetch nsfw-classifier` (about 350 MB, installed under
  `~/.zdraw/models/nsfw_image_detection`; `ZDRAW_SAFETY_MODEL` points elsewhere);
  without it that stage is skipped. `--safety off` disables both stages for
  `generate`, `session` and `bench`. A blocked prompt or image exits with status 3.
  Neither filter is complete.
- **Blank output:** a blank or low-content image is rejected, no file is written
  and the exit status is 2.

Useful environment variables, all optional: `ZDRAW_PROGRESS=quiet` or
`compact` changes the progress output; `ZDRAW_PREVIEW_DIR=DIR` writes a preview
PNG after every step (verbose progress only); `ZDRAW_MEMTRACE=1` prints the
process footprint, the resident mapped-weight pages and their total per stage;
`ZDRAW_METRICS=1` prints Metal dispatch counters; `ZDRAW_KLEIN4B_WEIGHTS`,
`ZDRAW_KLEIN_BASE4B_WEIGHTS` and `ZDRAW_ZIMAGE_WEIGHTS` name the weights the
session `model` command switches to; `NO_COLOR` disables colour. The full
[inventory](env-flags.md) lists every flag.

Generation is serialised because the runtime shares state. Developer checks
are in [testing](testing.md); report issues through [support](../SUPPORT.md).
