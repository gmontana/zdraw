# Supported features

The CLI supports Z-Image-Turbo, FLUX.2 Klein 4B and its undistilled base
checkpoint (50-step guided sampling) on Apple Silicon. Internal engine
modules are not a stable SDK.

| Feature | Support |
| --- | --- |
| Text-to-image | All models. |
| Instruction editing | Klein; two steps by default (eight on the base model). [Example](demo.md#editing-example). |
| Image-to-image and masks | Klein; adjustable strength, feathered mask boundaries. |
| Multiple seeds | Klein text-to-image with `--seeds`. |
| Progressive previews | Single-image Klein generation and sessions; approximate latent projections, then the final decoded image. |
| Interactive sessions | Keep models loaded, change prompts/settings, reroll, undo, switch models and save. |
| Terminal images | Kitty/iTerm protocols, with an ANSI colour-cell fallback. |
| Model downloads | Resumable downloads and default W16 packing. |
| Quantisation and LoRA | Developer pack tools; adapters are merged before rendering. |
| Deterministic output | The same request produces the same bytes; `bench` checks the fixed case against `certified/hashes.json`; ten-render censuses are part of release validation. |
| Safety filter | Prompt rules plus an optional image classifier (`zdraw fetch nsfw-classifier`); `--safety off` disables both; a block exits with status 3. |
| Decoder profiles | Z-Image `product` (default) or `strict` (bit-exact reference VAE attention, slow at 1024px). |
| Diagnostics | `doctor` (also `--model`/`--weights` to check one checkpoint), fixed-case `bench` (four steps; 50 on the base model), `inspect`, quality gates and reproducibility tools. |
| PNG metadata | `generate` and `bench` add a `Software` marker ("zdraw (AI-generated image)") and a `zdraw` JSON chunk with model, profile, prompt SHA-256, seed, render dimensions, steps and guidance, plus the init image's hash and strength for `--init-image`; `--edit` inputs are not hashed. Session PNGs currently omit metadata. |

[Usage](usage.md) covers the commands, dimensions and editing limitations.
[Resolution results](resolutions.md) show tested sizes, runtime, memory and images.
[Validation](validation.md) records the tested hardware and configurations.

Z-Image and multi-seed batches display only final images. Klein previews need
an interactive terminal with enough space; colours and detail can differ from
the final image. The experimental XRES route skips them.

This release does not include a desktop app, Swift SDK, draft-then-final workflow,
outpainting UI, clipboard integration or recipe links. Experimental model variants
and noise interpolation are outside the supported CLI.
