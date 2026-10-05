//! Command help. Keep examples runnable without developer environment overrides.
const args = @import("args.zig");

pub fn text(topic: args.Help) []const u8 {
    return switch (topic) {
        .overview => overview,
        .generate => generate ++ sampling ++
            "Example: zdraw generate --model flux2-klein-4b --prompt \"a red fox\" --out fox.png\n",
        .session => session ++ sampling ++
            "Example: zdraw session --model flux2-klein-4b --out-dir images\n",
        .fetch => fetch,
        .bench => bench,
        .doctor => doctor,
        .preview => preview,
        .inspect => "Usage: zdraw inspect FILE.safetensors\n\n" ++
            "Print tensor names, types, and shapes without loading a model.\n",
        .version => "Usage: zdraw version\n\n" ++
            "Print the engine version, revision, ABI, and Metal availability.\n",
    };
}

const overview =
    \\zdraw — local image generation and editing on Apple Silicon
    \\
    \\Usage: zdraw COMMAND [OPTIONS]
    \\
    \\  fetch     Download a model and prepare its weight pack
    \\  generate  Generate or edit an image
    \\  session   Open an interactive prompt with history and completion
    \\  bench     Run a fixed case and report its reference hash
    \\  doctor    Check the installation and Metal routes
    \\  preview   Check image output without running a model
    \\  inspect   List tensors in a safetensors file
    \\  version   Show the installed version
    \\
    \\Get started:
    \\  zdraw fetch flux2-klein-4b
    \\  zdraw generate --model flux2-klein-4b --prompt "a red fox" --out fox.png
    \\  zdraw bench --card --model flux2-klein-4b
    \\
    \\Use zdraw COMMAND --help or zdraw help COMMAND for options and examples.
    \\
;

const sampling =
    \\Model settings:
    \\  --model NAME       z-image-turbo (default), flux2-klein-4b or flux2-klein-base-4b
    \\  --weights DIR      Override the model's ~/.zdraw/models directory
    \\  --width N          Width: multiple of 16 for Z-Image, 32 for Klein
    \\  --height N         Height: same divisibility as width
    \\  --steps N          Positive step count; default 4 (2 for Klein edits);
    \\                     flux2-klein-base-4b: 50 (8 for edits)
    \\  --seed N           Nonnegative random seed; default 42
    \\  --guidance N       flux2-klein-base-4b: classifier-free guidance, default 4;
    \\                     the distilled models require 1
    \\  --profile NAME     Z-Image only: product (default) or strict
    \\  --safety on|off    Prompt and available image filtering; default on
    \\  --show             Show the image in a compatible terminal
    \\  --no-progressive   Show only the final image (Klein previews default on)
    \\
    \\Klein image area must be divisible by 8192 and at most 1,048,576 pixels.
    \\Square sizes include 128, 256, 512, 768 and 1024; 64x64 is unsupported.
    \\
;

const generate =
    \\Usage: zdraw generate --prompt TEXT --out FILE.png [OPTIONS]
    \\
    \\Generate a PNG. Dimensions default to 1024x1024.
    \\  --prompt TEXT      Description or editing instruction (required)
    \\  --out FILE.png     Output file (required)
    \\  --seeds 7,11,23    Klein text-to-image: outputs named FILE-s<seed>.png
    \\                     Use distinct values instead of --seed
    \\  --repeat N         Repeat with resident weights; default 1, save final result
    \\
    \\  --vae-reference    Z-Image diagnostic: whole-image f32 reference decode
    \\
    \\Klein editing (choose one mode):
    \\  --edit FILE        Instruction edit, e.g. --prompt "add a blue scarf"
    \\  --init-image FILE  Image-to-image variation
    \\  --strength N       With --init-image: noise fraction in [0,1]; default 0.6
    \\  --mask FILE        With --init-image: white selects the region to redraw
    \\
;

const session =
    \\Usage: zdraw session --out-dir DIR [OPTIONS]
    \\
    \\Interactive prompt editing with Stanza. Dimensions default to 512x512.
    \\  --out-dir DIR      Save numbered images in this directory
    \\  --no-auto-save     Keep images in memory until 'save FILE'; DIR optional
    \\  --no-show          Disable terminal image display (enabled by default)
    \\  --preview          Exercise the session without model weights
    \\
    \\Type help inside the session for prompt, settings, saving, and exit commands.
    \\
;

const fetch =
    \\Usage: zdraw fetch MODEL [--dir DIR] [--no-pack]
    \\
    \\Download resumable checkpoints and build the weight pack, without Python.
    \\  flux2-klein-4b    About 24 GB including the pack
    \\  flux2-klein-base-4b  About 24 GB including the pack; 50-step guided sampling
    \\  z-image-turbo     About 49 GB including the pack
    \\  nsfw-classifier  Optional output-image classifier
    \\  --dir DIR        Override ~/.zdraw/models/<model-directory>
    \\  --no-pack        Download only; rendering still needs a prepared pack
    \\
    \\Rerun the same command after an interrupted download. HF_TOKEN is supported.
    \\Example: zdraw fetch flux2-klein-4b
    \\
;

const bench =
    \\Usage: zdraw bench [--card] [--model MODEL] [--weights DIR] [OPTIONS]
    \\
    \\Generate a fixed 1024px image and print timing, routes, and a hash comparison.
    \\  --model NAME       z-image-turbo (default), flux2-klein-4b or flux2-klein-base-4b
    \\  --weights DIR      Override the model's default download directory
    \\  --card             Compatibility flag; a benchmark card is always printed
    \\  --repeat N         Number of resident runs; default 1
    \\  --out FILE.png     Override zdraw-bench-<model>.png
    \\  --profile NAME     Z-Image only: product (default) or strict
    \\  --safety on|off    Prompt and available image filtering; default on
    \\
    \\MATCH confirms this case matches the recorded reference, not every image.
    \\MISMATCH needs investigation; an absent reference is not a successful match.
    \\Inspect the PNG and keep the card when reporting a problem.
    \\Example: zdraw bench --card --model flux2-klein-4b
    \\
;

const doctor =
    \\Usage: zdraw doctor [--json] [--model MODEL --weights DIR]
    \\
    \\Report hardware, Metal library availability, routes, and environment overrides.
    \\  --json           Emit machine-readable diagnostics
    \\  --model NAME     Model to inspect; default z-image-turbo
    \\  --weights DIR    Also inspect this checkpoint directory
    \\
    \\Doctor does not generate an image. Use generate or bench to test inference.
    \\
;

const preview =
    \\Usage: zdraw preview --prompt TEXT --out FILE.png [OPTIONS]
    \\
    \\Write a synthetic image to check installation and PNG output. No model runs.
    \\  --width N --height N  Positive dimensions; default 1024x1024
    \\  --seed N             Nonnegative random seed; default 42
    \\  --show               Show the image in a compatible terminal
    \\
    \\Example: zdraw preview --prompt "a red boat" --out preview.png
    \\
;
