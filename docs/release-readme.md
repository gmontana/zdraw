# zdraw

zdraw generates and edits images locally on Apple Silicon using Zig and Metal.
This archive contains a compiled CLI for macOS 14 or later. Keep `zdraw` and
its `lib/` directory together; no compiler is needed to run it.

## Check the installation

From the extracted directory:

```sh
./zdraw version
./zdraw doctor
./zdraw preview --prompt "a red boat" --out preview.png
```

The preview checks image output without weights. It does not run a model.

## Generate an image

Klein 4B requires approximately 24 GB of disk space for its checkpoint and
weight pack. Download it once, then generate an image:

```sh
./zdraw fetch flux2-klein-4b
./zdraw generate --model flux2-klein-4b \
  --prompt "a red fox in deep snow" --out fox.png
```

Open `fox.png` to inspect the result. For a fixed test case and benchmark card:

```sh
./zdraw bench --model flux2-klein-4b
```

`flux2-klein-base-4b` (the undistilled checkpoint, 50 guided steps) and
`z-image-turbo` (about 49 GB) are also available; without `--model`, commands
use `z-image-turbo`.

Full-image measurements currently use an M4 Max with 128 GiB RAM; other chips
and older macOS versions are not yet validated;
smaller-memory Macs are not yet validated.
See [SUPPORT.md](SUPPORT.md) for reporting problems, and include the output
of `./zdraw version` and `./zdraw doctor --json`.

## Licence

[Apache-2.0](LICENSE), including commercial and proprietary use.
See [commercial use](COMMERCIAL.md) for details.
[NOTICE](NOTICE) lists third-party code; its licence texts are under `vendor/`
and `licenses/stanza/`. Downloaded model weights retain their upstream terms.
