# Security

Image generation runs locally. `zdraw fetch` uses HTTPS to download model
files and can use an `HF_TOKEN` supplied through the environment. Inputs include safetensors checkpoints, LoRA
adapters, `.zpack` sidecars, tokenizer files, and prompts. A malformed model
file must fail loudly, never execute or read outside the file.

## Reporting

Report a vulnerability privately through GitHub's "Report a vulnerability"
(Security tab) on this repository. Do not open a public issue for a bug that
lets a crafted file crash, read memory, or write outside the output path.

Include the file (or how to construct it), the command, the zdraw commit, and
the macOS and hardware versions. You will get an acknowledgement within a
week and a fix or a stated plan within a month for confirmed reports.

## Scope

In scope: parsing of input files, model downloading, the CLI and interactive
session, and environment-variable handling.

Out of scope: the models themselves (their licences and output policies are
upstream's), and denial of service by asking for an image too large for the
machine.
