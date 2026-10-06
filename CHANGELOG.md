# Changelog

## 0.1.1 (6 October 2026)

- The wrong Klein base 4B image recorded as a known issue in 0.1.0 is
  explained and prevented. The engine's arithmetic was never at fault. A Klein
  sidecar built from one checkpoint fits every shape of its sibling, and
  `ZDRAW_KLEIN_ZPACK` applied to any Klein model, so a harness that kept the
  distilled model's pack in the environment rendered the base model with the
  distilled model's block and embedder weights under the base model's own
  modulation weights: a deterministic image, byte-identical on every machine,
  with no warning. A sidecar is now verified against the checkpoint it is
  loaded with and refused when it belongs to another. `kleinpack` records a
  checkpoint identity in every pack it writes (the SHA-256 of the timestep
  embedder, a tensor no LoRA merge touches); packs from before this release
  are checked through the f16 image of their embedder and output projection,
  and an adapted pre-0.1.1 pack is accepted with a notice. `zdraw bench --card`
  records the sidecar path variables as environment overrides.
- Metal correctness, byte-identical on the certified cards and censuses. Ten
  kernels reused a reduction slot right after every thread had read it, with
  no barrier in between, so a lagging simdgroup could read another thread's
  partial sum as the mean or the softmax maximum: the row attention kernel,
  seven VAE normalisation kernels and two Z-Image residual-norm kernels now
  carry the barrier (the race surfaced as a flaky attention test once the row
  kernel's occupancy rose). The row attention kernel takes its score buffer as
  host-sized threadgroup memory instead of a static 6144-entry array, which
  Metal's shader validation layer rejected; weight views bind through their
  page-aligned mapping instead of unaligned no-copy wraps, as Metal documents;
  the placeholder buffer bound for an absent gate covers the float the kernels
  declare; a failed compile of the direct GEMM kernel is reported (the generic
  kernel it falls back to is bit-identical on the certified shapes, but the
  route change was invisible).
- Instruments: `ZDRAW_KLEIN_DUMP_STEPS` also writes the rope table, time
  embedding and modulation vectors at step 0; `ZDRAW_KLEIN_DUMP_BLOCKS` records
  per-block activation hashes of the first two forwards with the latents, the
  mapped sidecar and the embedder weights as bound (docs/env-flags.md).

## 0.1.0 (5 October 2026)

- FLUX.2 Klein base 4B (`flux2-klein-base-4b`, Apache-2.0) in the CLI: fetch,
  default directory, 50-step sampling with classifier-free guidance 4
  (`--guidance` is model-aware; the distilled models accept only 1) and guided
  instruction edits. Certified hash and quality gate recorded. The known
  issue noted at release (one specific wrong image in two of about thirty
  renders) is explained and fixed in 0.1.1.
- Memory, byte-identical: the streamed VAE's norm scratch is sized only by its
  readers (the product tier had wired a 1 GiB buffer at 1024px that nothing
  read); Klein's VAE mid-attention borrows its scratch from the idle resident
  DiT pool; the VAE's first convolution output is leased for one decode from an
  idle denoise buffer. Same protocol as 1 October, 1024×1024: Klein peak
  footprint 4.57 → 3.02 GiB (RSS 3.88 → 2.33), Z-Image 5.94 → 4.72 GiB
  (RSS 5.16 → 3.95); times are reported in docs/resolutions.md.
- Memory reporting: `bench --card` and `ZDRAW_MEMTRACE=1` also report the
  resident pages of the mapped weight files and the total, which RSS and
  footprint leave out.
- Experimental strip decoder behind `ZDRAW_VAE_STRIPMEM` (default off; the
  default render path is unchanged).

### Release candidate (1 October 2026)

Initial engine and CLI source release candidate.

- Z-Image-Turbo and FLUX.2 Klein 4B generation on Apple Silicon.
- Klein instruction editing, image-to-image variations, and masked edits.
- Early validation of Klein resolution alignment/capacity, with runtime, memory
  and visual evidence across tested dimensions.
- Model download and packing, diagnostics, fixed benchmark case, and Stanza sessions.
- Live approximate previews for single-image Klein generation and sessions.
- Apache-2.0 licensing; contributions use the same licence without a separate CLA.

Release checks passed on two M4 Max Macs with 128 GiB RAM. Smaller-memory
Macs, other chip generations and older macOS runtime versions remain
unvalidated. The source tree records details in `docs/validation.md`.
