# Changelog

## 0.1.0 (5 October 2026)

- FLUX.2 Klein base 4B (`flux2-klein-base-4b`, Apache-2.0) in the CLI: fetch,
  default directory, 50-step sampling with classifier-free guidance 4
  (`--guidance` is model-aware; the distilled models accept only 1) and guided
  instruction edits. Certified hash and quality gate recorded.
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
