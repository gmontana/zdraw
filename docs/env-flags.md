# ZDRAW_* environment flag inventory

Status: authoritative inventory of the environment override surface.

Typed profiles in `src/runtime/runtime_options.zig` and `src/vae/vae_mode.zig` own shipping
configuration. Environment variables are explicit deployment inputs,
development selectors, or laboratory overrides; absence must never select a
hidden product policy.

Classifications:

- **default-config**: a typed profile or promoted default owns the value;
- **deployment**: supplied by a machine, invocation, or evidence harness;
- **instrument**: diagnostic output or capture, default off;
- **bench-selector**: selects a development benchmark subset;
- **quarantine**: a fenced unsafe, vendor, or reference route;
- **candidate**: a live experiment awaiting a promotion or kill ruling;
- **killed**: retained only until the concrete removal dependency recorded in
  its disposition lands;
- **retired**: a deliberately documented historical name absent from code.

Every row names one complete flag. Grouped suffix notation is forbidden because
it cannot be checked mechanically. `tools/style/env_guard.py` compares this
table with tracked root Zig files and tracked files under `src/` and `tools/`.

| Flag | Canonical owner | Classification | Default source | Disposition |
|---|---|---|---|---|
| ZDRAW_ACT | `zrun.zig` | candidate | unset | Z-Image activation-format experiment |
| ZDRAW_ATTN | `mattn.zig` | default-config | attention dispatcher | owned attention kernel selection (rows/flash/block); the `sdpa` value no longer selects a framework route |
| ZDRAW_ATTNB | `gemmbench.zig` | bench-selector | unset | attention-scale benchmark |
| ZDRAW_ATTN_MFA | `metal_api.m` | quarantine | unset | vendored MFA reference route |
| ZDRAW_ATTN_STEEL | `metal_api.m` | default-config | `1` | Z-Image chain attention on the vendored MLX steel kernel for every head_dim-128 shape (MFA, then the owned kernels, as fallbacks); `0` restores the previous order. The MPSGraph SDPA fallthrough for shapes below 2048 tokens is gone |
| ZDRAW_ATTN_WIDE | `mattn.zig` | candidate | unset | owned wide-shape kernels (chunked MMA, flash_wide): byte-equal to the graph route, 2.9x slower on validate (attn-wide-mma-chunked, 2026-08-25); opt-in |
| ZDRAW_BIN | `tools/ballast_sweep.sh` | deployment | `./zig-out/bin/zdraw` | executable the sweep renders with |
| ZDRAW_CLEAR_WEIGHT_CACHE_EACH_STEP | `zdenoise.zig` | instrument | unset | cache-attribution probe |
| ZDRAW_COMPILE_TIMES | `metal_api.m` | instrument | unset | pipeline compilation timing |
| ZDRAW_CONVH | `gemmbench.zig` | bench-selector | unset | convolution-H benchmark |
| ZDRAW_CONVH8 | `gemmbench.zig` | bench-selector | unset | h4-vs-h8 conv kernel race (W7) |
| ZDRAW_GEMMBENCH_GATES | `gemmbench.zig` | deployment | none | run only the correctness gates (GEMM exact/half, W6, VAE norm split, conv windows, fragment map) for small-memory runners such as the 7 GiB GitHub macOS VM |
| ZDRAW_REF_CACHE | `zflux2_run.zig` | default-config | on | reference-image encode memo for instruction edits (path, size, mtime, render size) |
| ZDRAW_SCRATCH_POISON | `metal_api.m` | instrument | unset | fill the private attention scratch buffers with 0xFF (f16 NaN) at allocation instead of zero: `1` every slot, `0`..`4` one slot; exposes a read that precedes its write |
| ZDRAW_VAE_BUDGET_H | `gemmbench.zig` | bench-selector | unset | product f16 decoder per-kernel budget at the 1024 ladder |
| ZDRAW_WINO | `gemmbench.zig` | bench-selector | unset | Winograd F(4x4,3x3) vs direct h8 conv race |
| ZDRAW_MPP | `gemmbench.zig` | bench-selector | unset | Metal 4 tensor-path GEMM vs the production GEMM at the Klein shapes |
| ZDRAW_CONVH_ONLY | `gemmbench.zig` | bench-selector | unset | convolution-H-only subset |
| ZDRAW_COPY_WEIGHTS | `metal_api.m` | candidate | unset | copied-versus-mapped weight experiment |
| ZDRAW_DENSE | `runtime_options.zig` | default-config | typed product profile | dense GEMM backend; `ours-v2` is the opt-in direct-W variant (wash in-chain, ledger gemm-v2-inchain-final) |
| ZDRAW_DISCOVERY_TRACE | `ztrace_capture.zig` | instrument | unset | discovery trace capture |
| ZDRAW_DUMP_DIR | `gemmbench.zig` | instrument | unset | tensor-dump directory |
| ZDRAW_DUMP_PRED | `zdenoise.zig` | instrument | unset | prediction dump |
| ZDRAW_DUMP_RGBF | `vrgb.zig` | instrument | unset | float RGB dump |
| ZDRAW_DUMP_UP | `gemmbench.zig` | instrument | unset | upsample dump |
| ZDRAW_SAFETY_MODEL | `safety.zig` | deployment | `~/.zdraw/models/nsfw_image_detection` | directory of the image-stage safety classifier (`zdraw fetch nsfw-classifier`) |
| ZDRAW_ENGINE_REVISION | execution tools | deployment | evidence harness | overrides the built-in version+commit (`zdraw version`) reported by `doctor` and bench cards |
| ZDRAW_EXACT_STAGED | `metal_api.m` | candidate | unset | exact staged-GEMM experiment |
| ZDRAW_FFN_RESID_F16_FUSE | `metal_api.m` | candidate | unset | fused FFN residual experiment |
| ZDRAW_FLUX2 | `gemmbench.zig` | bench-selector | unset | Klein benchmark family |
| ZDRAW_FLUX2_9B | `gemmbench.zig` | bench-selector | unset | Klein 9B shapes |
| ZDRAW_FLUX2_DIT | `gemmbench.zig` | bench-selector | unset | Klein transformer oracle |
| ZDRAW_FLUX2_EMB | `gemmbench.zig` | bench-selector | unset | Klein embedding oracle |
| ZDRAW_FLUX2_EMB_FULL | `gemmbench.zig` | bench-selector | unset | full embedding oracle |
| ZDRAW_FLUX2_EMB_ONLY | `gemmbench.zig` | bench-selector | unset | embedding-only oracle |
| ZDRAW_FLUX2_LOOP | `gemmbench.zig` | bench-selector | unset | Klein loop oracle |
| ZDRAW_FLUX2_RES | `gemmbench.zig` | bench-selector | unset | Klein resident oracle |
| ZDRAW_FLUX2_SWAP | `gemmbench.zig` | bench-selector | unset | Klein swap oracle |
| ZDRAW_FLUX2_VAE | `gemmbench.zig` | bench-selector | unset | Klein VAE oracle |
| ZDRAW_GEMM | `gemm_mode.zig` | default-config | GEMM dispatcher | general GEMM backend |
| ZDRAW_GEMM_BF16 | `metal_api.m`, `mgemm.zig` | default-config | enabled | routes bf16 checkpoints (the Klein text encoder) to gemm_f16_direct via a bf16 staging variant instead of the slower gemm_half; byte-identical, `0` opts out |
| ZDRAW_GPU_TRACE | `metal_api.m` | instrument | unset | Metal GPU capture |
| ZDRAW_HALFD1 | `gemmbench.zig` | bench-selector | unset | half-d1 benchmark |
| ZDRAW_KLEIN_ACT | `zflux2_resident.zig` | default-config | f16 | half activations for Klein, single AND batched `--seeds` (byte-identical, -11% GPU busy); `f32` opts out; with a W6 pack the f16-A GEMM fails loudly (InvalidDType), including via `--seeds`: use f32 for W6 packs |
| ZDRAW_KLEIN_ALLOW_UNPACKED | `zflux2_pack.zig` | deployment | disabled | permits an unpacked Klein run |
| ZDRAW_KLEIN_ATTN_VARIANT | `metal_api.m` | bench-selector | unset | steel attention tiling variant for the bench: `bk32` (default instantiation otherwise) |
| ZDRAW_KLEIN_ATTN_STEEL | `zflux2_resident.zig` | default-config | `1` | Klein attention on the MFA route runs the vendored MLX steel attention kernel (lib/steel.metallib beside the binary or ZDRAW_STEEL_LIB; MFA fallback if absent); `0` restores MFA. Gated 2026-08-26 (census, klein_gate 8/8, content) |
| ZDRAW_KLEIN_QK_HM | `zflux2_resident.zig` | default-config | `1` | Klein MFA route: q/k emitted head-major half by the norm+rope kernel (skips 2 of 4 layout converts per attention); `0` restores the convert path for A/B |
| ZDRAW_KLEIN_ATTN_MFA | `zflux2_resident.zig` | default-config | enabled | vendored MFA attention above the token threshold; `0` falls back to block16 |
| ZDRAW_KLEIN_CONCURRENCY | `gemmbench.zig` | candidate | unset | batched multi-seed experiment |
| ZDRAW_KLEIN_CUSTOM_GEMM | `metal_api.m` | quarantine | unset | custom-GEMM research route |
| ZDRAW_KLEIN_DUMP_STEPS | `zflux2_run.zig` | instrument | unset | per-step drift dumps |
| ZDRAW_KLEIN_GEMM64 | `zflux2_resident.zig` | default-config | resident default | GEMM64 route |
| ZDRAW_KLEIN_MODS_GPU | `zflux2_resident.zig` | default-config | resident default (`1`) | the per-step adaLN modulation matvecs run on the GPU (kmodvec, CPU-exact arithmetic, byte-identical output); `0` = the CPU per-timestep cache (A/B arm) |
| ZDRAW_KLEIN_GEMM_MPP | `zflux2_resident.zig` | default-config | resident default (`1`) | Klein f16-A GEMMs on the Metal 4 `matmul2d` tensor path (macOS 26+; byte-identical to the direct kernel); `0` = the direct simdgroup_matrix kernel (A/B arm; also the counted, WARNING-announced fallback below Metal 4) |
| ZDRAW_KLEIN_ATTNBENCH | `gemmbench.zig` | bench-selector | unset | `1` runs the Klein attention headroom bench (MFA cuts + steel attention vs the MLX SDPA race) instead of the GEMM benches |
| ZDRAW_KLEIN_GEMMBENCH | `gemmbench.zig` | bench-selector | unset | Klein GEMM subset |
| ZDRAW_KLEIN_MFA_MIN_TOKENS | `zflux2_resident.zig` | default-config | 2048 | token count above which MFA takes the attention |
| ZDRAW_KLEIN_RES | `zflux2_run.zig` | default-config | resident default | resident execution route |
| ZDRAW_KLEIN_TE_PROJ | `zflux2_run.zig` | instrument | unset | forces the encoder projections onto the GEMM (`1`) or rows (`0`) route independently of TE_BATCH, to bisect the divergence |
| ZDRAW_QWEN_DUMP | `qwen_encoder.zig` | instrument | unset | writes layer-0 q/k/v and post-layer state for route diffing |
| ZDRAW_KLEIN_TE_BATCH | `zflux2_run.zig` | default-config | `1` | simdgroup-GEMM encoder route (per-op oracle path only) |
| ZDRAW_KLEIN_TE_RES | `zflux2_run.zig` | default-config | on | GPU-resident Qwen3 stacked encode: one batched command buffer per encode (readbacks 175 to 14, cos 1.000000 vs per-op); `0` restores the per-op oracle route, which TE_BATCH/TE_PROJ govern |
| ZDRAW_KLEIN_TE_DUMP | `zflux2_run.zig` | instrument | unset | text-encoder A/B dump |
| ZDRAW_KLEIN_TRACE | `zflux2_resident.zig` | instrument | unset | denoising attribution |
| ZDRAW_KLEIN_TRACE_SINGLE | `zflux2_resident.zig` | instrument | unset | single-pass attribution |
| ZDRAW_KLEIN_W6_ONLY | `gemmbench.zig` | bench-selector | unset | Klein W6 subset |
| ZDRAW_GEMM_M | `gemmbench.zig` | bench-selector | unset | rescales the Klein shapes' M (512/1024 for the edit-sized, bandwidth-bound rows of the kernel table); with ZDRAW_KLEIN_W6_ONLY |
| ZDRAW_KLEIN_XRES | `zflux2_run.zig` | candidate | unset | resident Euler experiment |
| ZDRAW_KLEIN_ZPACK | `zflux2_run.zig` | deployment | invocation | Klein sidecar path |
| ZDRAW_KLEIN_TEXT_ZPACK | `zflux2_run.zig` | deployment | invocation | Klein text pack path (4-bit encoder) |
| ZDRAW_LATENT_IN | `zdenoise.zig`, `zflux2_run.zig` | instrument | unset | shared-latent parity input (raw f32 x0 in the engine layout; single-seed renders, both models) |
| ZDRAW_MATH | `metal_api.m` | default-config | math dispatcher | safe-versus-fast math |
| ZDRAW_MEMTRACE | `metrics.zig` | instrument | unset | per-stage footprint, resident mapped-weight pages (mapped_res) and their total |
| ZDRAW_METRICS | `model_runtime.zig` | instrument | unset | Metal dispatch, command-buffer and readback counters |
| ZDRAW_MFA_METAL | `metal_api.m` | quarantine | unset | MFA Metal research route |
| ZDRAW_MFA_MIN_TOKENS | `metal_api.m` | quarantine | unset | MFA research threshold |
| ZDRAW_MFA_MULTIHEAD | `metal_api.m` | quarantine | unset | MFA multihead experiment |
| ZDRAW_NO_LOCK | `genlock.zig` | deployment | unset | opts a lab run out of the per-user generation lock (concurrent renders corrupt; ledger concurrent-render-corruption) |
| ZDRAW_MIX_PROBE | `metal_api.m` | instrument | unset | mixed-precision diagnostic |
| ZDRAW_MODEL_REVISION | `ztrace_capture.zig` | deployment | evidence harness | binds traced model revision |
| ZDRAW_MPS_F16A | `metal_api.m` | candidate | unset | MPS f16-A reference experiment |
| ZDRAW_MPS_F16C | `metal_api.m` | candidate | unset | MPS f16-C reference experiment |
| ZDRAW_NORM_F16A_FUSE | `metal_api.m` | candidate | unset | fused normalization experiment |
| ZDRAW_PREVIEW_DIR | `preview.zig` | deployment | unset | directory for the per-step preview PNGs (the app's progressive display); announced as `zdraw: preview <path> step k/n` in verbose mode |
| ZDRAW_PROGRESS | `progress.zig` | deployment | invocation | progress rendering policy: `quiet`/`off`, `compact`/`bar`, otherwise verbose |
| ZDRAW_QK_HM | `runtime_options.zig` | default-config | typed product profile | head-major Q/K writes |
| ZDRAW_RELEASE_TEXT_WEIGHTS | `zsample.zig` | candidate | unset | text-weight lifetime experiment |
| ZDRAW_REPLAY_DIR | `gemmbench.zig` | instrument | unset | convolution replay source |
| ZDRAW_REPLAY_UP | `gemmbench.zig` | instrument | unset | upsample replay selector |
| ZDRAW_REQUIRE_ZPACK | packed-weight loaders | deployment | invocation | disallows raw-weight fallback |
| ZDRAW_SHIFT | `scheduler.zig` | deployment | request override | sampler shift |
| ZDRAW_STACK_EXACT_ATTN_BELOW | `mstack_policy.zig` | candidate | unset | q/k/v/proj exact in layers below the bound (Q2 pin search axis) |
| ZDRAW_STACK_EXACT_DOWN_BELOW | `mstack_policy.zig` | candidate | unset | ffn_down exact in layers below the bound (Q2 pin search axis) |
| ZDRAW_STACK_EXACT_GATEUP_BELOW | `mstack_policy.zig` | candidate | unset | ffn_gate/up exact in layers below the bound (Q2 pin search axis) |
| ZDRAW_STACK_EXACT_OPS | `mstack_policy.zig` | candidate | unset | exact-operation selection experiment |
| ZDRAW_STACK_GEMM | `runtime_options.zig` | default-config | typed product profile | `exact` pins every stack matrix to W32; unset or unknown values inherit the resolved `ZDRAW_GEMM` base mode |
| ZDRAW_STACK_HALF_FROM | `mstack_policy.zig` | candidate | unset | first selected layer |
| ZDRAW_STACK_HALF_LAST | `mstack_policy.zig` | candidate | unset | selected tail-layer count |
| ZDRAW_STACK_HALF_TO | `mstack_policy.zig` | candidate | unset | exclusive selected-layer bound |
| ZDRAW_STACK_LAYER_LIMIT | `zstep_res.zig` | candidate | unset | reduced-stack experiment |
| ZDRAW_ACT_GEMM | `metal_api.m` | instrument | unset | `v2` pins the f16 activation tier to the flat-loop GEMMs (cold-weight latency A/B) |
| ZDRAW_STACK_ACT | `metal_api.m` | candidate | unset | `f16` runs the chain with f16-resident activations (half state/norm/gate, half-A direct GEMMs); default f32 regime is byte-identical by construction |
| ZDRAW_STACK_MEASURED_FFN_FROM | `mstack_policy.zig` | candidate | unset | measured-policy FFN boundary |
| ZDRAW_STACK_PROFILE | `metal_api.m` | candidate | unset | native stack profile experiment |
| ZDRAW_STACK_W16 | `runtime_options.zig` | default-config | typed product profile (`1` since the 2026-08-04 recertification) | W16 sidecar route, byte-identical to the generic tier |
| ZDRAW_STACK_W6_KINDS | `mstack_policy.zig` | default-config | typed product profile | W6 matrix-kind set |
| ZDRAW_STACK_W8_KINDS | `mstack_policy.zig` | candidate | unset | W8 matrix-kind set |
| ZDRAW_STACK_W8_LAST | `mstack_policy.zig` | candidate | unset | W8 tail-layer count |
| ZDRAW_STEEL | `metal_api.m` | quarantine | disabled | fenced Steel f16 route |
| ZDRAW_STEEL_LIB | `metal_api.m` | deployment | invocation | metallib path |
| ZDRAW_STEEL_W6 | `metal_api.m` | default-config | typed product profile | Steel W6 dequant route |
| ZDRAW_SWIGLU_F16A_FUSE | `metal_api.m` | candidate | unset | fused SwiGLU experiment |
| ZDRAW_TAE | `vae_mode.zig` | candidate | unset | TAEF1 preview tier |
| ZDRAW_TOMA | `toma.zig` | killed | disabled | remove after the ToMA v2 scaffolding-reuse decision; ToMA v1 failed the breadth gate |
| ZDRAW_TOMA_DESTINATIONS | `zstep_toma.zig` | killed | unset | remove after the ToMA v2 scaffolding-reuse decision; retained ToMA v1 destination control |
| ZDRAW_TOMA_FROM | `toma.zig` | killed | unset | remove after the ToMA v2 scaffolding-reuse decision; retained ToMA v1 layer bound |
| ZDRAW_TOMA_REGIONS | `zstep_toma.zig` | killed | unset | remove after the ToMA v2 scaffolding-reuse decision; retained ToMA v1 region control |
| ZDRAW_TOMA_ROUTE | `zstep_toma.zig` | killed | unset | remove after the ToMA v2 scaffolding-reuse decision; retained ToMA v1 route control |
| ZDRAW_TOMA_SCALE | `zstep_toma.zig` | killed | unset | remove after the ToMA v2 scaffolding-reuse decision; retained ToMA v1 scale control |
| ZDRAW_TOMA_TO | `toma.zig` | killed | unset | remove after the ToMA v2 scaffolding-reuse decision; retained ToMA v1 layer bound |
| ZDRAW_UNFUSE | `metal_api.m` | candidate | unset | FFN unfuse/overlap experiment |
| ZDRAW_VAE_ATTN_HALF | `mvattn_owned.zig` | default-config | typed VAE profile (product) | owned VAE mid attention with half Q/K/V^T/P operands (f32 accumulate); unset = exact f32 path (strict) |
| ZDRAW_VAE_CONV_H8 | `mvres_stream_chain.zig` | default-config | half VAE routes (product) | `0` keeps the 4-simdgroup conv kernels; unset/`1` = the 8-simdgroup staged-weight kernels (`conv2d_*_window_h8`, byte-identical) |
| ZDRAW_VAE_WINO | `vae_mode.zig` | default-config | typed VAE profile (product) | Winograd F(4x4,3x3) 3x3 convs on the product half route; `0` = the direct kernels (A/B arm) |
| ZDRAW_VAE_ATTN_OWNED | `mvattn_owned.zig` | default-config | `1` | VAE mid-block attention on the owned row-blocked exact-GEMM + f32 softmax path (no MPSGraph); `0` restores the graph route for A/B |
| ZDRAW_VAE | `vae_mode.zig` | default-config | typed quality profile | VAE tier (`raw` = the owned streamed decoder on both profiles; the `mpsgraph` value is gone since 2026-08-27) |
| ZDRAW_VAE_BUDGET | `gemmbench.zig` | bench-selector | unset | VAE budget benchmark |
| ZDRAW_VENCODE_GPU | `vencodegate.zig` | instrument | unset | `0` keeps the VAE encoder gate on the CPU reference tier (default: the Metal contexts) |
| ZDRAW_VAE_DUMP_STAGES | `vae_mode.zig` | instrument | unset | VAE stage dump |
| ZDRAW_VAE_F16 | `vae_mode.zig` | default-config | typed VAE profile (product) | f16 VAE path |
| ZDRAW_VAE_F16SIM | `vae_mode.zig` | instrument | unset | reduced-precision simulation |
| ZDRAW_VAE_F16_STAGES | `vae_mode.zig` | instrument | unset | reduced-precision stage selector |
| ZDRAW_VAE_FINAL_H | `vae_mode.zig` | default-config | typed VAE profile (product) | final half-precision stage |
| ZDRAW_VAE_FULL_H | `vae_mode.zig` | default-config | typed VAE profile (product) | full half-precision path |
| ZDRAW_VAE_ATTN_GPU | `vdecode.zig` | default-config | typed VAE profile | resident VAE mid-attention (GroupNorm/transposes/projections/residual on chain-pool buffers; the wide-SDPA dispatch surface unchanged); product tier sets it, strict keeps the CPU-orchestrated reference path; `0` restores it anywhere |
| ZDRAW_VAE_PQFOLD | `zflux2_vae.zig` | default-config | on | fused BN+unpack+post_quant Klein prepare (four per-subpixel folded 32x32 maps, one pass, exact algebra, float-order change only); `0` restores the three-pass reference |
| ZDRAW_VAE_MID_F16 | `vae_mode.zig` | default-config | typed VAE profile (product) | mid-block precision |
| ZDRAW_VAE_STATS512 | `mvres_stream_chain.zig` | candidate | unset | normalization-statistics fast path |
| ZDRAW_VAE_STATSSQ | `vae_mode.zig` | default-config | typed VAE profile (product) | sum-of-squares statistics |
| ZDRAW_VAE_STREAM | `vae_mode.zig` | default-config | typed VAE profile | streamed VAE path |
| ZDRAW_VAE_STRIP | `vae_mode.zig` | default-config | typed VAE profile | strip height |
| ZDRAW_VAE_STRIPMEM | `mvres_strip_chain.zig` | candidate | none (off) | memory-ladder wall 1: the strip-memory decode for the up blocks (product f16 FULL_H Winograd route only); `1` = strip scratch with whole-map input/output, `2` = the group's later blocks in place over their input (halo stash), `3` = 2 plus the finish in strips without the whole-map f32 norm scratch; awaiting certification |
| ZDRAW_VAE_UNFUSE | `vae_mode.zig` | candidate | unset | bit-identical unfused route awaiting ruling |
| ZDRAW_VAE_V7 | `vae_mode.zig` | default-config | typed VAE profile (product and strict) | the owned convolution implementation both profiles select |
| ZDRAW_W6_ONLY | `gemmbench.zig` | bench-selector | unset | Z-Image W6 subset |
| ZDRAW_W6_RACE | `gemmbench.zig` | bench-selector | unset | Z-Image W6 race benchmark |
| ZDRAW_WCACHE_DEBUG | `zflux2_resident.zig` | instrument | unset | weight-cache miss attribution |
| ZDRAW_WEIGHTS | benchmark and quality tools | deployment | invocation | Z-Image weight directory |
| ZDRAW_ZIMAGE_WEIGHTS | `session_model.zig` | deployment | ~/.zdraw/models/Z-Image-Turbo | Z-Image weights for the session `model` command |
| ZDRAW_KLEIN4B_WEIGHTS | `session_model.zig` | deployment | unset | FLUX.2 Klein 4B weights for the session `model` command; falls back to the launch `--weights` |
| ZDRAW_KLEIN9B_WEIGHTS | `session_model.zig` | deployment | unset | FLUX.2 Klein 9B weights for the session `model` command |
| ZDRAW_KLEIN9BKV_WEIGHTS | `session_model.zig` | deployment | unset | FLUX.2 Klein 9B KV weights for the session `model` command |
| ZDRAW_KLEIN_BASE4B_WEIGHTS | `session_model.zig` | deployment | unset | FLUX.2 Klein base 4B weights for the session `model` command |
| ZDRAW_KLEIN_BASE9B_WEIGHTS | `session_model.zig` | deployment | unset | FLUX.2 Klein base 9B weights for the session `model` command |
| ZDRAW_ZPACK | packed-weight loaders | deployment | invocation | Z-Image sidecar path |
| ZDRAW_PACK_READAHEAD | `zpack_file.zig` | default-config | `1` | F_RDADVISE read-ahead of the mmap'd sidecar at open, on by default (cold page cache one-shot: median -1.1 s, klein-oneshot-prefetch-20260827); `0` disables it |
| ZDRAW_ZSTEP_RES | `zstep.zig` | default-config | resident default | Z-Image resident stack |

`ZDRAW_BATCH_MAX_SPLITS` is a C compile-time constant, not an environment
variable, and is the sole explicit scanner exclusion.

Maintenance rule: a new flag and its inventory row land together. A killed
probe loses its flag and implementation in the same change unless its
disposition records a concrete deferred removal dependency. A promoted probe
becomes typed profile state with one explicit escape hatch.
