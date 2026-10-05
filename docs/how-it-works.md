# How zDraw works

A short tour for readers who want to know what the engine computes and how it
is built. Part 1 recalls the model in the terms the code uses; part 2 follows
one render through the engine; part 3 explains the technical choices. File
references point at the source in this repository.

## 1. The model

The models are latent diffusion models trained with flow matching. The
image lives in a compressed latent space produced by a variational
autoencoder (VAE): for Klein, 32 channels at one eighth of the image's width
and height, so a 1024×1024 image is a 128×128×32 latent. The transformer does
not see the latent as pixels but as tokens: Klein patches 2×2 latent cells
into one 128-channel token (`src/zflux2.zig:22`), so the 1024×1024 image
becomes 4,096 tokens, which is also the engine's admission limit
(`src/model_kind.zig:3-10`).

Generation starts from Gaussian noise and moves it toward an image. Flow
matching parameterises this as a straight path from noise to data, indexed
by a noise level σ from 1 to 0, and trains the network to predict the
velocity along that path. Sampling is then an ordinary differential equation
solved with Euler steps: at each step the network predicts the velocity v for
the current latent and the latent moves by v times the change in σ
(`src/zdenoise.zig:214-221`). The σ schedule is not uniform: Z-Image uses a
fixed shift of 3.0 (`src/scheduler.zig:5-12`) and Klein a shift that depends
on the token count and step count, as in diffusers
(`src/zflux2_schedule.zig:40-52`). The distilled models (Klein 4B,
Z-Image-Turbo) were trained to need only four such steps. The undistilled
Klein base checkpoint takes fifty, and uses classifier-free guidance: the
network is evaluated twice per step, with the prompt and with the empty
prompt, and the velocities are combined as
`v = v_empty + g · (v_prompt − v_empty)` with g = 4 by default
(`src/zflux2_run.zig:588-611`; defaults in `src/model_kind.zig:37-65`).

The prompt enters as conditioning tokens from a text encoder. All three models
use a Qwen3 language model for this (`src/qwen_encoder.zig:22-32`). Z-Image takes
the penultimate hidden state; Klein concatenates the hidden states after
blocks 9, 18 and 27 of its Qwen3-4B encoder for every token
(`src/qwen_encoder.zig:127-148`). The transformer itself, for Klein 4B, has a
hidden size of 3,072 with 24 heads of dimension 128, five double-stream
blocks in which image and text tokens attend jointly, then twenty
single-stream blocks over the concatenated sequence, with rotary position
embeddings (`src/zflux2.zig:28-35`).

After the last step the VAE decoder turns the latent back into pixels: a
stack of residual blocks (GroupNorm, SiLU, 3×3 convolutions) and upsampling
stages, with one self-attention block at the lowest resolution. At 1024×1024
that attention runs over 16,384 positions with a single head of dimension
512 (`src/mvattn_owned.zig:1-10`), which is why the decoder, not the
transformer, sets the memory high-water mark.

## 2. One render, stage by stage

**Weights.** `zdraw fetch` downloads the checkpoint and builds a `.zpack`
sidecar: one file with a small header and 16-byte-aligned tensor payloads
(`src/zpack_file.zig:6-56`). The default W16 pack stores every weight as IEEE
half precision (`src/zw16.zig`, `src/klein_packer.zig:423-431`). Optional
W6, W4 and W2 packs quantise symmetrically in groups of 64 along each row
with one f16 scale per group (`src/klein_packer.zig:393-415`); the W6 format
packs four 6-bit codes into three bytes (`src/zw6.zig`). At run time the pack
is memory-mapped read-only and wrapped as a Metal buffer without copying
(`src/zpack_file.zig:74-115`, `src/metal_api.m:597-625`). The file is read
ahead into the page cache (`F_RDADVISE`), but its pages belong to the file
cache, not to the process's footprint.

**Text.** The Klein encoder runs as one resident GPU batch with a single
embedding upload and a single readback (`src/qwen_resident.zig:1-18`), and
the result is memoised, so rerolls and multi-seed batches encode the prompt
once (`src/zflux2_run.zig:96-101`).

**Noise.** The initial latent is drawn on the CPU from Zig's default
pseudo-random generator seeded with `--seed`, one normal sample per element
(`src/zdenoise.zig:179-183`). The seed therefore means the same thing on
every Mac.

**Denoising.** Each transformer block is a sequence of GEMMs and one
attention. On macOS 26, Klein's GEMMs use Metal 4's `matmul2d` tensor
operation (`src/mgemm_mpp_shader.zig`); elsewhere a simdgroup kernel of
zDraw's own runs instead, and the engine counts that as a fallback
(`src/metal_api.m:459`). Z-Image's default GEMM is zDraw's `gemm_f16_direct`,
a 64×64 tile kernel with four simdgroups and ping-pong staging
(`src/metal_api.m:4406-4409`). Attention uses the steel kernels from Apple's
MLX, compiled ahead of time into `steel.metallib` (`build.zig:63-90`,
`vendor/steel/attn_entry.metal`), with a vendored metal-flash-attention
kernel as the counted fallback. Quantised packs run through steel GEMM
variants written for zDraw, whose B-operand loader dequantises 6-, 4- or
2-bit codes into the threadgroup tile
(`vendor/steel/include/.../steel_gemm_w6.h`, `loader_w2.h`). The Euler update
runs on the CPU between GPU steps (`src/zdenoise.zig:214-221`); keeping it on
the GPU is an experiment behind `ZDRAW_KLEIN_XRES`, off by default
(`src/zflux2_run.zig:846-858`).

**Previews.** After each Klein step the engine forms the predicted clean
latent `x − σ_next · v` and maps its 32 channels to RGB with a linear map
fitted on 36 renders (R² 0.765), on the CPU at latent resolution
(`src/preview.zig`, `src/preview_map.zig`). The terminal shows it through the
kitty or iTerm image protocol, or as true-colour cells
(`src/terminal_image.zig:5-37`).

**Decoding.** The default `product` route keeps the feature maps in f16 and
runs every stride-1 3×3 convolution as a Winograd F(4×4, 3×3) transform: 36
batched GEMMs with f32 transforms and f32 accumulation
(`src/mconv_wino_shader.zig:1-10`, `src/vae_mode.zig:50-78`). GroupNorm
statistics are one pass of sums and sums of squares with a fixed-order tree
reduction (`src/mvnorm_shader.zig:231-268`), and normalisation plus SiLU is
fused into the convolution's input read. The upsampling stages stream: the
feature map stays on the GPU and ping-pongs between two pooled buffers
(`src/mvres_stream_chain.zig:1-17`). The mid-block attention is zDraw's own
kernel: row-blocked QKᵀ, f32 softmax, and the product with V on the exact f32
GEMM, in one command buffer (`src/mvattn_owned.zig`). The `strict` route
(Z-Image only) is the f32 reference decoder, several times slower; the product
route is measured against it (48.7 dB PSNR at 1024×1024,
`src/vae_mode.zig:40-72`).

**Output.** The PNG is written with two text chunks: a `Software` marker and
a `zdraw` JSON recipe with the model, profile, prompt hash, seed, size, steps
and guidance (`src/recipe.zig:25-72`). The encoder writes no date, version or
path, so identical renders give identical files.

## 3. The choices

**Zig with one native bridge.** The engine is Zig; all Metal calls go through
one Objective-C file of plain C functions taking opaque pointers
(`src/metal_api.m:1-24`). Struct layouts shared across that boundary are not
written by hand: a build step generates a header from Zig's `@sizeOf`
(`src/abi_gen.zig`, `build.zig:25-36`) and the C side checks it with static
assertions. Most kernels are Metal Shading Language source held in Zig and
Objective-C string literals and compiled at start-up; the steel attention and GEMM kernels are
the exception, compiled ahead of time because they need Xcode's compiler.

**Determinism as a contract, not a hope.** Noise comes from the CPU, every
reduction kernel fixes its summation order, the render shaders use no
atomics, and every command buffer is committed and waited on before the next
depends on it. The result is that a request produces the same bytes on every
run. The claim is enforced rather than assumed: `certified/hashes.json` is
compiled into the binary, `zdraw bench` renders the certified case and prints
MATCH or MISMATCH (`src/bench_card.zig`), and the release process runs
ten-render censuses (`tools/quality/repro_census.sh`). The card also refuses
to name a fast route unless that route's fallback counter is zero
(`src/bench_card.zig:276-296`), so a silent fallback cannot be reported as
the kernel it replaced.

**Memory.** Three decisions set the footprint. Weights are mapped, not
copied, which is why the process figures in the README exclude them and why
`ZDRAW_MEMTRACE` reports the resident mapped pages separately, measured with
`mincore` over the registered mappings (`src/metal_api.m:655-726`).
Activations live in named, grow-only pools that last as long as the Metal
context, because macOS does not unwire a released buffer, so freeing and
reallocating only raises the high-water mark (`src/mchain_pool.zig:1-40`).
And the pools lend to each other: while the decoder runs, the transformer's
idle buffers are offered to it through `mlend.Offer`, which only ever hands
out existing buffers and never allocates (`src/mlend.zig:1-40`,
`src/zflux2_pool.zig:154-189`).

**Honest routes.** `zdraw doctor` probes the machine and prints the kernels
the defaults will use and why (`src/doctor.zig:142-181`). Apple's MPS
frameworks are linked but off every default render path; they survive only
behind non-default environment settings for comparison.

**What is not here.** No Python at run time, no desktop app or SDK in this
repository, and no claim beyond what the certified case and the measurement
protocol cover. The [architecture document](architecture.md) records the
ownership rules behind the modules named above, and the
[measurement protocol](measurement-protocol.md) the rules behind every number.
