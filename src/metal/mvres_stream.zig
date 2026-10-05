//! Exact-streaming VAE residual block (Exact Streaming VAE, Step 3).
//!
//! Composes the two proven bit-exact primitives - the GroupNorm+SiLU split
//! (vae_norm_stats + vae_norm_apply_silu_window, Step 1) and conv2d_window
//! (Step 2) - into ONE residual block whose output is produced in contiguous
//! row-strips of height `strip_rows`, while every statistic stays GLOBAL.
//!
//! The math is identical to the full resblock (norm1+SiLU -> conv1 ->
//! norm2+SiLU -> conv2 -> + skip): GroupNorm means/scales are reduced over the
//! whole frame (one threadgroup per group), and each strip's conv reads its halo
//! from the full resident input by global coordinates. Tiling is therefore a
//! pure scheduling choice - any `strip_rows` (including one that does not divide
//! the height) yields bit-identical output. Proven max|d|=0 by runVaeResStreamGate.
//!
//! Step 3 keeps the norm1 and conv1 outputs as FULL buffers (N1 must be read
//! globally by conv1's halo; C1 must be reduced globally by stats2). The 2-buffer
//! ping-pong squeeze is Step 4 - here we only prove the composition + tiling are
//! exact.

const mbuffer = @import("mbuffer.zig");

// Mirrors the windowed-norm MSL/host params (ZdrawVaeNormParams /
// ZdrawVaeNormWindowParams). dtype 3 = exact f32 weights/biases.
// NormParams is the ABI-asserted struct from mvres_param (abi_assert/abi_gen).
pub const NormParams = @import("mvres_param.zig").NormParams;

pub const NormWindowParams = extern struct {
    channels: u32,
    height: u32,
    width: u32,
    groups: u32,
    dtype: u32,
    bias_dtype: u32,
    eps: f32,
    weight_offset: u64,
    bias_offset: u64,
    row0: u32,
    row1: u32,
    col0: u32,
    col1: u32,
};

// Mirrors ZdrawConvWindowParams: ConvParams fields + output strip [row0, row1).
pub const ConvWindowParams = extern struct {
    in_ch: u32,
    out_ch: u32,
    height: u32,
    width: u32,
    ksize: u32,
    pad: u32,
    dtype: u32,
    bias_dtype: u32,
    has_bias: u32,
    pad1: u32 = 0,
    weight_offset: u64,
    bias_offset: u64,
    row0: u32,
    row1: u32,
};

// Mirrors ZdrawVaeAddWindowParams.
pub const AddWindowParams = extern struct {
    channels: u32,
    height: u32,
    width: u32,
    row0: u32,
    row1: u32,
};

pub extern fn zdraw_metal_run_vae_norm_stats(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    stats: *anyopaque,
    params: *const NormParams,
    thread_count: usize,
) c_int;

pub extern fn zdraw_metal_run_vae_norm_apply_window(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    stats: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const NormWindowParams,
    thread_count: usize,
) c_int;

pub extern fn zdraw_metal_run_conv2d_window(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const ConvWindowParams,
    thread_count: usize,
) c_int;

extern fn zdraw_metal_run_vae_add_window(
    queue: *anyopaque,
    pipeline: *anyopaque,
    output: *anyopaque,
    residual: *anyopaque,
    params: *const AddWindowParams,
    thread_count: usize,
) c_int;

// Pre-compiled pipelines (vae_norm_stats, vae_norm_apply_silu_window,
// conv2d_window, vae_add_window) plus their max threadgroup widths.
pub const Ctx = struct {
    queue: *anyopaque,
    stats_pipe: *anyopaque,
    apply_pipe: *anyopaque,
    conv_pipe: *anyopaque,
    add_pipe: *anyopaque,
    stats_threads: usize,
    apply_threads: usize,
    conv_threads: usize,
    add_threads: usize,
};

pub const Config = struct {
    in_ch: u32,
    out_ch: u32,
    height: u32,
    width: u32,
    groups: u32,
    eps: f32,
};

// dtype code (1=f16, 2=bf16, 3=f32) + byte offset into the bound buffer for one
// weight/bias tensor. The gate uploads exact f32 at offset 0 (dtype 3); the real
// decode binds mmap'd shard slices, so dtype follows the safetensor and offset is
// the slice start. Defaults reproduce the gate's exact-f32 binding.
pub const WeightFmt = struct {
    dtype: u32 = 3,
    offset: u64 = 0,
};

// Per-stage weight/bias formats. Each pair matches a Buffers handle; the kernels
// read with read_value(base + offset, idx, dtype), so this carries the dtype and
// offset that the resident mvres path passes via mvres_param.NormParams/ConvParams.
pub const Weights = struct {
    norm1_w: WeightFmt = .{},
    norm1_b: WeightFmt = .{},
    conv1_w: WeightFmt = .{},
    conv1_b: WeightFmt = .{},
    norm2_w: WeightFmt = .{},
    norm2_b: WeightFmt = .{},
    conv2_w: WeightFmt = .{},
    conv2_b: WeightFmt = .{},
    skip_w: WeightFmt = .{},
    skip_b: WeightFmt = .{},
};

// Device buffers for one streamed resblock. Weight/bias handles pair with the
// matching Weights entry (dtype + offset); when none is supplied they default to
// exact f32 at offset 0, matching how the gate uploads them.
pub const Buffers = struct {
    input: *anyopaque,
    output: *anyopaque,
    norm1_w: *anyopaque,
    norm1_b: *anyopaque,
    conv1_w: *anyopaque,
    conv1_b: *anyopaque,
    norm2_w: *anyopaque,
    norm2_b: *anyopaque,
    conv2_w: *anyopaque,
    conv2_b: *anyopaque,
    skip_w: ?*anyopaque = null,
    skip_b: ?*anyopaque = null,
    fmt: Weights = .{},
};

// Run the streamed resblock, writing the full result into `bufs.output`. The
// output is produced one row-strip at a time (height `strip_rows`, ragged tail
// allowed); all GroupNorm statistics are global. `strip_rows` must be > 0.
pub fn run(
    device: *anyopaque,
    ctx: Ctx,
    bufs: Buffers,
    cfg: Config,
    strip_rows: u32,
) !void {
    if (strip_rows == 0) return error.InvalidShape;
    if (cfg.in_ch != cfg.out_ch and bufs.skip_w == null) return error.InvalidShape;

    var s = try Scratch.make(device, cfg, bufs.skip_w != null);
    defer s.deinit();
    const fmt = bufs.fmt;

    // Steps 1-2: global stats over input, then materialize norm1+SiLU into N1.
    const np1 = normParams(cfg, cfg.in_ch, fmt.norm1_w, fmt.norm1_b);
    try normFull(ctx, np1, s.stats1.handle, .{
        .input = bufs.input,
        .weight = bufs.norm1_w,
        .bias = bufs.norm1_b,
        .output = s.n1.handle,
    });
    // Step 3: conv1 per strip over N1 (halo read globally) -> full C1.
    const conv1 = ConvStage{ .weight = bufs.conv1_w, .bias = bufs.conv1_b };
    const conv1_cfg = convCfg(cfg.in_ch, cfg.out_ch, cfg, 3, 1, fmt.conv1_w, fmt.conv1_b);
    try convStrips(ctx, s.n1.handle, s.c1.handle, conv1, conv1_cfg, strip_rows);

    // Steps 4-5: global stats over C1, then materialize norm2+SiLU into N2.
    const np2 = normParams(cfg, cfg.out_ch, fmt.norm2_w, fmt.norm2_b);
    try normFull(ctx, np2, s.stats2.handle, .{
        .input = s.c1.handle,
        .weight = bufs.norm2_w,
        .bias = bufs.norm2_b,
        .output = s.n2.handle,
    });
    // Step 6: conv2 per strip over N2 -> output strip.
    const conv2 = ConvStage{ .weight = bufs.conv2_w, .bias = bufs.conv2_b };
    const conv2_cfg = convCfg(cfg.out_ch, cfg.out_ch, cfg, 3, 1, fmt.conv2_w, fmt.conv2_b);
    try convStrips(ctx, s.n2.handle, bufs.output, conv2, conv2_cfg, strip_rows);

    // Optional skip projection per strip into skip_buf (1x1, no halo).
    if (bufs.skip_w) |skip_w| {
        const skip = ConvStage{ .weight = skip_w, .bias = bufs.skip_b.? };
        const skip_cfg = convCfg(cfg.in_ch, cfg.out_ch, cfg, 1, 0, fmt.skip_w, fmt.skip_b);
        try convStrips(ctx, bufs.input, s.skip.?.handle, skip, skip_cfg, strip_rows);
    }

    // Step 7: add the residual (input, or its 1x1 projection) per strip.
    const residual = if (s.skip) |b| b.handle else bufs.input;
    try addStrips(ctx, bufs.output, residual, cfg, strip_rows);
}

// Full intermediates plus per-group stats. N1 (norm1+SiLU of input) and C1
// (conv1 output) MUST be full buffers so conv1's halos and the stats2 reduction
// see the whole frame; N2 holds norm2+SiLU of C1. stats* hold [mean, scale] per
// group. skip is allocated only when a 1x1 skip projection is needed.
const Scratch = struct {
    n1: mbuffer.Buffer,
    c1: mbuffer.Buffer,
    n2: mbuffer.Buffer,
    stats1: mbuffer.Buffer,
    stats2: mbuffer.Buffer,
    skip: ?mbuffer.Buffer,

    fn make(device: *anyopaque, cfg: Config, has_skip: bool) !Scratch {
        const hw: usize = @as(usize, cfg.height) * cfg.width;
        const in_n: usize = @as(usize, cfg.in_ch) * hw;
        const out_n: usize = @as(usize, cfg.out_ch) * hw;
        const stats_n: usize = @as(usize, cfg.groups) * 2 * 4;
        return .{
            .n1 = try mbuffer.Buffer.empty(device, in_n * 4),
            .c1 = try mbuffer.Buffer.empty(device, out_n * 4),
            .n2 = try mbuffer.Buffer.empty(device, out_n * 4),
            .stats1 = try mbuffer.Buffer.empty(device, stats_n),
            .stats2 = try mbuffer.Buffer.empty(device, stats_n),
            .skip = if (has_skip) try mbuffer.Buffer.empty(device, out_n * 4) else null,
        };
    }

    fn deinit(self: *Scratch) void {
        self.n1.deinit();
        self.c1.deinit();
        self.n2.deinit();
        self.stats1.deinit();
        self.stats2.deinit();
        if (self.skip) |*b| b.deinit();
    }
};

// One global GroupNorm+SiLU: global stats pass into `stats`, then a full-frame
// (whole-height) windowed apply. The composition is bit-exact vs vae_norm_silu.
const NormIo = struct {
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
};

fn normFull(
    ctx: Ctx,
    np: NormParams,
    stats: *anyopaque,
    io: NormIo,
) !void {
    if (zdraw_metal_run_vae_norm_stats(
        ctx.queue,
        ctx.stats_pipe,
        io.input,
        stats,
        &np,
        ctx.stats_threads,
    ) != 0) return error.MetalDispatchFailed;
    const apply = normWindow(np, 0, np.height);
    if (zdraw_metal_run_vae_norm_apply_window(
        ctx.queue,
        ctx.apply_pipe,
        io.input,
        stats,
        io.weight,
        io.bias,
        io.output,
        &apply,
        ctx.apply_threads,
    ) != 0) return error.MetalDispatchFailed;
}

// Weight/bias handles for one convolution stage.
const ConvStage = struct {
    weight: *anyopaque,
    bias: *anyopaque,
};

// conv2d_window over every [row0, row1) strip that covers [0, height).
fn convStrips(
    ctx: Ctx,
    input: *anyopaque,
    output: *anyopaque,
    stage: ConvStage,
    base: ConvWindowParams,
    strip_rows: u32,
) !void {
    var row0: u32 = 0;
    while (row0 < base.height) : (row0 += strip_rows) {
        const row1 = @min(row0 + strip_rows, base.height);
        var p = base;
        p.row0 = row0;
        p.row1 = row1;
        if (zdraw_metal_run_conv2d_window(
            ctx.queue,
            ctx.conv_pipe,
            input,
            stage.weight,
            stage.bias,
            output,
            &p,
            ctx.conv_threads,
        ) != 0) return error.MetalDispatchFailed;
    }
}

// vae_add_window over every strip that covers [0, height).
fn addStrips(
    ctx: Ctx,
    output: *anyopaque,
    residual: *anyopaque,
    cfg: Config,
    strip_rows: u32,
) !void {
    var row0: u32 = 0;
    while (row0 < cfg.height) : (row0 += strip_rows) {
        const row1 = @min(row0 + strip_rows, cfg.height);
        const p = AddWindowParams{
            .channels = cfg.out_ch,
            .height = cfg.height,
            .width = cfg.width,
            .row0 = row0,
            .row1 = row1,
        };
        if (zdraw_metal_run_vae_add_window(
            ctx.queue,
            ctx.add_pipe,
            output,
            residual,
            &p,
            ctx.add_threads,
        ) != 0) return error.MetalDispatchFailed;
    }
}

fn normParams(cfg: Config, channels: u32, weight: WeightFmt, bias: WeightFmt) NormParams {
    return .{
        .channels = channels,
        .height = cfg.height,
        .width = cfg.width,
        .groups = cfg.groups,
        .dtype = weight.dtype,
        .bias_dtype = bias.dtype,
        .eps = cfg.eps,
        .weight_offset = weight.offset,
        .bias_offset = bias.offset,
    };
}

fn normWindow(np: NormParams, row0: u32, row1: u32) NormWindowParams {
    return .{
        .channels = np.channels,
        .height = np.height,
        .width = np.width,
        .groups = np.groups,
        .dtype = np.dtype,
        .bias_dtype = np.bias_dtype,
        .eps = np.eps,
        .weight_offset = np.weight_offset,
        .bias_offset = np.bias_offset,
        .row0 = row0,
        .row1 = row1,
        .col0 = 0,
        .col1 = np.width,
    };
}

fn convCfg(
    in_ch: u32,
    out_ch: u32,
    cfg: Config,
    ksize: u32,
    pad: u32,
    weight: WeightFmt,
    bias: WeightFmt,
) ConvWindowParams {
    return .{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = cfg.height,
        .width = cfg.width,
        .ksize = ksize,
        .pad = pad,
        .dtype = weight.dtype,
        .bias_dtype = bias.dtype,
        .has_bias = 1,
        .weight_offset = weight.offset,
        .bias_offset = bias.offset,
        .row0 = 0,
        .row1 = cfg.height,
    };
}
