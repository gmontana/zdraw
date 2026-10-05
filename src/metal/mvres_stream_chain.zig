//! Resident exact-streaming VAE residual-block chain (Exact Streaming VAE,
//! Step 4b).
//!
//! Step 4a (mvres_stream_run) proved the windowed split is bit-exact vs
//! ZDRAW_VAE=raw, but it round-tripped every resnet through CPU f32 buffers
//! (upload input, dispatch, read back output), so the per-resnet transient
//! buffers drove the 1024px phys_footprint to ~30 GB even though the live GPU
//! set was only ~4 GB. This module keeps the VAE feature map ON-GPU across a
//! whole resnet group, exactly like mvrchain does for the resident (non-stream)
//! path: allocate the group's GPU buffers ONCE, ping-pong the feature map
//! between two output buffers across the blocks, and read back only the final
//! result. The convs fuse GroupNorm+SiLU into their input read
//! (conv2d_prenorm_window), so no full normed buffer is materialized either -
//! the only full buffers live at the top resolution are the two ping-pong
//! feature buffers plus conv1_out (which stats2 + conv2's halo must read
//! globally). The math is byte-for-byte the mvres_stream strip pipeline, which
//! is byte-for-byte ZDRAW_VAE=raw.

const std = @import("std");

const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const metrics = @import("metrics.zig");
const mlend = @import("mlend.zig");
const mpipe = @import("mpipe.zig");
const mres_util = @import("mres_util.zig");
const mvfinal = @import("mvfinal.zig");
const bufs = @import("mvres_buf.zig");
const params = @import("mvres_param.zig");
const mconv_h8 = @import("mconv_h8_shader.zig");
const mvres_pool = @import("mvres_pool.zig");
const mvres_strip = @import("mvres_strip_chain.zig");
const mvres_wino = @import("mvres_wino.zig");
const tensor = @import("../pack/tensor.zig");
const vae_f16 = @import("../vae/vae_f16.zig");
const vnorm_shader = @import("mvnorm_shader.zig");
const vres = @import("../vae/vres.zig");

// Fused nearest-2x-upsample + 3x3 conv over an output row-strip, reading the
// low-res input directly (no full 2x intermediate). The conv routes through the
// hand kernel, matching the resident mvup path (upsample2 then conv2d) under
// ZDRAW_VAE=raw bit-for-bit; here it reads the resblock group's GPU output and
// writes a pool buffer, so the resnet -> upsample handoff never touches the CPU.
pub const ConvUpsampleWindowParams = extern struct {
    channels: u32,
    out_height: u32,
    out_width: u32,
    in_height: u32,
    in_width: u32,
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

pub extern fn zdraw_metal_run_conv2d_upsample_window(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const ConvUpsampleWindowParams,
    thread_count: usize,
    oc_x4: c_int,
) c_int;

pub const Config = struct {
    in_ch: usize,
    out_ch: usize,
    height: usize,
    width: usize,
};

// One streamed resblock's GPU buffers (mirrors ZdrawVaeResStreamBuffers). The
// feature ping-pong (input/output) is owned by the Set; conv1_out/stats/skip are
// shared scratch reused by every block in the group.
pub const StreamBuffers = extern struct {
    input: *anyopaque,
    output: *anyopaque,
    conv1_out: *anyopaque,
    stats1: *anyopaque,
    stats2: *anyopaque,
    skip: *anyopaque,
    norm1_w: *anyopaque,
    norm1_b: *anyopaque,
    conv1_w: *anyopaque,
    conv1_b: *anyopaque,
    norm2_w: *anyopaque,
    norm2_b: *anyopaque,
    conv2_w: *anyopaque,
    conv2_b: *anyopaque,
    skip_w: *anyopaque,
    skip_b: *anyopaque,
    // Null unless the unfuse route (the one C reader) is enabled; the C side
    // null-checks it before encoding.
    norm: ?*anyopaque,
};

extern fn zdraw_metal_run_f16_to_f32(
    queue: *anyopaque,
    device: *anyopaque,
    input: *anyopaque,
    output: *anyopaque,
    count: u32,
) c_int;

pub extern fn zdraw_metal_run_vae_res_stream_chain(
    queue: *anyopaque,
    stats_pipeline: *anyopaque,
    prenorm_pipeline: *anyopaque,
    conv_pipeline: *anyopaque,
    add_pipeline: *anyopaque,
    apply_pipeline: *anyopaque,
    apply_h_pipeline: ?*anyopaque,
    conv3_pipeline: *anyopaque,
    stats2_pipeline: ?*anyopaque,
    prenorm2_pipeline: ?*anyopaque,
    buffers: [*]const StreamBuffers,
    ps: [*]const params.ResParams,
    count: usize,
    strip_rows: u32,
    stats_threads: usize,
    conv_threads: usize,
    add_threads: usize,
    prenorm_x4: c_int,
    hgraph: c_int,
    wino: ?*const mvres_wino.Set,
) c_int;

// Compiled streaming pipelines: global stats, fused-prenorm windowed conv, plain
// windowed conv (the 1x1 skip), and windowed residual add. Built once, reused for
// every group.

// One-time bf16 -> f32 promotion cache for streamed-VAE conv weights
// (ZDRAW_VAE_V7). The VAE's conv tensors are bf16 in the file (~160 MB); the
// v7 scalar-FMA kernel requires f32 for its float4 weight staging. Promoting
// once costs ~320 MB device-side and preserves values exactly (bf16 -> f32 is
// a widening map), so streamed==raw bit-identity is unaffected.
const Promo = struct {
    keys: [96]usize = @splat(0),
    bufs: [96]?mbuffer.Buffer = @splat(null),
    count: usize = 0,
    half_mode: bool = false,

    fn handleFor(self: *Promo, device: *anyopaque, view: tensor.View) !*anyopaque {
        const key = @intFromPtr(view.bytes.ptr);
        for (self.keys[0..self.count], 0..) |k, i| {
            if (k == key) return self.bufs[i].?.handle;
        }
        if (self.count >= self.keys.len) return error.PromoCacheFull;
        if (view.dtype != .bf16) return error.PromoUnsupportedDtype;
        const alloc = std.heap.page_allocator;
        const n = try view.elems();
        const buf = if (self.half_mode) blk: {
            // bf16 -> f16 is lossless in the VAE's range (8 mantissa bits
            // embed in f16's 11; |w| << 65504): same values at half the bytes.
            const f16s = try alloc.alloc(f16, n);
            defer alloc.free(f16s);
            for (f16s, 0..) |*dst, i| dst.* = @floatCast(view.atF32Unchecked(i));
            break :blk try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(f16s));
        } else blk: {
            const f32s = try alloc.alloc(f32, n);
            defer alloc.free(f32s);
            for (f32s, 0..) |*dst, i| dst.* = view.atF32Unchecked(i);
            break :blk try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(f32s));
        };
        self.keys[self.count] = key;
        self.bufs[self.count] = buf;
        self.count += 1;
        return buf.handle;
    }

    fn deinit(self: *Promo) void {
        for (self.bufs[0..self.count]) |*slot| {
            if (slot.*) |*buf| buf.deinit();
            slot.* = null;
        }
        self.count = 0;
    }
};

// Per-stage f16-storage simulation mask for the mixed-precision strict VAE
// bisection. ZDRAW_VAE_F16_STAGES is a comma list from {mid,up0,up1,up2,up3}
// ("all" = everything); ZDRAW_VAE_F16SIM=1 keeps meaning all stages. Stages in
// the mask run the *_f16sim kernels (feature reads rounded to f16); the rest
// run the exact f32 kernels. The finish path is always f32.
const SimStages = struct {
    mid: bool = false,
    up: [4]bool = .{ false, false, false, false },

    fn fromEnv() SimStages {
        var st = SimStages{};
        if (std.c.getenv("ZDRAW_VAE_F16SIM") != null) {
            st.mid = true;
            st.up = .{ true, true, true, true };
            return st;
        }
        const raw = std.c.getenv("ZDRAW_VAE_F16_STAGES") orelse return st;
        const list = std.mem.span(raw);
        if (std.mem.eql(u8, list, "all")) {
            st.mid = true;
            st.up = .{ true, true, true, true };
            return st;
        }
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |tok| {
            if (std.mem.eql(u8, tok, "mid")) st.mid = true;
            inline for (0..4) |b| {
                const name = std.fmt.comptimePrint("up{d}", .{b});
                if (std.mem.eql(u8, tok, name)) st.up[b] = true;
            }
        }
        return st;
    }
};

pub const Ctx = struct {
    stats_pipe: *anyopaque,
    stats_fast_pipe: *anyopaque,
    stats_sq_pipe: *anyopaque,
    stats_sim_pipe: *anyopaque,
    prenorm_pipe: *anyopaque,
    prenorm_v3_pipe: *anyopaque,
    prenorm_v7_pipe: *anyopaque,
    prenorm_sim_pipe: *anyopaque,
    apply_h_pipe: *anyopaque,
    up_v7_pipe: *anyopaque,
    up_sim_pipe: *anyopaque,
    apply_pipe: *anyopaque,
    conv7_pipe: *anyopaque,
    prenorm_h_pipe: *anyopaque,
    prenorm_hc1_pipe: *anyopaque,
    prenorm_hc2_pipe: *anyopaque,
    up_h_pipe: *anyopaque,
    prenorm_h8_pipe: *anyopaque,
    up_h8_pipe: *anyopaque,
    wino: mvres_wino.Pipes,
    // Strip-memory pipes (wall 1), present when ZDRAW_VAE_STRIPMEM is on.
    strip: ?mvres_strip.Pipes,
    up2_h_pipe: *anyopaque,
    skip_h_pipe: *anyopaque,
    stats_h_pipe: *anyopaque,
    stats_h_fast_pipe: *anyopaque,
    stats_h_sq_pipe: *anyopaque,
    add_h_pipe: *anyopaque,
    conv_sim_pipe: *anyopaque,
    sim_stages: SimStages = .{},
    up_v3_pipe: *anyopaque,
    f32_to_f16_pipe: *anyopaque,
    conv_pipe: *anyopaque,
    add_pipe: *anyopaque,
    up_pipe: *anyopaque,
    stats_threads: usize,
    conv_threads: usize,
    add_threads: usize,
    up_threads: usize,
    device: *anyopaque,
    pool: mvres_pool.Pool = .{},
    promo: Promo = .{},
    promo_h: Promo = .{ .half_mode = true },

    pub fn init(ctx: *mconv.Context) !Ctx {
        var err: [1024]u8 = undefined;
        const dev = ctx.device;
        const norm_src = vnorm_shader.vnorm.ptr;
        const conv_src = mconv_h8.conv.ptr;

        const stats_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_pipe);
        const stats_fast_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats_fast", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_fast_pipe);
        const stats_sq_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats_sq", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_sq_pipe);
        const stats_sim_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats_f16sim", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_sim_pipe);
        const apply_pipe = try mpipe.required(dev, norm_src, "vae_norm_apply_silu_window", &err);
        errdefer c.zdraw_metal_release_pipeline(apply_pipe);
        const apply_h_pipe = try mpipe.required(dev, norm_src, "vae_norm_apply_silu_window_h", &err);
        errdefer c.zdraw_metal_release_pipeline(apply_h_pipe);
        const conv7_pipe = try mpipe.required(dev, conv_src, "conv2d_window_v7", &err);
        errdefer c.zdraw_metal_release_pipeline(conv7_pipe);
        const prenorm_h_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_h4", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_h_pipe);
        const prenorm_hc1_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_h4c1", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_hc1_pipe);
        const prenorm_hc2_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_h4c2", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_hc2_pipe);
        const up_h_pipe = try mpipe.required(dev, conv_src, "conv2d_upsample_window_h4", &err);
        errdefer c.zdraw_metal_release_pipeline(up_h_pipe);
        const prenorm_h8_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_h8", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_h8_pipe);
        const up_h8_pipe = try mpipe.required(dev, conv_src, "conv2d_upsample_window_h8", &err);
        errdefer c.zdraw_metal_release_pipeline(up_h8_pipe);
        const wino = try mvres_wino.Pipes.init(dev);
        errdefer wino.deinit();
        const strip: ?mvres_strip.Pipes = if (mvres_strip.enabled())
            try mvres_strip.Pipes.init(dev)
        else
            null;
        errdefer if (strip) |sp| sp.deinit();
        const up2_h_pipe = try mpipe.required(dev, norm_src, "vae_upsample2_h", &err);
        errdefer c.zdraw_metal_release_pipeline(up2_h_pipe);
        const skip_h_pipe = try mpipe.required(dev, conv_src, "conv1x1_h", &err);
        errdefer c.zdraw_metal_release_pipeline(skip_h_pipe);
        const stats_h_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats_h", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_h_pipe);
        const stats_h_fast_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats_h_fast", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_h_fast_pipe);
        const stats_h_sq_pipe = try mpipe.required(dev, norm_src, "vae_norm_stats_h_sq", &err);
        errdefer c.zdraw_metal_release_pipeline(stats_h_sq_pipe);
        const add_h_pipe = try mpipe.required(dev, norm_src, "vae_add_window_h", &err);
        errdefer c.zdraw_metal_release_pipeline(add_h_pipe);
        const prenorm_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_pipe);
        const prenorm_v3_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_v3", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_v3_pipe);
        const prenorm_v7_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_v7", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_v7_pipe);
        const prenorm_sim_pipe = try mpipe.required(dev, conv_src, "conv2d_prenorm_window_f16sim", &err);
        errdefer c.zdraw_metal_release_pipeline(prenorm_sim_pipe);
        const up_v7_pipe = try mpipe.required(dev, conv_src, "conv2d_upsample_window_v7", &err);
        errdefer c.zdraw_metal_release_pipeline(up_v7_pipe);
        const up_sim_pipe = try mpipe.required(dev, conv_src, "conv2d_upsample_window_f16sim", &err);
        errdefer c.zdraw_metal_release_pipeline(up_sim_pipe);
        const conv_pipe = try mpipe.required(dev, conv_src, "conv2d_window", &err);
        errdefer c.zdraw_metal_release_pipeline(conv_pipe);
        const conv_sim_pipe = try mpipe.required(dev, conv_src, "conv2d_window_f16sim", &err);
        errdefer c.zdraw_metal_release_pipeline(conv_sim_pipe);
        const add_pipe = try mpipe.required(dev, norm_src, "vae_add_window", &err);
        errdefer c.zdraw_metal_release_pipeline(add_pipe);
        const up_pipe = try mpipe.required(dev, conv_src, "conv2d_upsample_window", &err);
        errdefer c.zdraw_metal_release_pipeline(up_pipe);
        const up_v3_pipe = try mpipe.required(dev, conv_src, "conv2d_upsample_window_v3", &err);
        errdefer c.zdraw_metal_release_pipeline(up_v3_pipe);
        const f32_to_f16_pipe = try mpipe.required(dev, norm_src, "vae_f32_to_f16", &err);
        errdefer c.zdraw_metal_release_pipeline(f32_to_f16_pipe);
        return .{
            .stats_pipe = stats_pipe,
            .stats_fast_pipe = stats_fast_pipe,
            .stats_sq_pipe = stats_sq_pipe,
            .prenorm_pipe = prenorm_pipe,
            .prenorm_v3_pipe = prenorm_v3_pipe,
            .prenorm_v7_pipe = prenorm_v7_pipe,
            .prenorm_sim_pipe = prenorm_sim_pipe,
            .stats_sim_pipe = stats_sim_pipe,
            .apply_pipe = apply_pipe,
            .apply_h_pipe = apply_h_pipe,
            .conv7_pipe = conv7_pipe,
            .prenorm_h_pipe = prenorm_h_pipe,
            .prenorm_hc1_pipe = prenorm_hc1_pipe,
            .prenorm_hc2_pipe = prenorm_hc2_pipe,
            .up_h_pipe = up_h_pipe,
            .prenorm_h8_pipe = prenorm_h8_pipe,
            .up_h8_pipe = up_h8_pipe,
            .wino = wino,
            .strip = strip,
            .up2_h_pipe = up2_h_pipe,
            .skip_h_pipe = skip_h_pipe,
            .stats_h_pipe = stats_h_pipe,
            .stats_h_fast_pipe = stats_h_fast_pipe,
            .stats_h_sq_pipe = stats_h_sq_pipe,
            .add_h_pipe = add_h_pipe,
            .up_sim_pipe = up_sim_pipe,
            .conv_sim_pipe = conv_sim_pipe,
            .sim_stages = SimStages.fromEnv(),
            .up_v7_pipe = up_v7_pipe,
            .conv_pipe = conv_pipe,
            .add_pipe = add_pipe,
            .up_pipe = up_pipe,
            .up_v3_pipe = up_v3_pipe,
            .f32_to_f16_pipe = f32_to_f16_pipe,
            .stats_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(stats_pipe)),
            .conv_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(conv_pipe)),
            .add_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(add_pipe)),
            .up_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(up_pipe)),
            .device = dev,
        };
    }

    // Two resident pool buffers (feat1, output0) for the final norm+RGB conv to
    // reuse, so the finish faults in NO new top-resolution GPU buffers and the
    // final feature never leaves the GPU: `final` (the last group's pool output,
    // returned by runUpBlocks) is the finish's input as-is - no readback, no host
    // copy, no re-upload - and the norm scratch aliases whichever dead pool slot
    // is not `final`. Both slots already hold >= `bytes` (channels*h*w*4) from the
    // last block, but grow defensively in case it ran narrower. Returns null if
    // the pool was never populated (no group ran).
    pub fn residentFinish(
        self: *Ctx,
        final: *anyopaque,
        input_bytes: usize,
        norm_bytes: usize,
        input_f16: bool,
    ) !?mvfinal.Scratch {
        // conv1_out may be a lease this decode (no own buffer); it still counts
        // as populated. The finish only ever reads feat1/output0 here.
        const c1 = &self.pool.conv1_out;
        if (self.pool.feat1.buf == null or (c1.buf == null and c1.lent == null)) return null;
        const feat1_cur = self.pool.feat1.buf.?.handle;
        const out0_cur = if (self.pool.output0.buf) |b| b.handle else null;
        _ = input_bytes; // The caller owns final's current storage size.
        const norm = if (feat1_cur == final)
            // Own buffer: a lease may be smaller than this request and a
            // lent buffer must never be grown by the borrower.
            try self.pool.conv1_out.ownHandle(self.device, norm_bytes)
        else if (out0_cur != null and out0_cur.? == final)
            try self.pool.feat1.handle(self.device, norm_bytes)
        else
            try self.pool.feat1.handle(self.device, norm_bytes);
        return .{ .input = final, .norm = norm, .resident = true, .input_f16 = input_f16 };
    }

    // Free the persistent feature buffers (keeping the pipelines) once the finish
    // has run. The whole-VAE peak is the last up block's resident set; the finish
    // reads the final feature in place and reuses a dead slot for its norm scratch
    // (residentFinish) rather than faulting its own, so this just reclaims
    // everything at the very end.
    pub fn releasePool(self: *Ctx) void {
        self.pool.deinit();
        self.pool = .{};
    }

    /// Offer another phase's idle buffers for the decode that follows; the
    /// lease is decided at presize and lasts until endLease (one decode).
    pub fn beginLease(self: *Ctx, offer: ?mlend.Offer) void {
        self.pool.lender = offer;
    }

    /// End the decode's lease: the leased role returns to its own buffer.
    /// The offering pool unpins or releases its buffers itself.
    pub fn endLease(self: *Ctx) void {
        self.pool.conv1_out.lent = null;
        self.pool.conv1_chunk.lent = null;
        self.pool.conv1_strip.lent = null;
        self.pool.skip_strip.lent = null;
        self.pool.lender = null;
    }

    pub fn deinit(self: *Ctx) void {
        self.pool.deinit();
        c.zdraw_metal_release_pipeline(self.f32_to_f16_pipe);
        c.zdraw_metal_release_pipeline(self.up_v3_pipe);
        c.zdraw_metal_release_pipeline(self.up_pipe);
        c.zdraw_metal_release_pipeline(self.add_pipe);
        c.zdraw_metal_release_pipeline(self.conv_pipe);
        c.zdraw_metal_release_pipeline(self.prenorm_v3_pipe);
        self.promo_h.deinit();
        c.zdraw_metal_release_pipeline(self.prenorm_v7_pipe);
        c.zdraw_metal_release_pipeline(self.prenorm_sim_pipe);
        c.zdraw_metal_release_pipeline(self.stats_sim_pipe);
        c.zdraw_metal_release_pipeline(self.apply_h_pipe);
        c.zdraw_metal_release_pipeline(self.apply_pipe);
        c.zdraw_metal_release_pipeline(self.conv7_pipe);
        c.zdraw_metal_release_pipeline(self.prenorm_h_pipe);
        c.zdraw_metal_release_pipeline(self.prenorm_hc1_pipe);
        c.zdraw_metal_release_pipeline(self.prenorm_hc2_pipe);
        c.zdraw_metal_release_pipeline(self.up_h_pipe);
        c.zdraw_metal_release_pipeline(self.prenorm_h8_pipe);
        c.zdraw_metal_release_pipeline(self.up_h8_pipe);
        if (self.strip) |sp| sp.deinit();
        self.wino.deinit();
        c.zdraw_metal_release_pipeline(self.up2_h_pipe);
        c.zdraw_metal_release_pipeline(self.skip_h_pipe);
        c.zdraw_metal_release_pipeline(self.stats_h_pipe);
        c.zdraw_metal_release_pipeline(self.stats_h_fast_pipe);
        c.zdraw_metal_release_pipeline(self.stats_h_sq_pipe);
        c.zdraw_metal_release_pipeline(self.add_h_pipe);
        c.zdraw_metal_release_pipeline(self.up_v7_pipe);
        c.zdraw_metal_release_pipeline(self.up_sim_pipe);
        c.zdraw_metal_release_pipeline(self.conv_sim_pipe);
        self.promo.deinit();
        c.zdraw_metal_release_pipeline(self.prenorm_pipe);
        c.zdraw_metal_release_pipeline(self.stats_sq_pipe);
        c.zdraw_metal_release_pipeline(self.stats_fast_pipe);
        c.zdraw_metal_release_pipeline(self.stats_pipe);
        self.* = undefined;
    }
};

// Output strip height for the resident chain. Every VAE height divides 64, so
// this is the proven ts=64 schedule; correctness is independent of it (ts=full
// and ts=8 are also max|d|=0). conv1_out stays full regardless, so the strip is
// purely a dispatch-granularity knob, not a memory one.
pub const default_strip_rows: u32 = 64;

pub fn stripRows() u32 {
    const raw = std.c.getenv("ZDRAW_VAE_STRIP") orelse return default_strip_rows;
    const value = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch return default_strip_rows;
    return if (value == 0) default_strip_rows else value;
}

pub fn finalHEnabled() bool {
    if (!vae_f16.enabled()) return false;
    const raw = std.c.getenv("ZDRAW_VAE_FINAL_H") orelse return false;
    return raw[0] != '0';
}

pub fn midF16Enabled() bool {
    if (!vae_f16.enabled()) return false;
    const raw = std.c.getenv("ZDRAW_VAE_MID_F16") orelse return false;
    return raw[0] != '0';
}

fn stats512Enabled() bool {
    if (!vae_f16.enabled()) return false;
    const raw = std.c.getenv("ZDRAW_VAE_STATS512") orelse return false;
    return raw[0] != '0';
}

/// W7 kernel (8 simdgroups, 64 positions x 128 oc, staged weights): on by
/// default for the half routes; ZDRAW_VAE_CONV_H8=0 keeps the _h4 kernels.
fn convH8Enabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_CONV_H8") orelse return true;
    return raw[0] != '0';
}

/// The _h8 host contract: 64-position tiles inside one row, whole 128-oc
/// tiles, 8-channel blocks, half4 weight loads.
fn convH8Ok(width: usize, in_ch: usize, out_ch: usize, weight_offset: u64) bool {
    return convH8Enabled() and width % 64 == 0 and in_ch % 8 == 0 and
        out_ch % 128 == 0 and weight_offset % 8 == 0;
}

fn statsSqEnabled() bool {
    if (!vae_f16.enabled()) return false;
    const raw = std.c.getenv("ZDRAW_VAE_STATSSQ") orelse return false;
    return raw[0] != '0';
}

fn fullHEnabled() bool {
    if (!vae_f16.enabled()) return false;
    const raw = std.c.getenv("ZDRAW_VAE_FULL_H") orelse return false;
    return raw[0] != '0';
}

// Mirrors vae_unfuse_enabled() in metal_api.m (s[0] == '1', unlike the f16
// helpers above): the one norm-scratch consumer outside the f16 tier.
fn unfuseEnabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_UNFUSE") orelse return false;
    return raw[0] == '1';
}

// normbuf INVARIANT: the slot is sized only by code that reads it. Creation
// memsets the whole capacity (metal_api.m create_buffer), so an unread byte is
// pure dirty footprint: before this rule the product tier wired 1 GiB at 1024
// (up-3, 256ch x 1024^2 x f32) for a block norm scratch nothing consumed.
// Readers:
//   - the per-block norm scratch (Set.norm -> StreamBuffers.norm): read by the
//     C chain solely on the unfuse route (metal_api.m vae_unfuse_enabled &&
//     b->norm && f32-promoted weights); the fused prenorm kernels never touch
//     it and the hgraph half variant is dead (every caller passes false).
//   - f16->f32 homes that size themselves at their own handle() call:
//     featureToF32 (the resident mid-attention, presizeDecode), run()'s
//     readback, and the FINAL_H-off tail of runUpBlocks.
fn blockNormNeeded() bool {
    return unfuseEnabled();
}

// Run `views.len` streamed resblocks resident: the GPU feature map ping-pongs
// across the Ctx's persistent pool buffers (reused across every group, so phys
// does not climb), single readback into `out`. The caller owns `input` (freed
// via recycle once it is uploaded into the pooled input buffer) and `out`.
pub fn run(
    ctx: *mconv.Context,
    stream: *Ctx,
    out: []f32,
    input: []const f32,
    views: []const vres.Views,
    cfg: Config,
    recycle: ?mbuffer.Recycle,
    strip_rows: u32,
) !void {
    if (views.len == 0) return error.InvalidShape;
    if (strip_rows == 0) return error.InvalidShape;
    try check(out, input, views, cfg);

    const h16 = groupHEnabled(midF16Enabled(), views, cfg);
    var set = try Set.make(stream, input, cfg, recycle, false, h16, fullHEnabled(), false, false);
    defer set.deinit();
    const final = try dispatchGroup(ctx, stream, &set, views, cfg, strip_rows, stream.sim_stages.mid, h16);
    if (h16) {
        const n = try product3U32(cfg.out_ch, cfg.height, cfg.width);
        const bytes = @as(usize, n) * @sizeOf(f32);
        const dst = try stream.pool.normbuf.handle(stream.device, bytes);
        if (zdraw_metal_run_f16_to_f32(ctx.queue, stream.device, final, dst, n) != 0) {
            return error.MetalDispatchFailed;
        }
        readBack(dst, out);
    } else {
        readBack(final, out);
    }
}

/// Mid h16 predicate (exposed for the vdecode streamMid route so its
/// conversions agree with what the groups will actually store).
pub fn midH16(views: []const vres.Views, cfg: Config) bool {
    return groupHEnabled(midF16Enabled(), views, cfg);
}

pub const MidInput = union(enum) {
    host: []const f32,
    out0: void,
};

/// Mid groups for the resident streamMid sequence: like `run` but the result
/// stays on its pool buffer (out0 for the host-input group, feat1 for the
/// swapped group). h16 groups return their f16 handle unconverted; the
/// caller owns the boundary conversions.
pub fn runMidResident(
    ctx: *mconv.Context,
    stream: *Ctx,
    from: MidInput,
    views: []const vres.Views,
    cfg: Config,
    recycle: ?mbuffer.Recycle,
    strip_rows: u32,
) !ResidentOutput {
    if (views.len == 0) return error.InvalidShape;
    if (strip_rows == 0) return error.InvalidShape;
    try checkViews(views, cfg);
    const h16 = groupHEnabled(midF16Enabled(), views, cfg);
    var set = switch (from) {
        .host => |input| blk: {
            if (input.len != cfg.in_ch * cfg.height * cfg.width) return error.InvalidShape;
            break :blk try Set.make(
                stream,
                input,
                cfg,
                recycle,
                false,
                h16,
                fullHEnabled(),
                false,
                false,
            );
        },
        .out0 => try Set.makeSwapped(stream, cfg, h16, fullHEnabled(), false),
    };
    defer set.deinit();
    const sim = stream.sim_stages.mid;
    const final = try dispatchGroup(ctx, stream, &set, views, cfg, strip_rows, sim, h16);
    return .{ .handle = final, .f16 = h16 };
}

/// f16 pool feature to f32 in normbuf (no readback): the resident
/// mid-attention's f32 home. Returns the normbuf handle.
pub fn featureToF32(ctx: *mconv.Context, stream: *Ctx, src: *anyopaque, elems: u32) !*anyopaque {
    const dst = try stream.pool.normbuf.handle(stream.device, @as(usize, elems) * @sizeOf(f32));
    if (zdraw_metal_run_f16_to_f32(ctx.queue, stream.device, src, dst, elems) != 0) {
        return error.MetalDispatchFailed;
    }
    return dst;
}

/// f32 normbuf back to an f16 pool buffer (mid1's input in output0): the
/// plain RNE cast, matching the CPU @floatCast the Set upload performs.
pub fn featureToF16(
    ctx: *mconv.Context,
    stream: *Ctx,
    src: *anyopaque,
    dst: *anyopaque,
    elems: u32,
) !void {
    if (c.zdraw_metal_run_glue(
        ctx.queue,
        stream.f32_to_f16_pipe,
        src,
        dst,
        null,
        null,
        null,
        null,
        &elems,
        @sizeOf(u32),
        2,
        elems,
        256,
        0,
    ) != 0) return error.MetalDispatchFailed;
}

// One up block fed to the resident sequence: its 3 resnet views, the optional
// 2x upsampler conv weights (null only for the final up block), and the block's
// input shape. The runner derives the output shape (out_ch at the same h/w for a
// bare group, or the doubled h/w when an upsampler is present).
pub const UpView = struct {
    res: []const vres.Views,
    up_weight: ?tensor.View,
    up_bias: ?tensor.View,
    cfg: Config,
};

pub const ResidentOutput = struct {
    handle: *anyopaque,
    f16: bool,
};

// Run the whole up-block sequence as ONE resident flow: the VAE feature never
// leaves the GPU between up blocks. The first block's input is uploaded from the
// host once; every block runs its resnet group (dispatchGroup) and, when it has
// an upsampler, its fused 2x upsample+conv, with the upsample writing its result
// into the pool's feat1 - which is exactly the next block's resident input, so no
// readback or re-upload happens at the block boundary. Only the final block's
// result is read back into `out`. Bit-identical to running each block as a
// streamed group + mvup (the ZDRAW_VAE=raw path), with the per-up-block CPU
// round-trip removed. `out` is sized to the last block's output.
pub fn runUpBlocks(
    ctx: *mconv.Context,
    stream: *Ctx,
    input: []const f32,
    blocks: []const UpView,
    recycle: ?mbuffer.Recycle,
    strip_rows: u32,
    resident_first: bool,
) !ResidentOutput {
    if (blocks.len == 0) return error.InvalidShape;
    if (strip_rows == 0) return error.InvalidShape;

    const h16_sequence = vae_f16.enabled() and useHSequence(blocks);
    try presizeUpBlocks(stream, blocks, h16_sequence, fullHEnabled());
    const strip_plan = mvres_strip.planFor(stream, blocks, h16_sequence, fullHEnabled(), blockNormNeeded());
    const strip_up = strip_plan != null;
    var host_input = input;
    var host_recycle = recycle;
    var final: ?*anyopaque = null;
    for (blocks, 0..) |block, idx| {
        const block_start = metrics.now();
        if (block.res.len == 0) return error.InvalidShape;
        const cfg = block.cfg;
        try checkGroup(input.len, block, idx, resident_first);

        // Block 0 reads the host feature (upload); every later block's input is
        // already resident in feat1 from the prior block's upsample.
        const h16 = h16_sequence;
        const resident = idx != 0 or resident_first;
        var set = try Set.make(
            stream,
            host_input,
            cfg,
            host_recycle,
            resident,
            h16,
            fullHEnabled(),
            false,
            strip_up,
        );
        defer set.deinit();
        const stage_sim = idx < 4 and stream.sim_stages.up[idx];
        const res_start = metrics.now();
        const group_out = try dispatchGroup(ctx, stream, &set, block.res, cfg, strip_rows, stage_sim, h16);
        metrics.record(sup_res_time[@min(idx, sup_res_time.len - 1)], metrics.now() - res_start);
        if (block.up_weight) |up_weight| {
            // Resident fused upsample+conv: reads the group's GPU output (low-res,
            // pool.output0) via the 2x index map and writes the 2x result into
            // feat1 grown to the doubled resolution. feat1 (the input/slot-1 buffer)
            // is dead once the group finishes and must already hold the next block's
            // larger input, so it absorbs the 2x output for free. The result becomes
            // the next block's resident input - no CPU round-trip at the boundary.
            const up_start = metrics.now();
            _ = try dispatchUp(ctx, stream, group_out, cfg, up_weight, block.up_bias, strip_rows, stage_sim, h16);
            metrics.record(sup_conv_time[@min(idx, sup_conv_time.len - 1)], metrics.now() - up_start);
        } else {
            // Final block: no upsampler. Its result stays resident on its pool
            // buffer for the decode finish - no host round-trip.
            final = group_out;
        }
        // Later blocks take their input from the resident feat1, not the host.
        host_input = &.{};
        host_recycle = null;
        metrics.record(sup_time[@min(idx, sup_time.len - 1)], metrics.now() - block_start);
        metrics.memtrace(sup_stage[@min(idx, sup_stage.len - 1)]);
    }
    const out_handle = final orelse return error.InvalidShape;
    if (h16_sequence) {
        if (finalHEnabled()) return .{ .handle = out_handle, .f16 = true };
        // Boundary back to f32 for the finish pass: one conversion dispatch
        // into the (f32-sized) norm scratch.
        const last = blocks[blocks.len - 1];
        const last_elems = try product3U32(last.cfg.out_ch, last.cfg.height, last.cfg.width);
        const f32_bytes = @as(usize, last_elems) * @sizeOf(f32);
        const dst = try stream.pool.normbuf.handle(stream.device, f32_bytes);
        if (zdraw_metal_run_f16_to_f32(ctx.queue, stream.device, out_handle, dst, last_elems) != 0) {
            return error.MetalDispatchFailed;
        }
        return .{ .handle = dst, .f16 = false };
    }
    return .{ .handle = out_handle, .f16 = false };
}

// Grow every pool buffer to the sequence's high-water size BEFORE block 0. A
// ReuseBuf grow frees its old Metal buffer, but freed pages stay wired
// (phys_footprint keeps them), so growing inside the sequence abandons ~2 GB of
// wired intermediates at 1024px (feat1/output0/conv1_out/normbuf each double
// per block). Growing once up front caps pool wiring at the true high-water
// set; the per-block handle() calls then never reallocate, which also
// guarantees Set.make's resident_input cap assertion. Sizes mirror Set.make
// and dispatchUp exactly; dataflow and kernel order are untouched.
fn presizeUpBlocks(
    stream: *Ctx,
    blocks: []const UpView,
    h16: bool,
    full_h: bool,
) !void {
    var dims: [8]mvres_pool.BlockDims = undefined;
    if (blocks.len > dims.len) return error.InvalidShape;
    for (blocks, 0..) |b, i| dims[i] = .{
        .in_ch = b.cfg.in_ch,
        .out_ch = b.cfg.out_ch,
        .height = b.cfg.height,
        .width = b.cfg.width,
        .upsample = b.up_weight != null,
    };
    const plan = mvres_strip.planFor(stream, blocks, h16, full_h, blockNormNeeded());
    const block_norm = blockNormNeeded();
    const sizes = try mvres_pool.upBlockSizes(dims[0..blocks.len], h16, full_h, block_norm, finalHEnabled(), plan);
    try applySizes(stream, sizes);
}

fn reserveRole(stream: *Ctx, role: *mvres_pool.ReuseBuf, which: mlend.VaeRole, bytes: usize) !void {
    const offer = if (stream.pool.lender) |l| l.lend(.{ .vae = which }, bytes) else null;
    try role.reserve(stream.device, bytes, offer);
}

fn applySizes(stream: *Ctx, s: mvres_pool.Sizes) !void {
    const dev = stream.device;
    _ = try stream.pool.feat1.handle(dev, s.feat1);
    _ = try stream.pool.output0.handle(dev, s.output0);
    // conv1_out is per-group scratch: written and fully consumed inside each
    // group dispatch, never grown by the upsample, dead during the
    // mid-attention. It may therefore live in an idle buffer of another
    // phase for this decode (the Klein DiT `wide` slot or Z-Image's chain
    // `gateup`); the lender answers null when its buffer is too small
    // (strict's f32 conv1_out) and the own buffer is sized as before.
    if (s.conv1_out != 0) {
        const offer = if (stream.pool.lender) |l| l.lend(.{ .vae = .conv1_out }, s.conv1_out) else null;
        try stream.pool.conv1_out.reserve(dev, s.conv1_out, offer);
    }
    if (s.normbuf != 0) _ = try stream.pool.normbuf.handle(dev, s.normbuf);
    // The strip-memory roles take the same kind of lease as conv1_out did.
    const pool = &stream.pool;
    if (s.conv1_chunk != 0) try reserveRole(stream, &pool.conv1_chunk, .conv1_chunk, s.conv1_chunk);
    if (s.conv1_strip != 0) try reserveRole(stream, &pool.conv1_strip, .conv1_strip, s.conv1_strip);
    if (s.skip_strip != 0) try reserveRole(stream, &pool.skip_strip, .skip_strip, s.skip_strip);
    if (s.stash != 0) _ = try pool.stash.handle(dev, s.stash);
    if (s.norm_strip != 0) _ = try pool.norm_strip.handle(dev, s.norm_strip);
}

/// The up-sequence h16 predicate (exposed for the streamMid route: the mid
/// output's element type must agree with what block 0 will read).
pub fn upSequenceH16(blocks: []const UpView) bool {
    return vae_f16.enabled() and useHSequence(blocks);
}

/// Grow the pool to the WHOLE decode's high-water BEFORE mid0, so the
/// resident mid handoffs never reallocate a buffer holding the live feature
/// (the no-grow assertions in Set.make/makeSwapped then hold for the entire
/// decode). Up-block maxima dominate at every real resolution; the mid terms
/// are unioned in for completeness.
pub fn presizeDecode(
    stream: *Ctx,
    mid_cfg: Config,
    mid_h16: bool,
    blocks: []const UpView,
) !void {
    try presizeUpBlocks(stream, blocks, upSequenceH16(blocks), fullHEnabled());
    const dev = stream.device;
    const hw = try checkedMul(mid_cfg.height, mid_cfg.width);
    const el: usize = if (mid_h16) @sizeOf(f16) else @sizeOf(f32);
    const in_b = try checkedMul(try checkedMul(mid_cfg.in_ch, hw), el);
    const out_b = try checkedMul(try checkedMul(mid_cfg.out_ch, hw), el);
    // feat1 holds mid0's upload then mid1's output; output0 holds mid0's
    // output then mid1's input; normbuf is the attention's f32 home.
    _ = try stream.pool.feat1.handle(dev, @max(in_b, out_b));
    _ = try stream.pool.output0.handle(dev, @max(in_b, out_b));
    const c1_half = mid_h16 and (fullHEnabled() or false);
    const c1_el: usize = if (c1_half) @sizeOf(f16) else @sizeOf(f32);
    const c1_b = try checkedMul(try checkedMul(mid_cfg.out_ch, hw), c1_el);
    _ = try stream.pool.conv1_out.handle(dev, c1_b);
    const wide_ch = @max(mid_cfg.in_ch, mid_cfg.out_ch);
    const attn_home = try checkedMul(try checkedMul(wide_ch, hw), @sizeOf(f32));
    _ = try stream.pool.normbuf.handle(dev, attn_home);
}

// Per-up-block memtrace stages for the streamed path (parity with the untiled
// path's vae-up-N samples in vdecode).
const sup_stage = [_][]const u8{ "vae-sup-0", "vae-sup-1", "vae-sup-2", "vae-sup-3" };
const sup_time = [_][]const u8{ "vae-up-0", "vae-up-1", "vae-up-2", "vae-up-3" };
const sup_res_time = [_][]const u8{ "vae-res-0", "vae-res-1", "vae-res-2", "vae-res-3" };
const sup_conv_time = [_][]const u8{ "vae-upconv-0", "vae-upconv-1", "vae-upconv-2", "vae-upconv-3" };

// Resident fused 2x-upsample+conv over the group's GPU output. Writes the doubled
// (out_ch * 2h * 2w) result into the pool's feat1 and returns its handle. Shared
// by every up block in the sequence; the only per-block state (weight binds) is
// local. Bit-for-bit the mvup upsample2+conv2d that ZDRAW_VAE=raw runs.
fn dispatchUp(
    ctx: *mconv.Context,
    stream: *Ctx,
    group_out: *anyopaque,
    cfg: Config,
    up_weight: tensor.View,
    up_bias: ?tensor.View,
    strip_rows: u32,
    sim: bool,
    h16: bool,
) !*anyopaque {
    const up_h = try checkedMul(cfg.height, 2);
    const up_w = try checkedMul(cfg.width, 2);
    const dev = stream.device;
    const up_elems = try checkedProduct3(cfg.out_ch, up_h, up_w);
    const up_el: usize = if (h16) @sizeOf(f16) else @sizeOf(f32);
    const up_out = try stream.pool.feat1.handle(dev, try checkedMul(up_elems, up_el));
    if (group_out == up_out) return error.InvalidShape;

    var wtmp: ?mbuffer.Buffer = null;
    defer if (wtmp) |*b| b.deinit();
    var btmp: ?mbuffer.Buffer = null;
    defer if (btmp) |*b| b.deinit();
    const wb = try ctx.buffers.bindView(up_weight, &wtmp);
    const bb = try ctx.buffers.bindBias(up_bias, &btmp);
    var conv = try mconv.paramsFor(up_weight, up_bias, .{
        .in_ch = cfg.out_ch,
        .out_ch = cfg.out_ch,
        .height = up_h,
        .width = up_w,
        .kernel = 3,
        .pad = 1,
    }, wb, bb);
    // v7 upsample (same scalar-FMA kernel family, f32-only weight contract):
    // promote the bf16 conv weights once and re-point, mirroring promoteGroup.
    var up_wb = wb;
    const up_h16 = h16;
    var up_v7 = !up_h16 and std.c.getenv("ZDRAW_VAE_V7") != null;
    if (up_h16 and conv.dtype == 2) {
        up_wb = .{ .handle = try stream.promo_h.handleFor(dev, up_weight), .offset = 0 };
        conv.dtype = 1;
        conv.weight_offset = 0;
    } else if (up_v7 and conv.dtype == 2) {
        up_wb = .{ .handle = try stream.promo.handleFor(dev, up_weight), .offset = 0 };
        conv.dtype = 3;
        conv.weight_offset = 0;
    }
    const up_h_ok = up_h16 and conv.dtype == 1;
    if (up_h16 and !up_h_ok) return error.UnsupportedDType;
    up_v7 = up_v7 and conv.dtype == 3;

    var row0: u32 = 0;
    const up_h_u: u32 = try mres_util.toU32(up_h);
    if (up_h_ok and mvres_wino.enabled()) {
        const done = try mvres_wino.upsample(
            ctx.queue,
            &stream.pool.wino,
            stream.wino,
            dev,
            cfg,
            conv,
            up_h_u,
            try mres_util.toU32(up_w),
            group_out,
            up_wb.handle,
            bb.handle,
            up_out,
        );
        if (done) row0 = up_h_u; // whole map done: the strip loop is skipped
    }
    while (row0 < up_h_u) : (row0 += strip_rows) {
        const row1 = @min(row0 + strip_rows, up_h_u);
        const up_params = ConvUpsampleWindowParams{
            .channels = try mres_util.toU32(cfg.out_ch),
            .out_height = up_h_u,
            .out_width = try mres_util.toU32(up_w),
            .in_height = try mres_util.toU32(cfg.height),
            .in_width = try mres_util.toU32(cfg.width),
            .ksize = 3,
            .pad = 1,
            .dtype = conv.dtype,
            .bias_dtype = conv.bias_dtype,
            .has_bias = conv.has_bias,
            .weight_offset = conv.weight_offset,
            .bias_offset = conv.bias_offset,
            .row0 = row0,
            .row1 = row1,
        };
        // v2 (halo-tiled, bit-identical) when the out grid meets the contract.
        const up2_ok = up_w % 32 == 0 and cfg.out_ch % 8 == 0;
        const up_h8 = up_h_ok and up2_ok and
            convH8Ok(up_w, cfg.out_ch, cfg.out_ch, conv.weight_offset);
        if (zdraw_metal_run_conv2d_upsample_window(
            ctx.queue,
            if (up_h8)
                stream.up_h8_pipe
            else if (up_h_ok and up2_ok)
                stream.up_h_pipe
            else if (sim and up2_ok and up_v7)
                stream.up_sim_pipe
            else if (up2_ok and up_v7)
                stream.up_v7_pipe
            else if (up2_ok)
                stream.up_v3_pipe
            else
                stream.up_pipe,
            group_out,
            up_wb.handle,
            bb.handle,
            up_out,
            &up_params,
            stream.up_threads,
            if (up_h8) 2 else @intFromBool(up_h_ok or (up2_ok and !up_v7)),
        ) != 0) return error.MetalDispatchFailed;
    }
    return up_out;
}

fn groupHEnabled(requested: bool, views: []const vres.Views, cfg: Config) bool {
    if (!requested or !groupHContract(cfg)) return false;
    for (views) |view| {
        if (!resViewFitsH(view)) return false;
    }
    return true;
}

fn useHSequence(blocks: []const UpView) bool {
    for (blocks) |block| {
        if (!groupHEnabled(true, block.res, block.cfg)) return false;
        if (block.up_weight) |weight| {
            if (!weightFitsH(weight)) return false;
        }
    }
    return true;
}

fn groupHContract(cfg: Config) bool {
    return cfg.width % 32 == 0 and cfg.in_ch % 8 == 0 and cfg.out_ch % 8 == 0;
}

fn resViewFitsH(view: vres.Views) bool {
    if (!weightFitsH(view.conv1_w) or !weightFitsH(view.conv2_w)) return false;
    if (view.skip_w) |weight| {
        if (!weightFitsH(weight)) return false;
    }
    return true;
}

fn weightFitsH(view: tensor.View) bool {
    return view.dtype == .bf16 or view.dtype == .f16;
}

// Validate one up block's group the same way run()/check() does, but against the
// resident input shape: block 0 reads the host `input_len`; every later block's
// input is the prior block's upsampled output, which lives in the pool with its
// size derived from cfg by construction, so only the weight views need checking.
fn checkGroup(input_len: usize, block: UpView, idx: usize, resident_first: bool) !void {
    const cfg = block.cfg;
    if (idx == 0 and !resident_first and input_len != cfg.in_ch * cfg.height * cfg.width) {
        return error.InvalidShape;
    }
    try checkViews(block.res, cfg);
}

// Bind every block's weights, fill the StreamBuffers, and dispatch the resident
// resblock chain into one command buffer. Returns the final block's pool output
// handle (NOT read back), so callers can either read it or chain an upsample.
// Swap each block's bf16 conv weights for their one-time f32 promotions and
// re-point the params (see Promo). Returns true when the whole group ends up
// on the f32 contract that conv2d_prenorm_window_v7 requires.
// f16-weight promotion for the h-kernel group (lossless bf16 -> f16).
fn promoteGroupH(
    stream: *Ctx,
    views: []const vres.Views,
    ps: []params.ResParams,
    sbufs: []StreamBuffers,
) !bool {
    var ok = true;
    for (views, 0..) |view, idx| {
        if (ps[idx].conv1.dtype == 2) {
            sbufs[idx].conv1_w = try stream.promo_h.handleFor(stream.device, view.conv1_w);
            ps[idx].conv1.dtype = 1;
            ps[idx].conv1.weight_offset = 0;
        }
        if (ps[idx].conv2.dtype == 2) {
            sbufs[idx].conv2_w = try stream.promo_h.handleFor(stream.device, view.conv2_w);
            ps[idx].conv2.dtype = 1;
            ps[idx].conv2.weight_offset = 0;
        }
        if (ps[idx].has_skip != 0 and ps[idx].skip.dtype == 2) {
            if (view.skip_w) |sw| {
                sbufs[idx].skip_w = try stream.promo_h.handleFor(stream.device, sw);
                ps[idx].skip.dtype = 1;
                ps[idx].skip.weight_offset = 0;
            }
        }
        ok = ok and ps[idx].conv1.dtype == 1 and ps[idx].conv2.dtype == 1;
    }
    return ok;
}

fn promoteGroup(
    stream: *Ctx,
    views: []const vres.Views,
    ps: []params.ResParams,
    sbufs: []StreamBuffers,
) !bool {
    var ok = true;
    for (views, 0..) |view, idx| {
        if (ps[idx].conv1.dtype == 2) {
            sbufs[idx].conv1_w = try stream.promo.handleFor(stream.device, view.conv1_w);
            ps[idx].conv1.dtype = 3;
            ps[idx].conv1.weight_offset = 0;
        }
        if (ps[idx].conv2.dtype == 2) {
            sbufs[idx].conv2_w = try stream.promo.handleFor(stream.device, view.conv2_w);
            ps[idx].conv2.dtype = 3;
            ps[idx].conv2.weight_offset = 0;
        }
        ok = ok and ps[idx].conv1.dtype == 3 and ps[idx].conv2.dtype == 3;
    }
    return ok;
}

// Statistics pipes for one group. Hybrid is the conservative f16-VAE path:
// conv1_out stays f32, so the second norm reads f32; ZDRAW_VAE_FULL_H keeps
// conv1_out half and routes norm2 through the half kernels.
pub const StatsPipes = struct { stats: *anyopaque, stats2: ?*anyopaque, fast: bool };

fn statsPipes(stream: *Ctx, h_ok: bool, full_h: bool, sim: bool) StatsPipes {
    const sq = h_ok and statsSqEnabled();
    const fast = h_ok and (sq or stats512Enabled());
    const half = if (sq) stream.stats_h_sq_pipe else if (fast) stream.stats_h_fast_pipe else stream.stats_h_pipe;
    const f32p = if (sq) stream.stats_sq_pipe else if (fast) stream.stats_fast_pipe else stream.stats_pipe;
    const stats = if (h_ok) half else if (sim) stream.stats_sim_pipe else f32p;
    const stats2: ?*anyopaque = if (h_ok) (if (full_h) half else f32p) else null;
    return .{ .stats = stats, .stats2 = stats2, .fast = fast };
}

fn dispatchGroup(
    ctx: *mconv.Context,
    stream: *Ctx,
    set: *const Set,
    views: []const vres.Views,
    cfg: Config,
    strip_rows: u32,
    sim: bool,
    h16: bool,
) !*anyopaque {
    const alloc = std.heap.page_allocator;
    const temps = try alloc.alloc(bufs.Temps, views.len);
    @memset(temps, .{});
    // Free the temp uploads (LIFO: runs before the slice itself is released).
    defer alloc.free(temps);
    defer freeTemps(temps);

    const sbufs = try alloc.alloc(StreamBuffers, views.len);
    defer alloc.free(sbufs);
    const ps = try alloc.alloc(params.ResParams, views.len);
    defer alloc.free(ps);

    for (views, 0..) |view, idx| {
        const binds = try bufs.bindAll(ctx, view, &temps[idx]);
        ps[idx] = try params.make(view, binds, blockCfg(cfg, idx));
        sbufs[idx] = set.streamBuffers(idx, binds);
    }

    // v2 prenorm (halo-tiled, bit-identical) under its host contract; the VAE
    // always satisfies it (3x3 convs, widths %32, channels %8), but guard so
    // odd shapes fall back to the general kernel.
    const v2_ok = cfg.width % 32 == 0 and cfg.in_ch % 8 == 0 and cfg.out_ch % 8 == 0;
    // v7 (register-tiled scalar FMA, f32-only weight contract) behind
    // ZDRAW_VAE_V7: promote each block's bf16 conv weights to f32 once and
    // re-point the params; eligibility then requires dtype==3 across the
    // whole group, which the promotion guarantees.
    const h_ok = h16 and v2_ok and try promoteGroupH(stream, views, ps, sbufs);
    const v7_ok = !h16 and v2_ok and std.c.getenv("ZDRAW_VAE_V7") != null and
        try promoteGroup(stream, views, ps, sbufs);
    const full_h = h_ok and set.conv1_half;
    var h8 = full_h;
    for (ps, 0..) |p, idx| {
        const bc = blockCfg(cfg, idx);
        h8 = h8 and convH8Ok(bc.width, bc.in_ch, bc.out_ch, p.conv1.weight_offset) and
            convH8Ok(bc.width, bc.out_ch, bc.out_ch, p.conv2.weight_offset);
    }
    const hgraph = h_ok and set.norm_half;
    // v3 (4 simdgroups sharing the operand stages) when eligible, else v2/v1.
    const prenorm = if (h8)
        stream.prenorm_h8_pipe
    else if (h_ok)
        (if (full_h) stream.prenorm_h_pipe else stream.prenorm_hc1_pipe)
    else if (sim and v7_ok)
        stream.prenorm_sim_pipe
    else if (v7_ok)
        stream.prenorm_v7_pipe
    else if (v2_ok)
        stream.prenorm_v3_pipe
    else
        stream.prenorm_pipe;
    const sp = statsPipes(stream, h_ok, full_h, sim);
    const stats_pipe = sp.stats;
    const stats2_pipe = sp.stats2;
    const stats_fast = sp.fast;
    const conv_pipe = if (h_ok) stream.skip_h_pipe else if (sim) stream.conv_sim_pipe else stream.conv_pipe;
    const add_pipe = if (h_ok) stream.add_h_pipe else stream.add_pipe;
    const prenorm2_pipe: ?*anyopaque = if (h8)
        stream.prenorm_h8_pipe
    else if (h_ok)
        (if (full_h) stream.prenorm_h_pipe else stream.prenorm_hc2_pipe)
    else
        null;
    if (h16 and !h_ok) return error.UnsupportedDType;
    const wino_set: ?mvres_wino.Set = if (full_h and mvres_wino.enabled())
        try mvres_wino.set(&stream.pool.wino, stream.wino, ctx.device, cfg.in_ch, cfg.out_ch)
    else
        null;
    if (set.strip_up) {
        try mvres_strip.dispatchStrips(ctx, stream, sbufs, ps, cfg, sp, add_pipe, wino_set);
        return set.finalOutput(views.len);
    }
    if (zdraw_metal_run_vae_res_stream_chain(
        ctx.queue,
        stats_pipe,
        prenorm,
        conv_pipe,
        add_pipe,
        stream.apply_pipe,
        stream.apply_h_pipe,
        stream.conv7_pipe,
        stats2_pipe,
        prenorm2_pipe,
        sbufs.ptr,
        ps.ptr,
        views.len,
        strip_rows,
        if (stats_fast) 512 else stream.stats_threads,
        stream.conv_threads,
        stream.add_threads,
        if (h8) 2 else @intFromBool(h_ok or (v2_ok and !v7_ok)),
        @intFromBool(hgraph),
        if (wino_set) |*ws| ws else null,
    ) != 0) return error.MetalDispatchFailed;

    return set.finalOutput(views.len);
}

// Handles for one resident streamed group, all drawn from the Ctx's persistent
// pool (reused across groups). The feature map ping-pongs across output0/output1
// (block i's input is block i-1's output, dead once read); conv1_out stays full
// (stats2 + conv2's halo read it globally); skip is sized only when some block in
// the group needs the 1x1 projection. The two tiny per-group stats buffers are
// fresh (256 B each) - they hold the global GroupNorm [mean, scale] and must not
// alias across the chain's two norms.
const Set = struct {
    input: *anyopaque,
    output: [2]*anyopaque,
    conv1_out: *anyopaque,
    stats1: mbuffer.Buffer,
    stats2: mbuffer.Buffer,
    skip: *anyopaque,
    norm: ?*anyopaque,
    conv1_half: bool,
    norm_half: bool,
    // Strip-memory up block: conv1_out is untouched (its handle here is a
    // placeholder the strip chain never reads) and dispatchGroup routes to
    // mvres_strip_chain.
    strip_up: bool,

    // `resident_input` (set by the up-block sequence runner) means the residual
    // input is ALREADY in the pool's feat1 from the previous up-block's upsample,
    // so neither the host upload nor a recycle free happens: feat1 is sized to the
    // larger of this group's in/out, and the prior upsample already grew it to this
    // group's input size (the doubled-resolution out_ch*h*w == this in_ch*h*w), so
    // the handle() call returns the same buffer without reallocating - the resident
    // feature survives. The non-resident path (mid blocks, up-0) uploads the host
    // residual and drops the CPU copy at once. Slot 1 (feat1) is not written until
    // block 1, by which point block 0 has finished reading the input either way.
    fn make(
        stream: *Ctx,
        input: []const f32,
        cfg: Config,
        recycle: ?mbuffer.Recycle,
        resident_input: bool,
        h16: bool,
        full_h: bool,
        hgraph: bool,
        strip_up: bool,
    ) !Set {
        const dev = stream.device;
        const hw = cfg.height * cfg.width;
        // f16 feature mode (vae_f16): features store half; stats/norm scratch
        // stay f32; weights promote bf16->f16 losslessly.
        const el: usize = if (h16) @sizeOf(f16) else @sizeOf(f32);
        const out_bytes = cfg.out_ch * hw * el;
        const in_bytes = cfg.in_ch * hw * el;
        const stats_elems = 32 * 2; // groups=32 across the VAE; [mean, scale]/group.

        const prev_cap = stream.pool.feat1.cap;
        const feat1 = try stream.pool.feat1.handle(dev, @max(in_bytes, out_bytes));
        if (resident_input) {
            // The resident feature must not have been reallocated out from under us:
            // if feat1 grew here, the prior upsample's output (this group's input)
            // is gone. The sequence sizing guarantees no grow; assert it loudly.
            if (stream.pool.feat1.cap != prev_cap) return error.InvalidShape;
        } else if (h16) {
            // Host f32 -> f16 cast at the boundary (mid output enters the
            // f16 up-block region here).
            const alloc = std.heap.page_allocator;
            const tmp = try alloc.alloc(f16, cfg.in_ch * hw);
            defer alloc.free(tmp);
            for (tmp, input[0 .. cfg.in_ch * hw]) |*dst, src| dst.* = @floatCast(src);
            c.zdraw_metal_write_buffer(feat1, std.mem.sliceAsBytes(tmp).ptr, in_bytes);
            if (recycle) |r| r.allocator.free(r.input);
        } else {
            c.zdraw_metal_write_buffer(feat1, std.mem.sliceAsBytes(input).ptr, in_bytes);
            if (recycle) |r| r.allocator.free(r.input);
        }

        const out0 = try stream.pool.output0.handle(dev, out_bytes);
        // Hybrid f16 keeps conv1_out f32; FULL_H stores it as half and
        // uses half norm2/conv2 kernels, reducing traffic in the VAE hot path.
        const conv1_half = h16 and (full_h or hgraph);
        const c1_el: usize = if (conv1_half) @sizeOf(f16) else @sizeOf(f32);
        const c1_bytes = cfg.out_ch * hw * c1_el;
        const c1 = if (strip_up) feat1 else try stream.pool.conv1_out.handle(dev, c1_bytes);
        // Tier 2: block 0 writes output0 from the whole input; the later blocks
        // run in place in output0 (the strip chain sees output == input).
        const inplace = strip_up and mvres_strip.tier() >= 2;
        const norm_half = h16 and hgraph;
        const norm_el: usize = if (norm_half) @sizeOf(f16) else @sizeOf(f32);
        const normb: ?*anyopaque = if (blockNormNeeded())
            try stream.pool.normbuf.handle(dev, @max(cfg.in_ch, cfg.out_ch) * hw * norm_el)
        else
            null;
        // The 1x1 skip projection reuses conv1_out: conv2 has fully read conv1_out
        // before the skip conv writes it, and conv1_out is sized >= out_bytes
        // (it equals the skip's out_ch*hw, and is grown larger by the upsample).
        const skip = c1;
        var s1 = try mbuffer.Buffer.empty(dev, stats_elems * @sizeOf(f32));
        errdefer s1.deinit();
        const s2 = try mbuffer.Buffer.empty(dev, stats_elems * @sizeOf(f32));
        return .{
            .input = feat1,
            .output = if (inplace) .{ out0, out0 } else .{ out0, feat1 },
            .conv1_out = c1,
            .stats1 = s1,
            .stats2 = s2,
            .skip = skip,
            .norm = normb,
            .conv1_half = conv1_half,
            .norm_half = norm_half,
            .strip_up = strip_up,
        };
    }

    /// Mid1 variant: the input is ALREADY resident in output0 (the attention
    /// wrote it there), the output goes to feat1 (dead since mid0 consumed
    /// its upload), so the two roles swap. presizeDecode guarantees neither
    /// handle() call grows (a grow would drop the live input); assert it.
    fn makeSwapped(
        stream: *Ctx,
        cfg: Config,
        h16: bool,
        full_h: bool,
        hgraph: bool,
    ) !Set {
        const dev = stream.device;
        const hw = cfg.height * cfg.width;
        const el: usize = if (h16) @sizeOf(f16) else @sizeOf(f32);
        const out_bytes = cfg.out_ch * hw * el;
        const in_bytes = cfg.in_ch * hw * el;
        const in_cap = stream.pool.output0.cap;
        const in0 = try stream.pool.output0.handle(dev, in_bytes);
        if (stream.pool.output0.cap != in_cap) return error.InvalidShape;
        const f1_cap = stream.pool.feat1.cap;
        const feat1 = try stream.pool.feat1.handle(dev, out_bytes);
        if (stream.pool.feat1.cap != f1_cap) return error.InvalidShape;
        const conv1_half = h16 and (full_h or hgraph);
        const c1_el: usize = if (conv1_half) @sizeOf(f16) else @sizeOf(f32);
        const c1 = try stream.pool.conv1_out.handle(dev, cfg.out_ch * hw * c1_el);
        const norm_half = h16 and hgraph;
        const norm_el: usize = if (norm_half) @sizeOf(f16) else @sizeOf(f32);
        const normb: ?*anyopaque = if (blockNormNeeded())
            try stream.pool.normbuf.handle(dev, @max(cfg.in_ch, cfg.out_ch) * hw * norm_el)
        else
            null;
        var s1 = try mbuffer.Buffer.empty(dev, 32 * 2 * @sizeOf(f32));
        errdefer s1.deinit();
        const s2 = try mbuffer.Buffer.empty(dev, 32 * 2 * @sizeOf(f32));
        return .{
            .input = in0,
            .output = .{ feat1, in0 },
            .conv1_out = c1,
            .stats1 = s1,
            .stats2 = s2,
            .skip = c1,
            .norm = normb,
            .conv1_half = conv1_half,
            .norm_half = norm_half,
            .strip_up = false,
        };
    }

    fn streamBuffers(self: *const Set, idx: usize, binds: bufs.Binds) StreamBuffers {
        return .{
            .input = self.inputHandle(idx),
            .output = self.outputHandle(idx),
            .conv1_out = self.conv1_out,
            .stats1 = self.stats1.handle,
            .stats2 = self.stats2.handle,
            .skip = self.skip,
            .norm = self.norm,
            .norm1_w = binds.norm1_w.handle,
            .norm1_b = binds.norm1_b.handle,
            .conv1_w = binds.conv1_w.handle,
            .conv1_b = binds.conv1_b.handle,
            .norm2_w = binds.norm2_w.handle,
            .norm2_b = binds.norm2_b.handle,
            .conv2_w = binds.conv2_w.handle,
            .conv2_b = binds.conv2_b.handle,
            .skip_w = binds.skip_w.handle,
            .skip_b = binds.skip_b.handle,
        };
    }

    fn inputHandle(self: *const Set, idx: usize) *anyopaque {
        return if (idx == 0) self.input else self.outputHandle(idx - 1);
    }

    fn outputHandle(self: *const Set, idx: usize) *anyopaque {
        return self.output[idx % 2];
    }

    fn finalOutput(self: *const Set, count: usize) *anyopaque {
        return self.output[(count - 1) % 2];
    }

    // Only the two tiny per-group stats buffers are owned here; the full feature
    // buffers live in the Ctx pool and outlive the group.
    fn deinit(self: *Set) void {
        self.stats2.deinit();
        self.stats1.deinit();
    }
};

fn check(out: []const f32, input: []const f32, views: []const vres.Views, cfg: Config) !void {
    if (out.len != cfg.out_ch * cfg.height * cfg.width) return error.InvalidShape;
    if (input.len != cfg.in_ch * cfg.height * cfg.width) return error.InvalidShape;
    try checkViews(views, cfg);
}

fn checkViews(views: []const vres.Views, cfg: Config) !void {
    for (views, 0..) |view, idx| {
        try params.checkViews(view, blockCfg(cfg, idx));
    }
}

// Block 0 maps in_ch -> out_ch (and is the only one with a skip projection);
// every later block runs out_ch -> out_ch at the same spatial size.
pub fn blockCfg(cfg: Config, idx: usize) vres.Config {
    return .{
        .in_ch = if (idx == 0) cfg.in_ch else cfg.out_ch,
        .out_ch = cfg.out_ch,
        .height = cfg.height,
        .width = cfg.width,
    };
}

fn freeTemps(temps: []bufs.Temps) void {
    for (temps) |*t| t.deinit();
}

fn readBack(handle: *anyopaque, out: []f32) void {
    c.zdraw_metal_read_buffer(handle, std.mem.sliceAsBytes(out).ptr, out.len * @sizeOf(f32));
}

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidShape;
}

fn checkedProduct3(a: usize, b: usize, c_: usize) !usize {
    return checkedMul(try checkedMul(a, b), c_);
}

fn product3U32(a: usize, b: usize, c_: usize) !u32 {
    return mres_util.toU32(try checkedProduct3(a, b, c_));
}
