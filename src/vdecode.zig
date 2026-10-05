//! Z-Image VAE decoder in the official AutoencoderKL order.
const std = @import("std");
const conv_fast = @import("conv_fast.zig");
const env = @import("env.zig");
const mattn = @import("mattn.zig");
const mvattn = @import("mvattn.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const mlend = @import("mlend.zig");
const mlinear = @import("mlinear.zig");
const mvrchain = @import("mvrchain.zig");
const mvres = @import("mvres.zig");
const mvfinal = @import("mvfinal.zig");
const mvres_stream_chain = @import("mvres_stream_chain.zig");
const mvres_strip = @import("mvres_strip_chain.zig");
const mvup = @import("mvup.zig");
const upsample = @import("upsample.zig");
const metrics = @import("metrics.zig");
const vattn = @import("vattn.zig");
const vdecode_finish = @import("vdecode_finish.zig");
const vres = @import("vres.zig");
const vviews = @import("vviews.zig");
const Allocator = std.mem.Allocator;
pub const Config = struct { height: usize, width: usize };
const State = struct { data: []f32, ch: usize, h: usize, w: usize };
const up_ch = [_]usize{ 512, 512, 256, 128 };

// Reused norm/work arena for the CPU residual path, allocated once and sized to
// the largest norm (in_ch*h*w) and work (out_ch*h*w) any resblock needs, then
// handed to every CPU resblock instead of a fresh per-block alloc/free. On the
// Metal path the kernels keep their own GPU scratch, so this stays zero-sized.
const VaeScratch = struct {
    norm: []f32 = &.{},
    work: []f32 = &.{},

    fn init(allocator: Allocator, metal: ?*mconv.Context, cfg: Config) !VaeScratch {
        if (metal != null) return .{};
        const sizes = maxResBlock(cfg);
        const norm = try allocator.alloc(f32, sizes.norm);
        errdefer allocator.free(norm);
        const work = try allocator.alloc(f32, sizes.work);
        return .{ .norm = norm, .work = work };
    }

    fn deinit(self: *VaeScratch, allocator: Allocator) void {
        allocator.free(self.work);
        allocator.free(self.norm);
        self.* = undefined;
    }
};

const ResMax = struct { norm: usize, work: usize };

// Largest norm (in_ch*h*w) and work (out_ch*h*w) over every resblock: the two
// mid blocks plus each up block's three resnets at their doubling spatial size.
fn maxResBlock(cfg: Config) ResMax {
    var norm: usize = 512 * cfg.height * cfg.width;
    var work: usize = 512 * cfg.height * cfg.width;
    var h = cfg.height;
    var w = cfg.width;
    var prev: usize = 512;
    for (up_ch) |out_ch| {
        for (0..3) |idx| {
            const in_ch = if (idx == 0) prev else out_ch;
            norm = @max(norm, in_ch * h * w);
            work = @max(work, out_ch * h * w);
        }
        prev = out_ch;
        h *= 2;
        w *= 2;
    }
    return .{ .norm = norm, .work = work };
}

pub fn run(
    metal: ?*mconv.Context,
    linear: ?*mlinear.Context,
    attn: ?*mattn.Context,
    persistent: ?*mvres_stream_chain.Ctx,
    allocator: Allocator,
    out: []f32,
    latents: []const f32,
    views: vviews.Views,
    cfg: Config,
    // Idle buffers another phase lends the mid-attention for this decode
    // (the Klein DiT pool); null keeps every allocation in the chain pool.
    lender: ?mlend.Offer,
) !void {
    var t0 = metrics.now();
    var scratch = try VaeScratch.init(allocator, metal, cfg);
    defer scratch.deinit(allocator);

    // ZDRAW_VAE_STREAM (default OFF): route every VAE resblock through the
    // resident exact-streaming chain (mvres_stream_chain) instead of the resident
    // mvres / mvrchain path. The streamed convs fuse GroupNorm+SiLU and keep the
    // feature map ON-GPU across each group (no per-resnet CPU round-trip), with
    // the SAME hand-kernel math so the decode equals ZDRAW_VAE=raw bit-for-bit.
    // Only valid on the Metal path.
    // A caller-owned persistent ctx (the runtime's, reused across decodes so
    // the pool's GPU wiring is the high-water mark, not a per-decode accrual)
    // is preferred; otherwise create one for this decode only.
    var local_ctx: ?mvres_stream_chain.Ctx = null;
    defer if (local_ctx) |*s| s.deinit();
    if (persistent == null) {
        if (metal) |ctx| {
            if (streamEnabled()) local_ctx = try mvres_stream_chain.Ctx.init(ctx);
        }
    }
    const stream = persistent orelse if (local_ctx) |*s| s else null;
    // The lease ends with this decode: the offering pool may reshape after it.
    if (stream) |s| s.beginLease(lender);
    defer if (stream) |s| s.endLease();

    // The streamed decode's host feature buffers are large (up to ~1 GB at
    // 1024px) and short-lived; the std GPA keeps such freed pages mapped, so they
    // pile into phys_footprint as the resolution climbs. Route every internal VAE
    // host allocation through page_allocator on the streamed path so each freed
    // feature munmaps immediately, holding the host VAE footprint to what is
    // genuinely live. (The caller's `out` buffer is untouched.) Non-streamed paths
    // keep the caller's allocator.
    const vae_alloc = if (stream != null) std.heap.page_allocator else allocator;

    var state = try start(metal, vae_alloc, latents, views, cfg);
    defer vae_alloc.free(state.data);
    t0 = metrics.lap("vae-in", t0);
    dumpStage("conv_in", state.data);

    var mid_resident = false;
    if (stream) |s| {
        mid_resident = try streamMid(metal.?, s, linear, attn, lender, vae_alloc, &state, views);
    }
    if (!mid_resident) {
        try replaceRes(metal, stream, vae_alloc, &state, views.mid0, 512, &scratch);
        metrics.memtrace("vae-mid0");
        try replaceAttn(metal, linear, attn, lender, vae_alloc, &state, views.attn);
        metrics.memtrace("vae-attn");
        try replaceRes(metal, stream, vae_alloc, &state, views.mid1, 512, &scratch);
    }
    t0 = metrics.lap("vae-mid", t0);
    dumpStage("mid", state.data);
    metrics.memtrace("vae-mid");
    var final_feat: ?mvres_stream_chain.ResidentOutput = null;
    if (stream) |s| {
        // Resident up-block sequence: the feature never leaves the GPU between up
        // blocks (the prior block's upsample writes the pool's feat1, which is the
        // next block's resident input), so the per-up-block CPU round-trip - the
        // ~1 GB up-2->up-3 host feature at 1024px - is gone. The final feature is
        // not read back either; the finish reads its pool buffer in place.
        // Bit-identical to the per-block streamed path.
        final_feat = try streamUpBlocks(metal.?, s, vae_alloc, &state, &views.up, mid_resident);
    } else for (views.up, 0..) |up, idx| {
        try upBlock(metal, null, vae_alloc, &state, up, up_ch[idx], idx < 3, &scratch);
        memtraceUp(idx);
        var nm: [8]u8 = undefined;
        dumpStage(std.fmt.bufPrint(&nm, "up{d}", .{idx}) catch "up?", state.data);
    }
    t0 = metrics.lap("vae-up", t0);
    defer _ = metrics.lap("vae-finish", t0);
    // Streamed: the final feature is still resident on its pool buffer
    // (final_feat); the finish reads it in place (finishScratch).
    const strips = if (stream) |s|
        (if (final_feat) |f| mvres_strip.finishApplies(s, f.f16) else false)
    else
        false;
    const finish_scratch = if (strips) null else try finishScratch(stream, final_feat, &state);
    // A persistent ctx keeps its pool for the next decode (releasing would not
    // unwire the pages anyway; reuse makes the footprint the high-water mark).
    defer if (persistent == null) {
        if (stream) |s| s.releasePool();
    };
    if (strips) {
        const fc = mvfinal.Config{ .channels = state.ch, .height = state.h, .width = state.w };
        return mvres_strip.finishStrips(metal.?, stream.?, final_feat.?.handle, views, fc, out);
    }
    // Finish has a tiny readback; let the top-level defer own this last free.
    try vdecode_finish.run(
        metal,
        vae_alloc,
        out,
        state.data,
        state.ch,
        state.h,
        state.w,
        views,
        null,
        finish_scratch,
    );
}
// The whole-map finish's scratch: the final feature in place and the f32 norm
// scratch aliased onto a dead pool slot (residentFinish), so the finish faults
// in no new top-resolution buffer and the readback round-trip is gone.
fn finishScratch(
    stream: ?*mvres_stream_chain.Ctx,
    final_feat: ?mvres_stream_chain.ResidentOutput,
    state: *const State,
) !?mvfinal.Scratch {
    const s = stream orelse return null;
    const feat = final_feat orelse return null;
    const elems = state.ch * state.h * state.w;
    const in_elem: usize = if (feat.f16) @sizeOf(f16) else @sizeOf(f32);
    return try s.residentFinish(feat.handle, elems * in_elem, elems * @sizeOf(f32), feat.f16);
}

fn start(
    metal: ?*mconv.Context,
    allocator: Allocator,
    latents: []const f32,
    views: vviews.Views,
    cfg: Config,
) !State {
    if (cfg.height == 0 or cfg.width == 0) return error.InvalidShape;
    // Latent channel count comes from the checkpoint (conv_in's in dim):
    // 16 for Z-Image (with its scalar denorm), 32 for FLUX.2 Klein (whose
    // BN denorm happens upstream in zflux2_vae.prepare).
    const in_ch = views.conv_in_w.shape[1];
    if (latents.len != in_ch * cfg.height * cfg.width) return error.InvalidShape;

    const norm = try allocator.alloc(f32, latents.len);
    defer allocator.free(norm);
    if (in_ch == 16) denorm(norm, latents) else @memcpy(norm, latents);

    const out = try allocator.alloc(f32, 512 * cfg.height * cfg.width);
    errdefer allocator.free(out);
    try conv_fast.run(metal, out, norm, views.conv_in_w, views.conv_in_b, .{
        .in_ch = in_ch,
        .out_ch = 512,
        .height = cfg.height,
        .width = cfg.width,
        .kernel = 3,
        .pad = 1,
    });
    return .{ .data = out, .ch = 512, .h = cfg.height, .w = cfg.width };
}

/// The resident mid sequence: mid0 group, resident attention, mid1 group,
/// with the feature never leaving the GPU. mid1's output lands in feat1,
/// which is exactly runUpBlocks' resident block-0 input. Returns false
/// (state untouched) when the resident attention route is off or any mode/
/// shape contract fails up front, so the CPU-boundary path runs instead.
fn streamMid(
    ctx: *mconv.Context,
    s: *mvres_stream_chain.Ctx,
    linear: ?*mlinear.Context,
    attn: ?*mattn.Context,
    lender: ?mlend.Offer,
    allocator: Allocator,
    state: *State,
    views: vviews.Views,
) !bool {
    const lin = linear orelse return false;
    const att = attn orelse return false;
    const tokens = state.h * state.w;
    const mid_cfg = mvres_stream_chain.Config{
        .in_ch = state.ch,
        .out_ch = 512,
        .height = state.h,
        .width = state.w,
    };
    const mid0_views = [_]vres.Views{views.mid0};
    const mid1_views = [_]vres.Views{views.mid1};
    if (!streamMidReady(state, &mid0_views, &mid1_views, mid_cfg, views)) return false;
    const mid0_h16 = mvres_stream_chain.midH16(&mid0_views, mid_cfg);
    var blocks: [up_ch.len]mvres_stream_chain.UpView = undefined;
    buildUpViews(&blocks, 512, state.h, state.w, &views.up);
    try mvres_stream_chain.presizeDecode(s, mid_cfg, mid0_h16, blocks[0..views.up.len]);

    const input = state.data;
    state.data = &.{};
    const r0 = try mvres_stream_chain.runMidResident(
        ctx,
        s,
        .{ .host = input },
        &mid0_views,
        mid_cfg,
        recycleOf(allocator, input),
        mvres_stream_chain.stripRows(),
    );
    metrics.memtrace("vae-mid0");
    const elems: u32 = @intCast(512 * tokens);
    const feat_h = if (r0.f16)
        try mvres_stream_chain.featureToF32(ctx, s, r0.handle, elems)
    else
        r0.handle;
    try streamMidAttn(ctx, lin, att, lender, allocator, feat_h, views.attn, state.h, state.w);
    metrics.memtrace("vae-attn");
    if (r0.f16) try mvres_stream_chain.featureToF16(ctx, s, feat_h, r0.handle, elems);
    try streamMid1(ctx, s, &mid1_views, state);
    state.* = .{ .data = &.{}, .ch = 512, .h = state.h, .w = state.w };
    return true;
}

// mid1 on the swapped pool roles: input from output0 (where the attention
// left the feature), output into feat1 (the up sequence's resident input).
fn streamMid1(
    ctx: *mconv.Context,
    s: *mvres_stream_chain.Ctx,
    mid1_views: []const vres.Views,
    state: *State,
) !void {
    const mid1_cfg = mvres_stream_chain.Config{
        .in_ch = 512,
        .out_ch = 512,
        .height = state.h,
        .width = state.w,
    };
    _ = try mvres_stream_chain.runMidResident(
        ctx,
        s,
        .out0,
        mid1_views,
        mid1_cfg,
        null,
        mvres_stream_chain.stripRows(),
    );
}

// The resident attention leg of streamMid, with its SDPA boundary scratch.
fn streamMidAttn(
    ctx: *mconv.Context,
    lin: *mlinear.Context,
    att: *mattn.Context,
    lender: ?mlend.Offer,
    allocator: Allocator,
    feat_h: *anyopaque,
    attn_views: vattn.Views,
    h: usize,
    w: usize,
) !void {
    const count = 512 * h * w;
    const q = try allocator.alloc(f32, count);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, count);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, count);
    defer allocator.free(v);
    const mix = try allocator.alloc(f32, count);
    defer allocator.free(mix);
    try mvattn.runResident(ctx, lin, att, lender, feat_h, attn_views, .{
        .q = q,
        .k = k,
        .v = v,
        .mix = mix,
    }, .{ .channels = 512, .height = h, .width = w, .head_dim = 512 });
}

// The up-front streamMid contracts: resident attention on and eligible, and
// every element-size mode agreeing (mid0 == mid1 == the up sequence's block-0
// read; a mismatch would hand block 0 a wrong-typed feat1).
fn streamMidReady(
    state: *State,
    mid0_views: []const vres.Views,
    mid1_views: []const vres.Views,
    mid_cfg: mvres_stream_chain.Config,
    views: vviews.Views,
) bool {
    if (!mvattn.fits(state.ch, state.h * state.w)) return false;
    if (!env.flag("ZDRAW_VAE_ATTN_GPU", false)) return false;
    const mid0_h16 = mvres_stream_chain.midH16(mid0_views, mid_cfg);
    const mid1_h16 = mvres_stream_chain.midH16(mid1_views, mid_cfg);
    if (mid0_h16 != mid1_h16) return false;
    var blocks: [up_ch.len]mvres_stream_chain.UpView = undefined;
    buildUpViews(&blocks, 512, state.h, state.w, &views.up);
    return mid1_h16 == mvres_stream_chain.upSequenceH16(blocks[0..views.up.len]);
}

fn replaceRes(
    metal: ?*mconv.Context,
    stream: ?*mvres_stream_chain.Ctx,
    allocator: Allocator,
    state: *State,
    views: vres.Views,
    out_ch: usize,
    scratch: *VaeScratch,
) !void {
    const out = try allocator.alloc(f32, out_ch * state.h * state.w);
    errdefer allocator.free(out);
    const cfg = vres.Config{
        .in_ch = state.ch,
        .out_ch = out_ch,
        .height = state.h,
        .width = state.w,
    };
    // Detach the input so exactly one owner remains: the Metal/streamed runners
    // free it on GPU copy (recycle); the CPU runner reads then frees it here.
    const input = state.data;
    state.data = &.{};
    if (stream) |s| {
        const one = [_]vres.Views{views};
        try streamGroup(metal.?, s, out, input, &one, cfg, allocator);
    } else if (metal) |ctx| {
        try mvres.run(ctx, out, input, views, cfg, recycleOf(allocator, input));
    } else {
        const work = scratch.work[0..out.len];
        const sc = vres.Scratch{ .norm = scratch.norm[0..input.len], .work = work, .skip = work };
        try vres.run(null, out, input, views, sc, cfg);
        allocator.free(input);
    }
    state.* = .{ .data = out, .ch = out_ch, .h = state.h, .w = state.w };
}

// Run one resident streamed group (1 mid block or an up block's 3 resnets) GPU
// -> GPU, freeing the (recycled) input the instant it is uploaded. `cfg.out_ch`
// is the group's output channel count; block 0 maps state.ch -> out_ch, the rest
// run out_ch -> out_ch at the same spatial size.
fn streamGroup(
    ctx: *mconv.Context,
    stream: *mvres_stream_chain.Ctx,
    out: []f32,
    input: []f32,
    views: []const vres.Views,
    cfg: vres.Config,
    allocator: Allocator,
) !void {
    try mvres_stream_chain.run(ctx, stream, out, input, views, .{
        .in_ch = cfg.in_ch,
        .out_ch = cfg.out_ch,
        .height = cfg.height,
        .width = cfg.width,
    }, recycleOf(allocator, input), mvres_stream_chain.default_strip_rows);
}

fn recycleOf(allocator: Allocator, input: []f32) mbuffer.Recycle {
    return .{ .allocator = allocator, .input = input };
}

/// Resident mid-attention on chain-pool buffers (mvattn.zig). GroupNorm,
/// transposes, projections and the residual add stay on GPU; the SDPA
/// boundary scratch is allocated here (vdecode owns VAE CPU allocation).
fn residentAttn(
    conv: *mconv.Context,
    linear: *mlinear.Context,
    attn: *mattn.Context,
    lender: ?mlend.Offer,
    allocator: Allocator,
    state: *State,
    views: vattn.Views,
) !void {
    const count = state.data.len;
    const q = try allocator.alloc(f32, count);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, count);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, count);
    defer allocator.free(v);
    const mix = try allocator.alloc(f32, count);
    defer allocator.free(mix);
    try mvattn.run(conv, linear, attn, lender, state.data, views, .{
        .q = q,
        .k = k,
        .v = v,
        .mix = mix,
    }, attnCfg(state));
}

fn replaceAttn(
    conv: ?*mconv.Context,
    linear: ?*mlinear.Context,
    attn: ?*mattn.Context,
    lender: ?mlend.Offer,
    allocator: Allocator,
    state: *State,
    views: vattn.Views,
) !void {
    const count = state.data.len;
    // Resident route (ZDRAW_VAE_ATTN_GPU, set by the product VAE tier;
    // strict keeps the CPU-orchestrated path so its reference bytes are
    // unchanged). Eligibility is decided up front so there is no mid-flight
    // fallback with a stranded resident feature.
    if (conv != null and linear != null and attn != null and
        mvattn.fits(state.ch, state.h * state.w) and
        env.flag("ZDRAW_VAE_ATTN_GPU", false))
    {
        return residentAttn(conv.?, linear.?, attn.?, lender, allocator, state, views);
    }
    const out = try allocator.alloc(f32, count);
    errdefer allocator.free(out);
    const norm = try allocator.alloc(f32, count);
    defer allocator.free(norm);
    const q = try allocator.alloc(f32, count);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, count);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, count);
    defer allocator.free(v);
    const mix = try allocator.alloc(f32, count);
    defer allocator.free(mix);
    const scores = try allocator.alloc(f32, state.h * state.w);
    defer allocator.free(scores);

    try vattn.run(linear, attn, out, state.data, views, .{
        .norm = norm,
        .q = q,
        .k = k,
        .v = v,
        .mix = mix,
        .scores = scores,
    }, attnCfg(state));
    allocator.free(state.data);
    state.data = out;
}

// Assemble all VAE up blocks into ONE resident GPU sequence
// (mvres_stream_chain.runUpBlocks): the feature map never leaves the GPU between up
// blocks - each block's fused 2x upsample writes the pool's feat1, which is exactly
// the next block's resident input - so the per-up-block CPU round-trip (the ~1 GB
// up-2->up-3 host feature at 1024px) is gone. The final feature is NOT read back
// either: its resident pool handle is returned for the finish to read in place
// (state.data stays empty; only ch/h/w track the shape). up_ch gives each block's
// output channels; (in_ch, h, w) tracks the feature as it grows through the
// upsamplers. Bit-identical to the per-block streamed path.
// Fill the up-block views/configs from the current feature shape (shared by
// streamMid's mode check and the up-sequence runner).
fn buildUpViews(
    blocks: *[up_ch.len]mvres_stream_chain.UpView,
    start_ch: usize,
    start_h: usize,
    start_w: usize,
    ups: []const vviews.Up,
) void {
    var in_ch = start_ch;
    var h = start_h;
    var w = start_w;
    for (ups, 0..) |*up, idx| {
        const has_up = idx + 1 < ups.len;
        blocks[idx] = .{
            .res = &up.res,
            .up_weight = if (has_up) up.up_w else null,
            .up_bias = if (has_up) up.up_b else null,
            .cfg = .{ .in_ch = in_ch, .out_ch = up_ch[idx], .height = h, .width = w },
        };
        in_ch = up_ch[idx];
        if (has_up) {
            h *= 2;
            w *= 2;
        }
    }
}

fn streamUpBlocks(
    ctx: *mconv.Context,
    stream: *mvres_stream_chain.Ctx,
    allocator: Allocator,
    state: *State,
    ups: []const vviews.Up,
    resident_first: bool,
) !mvres_stream_chain.ResidentOutput {
    var blocks: [up_ch.len]mvres_stream_chain.UpView = undefined;
    buildUpViews(&blocks, state.ch, state.h, state.w, ups);
    var in_ch = state.ch;
    var h = state.h;
    var w = state.w;
    for (ups, 0..) |_, idx| {
        in_ch = up_ch[idx];
        if (idx + 1 < ups.len) {
            h *= 2;
            w *= 2;
        }
    }
    const input = state.data;
    state.data = &.{};
    const final = try mvres_stream_chain.runUpBlocks(
        ctx,
        stream,
        input,
        blocks[0..ups.len],
        recycleOf(allocator, input),
        mvres_stream_chain.stripRows(),
        resident_first,
    );
    state.* = .{ .data = &.{}, .ch = in_ch, .h = h, .w = w };
    metrics.memtrace("vae-up");
    return final;
}

fn upBlock(
    metal: ?*mconv.Context,
    stream: ?*mvres_stream_chain.Ctx,
    allocator: Allocator,
    state: *State,
    views: vviews.Up,
    out_ch: usize,
    has_up: bool,
    scratch: *VaeScratch,
) !void {
    if (metal) |ctx| {
        // The three resnets fuse into one resident dispatch, so the per-block
        // footprint peaks here (input + GPU chain working set + readback).
        try replaceResGroup(ctx, allocator, state, views.res, out_ch);
        memtraceRes(state.ch);
    } else for (views.res, 0..) |res, idx| {
        const res_ch = if (idx == 0) out_ch else state.ch;
        try replaceRes(metal, stream, allocator, state, res, res_ch, scratch);
        memtraceRes(state.ch);
    }
    if (has_up) try upsampleConv(metal, allocator, state, views);
}

fn replaceResGroup(
    ctx: *mconv.Context,
    allocator: Allocator,
    state: *State,
    views: [3]vres.Views,
    out_ch: usize,
) !void {
    const out = try allocator.alloc(f32, out_ch * state.h * state.w);
    errdefer allocator.free(out);
    const input = state.data;
    state.data = &.{};
    try mvrchain.run(ctx, out, input, views, .{
        .in_ch = state.ch,
        .out_ch = out_ch,
        .height = state.h,
        .width = state.w,
    }, recycleOf(allocator, input));
    state.* = .{ .data = out, .ch = out_ch, .h = state.h, .w = state.w };
}

fn upsampleConv(
    metal: ?*mconv.Context,
    allocator: Allocator,
    state: *State,
    views: vviews.Up,
) !void {
    const up_h = state.h * 2;
    const up_w = state.w * 2;
    const out = try allocator.alloc(f32, state.ch * up_h * up_w);
    errdefer allocator.free(out);
    const input = state.data;
    state.data = &.{};
    if (metal) |ctx| {
        // Recycle frees the (small) pre-upsample input as soon as it is on the
        // GPU, so it is gone before the 2x-larger upsample/conv working set.
        try mvup.run(ctx, out, input, views.up_w.?, views.up_b, .{
            .channels = state.ch,
            .height = state.h,
            .width = state.w,
        }, recycleOf(allocator, input));
    } else {
        const high = try allocator.alloc(f32, out.len);
        defer allocator.free(high);
        try upsample.run(high, input, .{
            .channels = state.ch,
            .height = state.h,
            .width = state.w,
        });
        allocator.free(input);
        try conv_fast.run(null, out, high, views.up_w.?, views.up_b, .{
            .in_ch = state.ch,
            .out_ch = state.ch,
            .height = up_h,
            .width = up_w,
            .kernel = 3,
            .pad = 1,
        });
    }
    state.* = .{ .data = out, .ch = state.ch, .h = up_h, .w = up_w };
}

// ZDRAW_VAE_DUMP_STAGES=<dir>: write each decode stage's feature map (raw
// f32) for oracle bisection. Debug instrument; zero cost when unset.
fn dumpStage(name: []const u8, data: []const f32) void {
    const dir = std.c.getenv("ZDRAW_VAE_DUMP_STAGES") orelse return;
    var buf: [512]u8 = undefined;
    const dir_s = std.mem.span(dir);
    const path = std.fmt.bufPrintZ(&buf, "{s}/zd_{s}.bin", .{ dir_s, name }) catch return;
    const fh = std.c.fopen(path, "wb") orelse return;
    const bytes = std.mem.sliceAsBytes(data);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, fh);
    _ = std.c.fclose(fh);
}

fn memtraceUp(idx: usize) void {
    if (!metrics.memtraceEnabled()) return;
    var buf: [32]u8 = undefined;
    const stage = std.fmt.bufPrint(&buf, "vae-up-{d}", .{idx}) catch return;
    metrics.memtrace(stage);
}

// Per-resblock footprint sample, labelled by output channel count.
fn memtraceRes(ch: usize) void {
    if (!metrics.memtraceEnabled()) return;
    var buf: [40]u8 = undefined;
    const stage = std.fmt.bufPrint(&buf, "vae-resblock-{d}ch", .{ch}) catch return;
    metrics.memtrace(stage);
}

// ZDRAW_VAE_STREAM=1 opts the whole VAE decode into the exact-streaming resblock
// path (mvres_stream). Default OFF; only the Metal backend honors it.
/// Lazily create the caller's persistent streaming ctx (one per runtime,
/// reused across decodes). Returns null when the streamed path is off.
pub fn ensureStream(
    metal: ?*mconv.Context,
    slot: *?mvres_stream_chain.Ctx,
) !?*mvres_stream_chain.Ctx {
    const ctx = metal orelse return null;
    if (!streamEnabled()) return null;
    if (slot.* == null) slot.* = try mvres_stream_chain.Ctx.init(ctx);
    return &slot.*.?;
}

fn streamEnabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_STREAM") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

pub fn denorm(out: []f32, latents: []const f32) void {
    for (out, latents) |*dst, value| dst.* = value / 0.3611 + 0.1159;
}
fn attnCfg(state: *const State) vattn.Config {
    return .{
        .channels = state.ch,
        .height = state.h,
        .width = state.w,
        .head_dim = state.ch,
    };
}
