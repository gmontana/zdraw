//! Optional Metal scaled dot-product attention.

const std = @import("std");

const attention = @import("../runtime/attention.zig");
const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const mqk = @import("mqk_shader.zig");
const mpipe = @import("mpipe.zig");
const mres_util = @import("mres_util.zig");
const block_shader = @import("mattn_block_shader.zig");
const shader = @import("mattn_shader.zig");

pub const max_tokens = 6144;
pub const flash_head_dim = 128;
pub const wide_head_dim = 512;

pub const Kernel = enum { rows, flash, block, wide };

fn kernelFromEnv() Kernel {
    const raw = std.c.getenv("ZDRAW_ATTN") orelse return .rows;
    const value = std.mem.span(raw);
    if (std.mem.eql(u8, value, "flash")) return .flash;
    if (std.mem.eql(u8, value, "block")) return .block;
    return .rows;
}

fn compileKernel(device: *anyopaque, kernel: Kernel, err: *[1024]u8) !*anyopaque {
    const entry: [*:0]const u8 = switch (kernel) {
        .rows => "attention_rows",
        .flash => "attention_flash_rows",
        .block => "attention_block16",
        .wide => "attention_flash_wide",
    };
    const source = switch (kernel) {
        .rows => shader.attn.ptr,
        .flash => shader.flash.ptr,
        .block => block_shader.block.ptr,
        .wide => shader.wide.ptr,
    };
    return c.zdraw_metal_compile(device, source, entry, err, err.len) orelse
        error.MetalCompileFailed;
}

pub const Context = struct {
    device: *anyopaque,
    queue: *anyopaque,
    pipeline: *anyopaque, // rows kernel: the preferred path under the cap
    flash_pipeline: *anyopaque, // uncapped path for long sequences
    block_pipeline: ?*anyopaque, // variant-B simdgroup-MMA kernel (head_dim 128)
    wide_pipeline: ?*anyopaque, // scalar flash variant for tokens>cap AND head_dim 129..512
    wide_mma_pipeline: ?*anyopaque, // MMA wide kernel (heads==1, head_dim==512, tokens%64==0)
    qk_pipeline: ?*anyopaque,
    kernel: Kernel,
    threads: usize,
    flash_threads: usize,
    wide_threads: usize,
    wide_mma_threads: usize,
    qk_threads: usize,
    // Hoisted env reads (the a080280 discipline: fixed after startup, and
    // widePick sits in the VAE dispatch path); also the test seam.
    wide_on: bool,

    pub fn init() !Context {
        return initKernel(kernelFromEnv());
    }

    pub fn initKernel(kernel: Kernel) !Context {
        const device = c.zdraw_metal_create_device() orelse return error.MetalNotAvailable;
        errdefer c.zdraw_metal_release_device(device);
        const queue = c.zdraw_metal_create_queue(device) orelse return error.MetalQueueFailed;
        errdefer c.zdraw_metal_release_queue(queue);
        var err: [1024]u8 = undefined;
        const rows_pipe = try compileKernel(device, .rows, &err);
        errdefer c.zdraw_metal_release_pipeline(rows_pipe);
        const flash_pipe = try compileKernel(device, .flash, &err);
        errdefer c.zdraw_metal_release_pipeline(flash_pipe);
        const block_pipe = compileKernel(device, .block, &err) catch null;
        errdefer if (block_pipe) |bp| c.zdraw_metal_release_pipeline(bp);
        const wide_pipe = mpipe.optional(device, shader.wide.ptr, "attention_flash_wide", &err);
        errdefer if (wide_pipe) |p| c.zdraw_metal_release_pipeline(p);
        const mma_src = shader.wide_mma.ptr;
        const wide_mma_pipe = mpipe.optional(device, mma_src, "attention_wide_mma", &err);
        errdefer if (wide_mma_pipe) |p| c.zdraw_metal_release_pipeline(p);
        const qk_pipe = mpipe.optional(device, mqk.qk.ptr, "qk_norm_rope", &err);
        errdefer if (qk_pipe) |p| c.zdraw_metal_release_pipeline(p);
        return .{
            .device = device,
            .queue = queue,
            .pipeline = rows_pipe,
            .flash_pipeline = flash_pipe,
            .block_pipeline = block_pipe,
            .wide_pipeline = wide_pipe,
            .wide_mma_pipeline = wide_mma_pipe,
            .qk_pipeline = qk_pipe,
            .kernel = kernel,
            .threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(rows_pipe)),
            .flash_threads = @min(
                flash_head_dim,
                c.zdraw_metal_pipeline_threads(flash_pipe),
            ),
            .wide_threads = if (wide_pipe) |p| @min(
                wide_head_dim,
                c.zdraw_metal_pipeline_threads(p),
            ) else 0,
            // Register receipt: rises as the kernel's register class falls;
            // widePick refuses the MMA pick when the PSO cannot resolve its
            // 128-thread groups (a register-fat compile would otherwise be
            // dispatched anyway and fail at encode time).
            .wide_mma_threads = if (wide_mma_pipe) |p|
                c.zdraw_metal_pipeline_threads(p)
            else
                0,
            .qk_threads = mpipe.optionalThreads(qk_pipe),
            .wide_on = wideForced(),
        };
    }

    pub fn deinit(self: *Context) void {
        if (self.block_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.wide_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.wide_mma_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.qk_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        c.zdraw_metal_release_pipeline(self.flash_pipeline);
        c.zdraw_metal_release_pipeline(self.pipeline);
        c.zdraw_metal_release_queue(self.queue);
        c.zdraw_metal_release_device(self.device);
        self.* = undefined;
    }

    pub const Pick = struct { kernel: Kernel, pipeline: *anyopaque, threads: usize };

    pub fn pick(self: *Context, tokens: usize, head_dim: usize) Pick {
        if (self.kernel == .block and head_dim == flash_head_dim) {
            if (self.block_pipeline) |bp| {
                return .{ .kernel = .block, .pipeline = bp, .threads = 64 };
            }
        }
        const flash_ok = head_dim <= flash_head_dim;
        const want_flash = self.kernel == .flash or tokens > max_tokens;
        if (flash_ok and want_flash) {
            return .{
                .kernel = .flash,
                .pipeline = self.flash_pipeline,
                .threads = self.flash_threads,
            };
        }
        return .{ .kernel = .rows, .pipeline = self.pipeline, .threads = self.threads };
    }

    pub fn run(
        self: *Context,
        out: []f32,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        cfg: attention.Config,
    ) !void {
        const s = Slices{ .out = out, .q = q, .k = k, .v = v };
        if (sdpaShape(cfg)) {
            if (self.widePick(cfg)) |p| return runWideInner(self, p, s, cfg);
            // Wide shapes without the owned wide kernels: the VAE mid-block
            // takes mvattn_owned (row-blocked GEMM attention) before this.
            return error.UnsupportedShape;
        }
        const picked = self.pick(cfg.tokens, cfg.head_dim);
        // Only attention_rows reads AttnParams.valid; routing a padded shape
        // anywhere else would silently drop the mask and corrupt the pad rows.
        if (cfg.valid != 0 and picked.kernel != .rows) return error.UnsupportedShape;
        try check(out, q, k, v, cfg, picked.kernel);
        return runFresh(self, .{ .kernel = picked }, s, cfg);
    }

    // Pooled twin of run() for the VAE mid-block (vattn.attend): activations
    // borrow the chain pool while the denoise chain is idle, because fresh
    // per-call MTLBuffers never return their GPU wiring on macOS. Slot roles
    // follow mres_buf: q/k/v in, mix out. Wide shapes ride the owned
    // deterministic kernels only when ZDRAW_ATTN_WIDE=1 (opt-in: the chunked
    // MMA kernel was byte-equal to the graph route but 2.9x slower on the
    // validate box, ledger attn-wide-mma-chunked 2026-08-25); the default is
    // run()'s fresh-buffer MPSGraph route.
    pub fn runPooled(
        self: *Context,
        metal: *mlinear.Context,
        out: []f32,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        cfg: attention.Config,
    ) !void {
        const picked = if (sdpaShape(cfg)) blk: {
            const wp = self.widePick(cfg) orelse return error.UnsupportedShape;
            try checkWide(out, q, k, v, cfg);
            break :blk wp;
        } else blk: {
            const p = self.pick(cfg.tokens, cfg.head_dim);
            try check(out, q, k, v, cfg, p.kernel);
            break :blk p;
        };
        const dev = metal.device;
        const pool = &metal.pool;
        const q_h = try pool.filled(dev, .q, std.mem.sliceAsBytes(q));
        const k_h = try pool.filled(dev, .k, std.mem.sliceAsBytes(k));
        const v_h = try pool.filled(dev, .v, std.mem.sliceAsBytes(v));
        const out_h = try pool.handle(dev, .mix, bytesF32(out.len));
        const params = try paramsFor(cfg);
        const code = c.zdraw_metal_run_attention(
            self.queue,
            picked.pipeline,
            q_h,
            k_h,
            v_h,
            out_h,
            &params,
            @intFromEnum(picked.kernel),
            picked.threads,
        );
        if (code != 0) return error.MetalDispatchFailed;
        c.zdraw_metal_read_buffer(out_h, std.mem.sliceAsBytes(out).ptr, bytesF32(out.len));
    }

    /// runPooled's dispatch on caller-owned GPU handles (the resident VAE
    /// mid-attention): identical kernel pick and wide/SDPA refusal logic, no
    /// pool fill and no readback. The caller owns handle shape correctness;
    /// an UnsupportedShape refusal leaves the handles untouched so the caller
    /// can fall back to the unchanged fresh-buffer route.
    pub fn runOnHandles(
        self: *Context,
        q_h: *anyopaque,
        k_h: *anyopaque,
        v_h: *anyopaque,
        out_h: *anyopaque,
        cfg: attention.Config,
    ) !void {
        const params = try paramsFor(cfg);
        const picked = if (sdpaShape(cfg))
            self.widePick(cfg) orelse return error.UnsupportedShape
        else
            self.pick(cfg.tokens, cfg.head_dim);
        const code = c.zdraw_metal_run_attention(
            self.queue,
            picked.pipeline,
            q_h,
            k_h,
            v_h,
            out_h,
            &params,
            @intFromEnum(picked.kernel),
            picked.threads,
        );
        if (code != 0) return error.MetalDispatchFailed;
    }

    // Owned wide-shape kernels, opt-in via ZDRAW_ATTN_WIDE=1; the promotion
    // campaign (ledger attn-wide-mma-chunked, 2026-08-25) killed the default
    // flip on speed. Preference order: the MMA kernel when its contract
    // holds (heads==1, head_dim==512, tokens%64==0 - every real mid-block
    // shape), else the scalar flash_wide (byte-equal to the SDPA modal
    // render but ~9x slower; ledger 2026-08-04), else null -> MPSGraph.
    fn widePick(self: *Context, cfg: attention.Config) ?Pick {
        if (!self.wide_on) return null;
        if (cfg.heads != 1) return null;
        if (self.wide_mma_pipeline) |mp| {
            if (cfg.head_dim == wide_head_dim and cfg.tokens % 64 == 0 and
                self.wide_mma_threads >= 128)
            {
                return .{ .kernel = .wide, .pipeline = mp, .threads = 128 };
            }
        }
        const wp = self.wide_pipeline orelse return null;
        if (cfg.head_dim > wide_head_dim) return null;
        if (self.wide_threads != wide_head_dim) return null;
        return .{ .kernel = .wide, .pipeline = wp, .threads = self.wide_threads };
    }
};

// Wide heads past the token cap (the VAE mid-block at 1024px+): the one-shot
// SDPA graph by default; the owned wide kernels are opt-in (widePick).
fn sdpaShape(cfg: attention.Config) bool {
    return !cfg.causal and cfg.heads == cfg.kv_heads and
        cfg.tokens > max_tokens and cfg.head_dim > flash_head_dim;
}

// ZDRAW_ATTN_WIDE=1: opt into the owned flash_wide kernel for wide shapes.
fn wideForced() bool {
    const raw = std.c.getenv("ZDRAW_ATTN_WIDE") orelse return false;
    return raw[0] == '1';
}

test "widePick routes the wide shapes by contract" {
    var ctx = Context.init() catch return; // no Metal: nothing to route
    defer ctx.deinit();
    ctx.wide_on = true;
    const mma_shape = attention.Config{
        .tokens = 6208,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 512,
        .causal = false,
    };
    if (ctx.wide_mma_pipeline != null) {
        // The register receipt and the guard's premise: the PSO must resolve
        // the 128-thread groups the grid formula assumes.
        try std.testing.expect(ctx.wide_mma_threads >= 128);
        const p = ctx.widePick(mma_shape) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(@as(usize, 128), p.threads);
    }
    if (ctx.wide_pipeline != null and ctx.wide_threads == wide_head_dim) {
        // %16 but not %64: must fall to the scalar flash_wide, never MPSGraph.
        var odd = mma_shape;
        odd.tokens = 6160;
        const p = ctx.widePick(odd) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(@as(usize, wide_head_dim), p.threads);
    }
    ctx.wide_on = false;
    try std.testing.expect(ctx.widePick(mma_shape) == null);
}

fn checkWide(
    out: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    cfg: attention.Config,
) !void {
    if (cfg.tokens == 0 or cfg.heads == 0 or cfg.kv_heads == 0) return error.InvalidShape;
    if (cfg.heads != 1) return error.UnsupportedShape;
    if (cfg.head_dim == 0 or cfg.head_dim > wide_head_dim) return error.UnsupportedShape;
    try checkLens(out, q, k, v, cfg);
}

fn runWideInner(
    self: *Context,
    p: Context.Pick,
    s: Slices,
    cfg: attention.Config,
) !void {
    try checkWide(s.out, s.q, s.k, s.v, cfg);
    return runFresh(self, .{ .kernel = p }, s, cfg);
}

const Slices = struct { out: []f32, q: []const f32, k: []const f32, v: []const f32 };

const Dispatch = union(enum) { kernel: Context.Pick };

// Shared fresh-buffer body: upload q/k/v, dispatch one kernel, read the
// result back. Buffer lifetimes live here and nowhere else.
fn runFresh(self: *Context, d: Dispatch, s: Slices, cfg: attention.Config) !void {
    var q_buf = try mbuffer.Buffer.fromBytes(self.device, std.mem.sliceAsBytes(s.q));
    defer q_buf.deinit();
    var k_buf = try mbuffer.Buffer.fromBytes(self.device, std.mem.sliceAsBytes(s.k));
    defer k_buf.deinit();
    var v_buf = try mbuffer.Buffer.fromBytes(self.device, std.mem.sliceAsBytes(s.v));
    defer v_buf.deinit();
    var out_buf = try mbuffer.Buffer.empty(self.device, std.mem.sliceAsBytes(s.out).len);
    defer out_buf.deinit();
    const params = try paramsFor(cfg);
    const code = switch (d) {
        .kernel => |p| c.zdraw_metal_run_attention(
            self.queue,
            p.pipeline,
            q_buf.handle,
            k_buf.handle,
            v_buf.handle,
            out_buf.handle,
            &params,
            @intFromEnum(p.kernel),
            p.threads,
        ),
    };
    if (code != 0) return error.MetalDispatchFailed;
    c.zdraw_metal_read_buffer(out_buf.handle, std.mem.sliceAsBytes(s.out).ptr, bytesF32(s.out.len));
}

fn check(
    out: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    cfg: attention.Config,
    kernel: Kernel,
) !void {
    if (cfg.tokens == 0) return error.UnsupportedShape;
    switch (kernel) {
        // pick()/widePick() already route by shape; these are backstops.
        .rows => if (cfg.tokens > max_tokens) return error.UnsupportedShape,
        .flash => if (cfg.head_dim > flash_head_dim) return error.UnsupportedShape,
        .block => if (cfg.head_dim != flash_head_dim) return error.UnsupportedShape,
        .wide => if (cfg.head_dim > wide_head_dim) return error.UnsupportedShape,
    }
    if (cfg.heads == 0 or cfg.kv_heads == 0 or cfg.head_dim == 0) return error.InvalidShape;
    try checkLens(out, q, k, v, cfg);
}

fn checkLens(
    out: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    cfg: attention.Config,
) !void {
    // Self-contained: callers may reach this with no prior guard, so the
    // zero check must precede the modulo (kv_heads 0 would divide by zero).
    if (cfg.heads == 0 or cfg.kv_heads == 0 or cfg.head_dim == 0) return error.InvalidShape;
    if (cfg.heads % cfg.kv_heads != 0) return error.InvalidShape;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    if (out.len != q_len or q.len != q_len) return error.InvalidShape;
    if (k.len != kv_len or v.len != kv_len) return error.InvalidShape;
}

fn paramsFor(cfg: attention.Config) !c.AttnParams {
    return .{
        .tokens = try mres_util.toU32(cfg.tokens),
        .heads = try mres_util.toU32(cfg.heads),
        .kv_heads = try mres_util.toU32(cfg.kv_heads),
        .head_dim = try mres_util.toU32(cfg.head_dim),
        .causal = if (cfg.causal) 1 else 0,
        .valid = try mres_util.toU32(cfg.valid),
    };
}

fn bytesF32(count: usize) usize {
    return count * @sizeOf(f32);
}
