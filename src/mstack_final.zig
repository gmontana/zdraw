//! Chain the transformer stack directly into final projection.

const std = @import("std");

const c = @import("metal_c.zig");
const bufs = @import("mblock_chain_buf.zig");
const chain_c = @import("mblock_chain_c.zig");
const final_c = @import("mstack_final_c.zig");
const mattn = @import("mattn.zig");
const mbuffer = @import("mbuffer.zig");
const mfinal = @import("mfinal.zig");
const mlinear = @import("mlinear.zig");
const mstack = @import("mstack_chain.zig");
const policy = @import("mstack_policy.zig");
const tensor = @import("tensor.zig");
const zblock = @import("zblock.zig");
const zrope = @import("zrope.zig");

pub const Config = mstack.Config;

pub fn run(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    out: []f32,
    state: []const f32,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    final: Final,
    band: mstack.Band,
) !void {
    var stack = try makeStack(allocator, metal, attn, state, views, adaln, pos, cfg, rope, band);
    defer stack.deinit(allocator);
    var fset = try prepareFinal(metal, out.len, state.len, final);
    defer fset.deinit();
    try dispatch(metal, attn, stack, fset);
    readOutput(fset.output, out);
}

pub const Final = struct {
    scale: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    cfg: mfinal.Config,
};

fn dispatch(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    stack: Stack,
    final: FinalSet,
) !void {
    const threads = chain_c.Threads{
        .block = metal.block_threads,
        .qk = attn.qk_threads,
        .attn = stack.pipes.attn_threads,
        .attn_kernel = stack.pipes.attn_kernel,
        .swiglu = metal.swiglu_threads,
    };
    const code = final_c.zdraw_metal_run_stack_final(
        metal.queue,
        stack.pipes.norm,
        stack.pipes.resid,
        stack.pipes.gemm_exact,
        stack.pipes.gemm_half,
        stack.pipes.gemm_w8,
        stack.pipes.qk,
        stack.pipes.attn,
        stack.pipes.swiglu,
        metal.swiglu_fused_pipeline,
        stack.pipes.resid_norm,
        metal.final_pipeline orelse return error.GemmUnavailable,
        metal.gemm_bias_pipeline orelse return error.GemmUnavailable,
        &stack.bufs.c,
        stack.set.cweights.ptr,
        stack.set.params.ptr,
        stack.set.filled,
        &final.c,
        &final.params.norm,
        &final.params.gemm,
        &final.params.bias,
        &threads,
        metal.final_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn makeStack(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    state: []const f32,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    band: mstack.Band,
) !Stack {
    if (views.len == 0) return error.InvalidShape;
    try mstack.validateBand(band, views.len);
    const stack_policy = policy.fromEnv();
    const select = policy.selectFromEnv();
    const modes = policy.modes(
        metal.gemm_mode,
        stack_policy,
        select,
        band.index(0),
        band.total,
    );
    if (policy.allOff(modes)) return error.GemmUnavailable;
    const attn_pick = attn.pick(cfg.tokens, cfg.head_dim);
    const pipes = try mstack.getPipes(metal, attn, stack_policy, attn_pick);
    var block_bufs = try bufs.make(metal, state, views[0], cfg);
    errdefer block_bufs.deinit();
    const set = try mstack.makeSet(
        allocator,
        metal,
        views,
        adaln,
        pos,
        cfg,
        rope,
        stack_policy,
        select,
        .main,
        band,
    );
    return .{ .bufs = block_bufs, .set = set, .pipes = pipes };
}

pub fn prepareFinal(
    metal: *mlinear.Context,
    out_len: usize,
    state_len: usize,
    final: Final,
) !FinalSet {
    var temps = Temps{};
    errdefer temps.deinit();
    var scale = try mbuffer.Buffer.fromBytes(metal.device, std.mem.sliceAsBytes(final.scale));
    errdefer scale.deinit();
    // batch/output come from the persistent chain pool (rewritten every step),
    // so the per-step final stack does not accrue freed-buffer GPU wiring.
    const batch = try metal.pool.handle(metal.device, .final_batch, state_len * @sizeOf(f32));
    const output = try metal.pool.handle(metal.device, .final_out, out_len * @sizeOf(f32));
    const wb = try metal.buffers.bindView(final.weight, &temps.weight);
    const bb = try metal.buffers.bindView(final.bias, &temps.bias);
    const params = try mfinal.makeParams(final.weight, final.bias, wb, bb, final.cfg);
    return .{
        .scale = scale,
        .output = output,
        .temps = temps,
        .params = params,
        .c = .{
            .scale = scale.handle,
            .weight = wb.handle,
            .bias = bb.handle,
            .batch = batch,
            .output = output,
        },
    };
}

pub fn readOutput(handle: *anyopaque, out: []f32) void {
    c.zdraw_metal_read_buffer(
        handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

pub const FinalSet = struct {
    scale: mbuffer.Buffer,
    // batch/output live in the chain pool; only scale + temps are owned here.
    output: *anyopaque,
    temps: Temps,
    params: mfinal.Params,
    c: final_c.FinalBufs,

    pub fn deinit(self: *FinalSet) void {
        self.temps.deinit();
        self.scale.deinit();
    }
};

const Stack = struct {
    bufs: bufs.Bufs,
    set: mstack.Set,
    pipes: mstack.Pipes,

    fn deinit(self: *Stack, allocator: std.mem.Allocator) void {
        self.set.deinit(allocator);
        self.bufs.deinit();
    }
};

const Temps = struct {
    weight: ?mbuffer.Buffer = null,
    bias: ?mbuffer.Buffer = null,

    fn deinit(self: *Temps) void {
        if (self.bias) |*buf| buf.deinit();
        if (self.weight) |*buf| buf.deinit();
    }
};
