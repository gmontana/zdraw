//! Tests for the Metal attention fast path.

const std = @import("std");

const attention = @import("../runtime/attention.zig");
const mattn = @import("mattn.zig");

test "Metal attention matches CPU attention" {
    var ctx = mattn.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();

    const cfg = attention.Config{
        .tokens = 2,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .causal = false,
    };
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const k = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const v = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    var got = [_]f32{0.0} ** 4;
    var want = [_]f32{0.0} ** 4;
    var scratch = [_]f32{0.0} ** 2;

    try ctx.run(&got, &q, &k, &v, cfg);
    try attention.run(&want, &q, &k, &v, &scratch, cfg);
    for (got, want) |g, w| try std.testing.expectApproxEqAbs(w, g, 0.0001);
}

// Tile tails, multi-tile rows, causal masking, GQA, and the full 128 width.
const flash_cases = [_]attention.Config{
    .{ .tokens = 1, .heads = 2, .kv_heads = 2, .head_dim = 8, .causal = false },
    .{ .tokens = 31, .heads = 2, .kv_heads = 1, .head_dim = 16, .causal = false },
    .{ .tokens = 32, .heads = 1, .kv_heads = 1, .head_dim = 128, .causal = false },
    .{ .tokens = 97, .heads = 3, .kv_heads = 3, .head_dim = 64, .causal = true },
    .{ .tokens = 257, .heads = 2, .kv_heads = 2, .head_dim = 128, .causal = false },
    // Wide heads (the VAE shape class) must route to the rows kernel.
    .{ .tokens = 16, .heads = 1, .kv_heads = 1, .head_dim = 256, .causal = false },
};

// Variant-B block kernel: head_dim 128 only, non-causal DiT shapes
// (tails within and across 16-row query blocks and 32-token KV tiles).
const block_cases = [_]attention.Config{
    .{ .tokens = 16, .heads = 1, .kv_heads = 1, .head_dim = 128, .causal = false },
    .{ .tokens = 48, .heads = 2, .kv_heads = 2, .head_dim = 128, .causal = false },
    .{ .tokens = 53, .heads = 3, .kv_heads = 3, .head_dim = 128, .causal = false },
    .{ .tokens = 288, .heads = 4, .kv_heads = 4, .head_dim = 128, .causal = false },
    .{ .tokens = 289, .heads = 2, .kv_heads = 2, .head_dim = 128, .causal = false },
};

test "block attention matches CPU across shapes" {
    var block = mattn.Context.initKernel(.block) catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer block.deinit();
    if (block.block_pipeline == null) return error.BlockKernelCompileFailed;
    for (block_cases) |cfg| try expectBlockCase(&block, cfg);
}

fn expectBlockCase(block: *mattn.Context, cfg: attention.Config) !void {
    const alloc = std.testing.allocator;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    const q = try alloc.alloc(f32, q_len);
    defer alloc.free(q);
    const k = try alloc.alloc(f32, kv_len);
    defer alloc.free(k);
    const v = try alloc.alloc(f32, kv_len);
    defer alloc.free(v);
    var prng = std.Random.DefaultPrng.init(cfg.tokens * 31 + cfg.heads);
    const rng = prng.random();
    for (q) |*x| x.* = rng.floatNorm(f32) * 0.5;
    for (k) |*x| x.* = rng.floatNorm(f32) * 0.5;
    for (v) |*x| x.* = rng.floatNorm(f32) * 0.5;
    const got = try alloc.alloc(f32, q_len);
    defer alloc.free(got);
    const want = try alloc.alloc(f32, q_len);
    defer alloc.free(want);
    const scratch = try alloc.alloc(f32, cfg.tokens);
    defer alloc.free(scratch);
    try attention.run(want, q, k, v, scratch, cfg);
    try block.run(got, q, k, v, cfg);
    for (got, want) |g, w| try std.testing.expectApproxEqAbs(w, g, 0.002);
}

test "flash attention matches the rows kernel and CPU across shapes" {
    var flash = mattn.Context.initKernel(.flash) catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer flash.deinit();
    var rows = try mattn.Context.initKernel(.rows);
    defer rows.deinit();
    for (flash_cases) |cfg| try expectFlashCase(&flash, &rows, cfg);
}

fn expectFlashCase(flash: *mattn.Context, rows: *mattn.Context, cfg: attention.Config) !void {
    const allocator = std.testing.allocator;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    const q = try randomSlice(allocator, q_len, 1);
    defer allocator.free(q);
    const k = try randomSlice(allocator, kv_len, 2);
    defer allocator.free(k);
    const v = try randomSlice(allocator, kv_len, 3);
    defer allocator.free(v);
    const got = try allocator.alloc(f32, q_len);
    defer allocator.free(got);
    const ref = try allocator.alloc(f32, q_len);
    defer allocator.free(ref);
    const want = try allocator.alloc(f32, q_len);
    defer allocator.free(want);
    const scratch = try allocator.alloc(f32, cfg.tokens);
    defer allocator.free(scratch);

    try flash.run(got, q, k, v, cfg);
    try rows.run(ref, q, k, v, cfg);
    try attention.run(want, q, k, v, scratch, cfg);
    for (got, ref, want) |g, r, w| {
        try std.testing.expectApproxEqAbs(r, g, 0.001); // flash vs rows
        try std.testing.expectApproxEqAbs(w, g, 0.001); // flash vs CPU
    }
}

fn randomSlice(allocator: std.mem.Allocator, len: usize, seed: u64) ![]f32 {
    const out = try allocator.alloc(f32, len);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (out) |*value| value.* = random.floatNorm(f32);
    return out;
}

test "chunked-D wide mma matches flash_wide and a sampled reference" {
    var ctx = mattn.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();
    if (ctx.wide_mma_pipeline == null or ctx.wide_pipeline == null) return;
    ctx.wide_on = true;
    const alloc = std.testing.allocator;
    // Smallest %64 shape past the tokens>6144 wide-route threshold; the last
    // rows are the grid-formula canary (an off-by-one leaves them unwritten).
    const tokens = 6208;
    const dim = 512;
    const n = tokens * dim;
    const q = try alloc.alloc(f32, n);
    defer alloc.free(q);
    const k = try alloc.alloc(f32, n);
    defer alloc.free(k);
    const v = try alloc.alloc(f32, n);
    defer alloc.free(v);
    var prng = std.Random.DefaultPrng.init(0x51de);
    const rand = prng.random();
    for (q) |*x| x.* = rand.floatNorm(f32) * 0.5;
    for (k) |*x| x.* = rand.floatNorm(f32) * 0.5;
    for (v) |*x| x.* = rand.floatNorm(f32) * 0.5;
    const cfg = attention.Config{
        .tokens = tokens,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = dim,
        .causal = false,
    };

    const got = try alloc.alloc(f32, n);
    defer alloc.free(got);
    try ctx.run(got, q, k, v, cfg);

    // Determinism: a second run must be byte-identical.
    const again = try alloc.alloc(f32, n);
    defer alloc.free(again);
    try ctx.run(again, q, k, v, cfg);
    try std.testing.expect(std.mem.eql(
        u8,
        std.mem.sliceAsBytes(got),
        std.mem.sliceAsBytes(again),
    ));

    // Near-byte agreement with the certified scalar flash_wide over the FULL
    // output (also catches any region the grid formula left unwritten).
    const fw = try alloc.alloc(f32, n);
    defer alloc.free(fw);
    const saved = ctx.wide_mma_pipeline;
    ctx.wide_mma_pipeline = null;
    try ctx.run(fw, q, k, v, cfg);
    ctx.wide_mma_pipeline = saved;
    for (got, fw) |a, b| {
        try std.testing.expectApproxEqAbs(b, a, 2e-3);
    }

    // Sampled-row CPU reference (natural exp; the kernel's base-2 form is
    // the same function): both row-blocks of the first, a middle, and the
    // LAST threadgroup.
    const rows = [_]usize{ 0, 7, 8, 15, 16, 3103, 6191, 6207 };
    const logits = try alloc.alloc(f32, tokens);
    defer alloc.free(logits);
    const scale = 1.0 / @sqrt(@as(f32, dim));
    for (rows) |row| {
        var m: f32 = -std.math.inf(f32);
        for (0..tokens) |j| {
            var dot: f32 = 0.0;
            for (0..dim) |d| dot += q[row * dim + d] * k[j * dim + d];
            const s = dot * scale;
            logits[j] = s;
            m = @max(m, s);
        }
        var denom: f32 = 0.0;
        for (logits) |*s| {
            s.* = @exp(s.* - m);
            denom += s.*;
        }
        for (0..dim) |d| {
            var acc: f32 = 0.0;
            for (0..tokens) |j| acc += logits[j] * v[j * dim + d];
            try std.testing.expectApproxEqAbs(acc / denom, got[row * dim + d], 2e-3);
        }
    }
}
