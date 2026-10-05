//! Substrate-decision microbenchmark: our simdgroup_matrix GEMM vs Apple MPS.
//!
//! For each Z-Image DiT matmul shape it times our kernel and MPS, reports
//! GFLOP/s + the ratio, and checks both against a CPU reference. The one number
//! that decides whether we own the kernels or fall back to a framework. Dev-only,
//! like refcheck/bench.

const std = @import("std");

const c = @import("src/metal_c.zig");
const mbuffer = @import("src/mbuffer.zig");
const mgemm_shader = @import("src/mgemm_shader.zig");
const zflux2 = @import("src/zflux2.zig");
const qenc = @import("src/qwen_encoder.zig");
const mattn_b = @import("src/mattn.zig");
const attention = @import("src/attention.zig");
const metal_c = @import("src/metal_c.zig");
const zflux2_dit = @import("src/zflux2_dit.zig");
const zflux2_schedule = @import("src/zflux2_schedule.zig");
const zflux2_vae = @import("src/zflux2_vae.zig");
const zflux2_res = @import("src/zflux2_resident.zig");
const vviews = @import("src/vviews.zig");
const vdecode = @import("src/vdecode.zig");
const tensor_file_b = @import("src/tensor_file.zig");
const mconv_b = @import("src/mconv.zig");
const mlinear = @import("src/mlinear.zig");
const qscratch = @import("src/qwen_scratch.zig");
const shards = @import("src/shards.zig");
const weight_index = @import("src/weight_index.zig");
const zconfig = @import("src/zimage_config.zig");
const mgemm_bench_shader = @import("src/mgemm_bench_shader.zig");
const mconv_shader = @import("src/mconv_shader.zig");
const mshader = @import("src/mshader.zig");
const mps_c = @import("src/mps_c.zig");
const mvres_param = @import("src/mvres_param.zig");
const mvres_stream = @import("src/mvres_stream.zig");
const mvres_chain = @import("src/mvres_stream_chain.zig");
const mconv_h8 = @import("src/mconv_h8_shader.zig");
const mvattn_owned = @import("src/mvattn_owned.zig");
const mconv_wino = @import("src/mconv_wino_shader.zig");
const mgemm_mpp = @import("src/mgemm_mpp_shader.zig");
const tensor = @import("src/tensor.zig");
const vnorm_shader = @import("src/mvnorm_shader.zig");
const zw2 = @import("src/zw2.zig");
const zw4 = @import("src/zw4.zig");
const zw6 = @import("src/zw6.zig");

const Shape = struct { m: u32, k: u32, n: u32 };
const NamedShape = struct { label: []const u8, shape: Shape };

const w6_zimage_shapes = [_]NamedShape{
    .{ .label = "zimage.gateup", .shape = .{ .m = 4128, .k = 3840, .n = 10240 } },
    .{ .label = "zimage.down  ", .shape = .{ .m = 4128, .k = 10240, .n = 3840 } },
    .{ .label = "zimage.qkv   ", .shape = .{ .m = 4128, .k = 3840, .n = 3840 } },
};

const w6_klein_shapes = [_]NamedShape{
    .{ .label = "single.qkv   ", .shape = .{ .m = 4608, .k = 3072, .n = 3072 } },
    .{ .label = "single.gateup", .shape = .{ .m = 4608, .k = 3072, .n = 18432 } },
    .{ .label = "single.out   ", .shape = .{ .m = 4608, .k = 12288, .n = 3072 } },
    .{ .label = "double.qkv   ", .shape = .{ .m = 4096, .k = 3072, .n = 3072 } },
    .{ .label = "double.gateup", .shape = .{ .m = 4096, .k = 3072, .n = 18432 } },
    .{ .label = "double.down  ", .shape = .{ .m = 4096, .k = 9216, .n = 3072 } },
};

const shapes = [_]Shape{
    .{ .m = 288, .k = 3840, .n = 10240 }, // 256px gate+up
    .{ .m = 288, .k = 10240, .n = 3840 }, // 256px down
    .{ .m = 288, .k = 3840, .n = 3840 }, // 256px qkv/proj
    .{ .m = 4128, .k = 3840, .n = 10240 }, // 1024px gate+up
    .{ .m = 4128, .k = 10240, .n = 3840 }, // 1024px down
    .{ .m = 4128, .k = 3840, .n = 3840 }, // 1024px qkv/proj
};

const Ctx = struct {
    device: *anyopaque,
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_ours: *anyopaque,
    c_mps: *anyopaque,
    mps: *anyopaque,
    params: c.GemmParams,
};

const Kind = enum { ours, mps };

pub fn main(init: std.process.Init) !void {
    const runs = parseRuns(init);
    run(init.io, init.gpa, runs) catch |err| {
        std.debug.print("gemmbench: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, runs: u32) !void {
    const device = c.zdraw_metal_create_device() orelse return error.MetalNotAvailable;
    defer c.zdraw_metal_release_device(device);
    const queue = c.zdraw_metal_create_queue(device) orelse return error.MetalQueueFailed;
    defer c.zdraw_metal_release_queue(queue);
    if (std.c.getenv("ZDRAW_CONVH_ONLY") != null) {
        try benchConvH(io, allocator, device, queue);
        return;
    }
    if (std.c.getenv("ZDRAW_CONVH8") != null) {
        try benchConvH8(io, allocator, device, queue);
        return;
    }
    if (std.c.getenv("ZDRAW_VAE_BUDGET_H") != null) {
        try runVaeBudgetH(io, allocator, device, queue);
        return;
    }
    if (std.c.getenv("ZDRAW_WINO") != null) {
        try benchWino(io, allocator, device, queue);
        return;
    }
    if (std.c.getenv("ZDRAW_MPP") != null) {
        try benchMpp(io, allocator, device, queue, runs);
        return;
    }
    if (std.c.getenv("ZDRAW_KLEIN_GEMMBENCH") != null) {
        try kleinGemmHeadroom(io, allocator, device, queue, runs);
        return;
    }
    if (std.c.getenv("ZDRAW_KLEIN_ATTNBENCH") != null) {
        try kleinAttnBench(io, allocator, device, queue, runs);
        return;
    }
    if (std.c.getenv("ZDRAW_FLUX2_EMB_ONLY") != null) {
        try flux2EmbedGate(io, allocator);
        return;
    }
    const klein_w6 = std.c.getenv("ZDRAW_KLEIN_W6_ONLY") != null;
    if (klein_w6 or std.c.getenv("ZDRAW_W6_ONLY") != null) {
        // ZDRAW_GEMM_M=<rows> rescales the Klein shapes' M (512/1024 for the
        // edit-sized, bandwidth-bound regime of the paper's kernel table).
        var klein_m = w6_klein_shapes;
        if (std.c.getenv("ZDRAW_GEMM_M")) |raw| {
            const m_over = try std.fmt.parseInt(u32, std.mem.span(raw), 10);
            for (&klein_m) |*ns| ns.shape.m = m_over;
        }
        const w6_shapes = if (klein_w6) klein_m[0..] else w6_zimage_shapes[0..];
        try runW6Gate(allocator, device, queue);
        try runW6Speed(io, allocator, device, queue, w6_shapes, runs);
        try runW6Steel(io, allocator, device, queue, w6_shapes, runs);
        if (klein_w6) try runPackedSteel(zw4, "steel_gemm_w4_64", io, allocator, device, queue, w6_shapes);
        if (klein_w6) try runPackedSteel(zw2, "steel_gemm_w2_64", io, allocator, device, queue, w6_shapes);
        return;
    }
    if (std.c.getenv("ZDRAW_KLEIN_CONCURRENCY") != null) {
        try kleinConcurrency(io, allocator, device, queue, runs);
        return;
    }
    if (std.c.getenv("ZDRAW_REPLAY_DIR")) |dir| {
        try replayPrenorm(allocator, device, queue, std.mem.span(dir));
        return;
    }
    if (std.c.getenv("ZDRAW_REPLAY_UP")) |dir| {
        try replayUp(allocator, device, queue, std.mem.span(dir));
        return;
    }
    const pipe = try compile(device, mgemm_bench_shader.gemm_f16.ptr, "gemm_f16");
    defer c.zdraw_metal_release_pipeline(pipe);
    const exact = try compile(device, mgemm_shader.gemm.ptr, "gemm_exact");
    defer c.zdraw_metal_release_pipeline(exact);
    const half = try compile(device, mgemm_shader.gemm.ptr, "gemm_half");
    defer c.zdraw_metal_release_pipeline(half);
    const linear = try compile(device, mshader.linear.ptr, "linear_rows");
    defer c.zdraw_metal_release_pipeline(linear);

    // Correctness gates only: for small-memory runners (GitHub's 7 GiB
    // paravirtual M1 cannot hold the 1024-resolution VAE races and chains).
    if (std.c.getenv("ZDRAW_GEMMBENCH_GATES") != null) {
        try runGates(allocator, device, queue, exact);
        try checkHalf(allocator, device, queue, half);
        try runW6Gate(allocator, device, queue);
        try runVaeNormSplitGate(allocator, device, queue);
        try runConvWindowGate(allocator, device, queue);
        try runConvPrenormWindowGate(allocator, device, queue);
        try runConvUpsampleWindowGate(allocator, device, queue);
        runFragMap(device, queue);
        return;
    }

    std.debug.print("\ngemmbench ({d} runs, f16):  M x K x N        ours      MPS   ours/MPS   err(ours,mps)\n", .{runs});
    for (shapes) |s| try benchShape(io, allocator, device, queue, pipe, s, runs);
    std.debug.print("  (GFLOP/s; substrate viable if ours/MPS >= ~0.8 with ours-err small)\n", .{});

    try runSteel(io, allocator, device, queue, runs);
    try runV2(io, allocator, device, queue, runs);
    try runConvRace(io, allocator, device, queue);
    try runVaeResChainGate(allocator, device, queue);
    if (std.c.getenv("ZDRAW_VAE_BUDGET") != null) try runVaeBudget(io, allocator, device, queue);
    if (std.c.getenv("ZDRAW_CONVH") != null) try benchConvH(io, allocator, device, queue);
    if (std.c.getenv("ZDRAW_HALFD1") != null) try benchHalfD1(io, allocator, device, queue);
    if (std.c.getenv("ZDRAW_FLUX2") != null) try flux2MapGate(io, allocator);
    if (std.c.getenv("ZDRAW_FLUX2_EMB") != null) try flux2EmbedGate(io, allocator);
    if (std.c.getenv("ZDRAW_ATTNB") != null) try benchAttnScale(io, allocator);
    if (std.c.getenv("ZDRAW_FLUX2_DIT") != null) try flux2DitGate(io, allocator);
    if (std.c.getenv("ZDRAW_FLUX2_LOOP") != null) try flux2LoopGate(io, allocator);
    if (std.c.getenv("ZDRAW_FLUX2_SWAP") != null) try flux2SwapGate(io, allocator);
    if (std.c.getenv("ZDRAW_FLUX2_VAE") != null) try flux2VaeGate(io, allocator);

    try runGates(allocator, device, queue, exact);
    try checkHalf(allocator, device, queue, half);
    try runSpeed(io, allocator, device, queue, exact, half, linear, runs);
    try runSpeedW16(io, allocator, device, queue, exact, half, runs);
    try runW6Gate(allocator, device, queue);
    try runVaeNormSplitGate(allocator, device, queue);
    try runConvWindowGate(allocator, device, queue);
    try runConvPrenormWindowGate(allocator, device, queue);
    try runConvUpsampleWindowGate(allocator, device, queue);
    try runConvUpsampleV7RealGate(allocator, device, queue);
    try runVaeResStreamGate(allocator, device, queue);
    try runW6Speed(io, allocator, device, queue, w6_zimage_shapes[0..], runs);
    try runW6Steel(io, allocator, device, queue, w6_zimage_shapes[0..], runs);
    runChainProbe(device, queue);
    runMixProbe(device, queue);
    runFragMap(device, queue);
    runAbProbe(device, queue, 288, 3840, 20480);
    runAbProbe(device, queue, 288, 10240, 3840);
    runAbChain(device, queue, 288, 3840, 20480);
    runAbAlt(device, queue, 288, 3840, 20480);
    runAbAlt(device, queue, 288, 10240, 3840);
}

// The MPS entry comes from metal_c (the canonical ABI module); redeclaring
// it locally is the exact drift class the metal_api.m header bans.
const zdraw_metal_run_gemm_mps = c.zdraw_metal_run_gemm_mps;

extern fn zdraw_metal_ours16_make(device: *anyopaque) ?*anyopaque;
extern fn zdraw_metal_ourscustom_make(device: *anyopaque) ?*anyopaque;
extern fn zdraw_metal_f16a_direct_make(device: *anyopaque) ?*anyopaque;
extern fn zdraw_metal_f16a_v2_make(device: *anyopaque) ?*anyopaque;
extern fn zdraw_metal_ours64_run(
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_buf: *anyopaque,
    m: c_int,
    n: c_int,
    k: c_int,
) c_int;
extern fn zdraw_metal_ours64_run_1d(
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_buf: *anyopaque,
    m: c_int,
    n: c_int,
    k: c_int,
) c_int;
extern fn zdraw_metal_steel_make(
    device: *anyopaque,
    path: [*:0]const u8,
    bm: c_int,
    align_m: c_int,
    align_n: c_int,
    align_k: c_int,
) ?*anyopaque;

// Concurrency micro-bench: dispatch `n` independent GEMMs (distinct c_bufs[i])
// through a serial or concurrent compute encoder; commit+wait. Bench-only.
extern fn zdraw_metal_bench_gemm_fanout(
    queue: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_bufs: [*]const ?*anyopaque,
    params: *const c.GemmParams,
    n: u32,
    concurrent: c_int,
) c_int;
extern fn zdraw_metal_steel_run(
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    b: *anyopaque,
    d: *anyopaque,
    m: c_int,
    n: c_int,
    k: c_int,
    bm: c_int,
) c_int;
extern fn zdraw_metal_steel_w6_make_cfg(
    device: *anyopaque,
    path: [*:0]const u8,
    fname: [*:0]const u8,
    align_m: c_int,
    align_n: c_int,
    align_k: c_int,
) ?*anyopaque;

extern fn zdraw_metal_steel_w6_run_cfg(
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    d: *anyopaque,
    m: c_int,
    n: c_int,
    k: c_int,
    bm: c_int,
    bn: c_int,
    bk: c_int,
    wm: c_int,
    wn: c_int,
) c_int;

extern fn zdraw_metal_steel_w6_make(
    device: *anyopaque,
    path: [*:0]const u8,
    align_m: c_int,
    align_n: c_int,
    align_k: c_int,
) ?*anyopaque;
extern fn zdraw_metal_steel_w6_run(
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    d: *anyopaque,
    m: c_int,
    n: c_int,
    k: c_int,
) c_int;

fn compile(device: *anyopaque, source: [*:0]const u8, entry: [*:0]const u8) !*anyopaque {
    var err: [4096]u8 = undefined;
    return c.zdraw_metal_compile(device, source, entry, &err, err.len) orelse {
        std.debug.print("gemm kernel compile error:\n{s}\n", .{std.mem.sliceTo(&err, 0)});
        return error.MetalCompileFailed;
    };
}

fn benchShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    pipe: *anyopaque,
    s: Shape,
    runs: u32,
) !void {
    const a_data = try allocator.alloc(f16, @as(usize, s.m) * s.k);
    defer allocator.free(a_data);
    const w_data = try allocator.alloc(f16, @as(usize, s.n) * s.k);
    defer allocator.free(w_data);
    var prng = std.Random.DefaultPrng.init(0x1234 + s.m + s.n + s.k);
    fillF16(a_data, prng.random());
    fillF16(w_data, prng.random());

    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w_data));
    defer w_buf.deinit();
    var c_ours = try mbuffer.Buffer.empty(device, @as(usize, s.m) * s.n * 4);
    defer c_ours.deinit();
    var c_mps = try mbuffer.Buffer.empty(device, @as(usize, s.m) * s.n * 2);
    defer c_mps.deinit();

    const mps_ctx = mps_c.zdraw_mps_gemm_make(device, a_buf.handle, w_buf.handle, c_mps.handle, s.m, s.n, s.k) orelse
        return error.MpsMakeFailed;
    defer mps_c.zdraw_mps_gemm_free(mps_ctx);

    var ctx = Ctx{
        .device = device,
        .queue = queue,
        .pipe = pipe,
        .a = a_buf.handle,
        .w = w_buf.handle,
        .c_ours = c_ours.handle,
        .c_mps = c_mps.handle,
        .mps = mps_ctx,
        .params = .{ .m = s.m, .k = s.k, .n = s.n, .dtype = 1, .mode = 1 },
    };

    const ours_ns = try timeKind(io, .ours, runs, &ctx);
    const mps_ns = try timeKind(io, .mps, runs, &ctx);

    const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
    const ours_g = flops / f(ours_ns);
    const mps_g = flops / f(mps_ns);
    const err = try checkErr(allocator, &ctx, a_data, w_data);

    std.debug.print("  {d:>4} x {d:>5} x {d:>5}   {d:>8.0} {d:>8.0}   {d:>7.2}   ({d:.4}, {d:.4})\n", .{
        s.m, s.k, s.n, ours_g, mps_g, ours_g / mps_g, err.ours, err.mps,
    });
}

// MPP-informed v2 probe (apple-gpu-notes section 7, pre-registered): the
// direct-access schedule at production occupancy (64x64 / 128 threads / 2x2
// simdgroups) vs the shipped staged ours16, plus cadence-barrier and
// locality-walk variants. Ratios vs MPS; correctness vs the staged kernel.
fn runV2(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    runs: u32,
) !void {
    std.debug.print(
        "\nv2 probe ({d} runs, ratio vs MPS):  M x K x N      mps GF/s  direct   v2a   v2b" ++
            "   v2c  f16a   err(v2a,direct)  err(f16a,v2a)\n",
        .{runs},
    );
    const v2_shapes = [_]Shape{
        .{ .m = 4128, .k = 3840, .n = 10240 },
        .{ .m = 4128, .k = 10240, .n = 3840 },
        .{ .m = 4128, .k = 3840, .n = 3840 },
        .{ .m = 288, .k = 3840, .n = 20480 },
    };
    for (v2_shapes) |s| try benchV2Shape(io, allocator, device, queue, s, runs);
}

fn benchV2Shape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    s: Shape,
    runs: u32,
) !void {
    const m = @as(usize, s.m);
    const n = @as(usize, s.n);
    const k = @as(usize, s.k);
    const a16 = try allocator.alloc(f16, m * k);
    defer allocator.free(a16);
    const a32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a32);
    const b16 = try allocator.alloc(f16, n * k);
    defer allocator.free(b16);
    var prng = std.Random.DefaultPrng.init(0x7261 + s.m + s.n + s.k);
    fillF16(a16, prng.random());
    fillF16(b16, prng.random());
    for (a16, a32) |hv, *fv| fv.* = @floatCast(hv);

    var a16_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
    defer a16_buf.deinit();
    var a32_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a32));
    defer a32_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(b16));
    defer w_buf.deinit();
    var c_direct = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_direct.deinit();
    var c_v2 = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_v2.deinit();
    var c_mps = try mbuffer.Buffer.empty(device, m * n * 2);
    defer c_mps.deinit();

    const mps_ctx = mps_c.zdraw_mps_gemm_make(device, a16_buf.handle, w_buf.handle, c_mps.handle, s.m, s.n, s.k) orelse
        return error.MpsMakeFailed;
    defer mps_c.zdraw_mps_gemm_free(mps_ctx);

    const direct = zdraw_metal_ours16_make(device) orelse return error.MakeFailed;
    const v2a = try compile(device, mgemm_bench_shader.v2.ptr, "gemm_f16_v2a");
    defer c.zdraw_metal_release_pipeline(v2a);
    const v2b = try compile(device, mgemm_bench_shader.v2.ptr, "gemm_f16_v2b");
    defer c.zdraw_metal_release_pipeline(v2b);
    const v2c = try compile(device, mgemm_bench_shader.v2.ptr, "gemm_f16_v2c");
    defer c.zdraw_metal_release_pipeline(v2c);
    // The SHIPPED half-A promotion (metal_api f16a_v2_pipeline): must match
    // the staged v2a bit-for-bit and hold its ratios, or the promotion is bad.
    const f16a = zdraw_metal_f16a_v2_make(device) orelse return error.MakeFailed;

    const mi: c_int = @intCast(s.m);
    const ni: c_int = @intCast(s.n);
    const ki: c_int = @intCast(s.k);

    _ = zdraw_metal_ours64_run(queue, direct, a32_buf.handle, w_buf.handle, c_direct.handle, mi, ni, ki);
    _ = zdraw_metal_ours64_run(queue, v2a, a16_buf.handle, w_buf.handle, c_v2.handle, mi, ni, ki);
    const od = try allocator.alloc(f32, m * n);
    defer allocator.free(od);
    const ov = try allocator.alloc(f32, m * n);
    defer allocator.free(ov);
    c.zdraw_metal_read_buffer(c_direct.handle, std.mem.sliceAsBytes(od).ptr, m * n * 4);
    c.zdraw_metal_read_buffer(c_v2.handle, std.mem.sliceAsBytes(ov).ptr, m * n * 4);
    var maxerr: f64 = 0;
    for (od, ov) |dv, vv| {
        const d = @abs(@as(f64, dv) - @as(f64, vv));
        if (d > maxerr) maxerr = d;
    }
    var c_f16a = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_f16a.deinit();
    _ = zdraw_metal_ours64_run(queue, f16a, a16_buf.handle, w_buf.handle, c_f16a.handle, mi, ni, ki);
    const oh = try allocator.alloc(f32, m * n);
    defer allocator.free(oh);
    c.zdraw_metal_read_buffer(c_f16a.handle, std.mem.sliceAsBytes(oh).ptr, m * n * 4);
    var f16a_err: f64 = 0;
    for (oh, ov) |hv, vv| {
        const d = @abs(@as(f64, hv) - @as(f64, vv));
        if (d > f16a_err) f16a_err = d;
    }

    const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
    const mps_g = flops / f(try timeMps(io, queue, mps_ctx, runs));
    const direct_g = flops / f(try timeOurs(io, queue, direct, &a32_buf, &w_buf, &c_direct, mi, ni, ki, runs));
    const v2a_g = flops / f(try timeOurs(io, queue, v2a, &a16_buf, &w_buf, &c_v2, mi, ni, ki, runs));
    const v2b_g = flops / f(try timeOurs(io, queue, v2b, &a16_buf, &w_buf, &c_v2, mi, ni, ki, runs));
    const v2c_g = flops / f(try timeOurs1d(io, queue, v2c, &a16_buf, &w_buf, &c_v2, mi, ni, ki, runs));
    const f16a_g = flops / f(try timeOurs(io, queue, f16a, &a16_buf, &w_buf, &c_f16a, mi, ni, ki, runs));
    std.debug.print(
        "  {d:>4} x {d:>5} x {d:>5}    {d:>7.0}  {d:>6.2}  {d:>4.2}  {d:>4.2}  {d:>4.2}  {d:>4.2}   {d:.4}  {d:.4}\n",
        .{
            s.m,              s.k,           s.n,           mps_g,
            direct_g / mps_g, v2a_g / mps_g, v2b_g / mps_g, v2c_g / mps_g,
            f16a_g / mps_g,   maxerr,        f16a_err,
        },
    );
}

fn timeOurs1d(io: std.Io, queue: *anyopaque, pipe: *anyopaque, a: *mbuffer.Buffer, w: *mbuffer.Buffer, cb: *mbuffer.Buffer, mi: c_int, ni: c_int, ki: c_int, runs: u32) !u64 {
    var wm: u32 = 0;
    while (wm < 3) : (wm += 1) _ = zdraw_metal_ours64_run_1d(queue, pipe, a.handle, w.handle, cb.handle, mi, ni, ki);
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = zdraw_metal_ours64_run_1d(queue, pipe, a.handle, w.handle, cb.handle, mi, ni, ki);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn runSteel(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    runs: u32,
) !void {
    std.debug.print("\nkernel race ({d} runs, ratio vs MPS):  M x K x N      direct  custom   steel   err(cust,direct)\n", .{runs});
    const ffn = [_]Shape{
        .{ .m = 4128, .k = 3840, .n = 10240 },
        .{ .m = 4128, .k = 10240, .n = 3840 },
        .{ .m = 4128, .k = 3840, .n = 3840 },
    };
    for (ffn) |s| try benchKernelsShape(io, allocator, device, queue, s, runs);
}

// The VAE decode's dominant conv shapes at 1024px (per-block resolutions).
const ConvShape = struct { ic: u32, oc: u32, h: u32, w: u32 };
const conv_shapes = [_]ConvShape{
    .{ .ic = 512, .oc = 512, .h = 128, .w = 128 },
    .{ .ic = 512, .oc = 512, .h = 256, .w = 256 },
    .{ .ic = 512, .oc = 256, .h = 512, .w = 512 },
    .{ .ic = 256, .oc = 256, .h = 512, .w = 512 },
    .{ .ic = 256, .oc = 128, .h = 1024, .w = 1024 },
    .{ .ic = 128, .oc = 128, .h = 1024, .w = 1024 },
};

// Race the streamed-VAE conv kernel (v1) against conv2d_window_v2 on the real
// decode shapes, full strip. v2 must be BIT-IDENTICAL (same K order, same MMA
// chunking) and meaningfully faster — the #44 gate.
fn runConvRace(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const v1 = try compile(device, mconv_shader.conv.ptr, "conv2d_window");
    defer c.zdraw_metal_release_pipeline(v1);
    const v2 = try compile(device, mconv_shader.conv.ptr, "conv2d_window_v2");
    defer c.zdraw_metal_release_pipeline(v2);
    std.debug.print(
        "\nVAE conv race (median of 3):  ic->oc @ HxW          v1      v2   v2/v1  bit|d|\n",
        .{},
    );
    for (conv_shapes) |s| try benchConvShape(io, allocator, device, queue, v1, v2, s);
    try benchGemmF32(io, allocator, device, queue);
    // The 1x1 skip projections (block-boundary resnets) run the v1 window
    // kernel (v2's contract is 3x3); time them so lever-3 work is sized by data.
    try benchSkipShape(io, allocator, device, queue, v1, .{ .ic = 512, .oc = 256, .h = 512, .w = 512 });
    try benchSkipShape(io, allocator, device, queue, v1, .{ .ic = 256, .oc = 128, .h = 1024, .w = 1024 });
}

// Kill-criterion control: the #22 direct-load schedule at PURE f32 on a
// conv-sized GEMM. Decides whether v5 continues (>=3-4 TF/s) or dies honestly.
fn benchGemmF32(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const M: u32 = 4096;
    const K: u32 = 4608;
    const N: u32 = 512;
    const a = try allocator.alloc(f32, @as(usize, M) * K);
    defer allocator.free(a);
    const w = try allocator.alloc(f32, @as(usize, N) * K);
    defer allocator.free(w);
    var prng = std.Random.DefaultPrng.init(0xF32);
    const rng = prng.random();
    for (a) |*v| v.* = rng.floatNorm(f32);
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w));
    defer w_buf.deinit();
    var c_buf = try mbuffer.Buffer.empty(device, @as(usize, M) * N * 4);
    defer c_buf.deinit();
    const pipe = try compile(device, mgemm_bench_shader.gemm_f32.ptr, "gemm_f32");
    defer c.zdraw_metal_release_pipeline(pipe);
    const params = c.GemmParams{ .m = M, .k = K, .n = N, .dtype = 3, .mode = 0 };
    if (c.zdraw_metal_run_gemm(queue, pipe, a_buf.handle, w_buf.handle, c_buf.handle, &params) != 0) {
        return error.DispatchFailed;
    }
    var samples: [3]u64 = undefined;
    for (&samples) |*sample| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (c.zdraw_metal_run_gemm(queue, pipe, a_buf.handle, w_buf.handle, c_buf.handle, &params) != 0) {
            return error.DispatchFailed;
        }
        sample.* = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, &samples, {}, std.sort.asc(u64));
    const flops: f64 = 2.0 * f(@as(usize, M)) * f(@as(usize, K)) * f(@as(usize, N));
    std.debug.print("  pure f32 GEMM (direct-load schedule) {d}x{d}x{d}: {d:>6.1} GFLOP/s\n", .{
        M, K, N, flops / f(samples[1]),
    });
}

// f16-VAE rate probe: time conv2d_window_h (half features x half weights,
// f32 accumulators) at the production shapes. The GFLOP/s column decides the
// f16-VAE project: >=5 TF/s -> GO (vae-up ~1.3-2s), ~2.5 -> dead.
extern fn zdraw_metal_run_conv2d_window_x4(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const ConvWindowParams,
) c_int;

// Attention scaling probe: the production pick() path at DiT shapes across
// token counts (the 2048px question: where does the rate cliff start?).
fn benchAttnScale(io: std.Io, allocator: std.mem.Allocator) !void {
    var ctx = try mattn_b.Context.init();
    defer ctx.deinit();
    const heads: usize = 30;
    const hd: usize = 128;
    const counts = [_]usize{ 4224, 8448, 16896 };
    std.debug.print("\nattention kernel rate (resident buffers):  tokens   ms      TFLOP/s\n", .{});
    for (counts) |n| {
        const len = n * heads * hd;
        const host = try allocator.alloc(f32, len);
        defer allocator.free(host);
        var prng = std.Random.DefaultPrng.init(0xA77);
        const rng = prng.random();
        for (host) |*x| x.* = rng.floatNorm(f32) * 0.1;
        var q_b = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(host));
        defer q_b.deinit();
        var k_b = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(host));
        defer k_b.deinit();
        var v_b = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(host));
        defer v_b.deinit();
        var o_b = try mbuffer.Buffer.empty(ctx.device, len * 4);
        defer o_b.deinit();
        const picked = ctx.pick(n, hd);
        const params = metal_c.AttnParams{
            .tokens = @intCast(n),
            .heads = @intCast(heads),
            .kv_heads = @intCast(heads),
            .head_dim = @intCast(hd),
            .causal = 0,
        };
        if (metal_c.zdraw_metal_run_attention(ctx.queue, picked.pipeline, q_b.handle, k_b.handle, v_b.handle, o_b.handle, &params, @intFromEnum(picked.kernel), picked.threads) != 0) {
            return error.DispatchFailed;
        }
        var best: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (metal_c.zdraw_metal_run_attention(ctx.queue, picked.pipeline, q_b.handle, k_b.handle, v_b.handle, o_b.handle, &params, @intFromEnum(picked.kernel), picked.threads) != 0) {
                return error.DispatchFailed;
            }
            const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            if (dt < best) best = dt;
        }
        const fl: f64 = 4.0 * f(n) * f(n) * f(hd) * f(heads);
        std.debug.print("  {d:>6} ({s})  {d:>7.1}  {d:>7.2}\n", .{ n, @tagName(picked.kernel), f(best) / 1e6, fl / f(best) / 1000.0 });
    }
}

// Klein VAE gate: decode zdraw's final latents through zdraw's OWN
// AutoencoderKL machinery (Klein VAE loads via the existing vviews names);
// dumps raw RGB f32 for the python image compare.
fn flux2VaeGate(io: std.Io, allocator: std.mem.Allocator) !void {
    const snap = std.mem.span(std.c.getenv("ZDRAW_FLUX2_VAE").?);
    const home = std.mem.span(std.c.getenv("HOME").?);
    var buf: [1024]u8 = undefined;
    var vf = try tensor_file_b.open(io, allocator, try std.fmt.bufPrint(
        &buf,
        "{s}/vae/diffusion_pytorch_model.safetensors",
        .{snap},
    ));
    defer vf.deinit(io, allocator);
    const views = try vviews.load(&vf);
    const lat = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/zdraw_final_latents.bin", .{home}));
    defer allocator.free(lat);
    const prepared = try zflux2_vae.prepare(allocator, lat, 64, 64, &vf);
    defer allocator.free(prepared);

    {
        const ph = std.c.fopen(try std.fmt.bufPrintZ(&buf, "{s}/anchors/klein4b/zdraw_prepared32.bin", .{home}), "wb") orelse return error.Unexpected;
        _ = std.c.fwrite(@as([*]const u8, @ptrCast(prepared.ptr)), 4, prepared.len, ph);
        _ = std.c.fclose(ph);
    }
    var conv_ctx = mconv_b.Context.init() catch null;
    defer if (conv_ctx) |*c2| c2.deinit();
    const cp: ?*mconv_b.Context = if (conv_ctx) |*c2| c2 else null;
    var lin_ctx = mlinear.Context.init() catch null;
    defer if (lin_ctx) |*m| m.deinit();
    const lp: ?*mlinear.Context = if (lin_ctx) |*m| m else null;
    var attn_ctx = try mattn_b.Context.init();
    defer attn_ctx.deinit();

    const out = try allocator.alloc(f32, 3 * 1024 * 1024);
    defer allocator.free(out);
    try vdecode.run(cp, lp, &attn_ctx, null, allocator, out, prepared, views, .{
        .height = 128,
        .width = 128,
    }, null);
    const fh = std.c.fopen(try std.fmt.bufPrintZ(&buf, "{s}/anchors/klein4b/zdraw_native_rgb.bin", .{home}), "wb") orelse return error.Unexpected;
    _ = std.c.fwrite(@as([*]const u8, @ptrCast(out.ptr)), 4, out.len, fh);
    _ = std.c.fclose(fh);
    std.debug.print("FLUX2-VAE native decode done\n", .{});
}

// Klein denoise-loop gate: full 4-step flow-match Euler// Klein denoise-loop gate: full 4-step flow-match Euler vs the per-step
// oracle latents; dumps the final latents for reference-VAE decode.
// Pool-contamination gate: run prompt A then prompt B on the SAME resident
// Ctx (reused pool), and assert B's velocity matches a FRESH-Ctx B. If the
// pooled embeds buffer were stale-cached (the old pointer-identity bug), the
// reused-pool B would carry A's conditioning and diverge.
fn flux2SwapGate(io: std.Io, allocator: std.mem.Allocator) !void {
    const snap = std.mem.span(std.c.getenv("ZDRAW_FLUX2_SWAP").?);
    const home = std.mem.span(std.c.getenv("HOME").?);
    var buf: [1024]u8 = undefined;
    var loaded = try zflux2.load(io, allocator, try std.fmt.bufPrint(&buf, "{s}/transformer", .{snap}), zflux2.Config.klein_4b);
    defer loaded.deinit(io, allocator);
    var rope = try zflux2_dit.buildRope(allocator, 64, 64, 2000.0);
    defer rope.deinit(allocator);
    const x = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/input_latents.bin", .{home}));
    defer allocator.free(x);
    const emb_a = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/prompt_embeds_f32.bin", .{home}));
    defer allocator.free(emb_a);
    // Prompt B = a distinct conditioning (negate A); different bytes, possibly
    // the same address once A is freed in a real run.
    const emb_b = try allocator.alloc(f32, emb_a.len);
    defer allocator.free(emb_b);
    for (emb_b, emb_a) |*d, v| d.* = -v;

    const v_a = try allocator.alloc(f32, 4096 * 128);
    defer allocator.free(v_a);
    const v_b_reused = try allocator.alloc(f32, 4096 * 128);
    defer allocator.free(v_b_reused);
    const v_b_fresh = try allocator.alloc(f32, 4096 * 128);
    defer allocator.free(v_b_fresh);

    // Shared Ctx: A then B on the same pool.
    var ctx = try zflux2_res.Ctx.init(allocator);
    try zflux2_res.forward(&ctx, v_a, x, emb_a, &loaded, rope, 1000.0, .{});
    try zflux2_res.forward(&ctx, v_b_reused, x, emb_b, &loaded, rope, 1000.0, .{});
    ctx.deinit();
    // Fresh Ctx: B from clean state.
    var ctx2 = try zflux2_res.Ctx.init(allocator);
    try zflux2_res.forward(&ctx2, v_b_fresh, x, emb_b, &loaded, rope, 1000.0, .{});
    ctx2.deinit();

    var max_ab: f32 = 0;
    var max_contam: f32 = 0;
    for (0..v_a.len) |k| {
        max_ab = @max(max_ab, @abs(v_a[k] - v_b_reused[k]));
        max_contam = @max(max_contam, @abs(v_b_reused[k] - v_b_fresh[k]));
    }
    std.debug.print("FLUX2-SWAP A-vs-B={d:.4} (>0 = B differs from A, sane)  reused-vs-fresh-B={d:.6} -> {s}\n", .{
        max_ab, max_contam, if (max_contam < 1e-3 and max_ab > 1e-3) "PASS" else "FAIL",
    });
}

fn flux2LoopGate(io: std.Io, allocator: std.mem.Allocator) !void {
    const snap = std.mem.span(std.c.getenv("ZDRAW_FLUX2_LOOP").?);
    const home = std.mem.span(std.c.getenv("HOME").?);
    var buf: [1024]u8 = undefined;
    var loaded = try zflux2.load(
        io,
        allocator,
        try std.fmt.bufPrint(&buf, "{s}/transformer", .{snap}),
        zflux2.Config.klein_4b,
    );
    defer loaded.deinit(io, allocator);
    var metal = mlinear.Context.init() catch null;
    defer if (metal) |*m| m.deinit();
    const mp: ?*mlinear.Context = if (metal) |*m| m else null;
    var attn_ctx = try mattn_b.Context.init();
    defer attn_ctx.deinit();
    var rope = try zflux2_dit.buildRope(allocator, 64, 64, 2000.0);
    defer rope.deinit(allocator);

    const emb = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/prompt_embeds_f32.bin", .{home}));
    defer allocator.free(emb);
    const x = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/input_latents.bin", .{home}));
    defer allocator.free(x);

    const sched = try zflux2_schedule.make(allocator, 4096, 4);
    defer sched.deinit(allocator);
    const refs = [_][]const u8{
        "latents_step0_t1000.0000",
        "latents_step1_t967.3840",
        "latents_step2_t908.1439",
        "latents_step3_t767.2000",
    };
    const v = try allocator.alloc(f32, 4096 * 128);
    defer allocator.free(v);
    var res_ctx: ?zflux2_res.Ctx = if (std.c.getenv("ZDRAW_FLUX2_RES") != null)
        try zflux2_res.Ctx.init(allocator)
    else
        null;
    defer if (res_ctx) |*rc| rc.deinit();
    for (0..4) |step| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (res_ctx) |*rc| {
            try zflux2_res.forward(rc, v, x, emb, &loaded, rope, sched.timesteps[step], .{});
        } else {
            try zflux2_dit.forward(allocator, mp, &attn_ctx, v, x, emb, &loaded, rope, sched.timesteps[step]);
        }
        const ms = @as(f64, @floatFromInt(t0.untilNow(io, .awake).toNanoseconds())) / 1e6;
        std.debug.print("  step {d} forward: {d:.0} ms\n", .{ step, ms });
        const dt = try zflux2_schedule.delta(sched, step);
        for (x, v) |*xi, vi| xi.* += dt * vi;
        const ref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/{s}.bin", .{ home, refs[step] }));
        defer allocator.free(ref);
        var nm: [16]u8 = undefined;
        report(try std.fmt.bufPrint(&nm, "step_{d}", .{step}), x, ref);
    }
    // dump final latents for the reference-VAE decode
    const fh = std.c.fopen(try std.fmt.bufPrintZ(&buf, "{s}/anchors/klein4b/zdraw_final_latents.bin", .{home}), "wb") orelse return error.Unexpected;
    _ = std.c.fwrite(@as([*]const u8, @ptrCast(x.ptr)), 4, x.len, fh);
    _ = std.c.fclose(fh);
    std.debug.print("FLUX2-LOOP final latents dumped\n", .{});
}

// Klein DiT stage-1 gate: embedders + time embed vs the block oracle// Klein DiT stage-1 gate: embedders + time embed vs the block oracle
// (~/anchors/klein4b/blocks). ZDRAW_FLUX2_DIT=<snapshot dir>.
fn flux2DitGate(io: std.Io, allocator: std.mem.Allocator) !void {
    const snap = std.mem.span(std.c.getenv("ZDRAW_FLUX2_DIT").?);
    const home = std.mem.span(std.c.getenv("HOME").?);
    var buf: [1024]u8 = undefined;
    var loaded = try zflux2.load(
        io,
        allocator,
        try std.fmt.bufPrint(&buf, "{s}/transformer", .{snap}),
        zflux2.Config.klein_4b,
    );
    defer loaded.deinit(io, allocator);
    var metal = mlinear.Context.init() catch null;
    defer if (metal) |*m| m.deinit();
    const mp: ?*mlinear.Context = if (metal) |*m| m else null;

    const D = "{s}/anchors/klein4b/blocks";
    _ = D;
    // input latents (4096 x 128) + oracles
    const lat = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/input_latents.bin", .{home}));
    defer allocator.free(lat);
    const x_ref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/x_embed.bin", .{home}));
    defer allocator.free(x_ref);
    const c_ref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/ctx_embed.bin", .{home}));
    defer allocator.free(c_ref);
    const t_ref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/time_embed.bin", .{home}));
    defer allocator.free(t_ref);
    const emb = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/prompt_embeds_f32.bin", .{home}));
    defer allocator.free(emb);

    const x_out = try allocator.alloc(f32, 4096 * 3072);
    defer allocator.free(x_out);
    try zflux2_dit.xEmbed(mp, x_out, lat, loaded.globals, 4096);
    report("x_embed", x_out, x_ref);

    const c_out = try allocator.alloc(f32, 512 * 3072);
    defer allocator.free(c_out);
    try zflux2_dit.contextEmbed(mp, c_out, emb, loaded.globals, 512);
    report("ctx_embed", c_out, c_ref);

    const t_out = try allocator.alloc(f32, 3072);
    defer allocator.free(t_out);
    try zflux2_dit.timeEmbed(allocator, mp, t_out, loaded.globals, 1000.0);
    report("time_embed(t=1000)", t_out, t_ref);

    // double block 0 with ORACLE inputs (isolated stage gate)
    const d0_ref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/double_0.bin", .{home}));
    defer allocator.free(d0_ref);
    var rope = try zflux2_dit.buildRope(allocator, 64, 64, 2000.0);
    defer rope.deinit(allocator);
    const mods_img = try zflux2_dit.modVectors(allocator, mp, loaded.globals.mod_img, t_ref, 2);
    defer allocator.free(mods_img);
    const mods_txt = try zflux2_dit.modVectors(allocator, mp, loaded.globals.mod_txt, t_ref, 2);
    defer allocator.free(mods_txt);
    var attn_ctx = try mattn_b.Context.init();
    defer attn_ctx.deinit();
    const img_s = try allocator.alloc(f32, 4096 * 3072);
    defer allocator.free(img_s);
    const txt_s = try allocator.alloc(f32, 512 * 3072);
    defer allocator.free(txt_s);
    @memcpy(img_s, x_ref);
    @memcpy(txt_s, c_ref);
    for (loaded.doubles, 0..) |blk, i| {
        try zflux2_dit.doubleBlock(allocator, mp, &attn_ctx, .{ .img = img_s, .txt = txt_s }, blk, mods_img, mods_txt, rope, zflux2_dit.Dims.fromConfig(zflux2.Config.klein_4b));
        const dref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/double_{d}.bin", .{ home, i }));
        defer allocator.free(dref);
        var nm: [24]u8 = undefined;
        report(try std.fmt.bufPrint(&nm, "double_{d} (txt)", .{i}), txt_s, dref);
    }
    // concat [txt, img] for the single chain
    const cat = try allocator.alloc(f32, 4608 * 3072);
    defer allocator.free(cat);
    @memcpy(cat[0 .. 512 * 3072], txt_s);
    @memcpy(cat[512 * 3072 ..], img_s);
    const mods_single = try zflux2_dit.modVectors(allocator, mp, loaded.globals.mod_single, t_ref, 1);
    defer allocator.free(mods_single);
    for (loaded.singles, 0..) |blk, i| {
        try zflux2_dit.singleBlock(allocator, mp, &attn_ctx, cat, blk, mods_single, rope, zflux2_dit.Dims.fromConfig(zflux2.Config.klein_4b));
        if (i % 5 == 4 or i == 0) {
            const sref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/single_{d}.bin", .{ home, i }));
            defer allocator.free(sref);
            var nm: [24]u8 = undefined;
            report(try std.fmt.bufPrint(&nm, "single_{d}", .{i}), cat, sref);
        }
    }
    // final: adaLN + proj on the IMG part vs dit_out
    const dit_ref = try readBin(io, allocator, try std.fmt.bufPrint(&buf, "{s}/anchors/klein4b/blocks/dit_out.bin", .{home}));
    defer allocator.free(dit_ref);
    const fin = try allocator.alloc(f32, 4096 * 128);
    defer allocator.free(fin);
    try zflux2_dit.finalProj(allocator, mp, fin, cat[512 * 3072 ..], loaded.globals, t_ref, 4096, true, zflux2_dit.Dims.fromConfig(zflux2.Config.klein_4b));
    report("dit_out (scale-first)", fin, dit_ref);
    try zflux2_dit.finalProj(allocator, mp, fin, cat[512 * 3072 ..], loaded.globals, t_ref, 4096, false, zflux2_dit.Dims.fromConfig(zflux2.Config.klein_4b));
    report("dit_out (shift-first)", fin, dit_ref);
}

fn report(name: []const u8, got: []const f32, ref: []const f32) void {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var max_abs: f64 = 0;
    const n = @min(got.len, ref.len);
    for (0..n) |i| {
        const a: f64 = got[i];
        const b: f64 = ref[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_abs = @max(max_abs, @abs(a - b));
    }
    const cos = dot / @max(@sqrt(na * nb), 1e-30);
    std.debug.print("FLUX2-DIT {s}: cos={d:.6} max|d|={d:.4} -> {s}\n", .{
        name, cos, max_abs, if (cos > 0.999) "PASS" else "FAIL",
    });
}

fn readBin(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const bytes = try readAll(io, allocator, path);
    defer allocator.free(bytes);
    const vals = std.mem.bytesAsSlice(f32, bytes);
    const out = try allocator.alloc(f32, vals.len);
    for (out, vals) |*d, v| d.* = v;
    return out;
}

// Klein embeds gate: run the shared Qwen3 encoder with the {9,18,27}
// stacked tap on the anchor token ids and compare against the diffusers
// oracle (bf16/MPS reference -> relative-error bar, not bit exactness).
// Default: CPU reference on the 15 real tokens. ZDRAW_KLEIN_TE_BATCH=1: the
// PRODUCT path — Metal contexts, full 512 padded ids, simdgroup-GEMM route —
// scored against the same oracle (all 512 rows).
fn flux2EmbedGate(io: std.Io, allocator: std.mem.Allocator) !void {
    const snap = std.mem.span(std.c.getenv("ZDRAW_FLUX2_EMB").?);
    var buf: [1024]u8 = undefined;

    const text = zflux2.textConfig(zflux2.Config.klein_4b);

    var index = try weight_index.read(
        io,
        allocator,
        try std.fmt.bufPrint(&buf, "{s}/text_encoder/model.safetensors.index.json", .{snap}),
    );
    defer index.deinit(allocator);
    var store = try shards.open(
        io,
        allocator,
        try std.fmt.bufPrint(&buf, "{s}/text_encoder", .{snap}),
        index,
    );
    defer store.deinit(io, allocator);

    // anchor token ids (already chat-templated by the oracle dump); trim pads
    const home = std.mem.span(std.c.getenv("HOME").?);
    const ids_bytes = try readAll(io, allocator, try std.fmt.bufPrint(
        &buf,
        "{s}/anchors/klein4b/token_ids_u32.bin",
        .{home},
    ));
    defer allocator.free(ids_bytes);
    const all_ids = std.mem.bytesAsSlice(u32, ids_bytes);
    const use_gemm = blk: {
        const v = std.c.getenv("ZDRAW_KLEIN_TE_BATCH") orelse break :blk false;
        break :blk !std.mem.eql(u8, std.mem.span(v), "0");
    };
    // Full padded length (the product encode shape) when the gemm path is on
    // or explicitly requested; the CPU reference default trims pads.
    const full = use_gemm or std.c.getenv("ZDRAW_FLUX2_EMB_FULL") != null;
    var real: usize = all_ids.len;
    while (real > 0 and all_ids[real - 1] == 151643) real -= 1;
    const used: usize = if (full) all_ids.len else real;
    const ids = try allocator.alloc(u32, used);
    defer allocator.free(ids);
    for (ids, all_ids[0..used]) |*d, v| d.* = v;

    const groups = [_]usize{ 9, 18, 27 };
    const hidden: usize = text.hidden_size;
    const out = try allocator.alloc(f32, used * groups.len * hidden);
    defer allocator.free(out);

    const cfg = qenc.attnConfig(text, used);
    var scratch = try qscratch.init(allocator, cfg, text.intermediate_size);
    defer scratch.deinit(allocator);

    var lin: ?mlinear.Context = if (full) mlinear.Context.init() catch null else null;
    defer if (lin) |*m| m.deinit();
    var attnc: ?mattn_b.Context = if (full) mattn_b.Context.init() catch null else null;
    defer if (attnc) |*a| a.deinit();

    // Padded runs pass the real-token mask, matching the product encode.
    const valid: usize = if (used > real) real else 0;
    try qenc.run(io, allocator, if (lin) |*m| m else null, if (attnc) |*a| a else null, out, ids, &store, index, text, .{
        .stacked = &groups,
    }, &scratch, use_gemm, null, valid);

    // oracle: (512, 7680) f32; compare the real-token rows
    const ref_bytes = try readAll(io, allocator, try std.fmt.bufPrint(
        &buf,
        "{s}/anchors/klein4b/prompt_embeds_f32.bin",
        .{home},
    ));
    defer allocator.free(ref_bytes);
    const ref = std.mem.bytesAsSlice(f32, ref_bytes);
    const width = groups.len * hidden;

    var max_rel: f64 = 0;
    var worst_cos: f64 = 1; // real-token rows: the certified bar
    var worst_pad: f64 = 1; // pad rows: known divergence class, reported only
    for (0..used) |t| {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (0..width) |j| {
            const a: f64 = out[t * width + j];
            const b: f64 = ref[t * width + j];
            dot += a * b;
            na += a * a;
            nb += b * b;
            if (t < real) {
                const denom = @max(@abs(b), 1.0);
                max_rel = @max(max_rel, @abs(a - b) / denom);
            }
        }
        const cos = dot / @max(@sqrt(na * nb), 1e-30);
        if (t < real) {
            worst_cos = @min(worst_cos, cos);
        } else {
            worst_pad = @min(worst_pad, cos);
        }
    }
    std.debug.print(
        "FLUX2 EMBEDS: tokens={d} real={d} max_rel={d:.5} worst_cos={d:.6} pad_cos={d:.6} -> {s}\n",
        .{ used, real, max_rel, worst_cos, worst_pad, if (worst_cos > 0.999) "PASS" else "FAIL" },
    );
}

fn readAll(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size: usize = @intCast((try file.stat(io)).size);
    const bytes = try allocator.alloc(u8, size);
    errdefer allocator.free(bytes);
    var reader = file.reader(io, &.{});
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

// Klein phase-1 gate: map + shape-check every transformer tensor of the// Klein phase-1 gate: map + shape-check every transformer tensor of the
// snapshot under ZDRAW_FLUX2 (path to the transformer component DIR).
fn flux2MapGate(io: std.Io, allocator: std.mem.Allocator) !void {
    const path = std.mem.span(std.c.getenv("ZDRAW_FLUX2").?);
    const cfg = if (std.c.getenv("ZDRAW_FLUX2_9B") != null)
        zflux2.Config.klein_9b
    else
        zflux2.Config.klein_4b;
    var loaded = try zflux2.load(io, allocator, path, cfg);
    defer loaded.deinit(io, allocator);
    try zflux2.check(&loaded);
    std.debug.print(
        "FLUX2 MAP OK: {d} doubles + {d} singles + globals, all shapes verified ({s})\n",
        .{ loaded.doubles.len, loaded.singles.len, path },
    );
}

// P0 lever-1 route (a): the production gemm_half dtype-1 (cached f16
// weights, wide direct-B path) rate at the refiner GEMM shapes.
fn benchHalfD1(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const pipe = try compile(device, mgemm_shader.gemm.ptr, "gemm_half");
    defer c.zdraw_metal_release_pipeline(pipe);
    const shp = [_]Shape{
        .{ .m = 4096, .k = 3840, .n = 11520 },
        .{ .m = 4096, .k = 3840, .n = 3840 },
        .{ .m = 4096, .k = 3840, .n = 20480 },
        .{ .m = 4096, .k = 10240, .n = 3840 },
    };
    std.debug.print("\ngemm_half dtype-1 rate:  M x K x N      GFLOP/s\n", .{});
    for (shp) |sh| {
        const a = try allocator.alloc(f32, @as(usize, sh.m) * sh.k);
        defer allocator.free(a);
        const w = try allocator.alloc(f16, @as(usize, sh.n) * sh.k);
        defer allocator.free(w);
        var prng = std.Random.DefaultPrng.init(0xD1);
        const rng = prng.random();
        for (a) |*v| v.* = rng.floatNorm(f32) * 0.5;
        for (w) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
        var ab = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
        defer ab.deinit();
        var wb = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w));
        defer wb.deinit();
        var cb = try mbuffer.Buffer.empty(device, @as(usize, sh.m) * sh.n * 4);
        defer cb.deinit();
        const gp = c.GemmParams{ .m = sh.m, .k = sh.k, .n = sh.n, .dtype = 1, .mode = 1 };
        if (c.zdraw_metal_run_gemm(queue, pipe, ab.handle, wb.handle, cb.handle, &gp) != 0) return error.DispatchFailed;
        var best: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (c.zdraw_metal_run_gemm(queue, pipe, ab.handle, wb.handle, cb.handle, &gp) != 0) return error.DispatchFailed;
            const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            if (dt < best) best = dt;
        }
        const fl: f64 = 2.0 * f(@as(usize, sh.m)) * f(@as(usize, sh.k)) * f(@as(usize, sh.n));
        std.debug.print("  {d} x {d} x {d}   {d:>7.0}\n", .{ sh.m, sh.k, sh.n, fl / f(best) });
    }
}

fn timeHalf(io: std.Io, queue: *anyopaque, pipe: *anyopaque, a: *mbuffer.Buffer, w: *mbuffer.Buffer, cb: *mbuffer.Buffer, gp: c.GemmParams, runs: u32) !u64 {
    var p = gp;
    var wmup: u32 = 0;
    while (wmup < 3) : (wmup += 1) if (c.zdraw_metal_run_gemm(queue, pipe, a.handle, w.handle, cb.handle, &p) != 0) return error.DispatchFailed;
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (c.zdraw_metal_run_gemm(queue, pipe, a.handle, w.handle, cb.handle, &p) != 0) return error.DispatchFailed;
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

// Klein resident GEMM headroom: gemm_half (the kernel Klein uses) vs steel vs
// MPS at Klein 4B's ACTUAL single/double qkv-gate and ffn-down shapes (1024px:
// img_len 4096, txt 512, tokens 4608; hidden 3072, ffn_inner 9216). ATTRIBUTION
// throughput only — never product speed. Kill bar: promote steel only if
// geomean(steel/half) >= 1.20 AND no shape regresses (>= 0.95). The in-chain
// LOOP cosine is the authoritative drift gate; the err column here is a rough
// upper bound (steel f16-A vs half f32-A). Gated by ZDRAW_KLEIN_GEMMBENCH=1.
fn kleinGemmHeadroom(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, runs: u32) !void {
    const KS = struct { label: []const u8, m: u32, k: u32, n: u32 };
    const klein = [_]KS{
        .{ .label = "single.qkv_gate(qkv) ", .m = 4608, .k = 3072, .n = 3072 },
        .{ .label = "single.qkv_gate(gate)", .m = 4608, .k = 3072, .n = 18432 },
        .{ .label = "single.ffn_down      ", .m = 4608, .k = 12288, .n = 3072 },
        .{ .label = "double.qkv_gate(qkv) ", .m = 4096, .k = 3072, .n = 3072 },
        .{ .label = "double.qkv_gate(gate)", .m = 4096, .k = 3072, .n = 18432 },
        .{ .label = "double.ffn_down      ", .m = 4096, .k = 9216, .n = 3072 },
    };
    // Reference and production arm: the f16-A kernel Klein's resident path
    // dispatches (gemmRunF16A). gemm_half is the dead 32x32 fallback, kept as
    // the historical column so the June rows stay comparable.
    const direct = zdraw_metal_f16a_direct_make(device) orelse return error.MakeFailed;
    defer c.zdraw_metal_release_pipeline(direct);
    const half = try compile(device, mgemm_shader.gemm.ptr, "gemm_half");
    defer c.zdraw_metal_release_pipeline(half);
    const lib_path = steelLibPath();

    std.debug.print("\nKlein GEMM headroom ({d} runs; GFLOP/s; ATTRIBUTION, not product speed):\n", .{runs});
    std.debug.print(
        "  ratios and max|d| are against the production f16-A kernel (f32 out);" ++
            " half = gemm_half fallback\n",
        .{},
    );
    std.debug.print("  shape                  direct  half  steel16  steel32  MPS\n", .{});

    var sum_log_s16: f64 = 0;
    var sum_log_s32: f64 = 0;
    var sum_log_mps: f64 = 0;
    var worst_s32: f64 = 1.0e9;
    var worst_err_s32: f64 = 0;
    var n_s16: usize = 0;
    var n_s32: usize = 0;
    for (klein) |s| {
        const mn: usize = @as(usize, s.m) * s.n;
        const a32 = try allocator.alloc(f32, @as(usize, s.m) * s.k);
        defer allocator.free(a32);
        const a16 = try allocator.alloc(f16, @as(usize, s.m) * s.k);
        defer allocator.free(a16);
        const w16 = try allocator.alloc(f16, @as(usize, s.n) * s.k);
        defer allocator.free(w16);
        var prng = std.Random.DefaultPrng.init(0x4b15 +% s.m +% s.n +% s.k);
        const rng = prng.random();
        for (a32, a16) |*v32, *v16| {
            const x = rng.floatNorm(f32) * 0.5;
            v32.* = x;
            v16.* = @floatCast(x);
        }
        for (w16) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
        var a32_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a32));
        defer a32_buf.deinit();
        var a16_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
        defer a16_buf.deinit();
        var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w16));
        defer w_buf.deinit();
        // Every output starts as NaN so an unwritten tile shows up as drift,
        // not as whatever the allocator left behind.
        var c_direct = try nanBuffer(allocator, device, mn, 4);
        defer c_direct.deinit();
        var c_half = try nanBuffer(allocator, device, mn, 4);
        defer c_half.deinit();
        var c_s16 = try nanBuffer(allocator, device, mn, 2);
        defer c_s16.deinit();
        var c_s32 = try nanBuffer(allocator, device, mn, 4);
        defer c_s32.deinit();
        var c_mps = try nanBuffer(allocator, device, mn, 2);
        defer c_mps.deinit();

        const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
        const mi: c_int = @intCast(s.m);
        const ni: c_int = @intCast(s.n);
        const ki: c_int = @intCast(s.k);

        const direct_ns =
            try timeOurs(io, queue, direct, &a16_buf, &w_buf, &c_direct, mi, ni, ki, runs);
        const direct_g = flops / f(direct_ns);
        const gp = c.GemmParams{ .m = s.m, .k = s.k, .n = s.n, .dtype = 1, .mode = 1 };
        const half_g = flops / f(try timeHalf(io, queue, half, &a32_buf, &w_buf, &c_half, gp, runs));

        const mps_ctx = mps_c.zdraw_mps_gemm_make(device, a16_buf.handle, w_buf.handle, c_mps.handle, s.m, s.n, s.k) orelse return error.MpsMakeFailed;
        defer mps_c.zdraw_mps_gemm_free(mps_ctx);
        const mps_g = flops / f(try timeMps(io, queue, mps_ctx, runs));

        const am: c_int = if (s.m % 64 == 0) 1 else 0;
        const an: c_int = if (s.n % 64 == 0) 1 else 0;
        const ak: c_int = if (s.k % 16 == 0) 1 else 0;
        var s16_g: f64 = 0;
        if (zdraw_metal_steel_make(device, lib_path, 64, am, an, ak)) |sp| {
            const ns = try timeSteel(io, queue, sp, &a16_buf, &w_buf, &c_s16, mi, ni, ki, 64, runs);
            s16_g = flops / f(ns);
        }
        var s32_g: f64 = 0;
        if (am == 1 and an == 1 and ak == 1) {
            const hf32 = "steel_gemm_hf32_64";
            if (zdraw_metal_steel_w6_make_cfg(device, lib_path, hf32, 1, 1, 1)) |hp| {
                const ns =
                    try timeSteelCfg(io, queue, hp, &a16_buf, &w_buf, &c_s32, mi, ni, ki, runs);
                s32_g = flops / f(ns);
            }
        }

        // Drift of every arm against the production output over the whole
        // matrix (NaN/inf anywhere reads as inf).
        const ref = try allocator.alloc(f32, mn);
        defer allocator.free(ref);
        c.zdraw_metal_read_buffer(c_direct.handle, std.mem.sliceAsBytes(ref).ptr, mn * 4);
        const e_half = try maxDiffF32(allocator, &c_half, ref);
        const e_s16 = try maxDiffF16(allocator, &c_s16, ref);
        const e_s32 = try maxDiffF32(allocator, &c_s32, ref);
        const e_mps = try maxDiffF16(allocator, &c_mps, ref);

        const r = struct {
            fn of(x: f64, base: f64) f64 {
                return if (base > 0 and x > 0) x / base else 0;
            }
        };
        std.debug.print(
            "  {s} direct={d:>6.0} half={d:>6.0}({d:.2}x e={d:.3})" ++
                " steel16={d:>6.0}({d:.2}x e={d:.3}) steel32={d:>6.0}({d:.2}x e={d:.3})" ++
                " mps={d:>6.0}({d:.2}x e={d:.3})\n",
            .{
                s.label,                direct_g, half_g,
                r.of(half_g, direct_g), e_half,   s16_g,
                r.of(s16_g, direct_g),  e_s16,    s32_g,
                r.of(s32_g, direct_g),  e_s32,    mps_g,
                r.of(mps_g, direct_g),  e_mps,
            },
        );
        if (direct_g > 0) {
            sum_log_mps += @log(r.of(mps_g, direct_g));
            if (s16_g > 0) {
                sum_log_s16 += @log(r.of(s16_g, direct_g));
                n_s16 += 1;
            }
            if (s32_g > 0) {
                const rr = r.of(s32_g, direct_g);
                sum_log_s32 += @log(rr);
                if (rr < worst_s32) worst_s32 = rr;
                if (e_s32 > worst_err_s32) worst_err_s32 = e_s32;
                n_s32 += 1;
            }
        }
    }
    const geo_mps = @exp(sum_log_mps / @as(f64, @floatFromInt(klein.len)));
    std.debug.print("  geomean MPS/direct={d:.2}x\n", .{geo_mps});
    if (n_s16 > 0) {
        const geo16 = @exp(sum_log_s16 / @as(f64, @floatFromInt(n_s16)));
        std.debug.print("  geomean steel16/direct={d:.2}x (f16 out, not for Klein)\n", .{geo16});
    }
    if (n_s32 > 0) {
        const geo = @exp(sum_log_s32 / @as(f64, @floatFromInt(n_s32)));
        const speed_ok = geo >= 1.10 and worst_s32 >= 0.95;
        const drift_ok = std.math.isFinite(worst_err_s32) and worst_err_s32 < 0.05;
        std.debug.print(
            "  geomean steel32/direct={d:.2}x  worst={d:.2}x  worst max|d|={d:.4}\n",
            .{ geo, worst_s32, worst_err_s32 },
        );
        const verdict: []const u8 = if (speed_ok and drift_ok) "PASS" else "FAIL";
        std.debug.print("  W3.1 bar (geomean>=1.10, worst>=0.95, max|d|<0.05): {s}\n", .{verdict});
    } else {
        std.debug.print("  steel32 unavailable ({s}; see tools/build_steel_lib.sh)\n", .{lib_path});
    }
}

fn nanBuffer(
    allocator: std.mem.Allocator,
    device: *anyopaque,
    count: usize,
    elem: usize,
) !mbuffer.Buffer {
    const bytes = try allocator.alloc(u8, count * elem);
    defer allocator.free(bytes);
    if (elem == 4) {
        const nan32: u32 = 0x7fc00000;
        for (0..count) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], nan32, .little);
    } else {
        const nan16: u16 = 0x7e00;
        for (0..count) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], nan16, .little);
    }
    return mbuffer.Buffer.fromBytes(device, bytes);
}

fn maxDiffF32(allocator: std.mem.Allocator, buf: *mbuffer.Buffer, ref: []const f32) !f64 {
    const out = try allocator.alloc(f32, ref.len);
    defer allocator.free(out);
    c.zdraw_metal_read_buffer(buf.handle, std.mem.sliceAsBytes(out).ptr, ref.len * 4);
    var worst: f64 = 0;
    for (out, ref) |v, r| {
        if (!std.math.isFinite(v)) return std.math.inf(f64);
        const d = @abs(@as(f64, v) - @as(f64, r));
        if (d > worst) worst = d;
    }
    return worst;
}

fn maxDiffF16(allocator: std.mem.Allocator, buf: *mbuffer.Buffer, ref: []const f32) !f64 {
    const out = try allocator.alloc(f16, ref.len);
    defer allocator.free(out);
    c.zdraw_metal_read_buffer(buf.handle, std.mem.sliceAsBytes(out).ptr, ref.len * 2);
    var worst: f64 = 0;
    for (out, ref) |v, r| {
        const vf: f32 = @floatCast(v);
        if (!std.math.isFinite(vf)) return std.math.inf(f64);
        const d = @abs(@as(f64, vf) - @as(f64, r));
        if (d > worst) worst = d;
    }
    return worst;
}

// max|a-b| between two half buffers of n elements (inf on any non-finite).
fn maxDiffF16Pair(allocator: std.mem.Allocator, a: *mbuffer.Buffer, b: *mbuffer.Buffer, n: usize) !f64 {
    const av = try allocator.alloc(f16, n);
    defer allocator.free(av);
    const bv = try allocator.alloc(f16, n);
    defer allocator.free(bv);
    c.zdraw_metal_read_buffer(a.handle, std.mem.sliceAsBytes(av).ptr, n * 2);
    c.zdraw_metal_read_buffer(b.handle, std.mem.sliceAsBytes(bv).ptr, n * 2);
    var worst: f64 = 0;
    for (av, bv) |x, y| {
        const xf: f32 = @floatCast(x);
        const yf: f32 = @floatCast(y);
        if (!std.math.isFinite(xf) or !std.math.isFinite(yf)) return std.math.inf(f64);
        const d = @abs(@as(f64, xf) - @as(f64, yf));
        if (d > worst) worst = d;
    }
    return worst;
}

fn timeSteelCfg(
    io: std.Io,
    queue: *anyopaque,
    pipe: *anyopaque,
    a: *mbuffer.Buffer,
    w: *mbuffer.Buffer,
    d: *mbuffer.Buffer,
    mi: c_int,
    ni: c_int,
    ki: c_int,
    runs: u32,
) !u64 {
    const steel_once = struct {
        fn once(
            q: *anyopaque,
            p: *anyopaque,
            ab: *anyopaque,
            wb: *anyopaque,
            db: *anyopaque,
            m: c_int,
            n: c_int,
            k: c_int,
        ) void {
            _ = zdraw_metal_steel_w6_run_cfg(q, p, ab, wb, db, m, n, k, 64, 64, 16, 2, 2);
        }
    };
    var wm: u32 = 0;
    while (wm < 3) : (wm += 1) {
        steel_once.once(queue, pipe, a.handle, w.handle, d.handle, mi, ni, ki);
    }
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        steel_once.once(queue, pipe, a.handle, w.handle, d.handle, mi, ni, ki);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

// min wall over `reps` of one fanout dispatch (warmup 3). Load-robust: the min
// approximates the unloaded run even on a busy box.
fn fanoutMinWall(io: std.Io, queue: *anyopaque, pipe: *anyopaque, a: *anyopaque, w: *anyopaque, cp: [*]const ?*anyopaque, params: *const c.GemmParams, n: u32, concurrent: c_int, reps: u32) !u64 {
    var wm: u32 = 0;
    while (wm < 3) : (wm += 1) if (zdraw_metal_bench_gemm_fanout(queue, pipe, a, w, cp, params, n, concurrent) != 0) return error.DispatchFailed;
    var best: u64 = std.math.maxInt(u64);
    var i: u32 = 0;
    while (i < reps) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (zdraw_metal_bench_gemm_fanout(queue, pipe, a, w, cp, params, n, concurrent) != 0) return error.DispatchFailed;
        const ns: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
        if (ns < best) best = ns;
    }
    return best;
}

// Does overlapping the INDEPENDENT qkv GEMMs beat serial, or is one Klein GEMM
// already saturating the GPU? Times N copies of the single-block qkv shape
// dispatched serial vs concurrent, min wall over `runs` reps. The last
// kernel-side Klein denoise lever; gated by ZDRAW_KLEIN_CONCURRENCY.
fn kleinConcurrency(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, runs: u32) !void {
    const m: u32 = 4608;
    const k: u32 = 3072;
    const n: u32 = 3072;
    const fan: u32 = 3;
    const half = try compile(device, mgemm_shader.gemm.ptr, "gemm_half");
    defer c.zdraw_metal_release_pipeline(half);

    const a16 = try allocator.alloc(f16, @as(usize, m) * k);
    defer allocator.free(a16);
    const w16 = try allocator.alloc(f16, @as(usize, n) * k);
    defer allocator.free(w16);
    var prng = std.Random.DefaultPrng.init(0x4b1c);
    const rng = prng.random();
    for (a16) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.5);
    for (w16) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w16));
    defer w_buf.deinit();
    const mn: usize = @as(usize, m) * n;
    var c0 = try mbuffer.Buffer.empty(device, mn * 4);
    defer c0.deinit();
    var c1 = try mbuffer.Buffer.empty(device, mn * 4);
    defer c1.deinit();
    var c2 = try mbuffer.Buffer.empty(device, mn * 4);
    defer c2.deinit();
    const cptrs = [3]?*anyopaque{ c0.handle, c1.handle, c2.handle };
    const cp: [*]const ?*anyopaque = &cptrs;

    const gp = c.GemmParams{ .m = m, .k = k, .n = n, .dtype = 1, .mode = 1 };
    const flops1: f64 = 2.0 * f(m) * f(k) * f(n);
    const t1 = try fanoutMinWall(io, queue, half, a_buf.handle, w_buf.handle, cp, &gp, 1, 0, runs);
    const ts = try fanoutMinWall(io, queue, half, a_buf.handle, w_buf.handle, cp, &gp, fan, 0, runs);
    const tc = try fanoutMinWall(io, queue, half, a_buf.handle, w_buf.handle, cp, &gp, fan, 1, runs);

    const tf = struct {
        fn g(flops: f64, ns: u64) f64 {
            return flops / f(ns) / 1000.0;
        }
    }.g;
    std.debug.print("\nKlein qkv concurrency ({d} reps, min wall; shape {d}x{d}x{d}, gemm_half):\n", .{ runs, m, k, n });
    std.debug.print("  1 GEMM              {d:>6.2} ms  {d:>5.1} TF/s\n", .{ f(t1) / 1.0e6, tf(flops1, t1) });
    std.debug.print("  {d} GEMMs serial      {d:>6.2} ms  {d:>5.1} TF/s\n", .{ fan, f(ts) / 1.0e6, tf(flops1 * f(fan), ts) });
    std.debug.print("  {d} GEMMs concurrent  {d:>6.2} ms  {d:>5.1} TF/s\n", .{ fan, f(tc) / 1.0e6, tf(flops1 * f(fan), tc) });
    std.debug.print("  concurrency speedup (serial/concurrent): {d:.2}x  (>1.0 = overlap helps)\n", .{f(ts) / f(tc)});
    std.debug.print("  serial scaling (serial/{d}x1): {d:.2}x  (1.0 = one GEMM already saturates the GPU)\n", .{ fan, f(ts) / (f(t1) * f(fan)) });
}

fn benchConvH(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_window_h");
    defer c.zdraw_metal_release_pipeline(pipe);
    const pipe4 = try compile(device, mconv_shader.conv.ptr, "conv2d_window_h4");
    defer c.zdraw_metal_release_pipeline(pipe4);
    const shapes_h = [_]ConvShape{
        .{ .ic = 512, .oc = 512, .h = 256, .w = 256 },
        .{ .ic = 256, .oc = 256, .h = 512, .w = 512 },
        .{ .ic = 128, .oc = 128, .h = 1024, .w = 1024 },
    };
    std.debug.print("\nf16-MMA conv probe:  ic->oc @ HxW        GFLOP/s\n", .{});
    for (shapes_h) |sh| {
        const hw: usize = @as(usize, sh.h) * sh.w;
        const in_n: usize = @as(usize, sh.ic) * hw;
        const w_n: usize = @as(usize, sh.oc) * sh.ic * 9;
        const input = try allocator.alloc(f16, in_n);
        defer allocator.free(input);
        const wts = try allocator.alloc(f16, w_n);
        defer allocator.free(wts);
        var prng = std.Random.DefaultPrng.init(0xF16C);
        const rng = prng.random();
        for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
        for (wts) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
        var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
        defer in_b.deinit();
        var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
        defer w_b.deinit();
        var out_b = try mbuffer.Buffer.empty(device, @as(usize, sh.oc) * hw * 4);
        defer out_b.deinit();
        const cp = ConvWindowParams{
            .in_ch = sh.ic,
            .out_ch = sh.oc,
            .height = sh.h,
            .width = sh.w,
            .ksize = 3,
            .pad = 1,
            .dtype = 1,
            .bias_dtype = 1,
            .has_bias = 0,
            .weight_offset = 0,
            .bias_offset = 0,
            .row0 = 0,
            .row1 = sh.h,
        };
        if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp, 32) != 0) {
            return error.DispatchFailed;
        }
        var best: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp, 32) != 0) {
                return error.DispatchFailed;
            }
            const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            if (dt < best) best = dt;
        }
        const flops: f64 = 2.0 * f(@as(usize, sh.oc)) * f(@as(usize, sh.ic)) * 9.0 * f(hw);
        // h4: 4-simdgroup variant
        if (zdraw_metal_run_conv2d_window_x4(queue, pipe4, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp) != 0) {
            return error.DispatchFailed;
        }
        var best4: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (zdraw_metal_run_conv2d_window_x4(queue, pipe4, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp) != 0) {
                return error.DispatchFailed;
            }
            const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            if (dt < best4) best4 = dt;
        }
        std.debug.print(
            "  {d}->{d} @ {d}x{d}   h={d:>6.0}  h4={d:>6.0}\n",
            .{ sh.ic, sh.oc, sh.h, sh.w, flops / f(best), flops / f(best4) },
        );
    }
}

// W7 instrument (vae-conv-h8-20260827): the 4-simdgroup half conv kernels
// against the 8-simdgroup staged-weight ones at the decoder's up-block
// shapes, interleaved (h4/h8/h8/h4 x3, best lap), plus max|d| between the
// two outputs (the h8 kernels must be byte-identical: same operands, same
// MMA order).
fn benchConvH8(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const pre4 = try compile(device, mconv_shader.conv.ptr, "conv2d_prenorm_window_h4");
    defer c.zdraw_metal_release_pipeline(pre4);
    const pre8 = try compile(device, mconv_h8.conv.ptr, "conv2d_prenorm_window_h8");
    defer c.zdraw_metal_release_pipeline(pre8);
    const up4 = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window_h4");
    defer c.zdraw_metal_release_pipeline(up4);
    const up8 = try compile(device, mconv_h8.conv.ptr, "conv2d_upsample_window_h8");
    defer c.zdraw_metal_release_pipeline(up8);
    const shapes_h8 = [_]ConvShape{
        .{ .ic = 512, .oc = 512, .h = 128, .w = 128 },
        .{ .ic = 512, .oc = 512, .h = 256, .w = 256 },
        .{ .ic = 256, .oc = 256, .h = 512, .w = 512 },
        .{ .ic = 128, .oc = 128, .h = 1024, .w = 1024 },
    };
    std.debug.print("\nh4 vs h8 conv (GFLOP/s, best of 3 interleaved; d = max|h8-h4|)\n", .{});
    for (shapes_h8) |sh| {
        try benchConvH8Shape(io, allocator, device, queue, pre4, pre8, sh, false);
        try benchConvH8Shape(io, allocator, device, queue, up4, up8, sh, true);
    }
}

fn benchConvH8Shape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    pipe4: *anyopaque,
    pipe8: *anyopaque,
    sh: ConvShape,
    upsample: bool,
) !void {
    // Upsample: the input is the half-size map, the conv runs at h x w.
    const in_h: usize = if (upsample) sh.h / 2 else sh.h;
    const in_w: usize = if (upsample) sh.w / 2 else sh.w;
    const hw: usize = @as(usize, sh.h) * sh.w;
    const in_n: usize = @as(usize, sh.ic) * in_h * in_w;
    const w_n: usize = @as(usize, sh.oc) * sh.ic * 9;
    const input = try allocator.alloc(f16, in_n);
    defer allocator.free(input);
    const wts = try allocator.alloc(f16, w_n);
    defer allocator.free(wts);
    const bias = try allocator.alloc(f16, sh.oc);
    defer allocator.free(bias);
    const norm_w = try allocator.alloc(f16, sh.ic);
    defer allocator.free(norm_w);
    const norm_b = try allocator.alloc(f16, sh.ic);
    defer allocator.free(norm_b);
    const stats = try allocator.alloc(f32, 64);
    defer allocator.free(stats);
    var prng = std.Random.DefaultPrng.init(0xC0DE + sh.ic);
    const rng = prng.random();
    for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
    for (wts) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
    for (bias) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.1);
    for (norm_w) |*v| v.* = @floatCast(1.0 + rng.floatNorm(f32) * 0.1);
    for (norm_b) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.1);
    for (0..32) |g| {
        stats[g * 2] = rng.floatNorm(f32) * 0.1;
        stats[g * 2 + 1] = 1.0 + rng.floatNorm(f32) * 0.1;
    }
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_b.deinit();
    var nw_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm_w));
    defer nw_b.deinit();
    var nb_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm_b));
    defer nb_b.deinit();
    var st_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(stats));
    defer st_b.deinit();
    var out4 = try mbuffer.Buffer.empty(device, @as(usize, sh.oc) * hw * 2);
    defer out4.deinit();
    var out8 = try mbuffer.Buffer.empty(device, @as(usize, sh.oc) * hw * 2);
    defer out8.deinit();
    const pp = ConvPrenormWindowParams{
        .in_ch = sh.ic,
        .out_ch = sh.oc,
        .height = sh.h,
        .width = sh.w,
        .ksize = 3,
        .pad = 1,
        .dtype = 1,
        .bias_dtype = 1,
        .has_bias = 1,
        .groups = 32,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = sh.h,
        .norm_dtype = 1,
        .norm_bias_dtype = 1,
        .norm_weight_offset = 0,
        .norm_bias_offset = 0,
    };
    const up = ConvUpsampleWindowParams{
        .channels = sh.oc,
        .out_height = sh.h,
        .out_width = sh.w,
        .in_height = @intCast(in_h),
        .in_width = @intCast(in_w),
        .ksize = 3,
        .pad = 1,
        .dtype = 1,
        .bias_dtype = 1,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = sh.h,
    };
    const Arm = struct {
        fn run(q: *anyopaque, pipe: *anyopaque, x4: c_int, is_up: bool, ib: *anyopaque, wb: *anyopaque, bb: *anyopaque, ob: *anyopaque, sb: *anyopaque, nwb: *anyopaque, nbb: *anyopaque, p_pre: *const ConvPrenormWindowParams, p_up: *const ConvUpsampleWindowParams) !void {
            const rc = if (is_up)
                mvres_chain.zdraw_metal_run_conv2d_upsample_window(q, pipe, ib, wb, bb, ob, p_up, 32, x4)
            else
                zdraw_metal_run_conv2d_prenorm_window_xn(q, pipe, ib, wb, bb, ob, sb, nwb, nbb, p_pre, x4);
            if (rc != 0) return error.DispatchFailed;
        }
    };
    var best4: u64 = std.math.maxInt(u64);
    var best8: u64 = std.math.maxInt(u64);
    for (0..3) |_| {
        const order = [_]u8{ 4, 8, 8, 4 };
        for (order) |arm| {
            const pipe = if (arm == 4) pipe4 else pipe8;
            const ob = if (arm == 4) out4.handle else out8.handle;
            const x4: c_int = if (arm == 4) 1 else 2;
            const t0 = std.Io.Timestamp.now(io, .awake);
            try Arm.run(queue, pipe, x4, upsample, in_b.handle, w_b.handle, b_b.handle, ob, st_b.handle, nw_b.handle, nb_b.handle, &pp, &up);
            const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            if (arm == 4) best4 = @min(best4, dt) else best8 = @min(best8, dt);
        }
    }
    const flops: f64 = 2.0 * f(@as(usize, sh.oc)) * f(@as(usize, sh.ic)) * 9.0 * f(hw);
    const d = try maxDiffF16Pair(allocator, &out4, &out8, @as(usize, sh.oc) * hw);
    std.debug.print(
        "  {s} {d}->{d} @ {d}x{d}   h4={d:>6.0}  h8={d:>6.0}  x{d:.2}  d={e:.2}\n",
        .{ if (upsample) "up " else "pre", sh.ic, sh.oc, sh.h, sh.w, flops / f(best4), flops / f(best8), f(best4) / f(best8), d },
    );
}

extern fn zdraw_metal_run_conv2d_prenorm_window_xn(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    stats: *anyopaque,
    norm_weight: *anyopaque,
    norm_bias: *anyopaque,
    params: *const ConvPrenormWindowParams,
    oc_x4: c_int,
) c_int;

// The product (owned f16) decoder budget: time every kernel class of the
// f16 res chain at the REAL 1024 ladder shapes and the production strip
// count (STRIP=256) and reconstruct each up-group's in-chain seconds. The
// isolated conv rate (9-11 TFLOP/s) explains under half of the chain laps
// (vae-conv-h8-20260827); sum-vs-observed per class names the rest.
fn runVaeBudgetH(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const stats_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_stats_h_sq");
    defer c.zdraw_metal_release_pipeline(stats_pipe);
    const add_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_add_window_h");
    defer c.zdraw_metal_release_pipeline(add_pipe);
    const pre8 = try compile(device, mconv_h8.conv.ptr, "conv2d_prenorm_window_h8");
    defer c.zdraw_metal_release_pipeline(pre8);
    const up8 = try compile(device, mconv_h8.conv.ptr, "conv2d_upsample_window_h8");
    defer c.zdraw_metal_release_pipeline(up8);
    const skip_pipe = try compile(device, mconv_shader.conv.ptr, "conv1x1_h");
    defer c.zdraw_metal_release_pipeline(skip_pipe);
    const stats_threads = clampThreads(c.zdraw_metal_pipeline_threads(stats_pipe));
    const add_threads = clampThreads(c.zdraw_metal_pipeline_threads(add_pipe));
    const strip: u32 = 256;
    // groups: (in_ch of the first block, ch, size, blocks, upsample after?)
    const Group = struct { ic0: u32, ch: u32, size: u32, blocks: u32, up: bool, name: []const u8 };
    const groups = [_]Group{
        .{ .ic0 = 512, .ch = 512, .size = 128, .blocks = 3, .up = true, .name = "up-0 512@128" },
        .{ .ic0 = 512, .ch = 512, .size = 256, .blocks = 3, .up = true, .name = "up-1 512@256" },
        .{ .ic0 = 512, .ch = 256, .size = 512, .blocks = 3, .up = true, .name = "up-2 256@512" },
        .{ .ic0 = 256, .ch = 128, .size = 1024, .blocks = 3, .up = false, .name = "up-3 128@1024" },
    };
    std.debug.print("\n=== product f16 decoder budget at 1024 (standalone, strip {d}, ms) ===\n", .{strip});
    // mid attention (16384 tokens x 512, one head) on the owned half route;
    // the chain lap vae-mid was 307 ms hot / ~200 ms cool.
    if (mlinear.Context.init() catch null) |lin_v| {
        var lin = lin_v;
        defer lin.deinit();
        const mid_ms = budgetMidAttn(io, allocator, &lin) catch |err| blk: {
            std.debug.print("mid attention: {s}\n", .{@errorName(err)});
            break :blk -1.0;
        };
        std.debug.print("mid attention 16384x512 (owned half): {d:.1} ms   (chain lap vae-mid 307 hot)\n", .{mid_ms});
    }
    std.debug.print("group            stats   conv(prenorm)   skip    add   upsample   sum   (chain lap k8_1 / k4_1)\n", .{});
    const laps = [_][2]f64{ .{ 119.9, 128.8 }, .{ 479.4, 506.7 }, .{ 680.7, 668.7 }, .{ 527.8, 482.7 } };
    for (groups, 0..) |g, gi| {
        const st_ms = try budgetStatsH(io, allocator, device, queue, stats_pipe, stats_threads, g.ch, g.size);
        const st_first = try budgetStatsH(io, allocator, device, queue, stats_pipe, stats_threads, g.ic0, g.size);
        // per block: stats over the input (first block: ic0 channels) + stats over conv1_out
        const stats_total = st_first + st_ms + f(g.blocks - 1) * 2.0 * st_ms;
        const conv_same = try budgetConvH(io, allocator, device, queue, pre8, g.ch, g.ch, g.size, strip);
        const conv_first = if (g.ic0 != g.ch) try budgetConvH(io, allocator, device, queue, pre8, g.ic0, g.ch, g.size, strip) else conv_same;
        const conv_total = conv_first + conv_same * f(2 * g.blocks - 1);
        const skip_ms = if (g.ic0 != g.ch) try budgetSkipH(io, allocator, device, queue, skip_pipe, g.ic0, g.ch, g.size, strip) else 0.0;
        const add_ms = try budgetAddH(io, device, queue, add_pipe, add_threads, g.ch, g.size, strip);
        const add_total = add_ms * f(g.blocks);
        const up_ms = if (g.up) try budgetUpH(io, allocator, device, queue, up8, g.ch, g.size, strip) else 0.0;
        const sum = stats_total + conv_total + skip_ms + add_total;
        std.debug.print("{s:<15} {d:7.1} {d:11.1}     {d:6.1} {d:6.1} {d:9.1} {d:7.1}   ({d:.1} / {d:.1})\n", .{ g.name, stats_total, conv_total, skip_ms, add_total, up_ms, sum, laps[gi][0], laps[gi][1] });
    }
}

fn budgetMidAttn(io: std.Io, allocator: std.mem.Allocator, lin: *mlinear.Context) !f64 {
    const tokens: usize = 16384;
    const ch: usize = 512;
    const n = tokens * ch;
    const data = try allocator.alloc(f16, n);
    defer allocator.free(data);
    var prng = std.Random.DefaultPrng.init(0x3a1d);
    const rng = prng.random();
    for (data) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.3);
    var q_b = try mbuffer.Buffer.fromBytes(lin.device, std.mem.sliceAsBytes(data));
    defer q_b.deinit();
    var k_b = try mbuffer.Buffer.fromBytes(lin.device, std.mem.sliceAsBytes(data));
    defer k_b.deinit();
    var v_b = try mbuffer.Buffer.fromBytes(lin.device, std.mem.sliceAsBytes(data));
    defer v_b.deinit();
    var o_b = try mbuffer.Buffer.empty(lin.device, n * 4);
    defer o_b.deinit();
    const cfg = attention.Config{ .tokens = tokens, .heads = 1, .kv_heads = 1, .head_dim = ch, .causal = false };
    var best: f64 = 1e18;
    for (0..4) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        try mvattn_owned.run(lin, null, q_b.handle, k_b.handle, v_b.handle, o_b.handle, cfg);
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (i > 0 and dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetStatsH(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, threads: usize, ch: u32, size: u32) !f64 {
    const n: usize = @as(usize, ch) * size * size;
    const input = try allocator.alloc(f16, n);
    defer allocator.free(input);
    var prng = std.Random.DefaultPrng.init(0x57a7 + ch);
    const rng = prng.random();
    for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var st_b = try mbuffer.Buffer.empty(device, 64 * 4);
    defer st_b.deinit();
    const np = VaeNormParams{ .channels = ch, .height = size, .width = size, .groups = 32, .dtype = 1, .bias_dtype = 1, .eps = 1e-6, .weight_offset = 0, .bias_offset = 0 };
    var best: f64 = 1e18;
    for (0..4) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (mvres_stream.zdraw_metal_run_vae_norm_stats(queue, pipe, in_b.handle, st_b.handle, &np, threads) != 0) return error.DispatchFailed;
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (i > 0 and dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetConvH(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, ic: u32, oc: u32, size: u32, strip: u32) !f64 {
    const hw: usize = @as(usize, size) * size;
    const input = try allocator.alloc(f16, @as(usize, ic) * hw);
    defer allocator.free(input);
    const wts = try allocator.alloc(f16, @as(usize, oc) * ic * 9);
    defer allocator.free(wts);
    const small = try allocator.alloc(f16, 512);
    defer allocator.free(small);
    var prng = std.Random.DefaultPrng.init(0xB0D6 + ic);
    const rng = prng.random();
    for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
    for (wts) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
    for (small) |*v| v.* = @floatCast(1.0);
    const stats = [_]f32{ 0, 1 } ** 32;
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var s_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(small));
    defer s_b.deinit();
    var st_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(&stats));
    defer st_b.deinit();
    var out_b = try mbuffer.Buffer.empty(device, @as(usize, oc) * hw * 2);
    defer out_b.deinit();
    var best: f64 = 1e18;
    for (0..4) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var row0: u32 = 0;
        while (row0 < size) : (row0 += strip) {
            const pp = ConvPrenormWindowParams{
                .in_ch = ic,
                .out_ch = oc,
                .height = size,
                .width = size,
                .ksize = 3,
                .pad = 1,
                .dtype = 1,
                .bias_dtype = 1,
                .has_bias = 1,
                .groups = 32,
                .weight_offset = 0,
                .bias_offset = 0,
                .row0 = row0,
                .row1 = @min(row0 + strip, size),
                .norm_dtype = 1,
                .norm_bias_dtype = 1,
                .norm_weight_offset = 0,
                .norm_bias_offset = 0,
            };
            if (zdraw_metal_run_conv2d_prenorm_window_xn(queue, pipe, in_b.handle, w_b.handle, s_b.handle, out_b.handle, st_b.handle, s_b.handle, s_b.handle, &pp, 2) != 0) return error.DispatchFailed;
        }
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (i > 0 and dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetUpH(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, ch: u32, in_size: u32, strip: u32) !f64 {
    const out_size = in_size * 2;
    const input = try allocator.alloc(f16, @as(usize, ch) * in_size * in_size);
    defer allocator.free(input);
    const wts = try allocator.alloc(f16, @as(usize, ch) * ch * 9);
    defer allocator.free(wts);
    var prng = std.Random.DefaultPrng.init(0x0b5 + ch);
    const rng = prng.random();
    for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
    for (wts) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var out_b = try mbuffer.Buffer.empty(device, @as(usize, ch) * out_size * out_size * 2);
    defer out_b.deinit();
    var best: f64 = 1e18;
    for (0..4) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var row0: u32 = 0;
        while (row0 < out_size) : (row0 += strip) {
            const up = ConvUpsampleWindowParams{ .channels = ch, .out_height = out_size, .out_width = out_size, .in_height = in_size, .in_width = in_size, .ksize = 3, .pad = 1, .dtype = 1, .bias_dtype = 1, .has_bias = 0, .weight_offset = 0, .bias_offset = 0, .row0 = row0, .row1 = @min(row0 + strip, out_size) };
            if (mvres_chain.zdraw_metal_run_conv2d_upsample_window(queue, pipe, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &up, 32, 2) != 0) return error.DispatchFailed;
        }
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (i > 0 and dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetSkipH(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, ic: u32, oc: u32, size: u32, strip: u32) !f64 {
    const hw: usize = @as(usize, size) * size;
    const input = try allocator.alloc(f16, @as(usize, ic) * hw);
    defer allocator.free(input);
    const wts = try allocator.alloc(f16, @as(usize, oc) * ic);
    defer allocator.free(wts);
    var prng = std.Random.DefaultPrng.init(0x5c1b + ic);
    const rng = prng.random();
    for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
    for (wts) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var out_b = try mbuffer.Buffer.empty(device, @as(usize, oc) * hw * 2);
    defer out_b.deinit();
    var best: f64 = 1e18;
    for (0..4) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var row0: u32 = 0;
        while (row0 < size) : (row0 += strip) {
            const cp = ConvWindowParams{ .in_ch = ic, .out_ch = oc, .height = size, .width = size, .ksize = 1, .pad = 0, .dtype = 1, .bias_dtype = 1, .has_bias = 0, .weight_offset = 0, .bias_offset = 0, .row0 = row0, .row1 = @min(row0 + strip, size) };
            if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp, 32) != 0) return error.DispatchFailed;
        }
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (i > 0 and dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetAddH(io: std.Io, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, threads: usize, ch: u32, size: u32, strip: u32) !f64 {
    const n: usize = @as(usize, ch) * size * size;
    var out_b = try mbuffer.Buffer.empty(device, n * 2);
    defer out_b.deinit();
    var res_b = try mbuffer.Buffer.empty(device, n * 2);
    defer res_b.deinit();
    var best: f64 = 1e18;
    for (0..4) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var row0: u32 = 0;
        while (row0 < size) : (row0 += strip) {
            const ap = mvres_stream.AddWindowParams{ .channels = ch, .height = size, .width = size, .row0 = row0, .row1 = @min(row0 + strip, size) };
            if (zdraw_metal_run_vae_add_window(queue, pipe, out_b.handle, res_b.handle, &ap, threads) != 0) return error.DispatchFailed;
        }
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (i > 0 and dt < best) best = dt;
    }
    return best / 1e6;
}

extern fn zdraw_metal_run_vae_add_window(
    queue: *anyopaque,
    pipeline: *anyopaque,
    output: *anyopaque,
    residual: *anyopaque,
    params: *const mvres_stream.AddWindowParams,
    thread_count: usize,
) c_int;

// Winograd F(4x4,3x3) vs the direct h8 prenorm conv at the decoder's shapes
// (vae-winograd-20260827): interleaved timing (d/w/w/d x3, best) and the
// error of the Winograd output against the direct half output (max|d|,
// relative RMS). The direct kernel is the verified route.
const WinoParams = extern struct {
    in_ch: u32,
    out_ch: u32,
    height: u32,
    width: u32,
    tiles_x: u32,
    tile0: u32,
    tile_count: u32,
    groups: u32,
    has_bias: u32,
    bias_dtype: u32,
    norm_dtype: u32,
    norm_bias_dtype: u32,
    weight_offset: u64,
    bias_offset: u64,
    norm_weight_offset: u64,
    norm_bias_offset: u64,
};

extern fn zdraw_metal_run_wino_conv(
    queue: *anyopaque,
    weight_pipe: *anyopaque,
    input_pipe: *anyopaque,
    gemm_pipe: *anyopaque,
    output_pipe: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    stats: *anyopaque,
    norm_w: *anyopaque,
    norm_b: *anyopaque,
    U: *anyopaque,
    V: *anyopaque,
    M: *anyopaque,
    output: *anyopaque,
    params: *const WinoParams,
    do_weight: c_int,
    batch_tiles: u32,
) c_int;

const WinoBufs = struct {
    in_b: *mbuffer.Buffer,
    w_b: *mbuffer.Buffer,
    b_b: *mbuffer.Buffer,
    st_b: *mbuffer.Buffer,
    nw_b: *mbuffer.Buffer,
    nb_b: *mbuffer.Buffer,
    u_b: *mbuffer.Buffer,
    v_b: *mbuffer.Buffer,
    m_b: *mbuffer.Buffer,
    out_b: *mbuffer.Buffer,
};

fn runWino(
    queue: *anyopaque,
    wp: *anyopaque,
    ip: *anyopaque,
    gp: *anyopaque,
    op: *anyopaque,
    b: WinoBufs,
    params: *const WinoParams,
    do_weight: c_int,
    batch: u32,
) c_int {
    return zdraw_metal_run_wino_conv(
        queue,
        wp,
        ip,
        gp,
        op,
        b.in_b.handle,
        b.w_b.handle,
        b.b_b.handle,
        b.st_b.handle,
        b.nw_b.handle,
        b.nb_b.handle,
        b.u_b.handle,
        b.v_b.handle,
        b.m_b.handle,
        b.out_b.handle,
        params,
        do_weight,
        batch,
    );
}

fn winoBatchTiles(ic: u32, oc: u32, total: u32) u32 {
    // V + M for the batch <= ~24 MB, a multiple of 64 tiles
    const per_tile: u32 = 36 * (ic + oc) * 2;
    var b: u32 = (24 * 1024 * 1024) / per_tile;
    b = (b / 64) * 64;
    if (b < 64) b = 64;
    if (b > total) b = total;
    return b;
}

fn benchWino(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
) !void {
    const pre8 = try compile(device, mconv_h8.conv.ptr, "conv2d_prenorm_window_h8");
    defer c.zdraw_metal_release_pipeline(pre8);
    const wp = try compile(device, mconv_wino.src.ptr, "wino_weight_h");
    defer c.zdraw_metal_release_pipeline(wp);
    const ip = try compile(device, mconv_wino.src.ptr, "wino_input_h");
    defer c.zdraw_metal_release_pipeline(ip);
    const gp = try compile(device, mconv_wino.src.ptr, "wino_gemm_h");
    defer c.zdraw_metal_release_pipeline(gp);
    const op = try compile(device, mconv_wino.src.ptr, "wino_output_h");
    defer c.zdraw_metal_release_pipeline(op);
    const shapes_w = [_]ConvShape{
        .{ .ic = 512, .oc = 512, .h = 128, .w = 128 },
        .{ .ic = 512, .oc = 512, .h = 256, .w = 256 },
        .{ .ic = 256, .oc = 256, .h = 512, .w = 512 },
        .{ .ic = 128, .oc = 128, .h = 1024, .w = 1024 },
    };
    std.debug.print("\nWinograd F(4x4,3x3) vs direct h8 " ++
        "(ms best of 3 interleaved; err vs direct half output)\n", .{});
    for (shapes_w) |sh| try benchWinoShape(io, allocator, device, queue, pre8, wp, ip, gp, op, sh);
}

fn benchWinoShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    pre8: *anyopaque,
    wp: *anyopaque,
    ip: *anyopaque,
    gp: *anyopaque,
    op: *anyopaque,
    sh: ConvShape,
) !void {
    const hw: usize = @as(usize, sh.h) * sh.w;
    const input = try allocator.alloc(f16, @as(usize, sh.ic) * hw);
    defer allocator.free(input);
    const wts = try allocator.alloc(f16, @as(usize, sh.oc) * sh.ic * 9);
    defer allocator.free(wts);
    const bias = try allocator.alloc(f16, sh.oc);
    defer allocator.free(bias);
    const norm_w = try allocator.alloc(f16, sh.ic);
    defer allocator.free(norm_w);
    const norm_b = try allocator.alloc(f16, sh.ic);
    defer allocator.free(norm_b);
    var stats: [64]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x3140 + sh.ic);
    const rng = prng.random();
    for (input) |*v| v.* = @floatCast(rng.floatNorm(f32));
    for (wts) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
    for (bias) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.1);
    for (norm_w) |*v| v.* = @floatCast(1.0 + rng.floatNorm(f32) * 0.1);
    for (norm_b) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.1);
    for (0..32) |g| {
        stats[g * 2] = rng.floatNorm(f32) * 0.1;
        stats[g * 2 + 1] = 1.0 + rng.floatNorm(f32) * 0.1;
    }
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_b.deinit();
    var nw_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm_w));
    defer nw_b.deinit();
    var nb_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm_b));
    defer nb_b.deinit();
    var st_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(&stats));
    defer st_b.deinit();
    var out_d = try mbuffer.Buffer.empty(device, @as(usize, sh.oc) * hw * 2);
    defer out_d.deinit();
    var out_w = try mbuffer.Buffer.empty(device, @as(usize, sh.oc) * hw * 2);
    defer out_w.deinit();
    const tiles_x: u32 = (sh.w + 3) / 4;
    const total: u32 = tiles_x * ((sh.h + 3) / 4);
    const batch = winoBatchTiles(sh.ic, sh.oc, total);
    var u_b = try mbuffer.Buffer.empty(device, 36 * @as(usize, sh.oc) * sh.ic * 2);
    defer u_b.deinit();
    var v_b = try mbuffer.Buffer.empty(device, 36 * @as(usize, sh.ic) * batch * 2);
    defer v_b.deinit();
    var m_b = try mbuffer.Buffer.empty(device, 36 * @as(usize, sh.oc) * batch * 2);
    defer m_b.deinit();
    const pp = ConvPrenormWindowParams{
        .in_ch = sh.ic,
        .out_ch = sh.oc,
        .height = sh.h,
        .width = sh.w,
        .ksize = 3,
        .pad = 1,
        .dtype = 1,
        .bias_dtype = 1,
        .has_bias = 1,
        .groups = 32,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = sh.h,
        .norm_dtype = 1,
        .norm_bias_dtype = 1,
        .norm_weight_offset = 0,
        .norm_bias_offset = 0,
    };
    const wpar = WinoParams{
        .in_ch = sh.ic,
        .out_ch = sh.oc,
        .height = sh.h,
        .width = sh.w,
        .tiles_x = tiles_x,
        .tile0 = 0,
        .tile_count = 0,
        .groups = 32,
        .has_bias = 1,
        .bias_dtype = 1,
        .norm_dtype = 1,
        .norm_bias_dtype = 1,
        .weight_offset = 0,
        .bias_offset = 0,
        .norm_weight_offset = 0,
        .norm_bias_offset = 0,
    };
    const bufs_w = WinoBufs{
        .in_b = &in_b,
        .w_b = &w_b,
        .b_b = &b_b,
        .st_b = &st_b,
        .nw_b = &nw_b,
        .nb_b = &nb_b,
        .u_b = &u_b,
        .v_b = &v_b,
        .m_b = &m_b,
        .out_b = &out_w,
    };
    // the weight transform alone (2), x10 in one buffer (3), an empty buffer (4)
    if (runWino(queue, wp, ip, gp, op, bufs_w, &wpar, 2, batch) != 0)
        return error.DispatchFailed;
    var tw: f64 = 1e18;
    var tw10: f64 = 1e18;
    var tempty: f64 = 1e18;
    for (0..3) |_| {
        const tw0 = std.Io.Timestamp.now(io, .awake);
        if (runWino(queue, wp, ip, gp, op, bufs_w, &wpar, 2, batch) != 0)
            return error.DispatchFailed;
        tw = @min(tw, @as(f64, @floatFromInt(tw0.untilNow(io, .awake).toNanoseconds())));
        const t10 = std.Io.Timestamp.now(io, .awake);
        if (runWino(queue, wp, ip, gp, op, bufs_w, &wpar, 3, batch) != 0)
            return error.DispatchFailed;
        tw10 = @min(tw10, @as(f64, @floatFromInt(t10.untilNow(io, .awake).toNanoseconds())));
        const te = std.Io.Timestamp.now(io, .awake);
        if (runWino(queue, wp, ip, gp, op, bufs_w, &wpar, 4, batch) != 0)
            return error.DispatchFailed;
        tempty = @min(tempty, @as(f64, @floatFromInt(te.untilNow(io, .awake).toNanoseconds())));
    }
    std.debug.print("    U alone {d:.2} ms, U x10 {d:.2} ms, empty buffer {d:.2} ms\n", .{ tw / 1e6, tw10 / 1e6, tempty / 1e6 });
    var best_d: f64 = 1e18;
    var best_w: f64 = 1e18;
    for (0..3) |_| {
        const order = [_]u8{ 'd', 'w', 'w', 'd' };
        for (order) |arm| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (arm == 'd') {
                if (zdraw_metal_run_conv2d_prenorm_window_xn(
                    queue,
                    pre8,
                    in_b.handle,
                    w_b.handle,
                    b_b.handle,
                    out_d.handle,
                    st_b.handle,
                    nw_b.handle,
                    nb_b.handle,
                    &pp,
                    2,
                ) != 0) return error.DispatchFailed;
            } else {
                if (runWino(queue, wp, ip, gp, op, bufs_w, &wpar, 0, batch) != 0)
                    return error.DispatchFailed;
            }
            const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
            if (arm == 'd') best_d = @min(best_d, dt) else best_w = @min(best_w, dt);
        }
    }
    // error vs the direct half output
    const n = @as(usize, sh.oc) * hw;
    const dv = try allocator.alloc(f16, n);
    defer allocator.free(dv);
    const wv = try allocator.alloc(f16, n);
    defer allocator.free(wv);
    c.zdraw_metal_read_buffer(out_d.handle, std.mem.sliceAsBytes(dv).ptr, n * 2);
    c.zdraw_metal_read_buffer(out_w.handle, std.mem.sliceAsBytes(wv).ptr, n * 2);
    var worst: f64 = 0;
    var se: f64 = 0;
    var ref2: f64 = 0;
    var bad: usize = 0;
    for (dv, wv) |a, b| {
        const af: f64 = @floatCast(a);
        const bf: f64 = @floatCast(b);
        if (!std.math.isFinite(bf)) {
            bad += 1;
            continue;
        }
        const d = @abs(af - bf);
        worst = @max(worst, d);
        se += d * d;
        ref2 += af * af;
    }
    const flops: f64 = 2.0 * f(@as(usize, sh.oc)) * f(@as(usize, sh.ic)) * 9.0 * f(hw);
    std.debug.print(
        "  {d}->{d} @ {d}x{d}  direct {d:7.2} ms ({d:5.0} GF/s)  wino {d:7.2} ms " ++
            "(x{d:.2}, batch {d} tiles, U {d:.1} ms)  max|d| {e:.2}  relRMS {e:.2}  nonfinite {d}\n",
        .{
            sh.ic,
            sh.oc,
            sh.h,
            sh.w,
            best_d / 1e6,
            flops / best_d,
            best_w / 1e6,
            best_d / best_w,
            batch,
            tw / 1e6,
            worst,
            @sqrt(se / @max(ref2, 1e-30)),
            bad,
        },
    );
}

// The Metal 4 tensor path (mgemm_mpp_shader.zig) against the production
// GEMM at the Klein shapes: rate (interleaved d/m64/m32/d, best of `runs`)
// and max|d| vs the production f32 output. Rigel (arXiv 2606.12765) puts
// matmul2d at 1.05-1.21x simdgroup_matrix on M4 Max.
extern fn zdraw_metal_compile_mpp(
    device: *anyopaque,
    source: [*:0]const u8,
    entry: [*:0]const u8,
) ?*anyopaque;
extern fn zdraw_metal_run_gemm_mpp(
    queue: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_out: *anyopaque,
    params: *const c.GemmParams,
    tn: u32,
) c_int;

fn benchMpp(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    runs: u32,
) !void {
    const direct = zdraw_metal_f16a_direct_make(device) orelse return error.MakeFailed;
    defer c.zdraw_metal_release_pipeline(direct);
    const mpp64 = zdraw_metal_compile_mpp(device, mgemm_mpp.src.ptr, "gemm_mpp64") orelse {
        std.debug.print("mpp: unavailable on this system (needs macOS 26 / Metal 4)\n", .{});
        return;
    };
    defer c.zdraw_metal_release_pipeline(mpp64);
    const mpp32 = zdraw_metal_compile_mpp(device, mgemm_mpp.src.ptr, "gemm_mpp32") orelse
        return error.MakeFailed;
    defer c.zdraw_metal_release_pipeline(mpp32);
    const KS = struct { label: []const u8, m: u32, k: u32, n: u32 };
    const klein = [_]KS{
        .{ .label = "single.qkv_gate(qkv) ", .m = 4608, .k = 3072, .n = 3072 },
        .{ .label = "single.qkv_gate(gate)", .m = 4608, .k = 3072, .n = 18432 },
        .{ .label = "single.ffn_down      ", .m = 4608, .k = 12288, .n = 3072 },
        .{ .label = "double.qkv_gate(qkv) ", .m = 4096, .k = 3072, .n = 3072 },
        .{ .label = "double.qkv_gate(gate)", .m = 4096, .k = 3072, .n = 18432 },
        .{ .label = "double.ffn_down      ", .m = 4096, .k = 9216, .n = 3072 },
    };
    std.debug.print("\nMetal 4 tensor path vs production GEMM " ++
        "({d} runs; GFLOP/s; max|d| vs direct)\n", .{runs});
    std.debug.print("  shape                   direct   mpp64   mpp32" ++
        "   x64   x32   d64      d32\n", .{});
    for (klein) |s| {
        const mk: usize = @as(usize, s.m) * s.k;
        const nk: usize = @as(usize, s.n) * s.k;
        const mn: usize = @as(usize, s.m) * s.n;
        const a16 = try allocator.alloc(f16, mk);
        defer allocator.free(a16);
        const w16 = try allocator.alloc(f16, nk);
        defer allocator.free(w16);
        var prng = std.Random.DefaultPrng.init(0x4B1E + s.n);
        const rng = prng.random();
        for (a16) |*v| v.* = @floatCast(rng.floatNorm(f32));
        for (w16) |*v| v.* = @floatCast(rng.floatNorm(f32) * 0.05);
        var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
        defer a_buf.deinit();
        var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w16));
        defer w_buf.deinit();
        var c_d = try nanBuffer(allocator, device, mn, 4);
        defer c_d.deinit();
        var c_64 = try nanBuffer(allocator, device, mn, 4);
        defer c_64.deinit();
        var c_32 = try nanBuffer(allocator, device, mn, 4);
        defer c_32.deinit();
        const mi: c_int = @intCast(s.m);
        const ni: c_int = @intCast(s.n);
        const ki: c_int = @intCast(s.k);
        const gp = c.GemmParams{ .m = s.m, .k = s.k, .n = s.n, .dtype = 1, .mode = 1 };
        // warm
        _ = try timeOurs(io, queue, direct, &a_buf, &w_buf, &c_d, mi, ni, ki, 1);
        if (zdraw_metal_run_gemm_mpp(
            queue,
            mpp64,
            a_buf.handle,
            w_buf.handle,
            c_64.handle,
            &gp,
            64,
        ) != 0) return error.DispatchFailed;
        if (zdraw_metal_run_gemm_mpp(
            queue,
            mpp32,
            a_buf.handle,
            w_buf.handle,
            c_32.handle,
            &gp,
            32,
        ) != 0) return error.DispatchFailed;
        var best_d: u64 = std.math.maxInt(u64);
        var best_64: u64 = std.math.maxInt(u64);
        var best_32: u64 = std.math.maxInt(u64);
        for (0..runs) |_| {
            const td = try timeOurs(io, queue, direct, &a_buf, &w_buf, &c_d, mi, ni, ki, 1);
            best_d = @min(best_d, td);
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (zdraw_metal_run_gemm_mpp(
                queue,
                mpp64,
                a_buf.handle,
                w_buf.handle,
                c_64.handle,
                &gp,
                64,
            ) != 0) return error.DispatchFailed;
            best_64 = @min(best_64, @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds())));
            const t1 = std.Io.Timestamp.now(io, .awake);
            if (zdraw_metal_run_gemm_mpp(
                queue,
                mpp32,
                a_buf.handle,
                w_buf.handle,
                c_32.handle,
                &gp,
                32,
            ) != 0) return error.DispatchFailed;
            best_32 = @min(best_32, @as(u64, @intCast(t1.untilNow(io, .awake).toNanoseconds())));
            const td2 = try timeOurs(io, queue, direct, &a_buf, &w_buf, &c_d, mi, ni, ki, 1);
            best_d = @min(best_d, td2);
        }
        const ref = try allocator.alloc(f32, mn);
        defer allocator.free(ref);
        c.zdraw_metal_read_buffer(c_d.handle, std.mem.sliceAsBytes(ref).ptr, mn * 4);
        const d64 = try maxDiffF32(allocator, &c_64, ref);
        const d32 = try maxDiffF32(allocator, &c_32, ref);
        const flops: f64 = 2.0 * f(mn) * f(@as(usize, s.k));
        std.debug.print("  {s} {d:7.0} {d:7.0} {d:7.0}  {d:4.2}  {d:4.2}  {e:.1}  {e:.1}\n", .{
            s.label,                flops / f(best_d),      flops / f(best_64), flops / f(best_32),
            f(best_d) / f(best_64), f(best_d) / f(best_32), d64,                d32,
        });
    }
}

fn benchSkipShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    v1: *anyopaque,
    s: ConvShape,
) !void {
    const hw: usize = @as(usize, s.h) * s.w;
    const in_n: usize = @as(usize, s.ic) * hw;
    const out_n: usize = @as(usize, s.oc) * hw;
    const w_n: usize = @as(usize, s.oc) * s.ic;
    var prng = std.Random.DefaultPrng.init(0x5717 + s.ic);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32);
    const weight = try allocator.alloc(f32, w_n);
    defer allocator.free(weight);
    for (weight) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const bias = try allocator.alloc(f32, s.oc);
    defer allocator.free(bias);
    for (bias) |*v| v.* = 0.0;
    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(weight));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var o_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o_buf.deinit();
    const cp = c.ConvParams{
        .in_ch = s.ic,
        .out_ch = s.oc,
        .height = s.h,
        .width = s.w,
        .ksize = 1,
        .pad = 0,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    const wp = convWindowParams(cp, 0, s.h);
    const ns = try timeConv(io, queue, v1, in_buf.handle, w_buf.handle, b_buf.handle, o_buf.handle, &wp, 32);
    std.debug.print("  skip 1x1 {d:>3}->{d:>3} @ {d:>4}x{d:<4}  {d:>6.1} ms\n", .{
        s.ic, s.oc, s.h, s.w, f(ns) / 1e6,
    });
}

fn benchConvShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    v1: *anyopaque,
    v2: *anyopaque,
    s: ConvShape,
) !void {
    const hw: usize = @as(usize, s.h) * s.w;
    const in_n: usize = @as(usize, s.ic) * hw;
    const out_n: usize = @as(usize, s.oc) * hw;
    const w_n: usize = @as(usize, s.oc) * s.ic * 9;

    var prng = std.Random.DefaultPrng.init(0xC0A + s.ic + s.h);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32);
    const weight = try allocator.alloc(f32, w_n);
    defer allocator.free(weight);
    for (weight) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const bias = try allocator.alloc(f32, s.oc);
    defer allocator.free(bias);
    for (bias) |*v| v.* = rng.floatNorm(f32) * 0.1;

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(weight));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var o1_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o1_buf.deinit();
    var o2_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o2_buf.deinit();

    const cp = c.ConvParams{
        .in_ch = s.ic,
        .out_ch = s.oc,
        .height = s.h,
        .width = s.w,
        .ksize = 3,
        .pad = 1,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    const wp = convWindowParams(cp, 0, s.h);
    const threads: usize = 32;

    const v1_ns = try timeConv(io, queue, v1, in_buf.handle, w_buf.handle, b_buf.handle, o1_buf.handle, &wp, threads);
    const v2_ns = try timeConv(io, queue, v2, in_buf.handle, w_buf.handle, b_buf.handle, o2_buf.handle, &wp, threads);

    const o1 = try allocator.alloc(f32, out_n);
    defer allocator.free(o1);
    c.zdraw_metal_read_buffer(o1_buf.handle, std.mem.sliceAsBytes(o1).ptr, out_n * 4);
    const o2 = try allocator.alloc(f32, out_n);
    defer allocator.free(o2);
    c.zdraw_metal_read_buffer(o2_buf.handle, std.mem.sliceAsBytes(o2).ptr, out_n * 4);
    const bit_d = maxBitDiff(o1, o2);

    const flops: f64 = 2.0 * f(out_n) * f(@as(usize, s.ic) * 9);
    std.debug.print(
        "  {d:>3}->{d:>3} @ {d:>4}x{d:<4}  {d:>6.1} {d:>6.1}  {d:>5.2}x  {d}\n",
        .{ s.ic, s.oc, s.h, s.w, flops / f(v1_ns), flops / f(v2_ns), f(v1_ns) / f(v2_ns), bit_d },
    );
}

fn timeConv(
    io: std.Io,
    queue: *anyopaque,
    pipe: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    wp: *const ConvWindowParams,
    threads: usize,
) !u64 {
    if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, input, weight, bias, output, wp, threads) != 0) {
        return error.DispatchFailed;
    }
    var samples: [3]u64 = undefined;
    for (&samples) |*sample| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, input, weight, bias, output, wp, threads) != 0) {
            return error.DispatchFailed;
        }
        sample.* = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, &samples, {}, std.sort.asc(u64));
    return samples[1];
}

fn benchKernelsShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    s: Shape,
    runs: u32,
) !void {
    const m = @as(usize, s.m);
    const n = @as(usize, s.n);
    const k = @as(usize, s.k);
    const a16 = try allocator.alloc(f16, m * k);
    defer allocator.free(a16);
    const a32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a32);
    const b16 = try allocator.alloc(f16, n * k);
    defer allocator.free(b16);
    var prng = std.Random.DefaultPrng.init(0x5733 + s.m + s.n + s.k);
    fillF16(a16, prng.random());
    fillF16(b16, prng.random());
    for (a16, a32) |hv, *fv| fv.* = @floatCast(hv);

    var a16_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
    defer a16_buf.deinit();
    var a32_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a32));
    defer a32_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(b16));
    defer w_buf.deinit();
    var c_direct = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_direct.deinit();
    var c_custom = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_custom.deinit();
    var c_steel = try mbuffer.Buffer.empty(device, m * n * 2);
    defer c_steel.deinit();
    var c_mps = try mbuffer.Buffer.empty(device, m * n * 2);
    defer c_mps.deinit();

    const mps_ctx = mps_c.zdraw_mps_gemm_make(device, a16_buf.handle, w_buf.handle, c_mps.handle, s.m, s.n, s.k) orelse
        return error.MpsMakeFailed;
    defer mps_c.zdraw_mps_gemm_free(mps_ctx);

    const direct = zdraw_metal_ours16_make(device) orelse return error.MakeFailed;
    const custom = zdraw_metal_ourscustom_make(device) orelse return error.MakeFailed;
    const am: c_int = if (s.m % 64 == 0) 1 else 0;
    const an: c_int = if (s.n % 64 == 0) 1 else 0;
    const ak: c_int = if (s.k % 16 == 0) 1 else 0;
    const steel = zdraw_metal_steel_make(device, steelLibPath(), 64, am, an, ak);

    const mi: c_int = @intCast(s.m);
    const ni: c_int = @intCast(s.n);
    const ki: c_int = @intCast(s.k);

    // Correctness: custom (f16 A) vs direct (f32 A), both f32 out, same math.
    _ = zdraw_metal_ours64_run(queue, direct, a32_buf.handle, w_buf.handle, c_direct.handle, mi, ni, ki);
    _ = zdraw_metal_ours64_run(queue, custom, a16_buf.handle, w_buf.handle, c_custom.handle, mi, ni, ki);
    const od = try allocator.alloc(f32, m * n);
    defer allocator.free(od);
    const oc = try allocator.alloc(f32, m * n);
    defer allocator.free(oc);
    c.zdraw_metal_read_buffer(c_direct.handle, std.mem.sliceAsBytes(od).ptr, m * n * 4);
    c.zdraw_metal_read_buffer(c_custom.handle, std.mem.sliceAsBytes(oc).ptr, m * n * 4);
    var maxerr: f64 = 0;
    for (od, oc) |dv, cv| {
        const d = @abs(@as(f64, dv) - @as(f64, cv));
        if (d > maxerr) maxerr = d;
    }

    const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
    const mps_g = flops / f(try timeMps(io, queue, mps_ctx, runs));
    const direct_g = flops / f(try timeOurs(io, queue, direct, &a32_buf, &w_buf, &c_direct, mi, ni, ki, runs));
    const custom_g = flops / f(try timeOurs(io, queue, custom, &a16_buf, &w_buf, &c_custom, mi, ni, ki, runs));
    var steel_r: f64 = 0;
    if (steel) |sp| steel_r = (flops / f(try timeSteel(io, queue, sp, &a16_buf, &w_buf, &c_steel, mi, ni, ki, 64, runs))) / mps_g;
    std.debug.print("  {d:>4} x {d:>5} x {d:>5}     {d:>6.2}  {d:>6.2}  {d:>6.2}   {d:.4}\n", .{
        s.m, s.k, s.n, direct_g / mps_g, custom_g / mps_g, steel_r, maxerr,
    });
}

fn timeOurs(io: std.Io, queue: *anyopaque, pipe: *anyopaque, a: *mbuffer.Buffer, w: *mbuffer.Buffer, cb: *mbuffer.Buffer, mi: c_int, ni: c_int, ki: c_int, runs: u32) !u64 {
    var wm: u32 = 0;
    while (wm < 3) : (wm += 1) _ = zdraw_metal_ours64_run(queue, pipe, a.handle, w.handle, cb.handle, mi, ni, ki);
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = zdraw_metal_ours64_run(queue, pipe, a.handle, w.handle, cb.handle, mi, ni, ki);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn timeSteel(io: std.Io, queue: *anyopaque, pipe: *anyopaque, a: *mbuffer.Buffer, b: *mbuffer.Buffer, d: *mbuffer.Buffer, mi: c_int, ni: c_int, ki: c_int, bmi: c_int, runs: u32) !u64 {
    var w: u32 = 0;
    while (w < 3) : (w += 1) _ = zdraw_metal_steel_run(queue, pipe, a.handle, b.handle, d.handle, mi, ni, ki, bmi);
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = zdraw_metal_steel_run(queue, pipe, a.handle, b.handle, d.handle, mi, ni, ki, bmi);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn timeMps(io: std.Io, queue: *anyopaque, ctx: *anyopaque, runs: u32) !u64 {
    var w: u32 = 0;
    while (w < 3) : (w += 1) _ = mps_c.zdraw_mps_gemm_run(queue, ctx);
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = mps_c.zdraw_mps_gemm_run(queue, ctx);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn timeKind(io: std.Io, kind: Kind, runs: u32, ctx: *Ctx) !u64 {
    var w: u32 = 0;
    while (w < 3) : (w += 1) try dispatch(kind, ctx);
    var samples: [512]u64 = undefined;
    const count = @min(runs, 512);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        try dispatch(kind, ctx);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn dispatch(kind: Kind, ctx: *Ctx) !void {
    const code = switch (kind) {
        .ours => c.zdraw_metal_run_gemm(ctx.queue, ctx.pipe, ctx.a, ctx.w, ctx.c_ours, &ctx.params),
        .mps => mps_c.zdraw_mps_gemm_run(ctx.queue, ctx.mps),
    };
    if (code != 0) return error.DispatchFailed;
}

const Err = struct { ours: f64, mps: f64 };

fn checkErr(allocator: std.mem.Allocator, ctx: *Ctx, a_data: []const f16, w_data: []const f16) !Err {
    const m = ctx.params.m;
    const n = ctx.params.n;
    const k = ctx.params.k;
    const ours = try allocator.alloc(f32, @as(usize, m) * n);
    defer allocator.free(ours);
    const mps = try allocator.alloc(f16, @as(usize, m) * n);
    defer allocator.free(mps);
    c.zdraw_metal_read_buffer(ctx.c_ours, std.mem.sliceAsBytes(ours).ptr, @as(usize, m) * n * 4);
    c.zdraw_metal_read_buffer(ctx.c_mps, std.mem.sliceAsBytes(mps).ptr, @as(usize, m) * n * 2);

    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();
    var eo: f64 = 0;
    var em: f64 = 0;
    var s: u32 = 0;
    while (s < 32) : (s += 1) {
        const mi = rng.uintLessThan(u32, m);
        const ni = rng.uintLessThan(u32, n);
        var ref: f64 = 0;
        var kk: u32 = 0;
        while (kk < k) : (kk += 1) {
            const av: f64 = a_data[@as(usize, mi) * k + kk];
            const wv: f64 = w_data[@as(usize, ni) * k + kk];
            ref += av * wv;
        }
        const idx = @as(usize, mi) * n + ni;
        const ov: f64 = ours[idx];
        const mv: f64 = mps[idx];
        eo = @max(eo, relErr(ov, ref));
        em = @max(em, relErr(mv, ref));
    }
    return .{ .ours = eo, .mps = em };
}

fn relErr(val: f64, ref: f64) f64 {
    return @abs(val - ref) / (@abs(ref) + 1e-6);
}

// --- Fair correctness gates for the production gemm_exact kernel ---------------
// f32 activations, bf16/f16 weights read at a byte offset, checked against a CPU
// reference computed from the same weight bits. Small shapes are checked in full;
// large shapes are checked at corners plus random interior cells. MPS is the
// timing oracle above; these gates depend only on the CPU reference.

const Gate = struct {
    m: u32,
    k: u32,
    n: u32,
    dtype: u32, // 1 = f16, 2 = bf16, 3 = f32
    pad: u32, // leading weight elements before the matrix (offset path)
    full: bool, // full check (small) vs stratified (large)
    label: []const u8,
};

const gates = [_]Gate{
    .{ .m = 32, .k = 8, .n = 32, .dtype = 2, .pad = 0, .full = true, .label = "bf16 small  full  " },
    .{ .m = 32, .k = 64, .n = 64, .dtype = 1, .pad = 0, .full = true, .label = "f16  small  full  " },
    .{ .m = 64, .k = 3840, .n = 128, .dtype = 2, .pad = 0, .full = true, .label = "bf16 tall   full  " },
    .{ .m = 256, .k = 3840, .n = 10240, .dtype = 2, .pad = 0, .full = false, .label = "bf16 ffn-up strat " },
    .{ .m = 256, .k = 10240, .n = 3840, .dtype = 2, .pad = 0, .full = false, .label = "bf16 ffn-dn strat " },
    .{ .m = 1024, .k = 3840, .n = 10240, .dtype = 2, .pad = 0, .full = false, .label = "bf16 up@1k  strat " },
    .{ .m = 32, .k = 8, .n = 32, .dtype = 2, .pad = 24, .full = true, .label = "bf16 offset full  " },
    .{ .m = 32, .k = 64, .n = 64, .dtype = 3, .pad = 0, .full = true, .label = "f32  small  full  " },
    .{ .m = 64, .k = 3840, .n = 128, .dtype = 3, .pad = 0, .full = true, .label = "f32  tall   full  " },
    .{ .m = 256, .k = 3840, .n = 10240, .dtype = 3, .pad = 0, .full = false, .label = "f32  ffn-up strat " },
    .{ .m = 256, .k = 10240, .n = 3840, .dtype = 3, .pad = 17, .full = false, .label = "f32  ffn-dn off   " },
};

const GateErr = struct { rel: f64, abs: f64 };

fn runGates(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque) !void {
    std.debug.print("\ngemm_exact gates (f32 act, f32/bf16/f16 weight):   M x K x N         max-rel  max-abs  verdict\n", .{});
    var all_pass = true;
    for (gates) |g| {
        if (!try runGate(allocator, device, queue, pipe, g)) all_pass = false;
    }
    std.debug.print("  {s}\n", .{if (all_pass) "all gates PASS" else "GATES FAILED"});
    if (!all_pass) return error.GemmGateFailed;
}

fn runGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, g: Gate) !bool {
    const a = try allocator.alloc(f32, @as(usize, g.m) * g.k);
    defer allocator.free(a);
    var prng = std.Random.DefaultPrng.init(0xA11 + g.m + g.k + g.n + g.pad);
    const rng = prng.random();
    for (a) |*v| v.* = rng.float(f32) * 0.2 - 0.1;

    const esize = dsize(g.dtype);
    const wbytes = try allocator.alloc(u8, (@as(usize, g.pad) + @as(usize, g.n) * g.k) * esize);
    defer allocator.free(wbytes);
    const wvals = try allocator.alloc(f32, @as(usize, g.n) * g.k);
    defer allocator.free(wvals);
    fillWeight(wbytes, wvals, g, rng);

    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, wbytes);
    defer w_buf.deinit();
    var c_buf = try mbuffer.Buffer.empty(device, @as(usize, g.m) * g.n * 4);
    defer c_buf.deinit();

    var params = c.GemmParams{
        .m = g.m,
        .k = g.k,
        .n = g.n,
        .dtype = g.dtype,
        .mode = 1,
        .weight_offset = @as(u64, g.pad) * esize,
    };
    if (c.zdraw_metal_run_gemm(queue, pipe, a_buf.handle, w_buf.handle, c_buf.handle, &params) != 0) {
        return error.DispatchFailed;
    }

    const out = try allocator.alloc(f32, @as(usize, g.m) * g.n);
    defer allocator.free(out);
    c.zdraw_metal_read_buffer(c_buf.handle, std.mem.sliceAsBytes(out).ptr, @as(usize, g.m) * g.n * 4);

    const err = checkGate(a, wvals, out, g);
    const ok = err.abs < 0.02 and err.rel < 0.02;
    std.debug.print("  {s}  {d:>5} x {d:>5} x {d:>5}   {d:.5}  {d:.5}  {s}\n", .{
        g.label, g.m, g.k, g.n, err.rel, err.abs, if (ok) "PASS" else "FAIL",
    });
    return ok;
}

fn dsize(dtype: u32) usize {
    return if (dtype == 3) 4 else 2;
}

fn fillWeight(bytes: []u8, vals: []f32, g: Gate, rng: std.Random) void {
    const esize = dsize(g.dtype);
    for (0..@as(usize, g.pad) * esize) |i| bytes[i] = 0x5A; // junk before the offset
    for (0..@as(usize, g.n) * g.k) |i| {
        const raw = rng.float(f32) * 0.2 - 0.1;
        const at = (@as(usize, g.pad) + i) * esize;
        if (g.dtype == 3) {
            const word: u32 = @bitCast(raw);
            bytes[at] = @truncate(word);
            bytes[at + 1] = @truncate(word >> 8);
            bytes[at + 2] = @truncate(word >> 16);
            bytes[at + 3] = @truncate(word >> 24);
            vals[i] = raw;
        } else if (g.dtype == 2) {
            const hi: u16 = @truncate(@as(u32, @bitCast(raw)) >> 16);
            bytes[at] = @truncate(hi);
            bytes[at + 1] = @truncate(hi >> 8);
            vals[i] = @bitCast(@as(u32, hi) << 16);
        } else {
            const h: f16 = @floatCast(raw);
            const hb: u16 = @bitCast(h);
            bytes[at] = @truncate(hb);
            bytes[at + 1] = @truncate(hb >> 8);
            vals[i] = h;
        }
    }
}

fn checkGate(a: []const f32, wvals: []const f32, out: []const f32, g: Gate) GateErr {
    var worst = GateErr{ .rel = 0, .abs = 0 };
    if (g.full) {
        for (0..g.m) |mi| {
            for (0..g.n) |ni| accum(&worst, cellErr(a, wvals, out, g, mi, ni));
        }
        return worst;
    }
    const corners = [_][2]usize{
        .{ 0, 0 },
        .{ 0, g.n - 1 },
        .{ g.m - 1, 0 },
        .{ g.m - 1, g.n - 1 },
    };
    for (corners) |cell| accum(&worst, cellErr(a, wvals, out, g, cell[0], cell[1]));
    var prng = std.Random.DefaultPrng.init(0x5A + g.m + g.n);
    const rng = prng.random();
    var s: u32 = 0;
    while (s < 64) : (s += 1) {
        const mi = rng.uintLessThan(u32, g.m);
        const ni = rng.uintLessThan(u32, g.n);
        accum(&worst, cellErr(a, wvals, out, g, mi, ni));
    }
    return worst;
}

fn accum(worst: *GateErr, e: GateErr) void {
    worst.rel = @max(worst.rel, e.rel);
    worst.abs = @max(worst.abs, e.abs);
}

fn cellErr(a: []const f32, wvals: []const f32, out: []const f32, g: Gate, mi: usize, ni: usize) GateErr {
    const k: usize = g.k;
    const n: usize = g.n;
    var ref: f64 = 0;
    var kk: usize = 0;
    while (kk < k) : (kk += 1) ref += @as(f64, a[mi * k + kk]) * @as(f64, wvals[ni * k + kk]);
    const got: f64 = out[mi * n + ni];
    const abs = @abs(got - ref);
    return .{ .rel = abs / (@abs(ref) + 1e-6), .abs = abs };
}

// gemm_half is lossy, so it gets its own loose check against the f32 CPU
// reference: ~1e-2 relative error is expected half-precision rounding; a much
// larger error means the kernel itself is wrong.
fn checkHalf(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque) !void {
    std.debug.print("\ngemm_half correctness vs f32 CPU ref:   M x K x N          max-rel  max-abs\n", .{});
    const half_shapes = [_]Gate{
        .{ .m = 1056, .k = 3840, .n = 64, .dtype = 3, .pad = 0, .full = false, .label = "half m1056 " },
        .{ .m = 1536, .k = 3840, .n = 64, .dtype = 3, .pad = 0, .full = false, .label = "half m1536 " },
        .{ .m = 2048, .k = 3840, .n = 64, .dtype = 3, .pad = 0, .full = false, .label = "half m2048 " },
        .{ .m = 2336, .k = 3840, .n = 64, .dtype = 3, .pad = 0, .full = false, .label = "half m2336 " },
        .{ .m = 4128, .k = 3840, .n = 64, .dtype = 3, .pad = 0, .full = false, .label = "half m4128 " },
        .{ .m = 1056, .k = 64, .n = 3840, .dtype = 3, .pad = 0, .full = false, .label = "half patch1" },
        .{ .m = 2336, .k = 64, .n = 3840, .dtype = 3, .pad = 0, .full = false, .label = "half patch2" },
        .{ .m = 4128, .k = 64, .n = 3840, .dtype = 3, .pad = 0, .full = false, .label = "half patch4" },
        .{ .m = 64, .k = 3840, .n = 128, .dtype = 3, .pad = 0, .full = true, .label = "half tall  " },
        .{ .m = 256, .k = 3840, .n = 10240, .dtype = 3, .pad = 0, .full = false, .label = "half ffn-up" },
        .{ .m = 256, .k = 10240, .n = 3840, .dtype = 3, .pad = 0, .full = false, .label = "half ffn-dn" },
    };
    for (half_shapes) |g| {
        const err = try halfErr(allocator, device, queue, pipe, g);
        std.debug.print("  {s}  {d:>5} x {d:>5} x {d:>5}   {d:.5}  {d:.5}\n", .{
            g.label, g.m, g.k, g.n, err.rel, err.abs,
        });
    }
    std.debug.print("  (rel ~1e-2 is expected half rounding; rel >> 0.1 means a kernel bug)\n", .{});
}

fn halfErr(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, pipe: *anyopaque, g: Gate) !GateErr {
    const a = try allocator.alloc(f32, @as(usize, g.m) * g.k);
    defer allocator.free(a);
    var prng = std.Random.DefaultPrng.init(0xB22 + g.m + g.k + g.n);
    const rng = prng.random();
    for (a) |*v| v.* = rng.float(f32) * 0.2 - 0.1;
    const wbytes = try allocator.alloc(u8, @as(usize, g.n) * g.k * dsize(g.dtype));
    defer allocator.free(wbytes);
    const wvals = try allocator.alloc(f32, @as(usize, g.n) * g.k);
    defer allocator.free(wvals);
    fillWeight(wbytes, wvals, g, rng);
    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, wbytes);
    defer w_buf.deinit();
    var c_buf = try mbuffer.Buffer.empty(device, @as(usize, g.m) * g.n * 4);
    defer c_buf.deinit();
    var params = c.GemmParams{
        .m = g.m,
        .k = g.k,
        .n = g.n,
        .dtype = g.dtype,
        .mode = 2,
        .weight_offset = 0,
    };
    if (c.zdraw_metal_run_gemm(queue, pipe, a_buf.handle, w_buf.handle, c_buf.handle, &params) != 0) {
        return error.DispatchFailed;
    }
    const out = try allocator.alloc(f32, @as(usize, g.m) * g.n);
    defer allocator.free(out);
    c.zdraw_metal_read_buffer(c_buf.handle, std.mem.sliceAsBytes(out).ptr, @as(usize, g.m) * g.n * 4);
    return checkGate(a, wvals, out, g);
}

// --- Kernel speed: our f32 gemm_exact vs the naive linear_rows GEMV ------------
// The real Z-Image transformer is f32, so this is the comparison that matters for
// the model path: does our simdgroup kernel beat the existing per-row GEMV on the
// actual FFN shapes? Synthetic f32 data, no model load. GFLOP/s + ours/naive.

const SpeedKind = enum { gemm, half, naive };

const SpeedCtx = struct {
    queue: *anyopaque,
    gemm_pipe: *anyopaque,
    half_pipe: *anyopaque,
    linear_pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    out: *anyopaque,
    bias: *anyopaque,
    gparams: c.GemmParams,
    lparams: c.LinearParams,
    threads: usize,
};

fn runSpeed(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    gemm_pipe: *anyopaque,
    half_pipe: *anyopaque,
    linear_pipe: *anyopaque,
    runs: u32,
) !void {
    std.debug.print("\nkernel speed (f32 weights, GFLOP/s):  M x K x N    exact     half    naive   half/naive\n", .{});
    const threads = clampThreads(c.zdraw_metal_pipeline_threads(linear_pipe));
    for (shapes) |s| {
        if (s.m % 32 != 0 or s.n % 32 != 0 or s.k % 8 != 0) continue;
        try speedShape(io, allocator, device, queue, gemm_pipe, half_pipe, linear_pipe, threads, s, runs);
    }
    std.debug.print("  (half is lossy; gate it with `zig build quality -- --mode half`)\n", .{});
}

fn speedShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    gemm_pipe: *anyopaque,
    half_pipe: *anyopaque,
    linear_pipe: *anyopaque,
    threads: usize,
    s: Shape,
    runs: u32,
) !void {
    const a = try allocator.alloc(f32, @as(usize, s.m) * s.k);
    defer allocator.free(a);
    const w = try allocator.alloc(f32, @as(usize, s.n) * s.k);
    defer allocator.free(w);
    var prng = std.Random.DefaultPrng.init(0x9E + s.m + s.k + s.n);
    const rng = prng.random();
    for (a) |*v| v.* = rng.float(f32) * 0.2 - 0.1;
    for (w) |*v| v.* = rng.float(f32) * 0.2 - 0.1;

    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w));
    defer w_buf.deinit();
    var out_buf = try mbuffer.Buffer.empty(device, @as(usize, s.m) * s.n * 4);
    defer out_buf.deinit();
    var bias_buf = try mbuffer.Buffer.empty(device, @as(usize, s.n) * 4);
    defer bias_buf.deinit();

    var ctx = SpeedCtx{
        .queue = queue,
        .gemm_pipe = gemm_pipe,
        .half_pipe = half_pipe,
        .linear_pipe = linear_pipe,
        .a = a_buf.handle,
        .w = w_buf.handle,
        .out = out_buf.handle,
        .bias = bias_buf.handle,
        .gparams = .{ .m = s.m, .k = s.k, .n = s.n, .dtype = 3, .mode = 1, .weight_offset = 0 },
        .lparams = .{
            .rows = s.n,
            .cols = s.k,
            .batch = s.m,
            .dtype = 3,
            .bias_dtype = 1,
            .has_bias = 0,
            .pad = 0,
            .weight_offset = 0,
            .bias_offset = 0,
        },
        .threads = threads,
    };

    const exact_ns = try speedTime(io, .gemm, runs, &ctx);
    const half_ns = try speedTime(io, .half, runs, &ctx);
    const naive_ns = try speedTime(io, .naive, runs, &ctx);
    const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
    std.debug.print("  {d:>4} x {d:>5} x {d:>5}   {d:>7.0} {d:>8.0} {d:>8.0}   {d:>7.2}\n", .{
        s.m, s.k, s.n, flops / f(exact_ns), flops / f(half_ns), flops / f(naive_ns), f(naive_ns) / f(half_ns),
    });
}

fn speedTime(io: std.Io, kind: SpeedKind, runs: u32, ctx: *SpeedCtx) !u64 {
    var warm: u32 = 0;
    while (warm < 3) : (warm += 1) try speedDispatch(kind, ctx);
    var samples: [512]u64 = undefined;
    const count = @min(runs, 512);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        try speedDispatch(kind, ctx);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

// --- W16 decision table: same kernels fed f16 weight bytes --------------------
// The production stack is weight-byte-bound; this measures what f16 weights in
// memory buy each substrate (our exact/half kernels and mixed-input MPS) on the
// real shapes. Correctness for the MPS mixed path is gated against the CPU ref.

const W16Ctx = struct {
    queue: *anyopaque,
    exact_pipe: *anyopaque,
    half_pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    out: *anyopaque,
    p_exact: c.GemmParams,
    p_half: c.GemmParams,
};

const W16Kind = enum { exact, half, mps };

fn runSpeedW16(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    exact_pipe: *anyopaque,
    half_pipe: *anyopaque,
    runs: u32,
) !void {
    std.debug.print(
        "\nkernel speed (f16 weights, GFLOP/s):  M x K x N    exact     half      MPS   mps-err\n",
        .{},
    );
    for (shapes) |s| {
        if (s.m != 288) continue; // the production-M trio decides the W16 substrate
        try w16Shape(io, allocator, device, queue, exact_pipe, half_pipe, s, runs);
    }
    std.debug.print("  (MPS row is mixed f32 activations x f16 weights -> f32 out)\n", .{});
}

fn w16Shape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    exact_pipe: *anyopaque,
    half_pipe: *anyopaque,
    s: Shape,
    runs: u32,
) !void {
    const g = Gate{
        .m = s.m,
        .k = s.k,
        .n = s.n,
        .dtype = 1,
        .pad = 0,
        .full = false,
        .label = "",
    };
    const a = try allocator.alloc(f32, @as(usize, s.m) * s.k);
    defer allocator.free(a);
    var prng = std.Random.DefaultPrng.init(0xF16 + s.m + s.k + s.n);
    const rng = prng.random();
    for (a) |*v| v.* = rng.float(f32) * 0.2 - 0.1;
    const wbytes = try allocator.alloc(u8, @as(usize, s.n) * s.k * 2);
    defer allocator.free(wbytes);
    const wvals = try allocator.alloc(f32, @as(usize, s.n) * s.k);
    defer allocator.free(wvals);
    fillWeight(wbytes, wvals, g, rng);

    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, wbytes);
    defer w_buf.deinit();
    var out_buf = try mbuffer.Buffer.empty(device, @as(usize, s.m) * s.n * 4);
    defer out_buf.deinit();
    var ctx = W16Ctx{
        .queue = queue,
        .exact_pipe = exact_pipe,
        .half_pipe = half_pipe,
        .a = a_buf.handle,
        .w = w_buf.handle,
        .out = out_buf.handle,
        .p_exact = .{ .m = s.m, .k = s.k, .n = s.n, .dtype = 1, .mode = 1, .weight_offset = 0 },
        .p_half = .{ .m = s.m, .k = s.k, .n = s.n, .dtype = 1, .mode = 2, .weight_offset = 0 },
    };
    try w16Report(io, allocator, &ctx, a, wvals, g, s, runs);
}

fn w16Report(
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *W16Ctx,
    a: []const f32,
    wvals: []const f32,
    g: Gate,
    s: Shape,
    runs: u32,
) !void {
    const exact_ns = try w16Time(io, .exact, runs, ctx);
    const half_ns = try w16Time(io, .half, runs, ctx);
    const mps_ns = try w16Time(io, .mps, runs, ctx);
    const out = try allocator.alloc(f32, @as(usize, s.m) * s.n);
    defer allocator.free(out);
    c.zdraw_metal_read_buffer(ctx.out, std.mem.sliceAsBytes(out).ptr, out.len * 4);
    const err = checkGate(a, wvals, out, g); // out holds the last (MPS) result
    const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
    std.debug.print("  {d:>4} x {d:>5} x {d:>5}   {d:>7.0} {d:>8.0} {d:>8.0}   {d:.5}\n", .{
        s.m, s.k, s.n, flops / f(exact_ns), flops / f(half_ns), flops / f(mps_ns), err.rel,
    });
}

fn w16Time(io: std.Io, kind: W16Kind, runs: u32, ctx: *W16Ctx) !u64 {
    var warm: u32 = 0;
    while (warm < 3) : (warm += 1) try w16Dispatch(kind, ctx);
    var samples: [512]u64 = undefined;
    const count = @min(runs, 512);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        try w16Dispatch(kind, ctx);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn w16Dispatch(kind: W16Kind, ctx: *W16Ctx) !void {
    const code = switch (kind) {
        .exact => c.zdraw_metal_run_gemm(ctx.queue, ctx.exact_pipe, ctx.a, ctx.w, ctx.out, &ctx.p_exact),
        .half => c.zdraw_metal_run_gemm(ctx.queue, ctx.half_pipe, ctx.a, ctx.w, ctx.out, &ctx.p_half),
        .mps => zdraw_metal_run_gemm_mps(ctx.queue, ctx.a, ctx.w, ctx.out, &ctx.p_half),
    };
    if (code != 0) return error.DispatchFailed;
}

fn speedDispatch(kind: SpeedKind, ctx: *SpeedCtx) !void {
    var half_params = ctx.gparams;
    half_params.mode = 2;
    const code = switch (kind) {
        .gemm => c.zdraw_metal_run_gemm(ctx.queue, ctx.gemm_pipe, ctx.a, ctx.w, ctx.out, &ctx.gparams),
        .half => c.zdraw_metal_run_gemm(ctx.queue, ctx.half_pipe, ctx.a, ctx.w, ctx.out, &half_params),
        .naive => c.zdraw_metal_run_linear(
            ctx.queue,
            ctx.linear_pipe,
            ctx.a,
            ctx.w,
            ctx.bias,
            ctx.out,
            &ctx.lparams,
            ctx.threads,
        ),
    };
    if (code != 0) return error.DispatchFailed;
}

fn clampThreads(max: usize) usize {
    if (max >= 256) return 256;
    if (max >= 128) return 128;
    if (max >= 64) return 64;
    if (max >= 32) return 32;
    return 16;
}

fn fillF16(buf: []f16, rng: std.Random) void {
    for (buf) |*v| v.* = @floatCast(rng.float(f32) * 0.2 - 0.1);
}

fn f(x: anytype) f64 {
    return @floatFromInt(x);
}

fn parseRuns(init: std.process.Init) u32 {
    var iter = std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa) catch return 50;
    defer iter.deinit();
    _ = iter.next();
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--runs")) {
            if (iter.next()) |v| return std.fmt.parseInt(u32, v, 10) catch 50;
        }
    }
    return 50;
}

extern fn zdraw_metal_chain_probe(
    device: *anyopaque,
    queue: *anyopaque,
    dependent: c_int,
    ops: c_int,
) u64;

extern fn zdraw_metal_mix_probe(device: *anyopaque, queue: *anyopaque, layers: c_int) u64;

extern fn zdraw_metal_ab_probe(
    device: *anyopaque,
    queue: *anyopaque,
    m: u32,
    k: u32,
    n: u32,
    pairs: c_int,
    a_ns: *u64,
    b_ns: *u64,
) c_int;

fn runAbProbe(device: *anyopaque, queue: *anyopaque, m: u32, k: u32, n: u32) void {
    var a: u64 = 0;
    var b: u64 = 0;
    if (zdraw_metal_ab_probe(device, queue, m, k, n, 40, &a, &b) != 0) {
        std.debug.print("ab probe {d}x{d}x{d}: unavailable\n", .{ m, k, n });
        return;
    }
    const fa = @as(f64, @floatFromInt(a));
    const fb = @as(f64, @floatFromInt(b));
    std.debug.print(
        "ab probe {d}x{d}x{d} (interleaved, clock-fair): base {d:.0} us, wide {d:.0} us, wide/base {d:.3}\n",
        .{ m, k, n, fa / 1000.0, fb / 1000.0, fb / fa },
    );
}

fn runAbChain(device: *anyopaque, queue: *anyopaque, m: u32, k: u32, n: u32) void {
    var a: u64 = 0;
    var b: u64 = 0;
    if (zdraw_metal_ab_probe(device, queue, m, k, n, -40, &a, &b) != 0) {
        std.debug.print("ab chain-style probe: unavailable\n", .{});
        return;
    }
    const fa = @as(f64, @floatFromInt(a));
    const fb = @as(f64, @floatFromInt(b));
    std.debug.print(
        "ab chain-style {d}x{d}x{d}: pristine-A {d:.0} us, convert-fed-A {d:.0} us, ratio {d:.3}\n",
        .{ m, k, n, fa / 1000.0, fb / 1000.0, fb / fa },
    );
}

fn runAbAlt(device: *anyopaque, queue: *anyopaque, m: u32, k: u32, n: u32) void {
    var a: u64 = 0;
    var b: u64 = 0;
    if (zdraw_metal_ab_probe(device, queue, m, k, n, -10040, &a, &b) != 0) {
        std.debug.print("ab alternation probe: unavailable\n", .{});
        return;
    }
    std.debug.print(
        "ab alternation {d}x{d}x{d}: square-arm {d:.0} us, fat-arm {d:.0} us\n",
        .{ m, k, n, @as(f64, @floatFromInt(a)) / 1000.0, @as(f64, @floatFromInt(b)) / 1000.0 },
    );
}

fn runMixProbe(device: *anyopaque, queue: *anyopaque) void {
    _ = zdraw_metal_mix_probe(device, queue, 30); // warmup: clocks + pipelines
    var samples: [5]u64 = undefined;
    for (&samples) |*s| {
        s.* = zdraw_metal_mix_probe(device, queue, 30);
        if (s.* == 0) {
            std.debug.print("mix probe: unavailable\n", .{});
            return;
        }
    }
    std.mem.sort(u64, &samples, {}, std.sort.asc(u64));
    const ns = samples[2];
    const per_layer = @as(f64, @floatFromInt(ns)) / 30.0 / 1e6;
    std.debug.print(
        "mix probe (30 layers, median of 5): {d:.2} ms/layer, {d:.0} ms/step, spread {d:.0}-{d:.0} ms\n",
        .{
            per_layer,
            @as(f64, @floatFromInt(ns)) / 1e6,
            @as(f64, @floatFromInt(samples[0])) / 1e6,
            @as(f64, @floatFromInt(samples[4])) / 1e6,
        },
    );
}

// VAE-shape probe: MPSGraph SDPA at head_dim 512 (single head) vs CPU.

extern fn zdraw_metal_run_gemm_w6(
    queue: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_buf: *anyopaque,
    params: *const c.GemmParams,
) c_int;

// A packed steel kernel (loader_w4.h / loader_w2.h) at the same shapes: speed
// and the worst relative error against a CPU decode reference that mirrors
// the kernel's arithmetic (half(code) * half(scale), f16 A, f32 accumulate).
fn runPackedSteel(
    comptime Codec: type,
    name: [*:0]const u8,
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    bench_shapes: []const NamedShape,
) !void {
    std.debug.print("\n{s}:  M x K x N        GFLOP/s   err(vs decode)\n", .{name});
    for (bench_shapes) |item| {
        const s = item.shape;
        const m = @as(usize, s.m);
        const n = @as(usize, s.n);
        const k = @as(usize, s.k);
        const a16 = try allocator.alloc(f16, m * k);
        defer allocator.free(a16);
        const w32 = try allocator.alloc(f32, n * k);
        defer allocator.free(w32);
        var prng = std.Random.DefaultPrng.init(0x4BEEF + s.m + s.n + s.k);
        for (a16) |*hv| hv.* = @floatCast(prng.random().floatNorm(f32) * 0.5);
        for (w32) |*x| x.* = prng.random().floatNorm(f32) * 0.05;
        const shape = [_]usize{ s.n, s.k };
        const view = tensor.View{ .dtype = .f32, .shape = &shape, .bytes = std.mem.sliceAsBytes(w32) };
        var packed_w = try Codec.pack(allocator, view, 64);
        defer packed_w.deinit(allocator);
        var a16_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
        defer a16_buf.deinit();
        var w_buf = try mbuffer.Buffer.fromBytes(device, packed_w.bytes);
        defer w_buf.deinit();
        var c_w4 = try mbuffer.Buffer.empty(device, m * n * 2);
        defer c_w4.deinit();
        const am: c_int = if (s.m % 64 == 0) 1 else 0;
        const an: c_int = if (s.n % 64 == 0) 1 else 0;
        const ak: c_int = if (s.k % 16 == 0) 1 else 0;
        const pipe = zdraw_metal_steel_w6_make_cfg(device, steelLibPath(), name, am, an, ak) orelse {
            std.debug.print("  {s}: pipeline make failed (metallib missing it?)\n", .{name});
            return;
        };
        defer c.zdraw_metal_release_pipeline(pipe);
        const mi: c_int = @intCast(s.m);
        const ni: c_int = @intCast(s.n);
        const ki: c_int = @intCast(s.k);
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..4) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            const rc = zdraw_metal_steel_w6_run_cfg(
                queue,
                pipe,
                a16_buf.handle,
                w_buf.handle,
                c_w4.handle,
                mi,
                ni,
                ki,
                64,
                64,
                16,
                2,
                2,
            );
            if (rc != 0) return error.PackedSteelDispatchFailed;
            best_ns = @min(best_ns, @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds())));
        }
        const got16 = try allocator.alloc(f16, m * n);
        defer allocator.free(got16);
        c.zdraw_metal_read_buffer(c_w4.handle, std.mem.sliceAsBytes(got16).ptr, m * n * 2);
        const worst = decodeError(Codec, got16, a16, packed_w, m, n, k);
        const flops: f64 = 2.0 * f(m) * f(k) * f(n);
        std.debug.print("  {s:<14} {d:>5}x{d:<5}x{d:<5} {d:>10.0}          {d:.4}\n", .{
            item.label, s.m, s.k, s.n, flops / f(best_ns), worst,
        });
    }
}

// Worst relative error of a packed steel output against the CPU decode
// reference on a sampled grid of rows and columns.
fn decodeError(
    comptime Codec: type,
    got16: []const f16,
    a16: []const f16,
    w4: Codec.Packed,
    m: usize,
    n: usize,
    k: usize,
) f64 {
    var worst: f64 = 0.0;
    var ri: usize = 0;
    while (ri < m) : (ri += 257) {
        var cj: usize = 0;
        while (cj < n) : (cj += 131) {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                const wv: f16 = @floatCast(w4.decode(cj, kk));
                acc += @as(f32, @floatCast(a16[ri * k + kk])) * @as(f32, @floatCast(wv));
            }
            const gv: f64 = @floatCast(got16[ri * n + cj]);
            const rel = @abs(gv - @as(f64, acc)) / (@abs(@as(f64, acc)) + 1e-3);
            worst = @max(worst, rel);
        }
    }
    return worst;
}

// Correctness gate: W6 kernel vs CPU decode+matmul on a small case.
fn runW6Speed(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    bench_shapes: []const NamedShape,
    runs: u32,
) !void {
    std.debug.print("\nW6 GEMM speed ({d} runs):  M x K x N         W6 GFLOP/s   (mflux bf16 ~14400)\n", .{runs});
    for (bench_shapes) |item| {
        const s = item.shape;
        const a = try allocator.alloc(f32, @as(usize, s.m) * s.k);
        defer allocator.free(a);
        const w = try allocator.alloc(f32, @as(usize, s.n) * s.k);
        defer allocator.free(w);
        var prng = std.Random.DefaultPrng.init(0x96 + s.m + s.n + s.k);
        for (a) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
        for (w) |*x| x.* = prng.random().floatNorm(f32) * 0.05;
        const view = tensor.View{ .dtype = .f32, .shape = &.{ s.n, s.k }, .bytes = std.mem.sliceAsBytes(w) };
        var packed_w = try zw6.packW6(allocator, view, 64);
        defer packed_w.deinit(allocator);
        var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
        defer a_buf.deinit();
        var w_buf = try mbuffer.Buffer.fromBytes(device, packed_w.bytes);
        defer w_buf.deinit();
        var c_out = try mbuffer.Buffer.empty(device, @as(usize, s.m) * s.n * 4);
        defer c_out.deinit();
        const params = c.GemmParams{ .m = s.m, .k = s.k, .n = s.n, .dtype = 4, .mode = 4 };
        var wm: u32 = 0;
        while (wm < 3) : (wm += 1) _ = zdraw_metal_run_gemm_w6(queue, a_buf.handle, w_buf.handle, c_out.handle, &params);
        var samples: [256]u64 = undefined;
        const count = @min(runs, 256);
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const t0 = std.Io.Timestamp.now(io, .awake);
            _ = zdraw_metal_run_gemm_w6(queue, a_buf.handle, w_buf.handle, c_out.handle, &params);
            samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
        }
        std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
        const flops: f64 = 2.0 * f(s.m) * f(s.k) * f(s.n);
        std.debug.print("  {s} {d:>4} x {d:>5} x {d:>5}      {d:>8.0}\n", .{
            item.label,
            s.m,
            s.k,
            s.n,
            flops / f(samples[count / 2]),
        });
    }
}

// W6 steel kernel: correctness (vs slow W6 oracle + CPU decode ref) and speed
// (GFLOP/s) vs the f16 steel and the slow gemm_w6_staged. A is f16, weight is
// W6 (group 64), output is f16. The slow gemm_w6_staged uses f32 A / f32 out.
fn runW6Steel(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    bench_shapes: []const NamedShape,
    runs: u32,
) !void {
    std.debug.print("\nW6 steel GEMM ({d} runs):  M x K x N        W6-steel  steel-f16  slow-W6   err(vs oracle, vs decode)\n", .{runs});
    for (bench_shapes) |item| {
        const s = item.shape;
        const m = @as(usize, s.m);
        const n = @as(usize, s.n);
        const k = @as(usize, s.k);

        // Random f16 activations (also as f32 for the slow oracle) and a random
        // f32 weight packed to W6.
        const a16 = try allocator.alloc(f16, m * k);
        defer allocator.free(a16);
        const a32 = try allocator.alloc(f32, m * k);
        defer allocator.free(a32);
        const w32 = try allocator.alloc(f32, n * k);
        defer allocator.free(w32);
        var prng = std.Random.DefaultPrng.init(0x6BEEF + s.m + s.n + s.k);
        for (a16, a32) |*hv, *fv| {
            const x = prng.random().floatNorm(f32) * 0.5;
            hv.* = @floatCast(x);
            fv.* = @floatCast(@as(f16, @floatCast(x))); // match the f16 A the steel kernel sees
        }
        for (w32) |*x| x.* = prng.random().floatNorm(f32) * 0.05;
        const view = tensor.View{ .dtype = .f32, .shape = &.{ s.n, s.k }, .bytes = std.mem.sliceAsBytes(w32) };
        var packed_w = try zw6.packW6(allocator, view, 64);
        defer packed_w.deinit(allocator);

        var a16_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a16));
        defer a16_buf.deinit();
        var a32_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a32));
        defer a32_buf.deinit();
        var w_buf = try mbuffer.Buffer.fromBytes(device, packed_w.bytes);
        defer w_buf.deinit();
        var c_w6 = try mbuffer.Buffer.empty(device, m * n * 2); // f16 out
        defer c_w6.deinit();
        var c_slow = try mbuffer.Buffer.empty(device, m * n * 4); // f32 out
        defer c_slow.deinit();

        // Config-variant race (same packed weights/activations).
        if (std.c.getenv("ZDRAW_W6_RACE") != null) {
            const mi2: c_int = @intCast(s.m);
            const ni2: c_int = @intCast(s.n);
            const ki2: c_int = @intCast(s.k);
            const Cfg = struct { name: [*:0]const u8, bm: c_int, bn: c_int, bk: c_int, wm: c_int, wn: c_int };
            const cfgs = [_]Cfg{
                .{ .name = "steel_gemm_w6_64", .bm = 64, .bn = 64, .bk = 16, .wm = 2, .wn = 2 },
                .{ .name = "steel_gemm_w6_128x64", .bm = 128, .bn = 64, .bk = 16, .wm = 4, .wn = 2 },
                .{ .name = "steel_gemm_w6_64x128", .bm = 64, .bn = 128, .bk = 16, .wm = 2, .wn = 4 },
                .{ .name = "steel_gemm_w6_128x128", .bm = 128, .bn = 128, .bk = 16, .wm = 4, .wn = 4 },
                .{ .name = "steel_gemm_w6_bk32", .bm = 64, .bn = 64, .bk = 32, .wm = 2, .wn = 2 },
            };
            const flops: f64 = 2.0 * f(@as(usize, s.m)) * f(@as(usize, s.k)) * f(@as(usize, s.n));
            for (cfgs) |cf| {
                const amr: c_int = if (s.m % @as(usize, @intCast(cf.bm)) == 0) 1 else 0;
                const anr: c_int = if (s.n % @as(usize, @intCast(cf.bn)) == 0) 1 else 0;
                const akr: c_int = if (s.k % @as(usize, @intCast(cf.bk)) == 0) 1 else 0;
                const cp = zdraw_metal_steel_w6_make_cfg(device, steelLibPath(), cf.name, amr, anr, akr) orelse {
                    std.debug.print("    cfg {s}: make failed\n", .{cf.name});
                    continue;
                };
                defer c.zdraw_metal_release_pipeline(cp);
                if (zdraw_metal_steel_w6_run_cfg(queue, cp, a16_buf.handle, w_buf.handle, c_w6.handle, mi2, ni2, ki2, cf.bm, cf.bn, cf.bk, cf.wm, cf.wn) != 0) {
                    std.debug.print("    cfg {s}: dispatch failed\n", .{cf.name});
                    continue;
                }
                var best_ns: u64 = std.math.maxInt(u64);
                for (0..3) |_| {
                    const t0 = std.Io.Timestamp.now(io, .awake);
                    if (zdraw_metal_steel_w6_run_cfg(queue, cp, a16_buf.handle, w_buf.handle, c_w6.handle, mi2, ni2, ki2, cf.bm, cf.bn, cf.bk, cf.wm, cf.wn) != 0) break;
                    const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
                    if (dt < best_ns) best_ns = dt;
                }
                std.debug.print("    cfg {s}: {d:>6.0} GFLOP/s\n", .{ cf.name, flops / f(best_ns) });
            }
        }
        const am: c_int = if (s.m % 64 == 0) 1 else 0;
        const an: c_int = if (s.n % 64 == 0) 1 else 0;
        const ak: c_int = if (s.k % 16 == 0) 1 else 0;
        const w6pipe = zdraw_metal_steel_w6_make(device, steelLibPath(), am, an, ak) orelse {
            std.debug.print("  w6 steel: pipeline make failed (metallib missing steel_gemm_w6_64?)\n", .{});
            return error.W6SteelMakeFailed;
        };
        const mi: c_int = @intCast(s.m);
        const ni: c_int = @intCast(s.n);
        const ki: c_int = @intCast(s.k);

        // Run both kernels once for correctness.
        if (zdraw_metal_steel_w6_run(queue, w6pipe, a16_buf.handle, w_buf.handle, c_w6.handle, mi, ni, ki) != 0)
            return error.W6SteelDispatchFailed;
        const slow_params = c.GemmParams{ .m = s.m, .k = s.k, .n = s.n, .dtype = 4, .mode = 4 };
        _ = zdraw_metal_run_gemm_w6(queue, a32_buf.handle, w_buf.handle, c_slow.handle, &slow_params);

        const got16 = try allocator.alloc(f16, m * n);
        defer allocator.free(got16);
        const slow = try allocator.alloc(f32, m * n);
        defer allocator.free(slow);
        c.zdraw_metal_read_buffer(c_w6.handle, std.mem.sliceAsBytes(got16).ptr, m * n * 2);
        c.zdraw_metal_read_buffer(c_slow.handle, std.mem.sliceAsBytes(slow).ptr, m * n * 4);

        // Error vs the slow W6 oracle (the correctness oracle), and vs a CPU
        // decode reference, both relative (outputs scale with K). Sample a grid
        // of rows/cols for the decode ref to keep the CPU pass cheap.
        var worst_oracle: f64 = 0.0;
        for (got16, slow) |gv, sv| {
            const rel = @abs(@as(f64, @floatCast(gv)) - @as(f64, sv)) / (@abs(@as(f64, sv)) + 1e-3);
            worst_oracle = @max(worst_oracle, rel);
        }
        // Decode ref mirrors the kernel's arithmetic: dequant weight in f16
        // (half(code)*half(scale)), multiply by the f16 activation, accumulate
        // in f32 (the GPU's accumulator type). This isolates real correctness
        // from the f16-vs-f64 rounding gap, which over K=3840 terms is several
        // percent and is NOT a kernel bug.
        var worst_decode: f64 = 0.0;
        var ri: usize = 0;
        while (ri < m) : (ri += 257) {
            var cj: usize = 0;
            while (cj < n) : (cj += 263) {
                var ref: f32 = 0.0;
                for (0..k) |kk| {
                    const sc: f16 = @floatCast(packed_w.scale(cj, kk / 64));
                    const code: f16 = @floatFromInt(decodeCode(packed_w, cj, kk));
                    ref += @as(f32, a16[ri * k + kk]) * @as(f32, code * sc);
                }
                const rel = @abs(@as(f64, ref) - @as(f64, @floatCast(got16[ri * n + cj]))) / (@abs(@as(f64, ref)) + 1e-3);
                worst_decode = @max(worst_decode, rel);
            }
        }

        // Speed: median of `runs`.
        const w6_g = flopsG(s, try timeW6Steel(io, queue, w6pipe, &a16_buf, &w_buf, &c_w6, mi, ni, ki, runs));
        const slow_g = flopsG(s, try timeW6Slow(io, queue, &a32_buf, &w_buf, &c_slow, &slow_params, runs));
        var steel_g: f64 = 0.0;
        {
            // f16 steel on a dense f16 weight, same shape, for the speed bar.
            const b16 = try allocator.alloc(f16, n * k);
            defer allocator.free(b16);
            for (b16) |*x| x.* = 0.01;
            var b16_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(b16));
            defer b16_buf.deinit();
            var c_st = try mbuffer.Buffer.empty(device, m * n * 2);
            defer c_st.deinit();
            const stpipe = zdraw_metal_steel_make(device, steelLibPath(), 64, am, an, ak);
            if (stpipe) |sp| steel_g = flopsG(s, try timeSteel(io, queue, sp, &a16_buf, &b16_buf, &c_st, mi, ni, ki, 64, runs));
        }

        std.debug.print("  {s} {d:>4} x {d:>5} x {d:>5}   {d:>8.0}  {d:>8.0}  {d:>8.0}   ({d:.4}, {d:.4}) -> {s}\n", .{
            item.label,
            s.m,
            s.k,
            s.n,
            w6_g,
            steel_g,
            slow_g,
            worst_oracle,
            worst_decode,
            if (worst_oracle < 0.02 and worst_decode < 0.05) "PASS" else "FAIL",
        });
        if (worst_oracle >= 0.02 or worst_decode >= 0.05) return error.W6SteelMismatch;
    }
}

fn flopsG(s: Shape, ns: u64) f64 {
    return (2.0 * f(s.m) * f(s.k) * f(s.n)) / f(ns);
}

// Signed integer code (-31..31) at (row, col) from a packed W6, mirroring the
// kernel's bit extraction (zw6.W6.decode without the scale multiply).
fn decodeCode(w: zw6.W6, row: usize, col: usize) i32 {
    const cbpr = w.groupsPerRow() * w.group / 4 * 3;
    const quad = col / 4;
    const at = row * cbpr + quad * 3;
    const lane: u5 = @intCast((col % 4) * 6);
    const word: u32 = @as(u32, w.bytes[at]) |
        (@as(u32, w.bytes[at + 1]) << 8) |
        (@as(u32, w.bytes[at + 2]) << 16);
    const raw: u32 = (word >> lane) & 0x3F;
    return if (raw < 32) @intCast(raw) else @as(i32, @intCast(raw)) - 64;
}

fn timeW6Steel(io: std.Io, queue: *anyopaque, pipe: *anyopaque, a: *mbuffer.Buffer, w: *mbuffer.Buffer, d: *mbuffer.Buffer, mi: c_int, ni: c_int, ki: c_int, runs: u32) !u64 {
    var wm: u32 = 0;
    while (wm < 3) : (wm += 1) _ = zdraw_metal_steel_w6_run(queue, pipe, a.handle, w.handle, d.handle, mi, ni, ki);
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = zdraw_metal_steel_w6_run(queue, pipe, a.handle, w.handle, d.handle, mi, ni, ki);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn timeW6Slow(io: std.Io, queue: *anyopaque, a: *mbuffer.Buffer, w: *mbuffer.Buffer, d: *mbuffer.Buffer, params: *const c.GemmParams, runs: u32) !u64 {
    var wm: u32 = 0;
    while (wm < 3) : (wm += 1) _ = zdraw_metal_run_gemm_w6(queue, a.handle, w.handle, d.handle, params);
    var samples: [256]u64 = undefined;
    const count = @min(runs, 256);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = zdraw_metal_run_gemm_w6(queue, a.handle, w.handle, d.handle, params);
        samples[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn runW6Gate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const m = 32;
    const k = 128;
    const n = 64;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    for (a) |*x| x.* = rng.floatNorm(f32) * 0.5;
    for (w) |*x| x.* = rng.floatNorm(f32) * 0.05;
    const view = tensor.View{ .dtype = .f32, .shape = &.{ n, k }, .bytes = std.mem.sliceAsBytes(w) };
    var packed_w = try zw6.packW6(allocator, view, 64);
    defer packed_w.deinit(allocator);
    var a_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(a));
    defer a_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, packed_w.bytes);
    defer w_buf.deinit();
    var c_out = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_out.deinit();
    const params = c.GemmParams{ .m = m, .k = k, .n = n, .dtype = 4, .mode = 4 };
    if (zdraw_metal_run_gemm_w6(queue, a_buf.handle, w_buf.handle, c_out.handle, &params) != 0) {
        std.debug.print("w6 gate: dispatch failed\n", .{});
        return error.DispatchFailed;
    }
    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    c.zdraw_metal_read_buffer(c_out.handle, std.mem.sliceAsBytes(got).ptr, m * n * 4);
    var worst: f64 = 0.0;
    for (0..m) |i| for (0..n) |j| {
        var ref: f64 = 0.0;
        for (0..k) |kk| ref += @as(f64, a[i * k + kk]) * packed_w.decode(j, kk);
        const err = @abs(ref - got[i * n + j]);
        worst = @max(worst, err);
    };
    std.debug.print("w6 gate: max |err| vs decoded ref = {d:.6} -> {s}\n", .{
        worst, if (worst < 1e-2) "PASS" else "FAIL",
    });
    if (worst >= 1e-2) return error.W6Mismatch;

    // Split-scales variant: same pack, scales bound at their absolute offset.
    // Identical math -> must be bit-identical to the staged kernel's output.
    var c_split = try mbuffer.Buffer.empty(device, m * n * 4);
    defer c_split.deinit();
    const scales_off: u64 = zw6.scalesBase(n, k, 64);
    if (c.zdraw_metal_run_gemm_w6_split(queue, a_buf.handle, w_buf.handle, c_split.handle, &params, scales_off) != 0) {
        std.debug.print("w6 split gate: dispatch failed\n", .{});
        return error.DispatchFailed;
    }
    const got_split = try allocator.alloc(f32, m * n);
    defer allocator.free(got_split);
    c.zdraw_metal_read_buffer(c_split.handle, std.mem.sliceAsBytes(got_split).ptr, m * n * 4);
    var split_diff: usize = 0;
    for (got, got_split) |x, y| {
        if (x != y) split_diff += 1;
    }
    std.debug.print("w6 split gate: {d} elems differ vs staged -> {s}\n", .{
        split_diff, if (split_diff == 0) "PASS" else "FAIL",
    });
    if (split_diff != 0) return error.W6Mismatch;

    // Fused sub-matrix: one packed [256, k] matrix consumed as four 64-row
    // windows via derived codes/scales offsets (the single-block qkv_mlp
    // pattern). Each window must match the CPU decode reference.
    const total_rows = 256;
    const wf = try allocator.alloc(f32, total_rows * k);
    defer allocator.free(wf);
    for (wf) |*x| x.* = rng.floatNorm(f32) * 0.05;
    const fused_view = tensor.View{ .dtype = .f32, .shape = &.{ total_rows, k }, .bytes = std.mem.sliceAsBytes(wf) };
    var fused_w = try zw6.packW6(allocator, fused_view, 64);
    defer fused_w.deinit(allocator);
    var fused_buf = try mbuffer.Buffer.fromBytes(device, fused_w.bytes);
    defer fused_buf.deinit();
    const gpr = (k + 63) / 64;
    var window: usize = 0;
    var fused_worst: f64 = 0.0;
    while (window < 4) : (window += 1) {
        const row0 = window * 64;
        const sub = c.GemmParams{
            .m = m,
            .k = k,
            .n = 64,
            .dtype = 4,
            .mode = 4,
            .weight_offset = row0 * zw6.codesPerRow(k, 64),
        };
        const sub_scales: u64 = zw6.scalesBase(total_rows, k, 64) + row0 * gpr * 2;
        if (c.zdraw_metal_run_gemm_w6_split(queue, a_buf.handle, fused_buf.handle, c_split.handle, &sub, sub_scales) != 0) {
            std.debug.print("w6 fused gate: dispatch failed (window {d})\n", .{window});
            return error.DispatchFailed;
        }
        const got_sub = try allocator.alloc(f32, m * 64);
        defer allocator.free(got_sub);
        c.zdraw_metal_read_buffer(c_split.handle, std.mem.sliceAsBytes(got_sub).ptr, m * 64 * 4);
        for (0..m) |i| for (0..64) |j| {
            var ref: f64 = 0.0;
            for (0..k) |kk| ref += @as(f64, a[i * k + kk]) * fused_w.decode(row0 + j, kk);
            fused_worst = @max(fused_worst, @abs(ref - got_sub[i * 64 + j]));
        };
    }
    std.debug.print("w6 fused sub-bind gate: max |err| vs decoded ref = {d:.6} -> {s}\n", .{
        fused_worst, if (fused_worst < 1e-2) "PASS" else "FAIL",
    });
    if (fused_worst >= 1e-2) return error.W6Mismatch;
}

// --- Exact Streaming VAE, Step 1: GroupNorm+SiLU kernel split gate -------------
// Proves the split (vae_norm_stats + vae_norm_apply_silu_window) is BIT-EXACT vs
// the fused vae_norm_silu: max|d| must be exactly 0. The stats kernel copies the
// fused reduction verbatim (same order), so mean/scale match bit-for-bit; the
// windowed apply reuses the identical per-pixel expression, so it matches on
// every written pixel and leaves the rest untouched.

const VaeNormParams = mvres_stream.NormParams;

const VaeNormWindowParams = mvres_stream.NormWindowParams;

extern fn zdraw_metal_run_vae_norm(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const VaeNormParams,
    thread_count: usize,
) c_int;

// ConvParams fields + the contiguous output row-strip [row0, row1).
const ConvWindowParams = mvres_stream.ConvWindowParams;

// Full-frame residual add (oracle reference for the streamed resblock gate).
extern fn zdraw_metal_run_vae_add(
    queue: *anyopaque,
    pipeline: *anyopaque,
    output: *anyopaque,
    residual: *anyopaque,
    count: u32,
    thread_count: usize,
) c_int;

fn runVaeNormSplitGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const channels: u32 = 512;
    const groups: u32 = 32;
    const height: u32 = 64;
    const width: u32 = 64;
    const eps: f32 = 0.000001; // matches src/vres.zig VAE GroupNorm eps
    const hw: usize = @as(usize, height) * width;
    const n: usize = @as(usize, channels) * hw;

    const ref_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_silu");
    defer c.zdraw_metal_release_pipeline(ref_pipe);
    const stats_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_stats");
    defer c.zdraw_metal_release_pipeline(stats_pipe);
    const apply_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_apply_silu_window");
    defer c.zdraw_metal_release_pipeline(apply_pipe);
    const threads = clampThreads(c.zdraw_metal_pipeline_threads(ref_pipe));

    var prng = std.Random.DefaultPrng.init(0x5EA);
    const rng = prng.random();
    const input = try allocator.alloc(f32, n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32) * 1.7;
    const weight = try allocator.alloc(f32, channels);
    defer allocator.free(weight);
    const bias = try allocator.alloc(f32, channels);
    defer allocator.free(bias);
    for (weight) |*v| v.* = rng.floatNorm(f32) * 0.3 + 1.0;
    for (bias) |*v| v.* = rng.floatNorm(f32) * 0.2;

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(weight));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var ref_buf = try mbuffer.Buffer.empty(device, n * 4);
    defer ref_buf.deinit();
    var stats_buf = try mbuffer.Buffer.empty(device, @as(usize, groups) * 2 * 4);
    defer stats_buf.deinit();
    // Output seeded with a sentinel so an untouched (outside-window) pixel is
    // distinguishable from any value the apply kernel could legitimately write.
    const seed = try allocator.alloc(f32, n);
    defer allocator.free(seed);
    @memset(seed, sentinel);

    // dtype 3 = f32 for both weight and bias: read_value is exact, so the only
    // thing under test is the reduction order and apply expression.
    const np = VaeNormParams{
        .channels = channels,
        .height = height,
        .width = width,
        .groups = groups,
        .dtype = 3,
        .bias_dtype = 3,
        .eps = eps,
        .weight_offset = 0,
        .bias_offset = 0,
    };

    // 1) Reference: fused vae_norm_silu.
    if (zdraw_metal_run_vae_norm(queue, ref_pipe, in_buf.handle, w_buf.handle, b_buf.handle, ref_buf.handle, &np, threads) != 0) {
        return error.DispatchFailed;
    }
    const ref = try allocator.alloc(f32, n);
    defer allocator.free(ref);
    c.zdraw_metal_read_buffer(ref_buf.handle, std.mem.sliceAsBytes(ref).ptr, n * 4);

    // 2) Split with a FULL window: stats then apply over the whole frame.
    if (mvres_stream.zdraw_metal_run_vae_norm_stats(queue, stats_pipe, in_buf.handle, stats_buf.handle, &np, threads) != 0) {
        return error.DispatchFailed;
    }
    const full = windowParams(np, 0, height, 0, width);
    var full_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(seed));
    defer full_buf.deinit();
    if (mvres_stream.zdraw_metal_run_vae_norm_apply_window(queue, apply_pipe, in_buf.handle, stats_buf.handle, w_buf.handle, b_buf.handle, full_buf.handle, &full, threads) != 0) {
        return error.DispatchFailed;
    }
    const o_full = try allocator.alloc(f32, n);
    defer allocator.free(o_full);
    c.zdraw_metal_read_buffer(full_buf.handle, std.mem.sliceAsBytes(o_full).ptr, n * 4);
    const full_d = maxBitDiff(ref, o_full);

    // 3) Split with a SUB-window: exact inside, untouched (sentinel) outside.
    const row0: u32 = 17;
    const row1: u32 = 41;
    const col0: u32 = 5;
    const col1: u32 = 60;
    const sub = windowParams(np, row0, row1, col0, col1);
    var sub_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(seed));
    defer sub_buf.deinit();
    if (mvres_stream.zdraw_metal_run_vae_norm_apply_window(queue, apply_pipe, in_buf.handle, stats_buf.handle, w_buf.handle, b_buf.handle, sub_buf.handle, &sub, threads) != 0) {
        return error.DispatchFailed;
    }
    const o_sub = try allocator.alloc(f32, n);
    defer allocator.free(o_sub);
    c.zdraw_metal_read_buffer(sub_buf.handle, std.mem.sliceAsBytes(o_sub).ptr, n * 4);

    var sub_in_d: f64 = 0;
    var outside_touched: usize = 0;
    for (0..channels) |ch| {
        for (0..height) |row| {
            for (0..width) |col| {
                const idx = ch * hw + row * @as(usize, width) + col;
                const inside = row >= row0 and row < row1 and col >= col0 and col < col1;
                if (inside) {
                    const d = @abs(@as(f64, o_sub[idx]) - @as(f64, ref[idx]));
                    sub_in_d = @max(sub_in_d, d);
                } else if (o_sub[idx] != sentinel) {
                    outside_touched += 1;
                }
            }
        }
    }

    const pass = full_d == 0 and sub_in_d == 0 and outside_touched == 0;
    std.debug.print(
        "\nvae norm split: full max|d|={d} sub-window max|d|={d} outside-touched={d} -> {s}\n",
        .{ full_d, sub_in_d, outside_touched, if (pass) "PASS" else "FAIL" },
    );
    if (!pass) return error.VaeNormSplitMismatch;
}

const sentinel: f32 = -123456.5;

// Exact-streaming VAE Step 2: prove conv2d_window (output restricted to a
// contiguous row-strip) is bit-exact against the full conv2d. The strip reads
// its halo from the full resident input by global coords, so interior strip
// boundaries read real neighbor rows and only true frame edges zero-pad.
fn runConvWindowGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const in_ch: u32 = 128;
    const out_ch: u32 = 128;
    const height: u32 = 64;
    const width: u32 = 64;
    const ksize: u32 = 3;
    const pad: u32 = 1;
    const hw: usize = @as(usize, height) * width;
    const in_n: usize = @as(usize, in_ch) * hw;
    const out_n: usize = @as(usize, out_ch) * hw;
    const w_n: usize = @as(usize, out_ch) * in_ch * ksize * ksize;

    const ref_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d");
    defer c.zdraw_metal_release_pipeline(ref_pipe);
    const win_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_window");
    defer c.zdraw_metal_release_pipeline(win_pipe);
    const threads = clampThreads(c.zdraw_metal_pipeline_threads(ref_pipe));

    var prng = std.Random.DefaultPrng.init(0xC07);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32) * 1.3;
    const weight = try allocator.alloc(f32, w_n);
    defer allocator.free(weight);
    for (weight) |*v| v.* = rng.floatNorm(f32) * 0.2;
    const bias = try allocator.alloc(f32, out_ch);
    defer allocator.free(bias);
    for (bias) |*v| v.* = rng.floatNorm(f32) * 0.1;

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(weight));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var ref_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer ref_buf.deinit();
    // Output seeded with a sentinel so an untouched (outside-strip) pixel is
    // distinguishable from any value the window kernel could legitimately write.
    const seed = try allocator.alloc(f32, out_n);
    defer allocator.free(seed);
    @memset(seed, sentinel);

    // dtype 3 = f32 weight (and bias): read_value is exact, so the only thing
    // under test is the windowed indexing, not any narrowing conversion.
    const cp = c.ConvParams{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = height,
        .width = width,
        .ksize = ksize,
        .pad = pad,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
    };

    // 1) Reference: full conv2d over the whole frame.
    if (c.zdraw_metal_run_conv(queue, ref_pipe, in_buf.handle, w_buf.handle, b_buf.handle, ref_buf.handle, &cp, threads) != 0) {
        return error.DispatchFailed;
    }
    const ref = try allocator.alloc(f32, out_n);
    defer allocator.free(ref);
    c.zdraw_metal_read_buffer(ref_buf.handle, std.mem.sliceAsBytes(ref).ptr, out_n * 4);

    // 2) Window over the FULL strip [0, height): must equal the full conv exactly.
    const full = convWindowParams(cp, 0, height);
    var full_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(seed));
    defer full_buf.deinit();
    if (mvres_stream.zdraw_metal_run_conv2d_window(queue, win_pipe, in_buf.handle, w_buf.handle, b_buf.handle, full_buf.handle, &full, threads) != 0) {
        return error.DispatchFailed;
    }
    const o_full = try allocator.alloc(f32, out_n);
    defer allocator.free(o_full);
    c.zdraw_metal_read_buffer(full_buf.handle, std.mem.sliceAsBytes(o_full).ptr, out_n * 4);
    const full_d = maxBitDiff(ref, o_full);

    // 3) Window over a SUB-strip [17, 41): exact inside, untouched (sentinel)
    // outside. Rows 16 and 41 are read as input halo, so a 0 here proves the
    // interior strip boundaries are NOT zero-padded - they read real neighbors.
    const row0: u32 = 17;
    const row1: u32 = 41;
    const sub = convWindowParams(cp, row0, row1);
    var sub_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(seed));
    defer sub_buf.deinit();
    if (mvres_stream.zdraw_metal_run_conv2d_window(queue, win_pipe, in_buf.handle, w_buf.handle, b_buf.handle, sub_buf.handle, &sub, threads) != 0) {
        return error.DispatchFailed;
    }
    const o_sub = try allocator.alloc(f32, out_n);
    defer allocator.free(o_sub);
    c.zdraw_metal_read_buffer(sub_buf.handle, std.mem.sliceAsBytes(o_sub).ptr, out_n * 4);

    var sub_in_d: f64 = 0;
    var outside_touched: usize = 0;
    for (0..out_ch) |ch| {
        for (0..height) |row| {
            for (0..width) |col| {
                const idx = ch * hw + row * @as(usize, width) + col;
                const inside = row >= row0 and row < row1;
                if (inside) {
                    const d = @abs(@as(f64, o_sub[idx]) - @as(f64, ref[idx]));
                    sub_in_d = @max(sub_in_d, d);
                } else if (o_sub[idx] != sentinel) {
                    outside_touched += 1;
                }
            }
        }
    }

    const pass = full_d == 0 and sub_in_d == 0 and outside_touched == 0;
    std.debug.print(
        "\nconv2d window: full max|d|={d} strip max|d|={d} outside-touched={d} -> {s}\n",
        .{ full_d, sub_in_d, outside_touched, if (pass) "PASS" else "FAIL" },
    );
    if (!pass) return error.ConvWindowMismatch;
}

// Mirrors MSL ConvPrenormWindowParams / the C ZdrawConvPrenormWindowParams.
const ConvPrenormWindowParams = extern struct {
    in_ch: u32,
    out_ch: u32,
    height: u32,
    width: u32,
    ksize: u32,
    pad: u32,
    dtype: u32,
    bias_dtype: u32,
    has_bias: u32,
    groups: u32,
    weight_offset: u64,
    bias_offset: u64,
    row0: u32,
    row1: u32,
    norm_dtype: u32,
    norm_bias_dtype: u32,
    norm_weight_offset: u64,
    norm_bias_offset: u64,
};

extern fn zdraw_metal_run_conv2d_prenorm_window(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    stats: *anyopaque,
    norm_weight: *anyopaque,
    norm_bias: *anyopaque,
    params: *const ConvPrenormWindowParams,
    thread_count: usize,
) c_int;

// Exact-streaming VAE Step 4b: prove conv2d_prenorm_window (GroupNorm+SiLU fused
// into the conv input read) is BIT-EXACT (max|d| == 0) against the two-step
// reference apply-then-conv: vae_norm_stats -> vae_norm_apply_silu_window over
// the full frame into N, then conv2d_window over N. The fused kernel must use the
// identical float op order so its inline transform of each loaded residual
// element matches the value the reference stores to N and the conv reads back.
fn runConvPrenormWindowGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const in_ch: u32 = 512;
    const out_ch: u32 = 512;
    const groups: u32 = 32;
    const height: u32 = 64;
    const width: u32 = 64;
    const ksize: u32 = 3;
    const pad: u32 = 1;
    const eps: f32 = 0.000001; // matches src/vres.zig VAE GroupNorm eps
    const hw: usize = @as(usize, height) * width;
    const in_n: usize = @as(usize, in_ch) * hw;
    const out_n: usize = @as(usize, out_ch) * hw;
    const w_n: usize = @as(usize, out_ch) * in_ch * ksize * ksize;

    const stats_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_stats");
    defer c.zdraw_metal_release_pipeline(stats_pipe);
    const apply_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_apply_silu_window");
    defer c.zdraw_metal_release_pipeline(apply_pipe);
    const conv_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_window");
    defer c.zdraw_metal_release_pipeline(conv_pipe);
    const prenorm_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_prenorm_window_v7");
    defer c.zdraw_metal_release_pipeline(prenorm_pipe);
    const stats_threads = clampThreads(c.zdraw_metal_pipeline_threads(stats_pipe));
    const apply_threads = clampThreads(c.zdraw_metal_pipeline_threads(apply_pipe));
    const conv_threads = clampThreads(c.zdraw_metal_pipeline_threads(conv_pipe));
    const prenorm_threads = clampThreads(c.zdraw_metal_pipeline_threads(prenorm_pipe));

    var prng = std.Random.DefaultPrng.init(0x9E3779B9);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32) * 1.3;
    const nweight = try allocator.alloc(f32, in_ch);
    defer allocator.free(nweight);
    for (nweight) |*v| v.* = rng.floatNorm(f32) * 0.3 + 1.0;
    const nbias = try allocator.alloc(f32, in_ch);
    defer allocator.free(nbias);
    for (nbias) |*v| v.* = rng.floatNorm(f32) * 0.2;
    const weight = try allocator.alloc(f32, w_n);
    defer allocator.free(weight);
    for (weight) |*v| v.* = rng.floatNorm(f32) * 0.12;
    const bias = try allocator.alloc(f32, out_ch);
    defer allocator.free(bias);
    for (bias) |*v| v.* = rng.floatNorm(f32) * 0.1;

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var nw_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(nweight));
    defer nw_buf.deinit();
    var nb_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(nbias));
    defer nb_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(weight));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var stats_buf = try mbuffer.Buffer.empty(device, @as(usize, groups) * 2 * 4);
    defer stats_buf.deinit();
    var n_buf = try mbuffer.Buffer.empty(device, in_n * 4);
    defer n_buf.deinit();
    var ref_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer ref_buf.deinit();
    var got_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer got_buf.deinit();

    const np = VaeNormParams{
        .channels = in_ch,
        .height = height,
        .width = width,
        .groups = groups,
        .dtype = 3,
        .bias_dtype = 3,
        .eps = eps,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    // Reference: global stats, full-frame norm+SiLU into N, then conv2d_window.
    if (mvres_stream.zdraw_metal_run_vae_norm_stats(queue, stats_pipe, in_buf.handle, stats_buf.handle, &np, stats_threads) != 0) {
        return error.DispatchFailed;
    }
    const apply = windowParams(np, 0, height, 0, width);
    if (mvres_stream.zdraw_metal_run_vae_norm_apply_window(queue, apply_pipe, in_buf.handle, stats_buf.handle, nw_buf.handle, nb_buf.handle, n_buf.handle, &apply, apply_threads) != 0) {
        return error.DispatchFailed;
    }
    const cwp = ConvWindowParams{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = height,
        .width = width,
        .ksize = ksize,
        .pad = pad,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = height,
    };
    if (mvres_stream.zdraw_metal_run_conv2d_window(queue, conv_pipe, n_buf.handle, w_buf.handle, b_buf.handle, ref_buf.handle, &cwp, conv_threads) != 0) {
        return error.DispatchFailed;
    }
    const ref = try allocator.alloc(f32, out_n);
    defer allocator.free(ref);
    c.zdraw_metal_read_buffer(ref_buf.handle, std.mem.sliceAsBytes(ref).ptr, out_n * 4);

    // Fused: same stats, then conv2d_prenorm_window straight over the raw input.
    // Two strips (ragged TS=24) to also exercise the windowed path, but stats are
    // global so any TS is bit-identical.
    var full_d: f64 = 0;
    var strip_d: f64 = 0;
    for ([_]u32{ height, 24 }) |ts| {
        var row0: u32 = 0;
        while (row0 < height) : (row0 += ts) {
            const row1 = @min(row0 + ts, height);
            const pp = ConvPrenormWindowParams{
                .in_ch = in_ch,
                .out_ch = out_ch,
                .height = height,
                .width = width,
                .ksize = ksize,
                .pad = pad,
                .dtype = 3,
                .bias_dtype = 3,
                .has_bias = 1,
                .groups = groups,
                .weight_offset = 0,
                .bias_offset = 0,
                .row0 = row0,
                .row1 = row1,
                .norm_dtype = 3,
                .norm_bias_dtype = 3,
                .norm_weight_offset = 0,
                .norm_bias_offset = 0,
            };
            if (zdraw_metal_run_conv2d_prenorm_window(queue, prenorm_pipe, in_buf.handle, w_buf.handle, b_buf.handle, got_buf.handle, stats_buf.handle, nw_buf.handle, nb_buf.handle, &pp, prenorm_threads) != 0) {
                return error.DispatchFailed;
            }
        }
        const got = try allocator.alloc(f32, out_n);
        defer allocator.free(got);
        c.zdraw_metal_read_buffer(got_buf.handle, std.mem.sliceAsBytes(got).ptr, out_n * 4);
        const d = maxBitDiff(ref, got);
        if (ts == height) full_d = d else strip_d = d;
    }

    const pass = full_d == 0 and strip_d == 0;
    std.debug.print(
        "\nconv2d prenorm window: full max|d|={d} ts=24 max|d|={d} -> {s}\n",
        .{ full_d, strip_d, if (pass) "PASS" else "FAIL" },
    );
    if (!pass) return error.ConvPrenormWindowMismatch;
}

// Mirrors MSL ConvUpsampleWindowParams / the C ZdrawConvUpsampleWindowParams.
const ConvUpsampleWindowParams = mvres_chain.ConvUpsampleWindowParams;

// Exact-streaming VAE Step 4b: prove conv2d_upsample_window (nearest-2x upsample
// fused into the conv input read) is BIT-EXACT (max|d| == 0) against the two-step
// reference: nearest 2x upsample into a full `high` buffer, then conv2d over it.
// The fused kernel reads the low-res input via the upsample index map, so it must
// reproduce upsample2-then-conv exactly. f32 weights + integer division make every
// MAC input identical; only the materialization of `high` differs.
// Real-dtype upsample-v7 gate: the production chain hands the upsample conv
// BF16 weights (promoted to f32 for v7) and a BF16 bias at a bind offset.
// Compares v1 reading bf16 directly vs v7 reading the promoted weights at the
// first up block's exact config (512ch, 64->128, strip 64 and 24).
fn runConvUpsampleV7RealGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const channels: u32 = 512;
    const in_h: u32 = 64;
    const in_w: u32 = 64;
    const out_h: u32 = in_h * 2;
    const out_w: u32 = in_w * 2;
    const in_n: usize = @as(usize, channels) * in_h * in_w;
    const out_n: usize = @as(usize, channels) * out_h * out_w;
    const w_n: usize = @as(usize, channels) * channels * 9;

    const v1 = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window");
    defer c.zdraw_metal_release_pipeline(v1);
    const v7 = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window_v7");
    defer c.zdraw_metal_release_pipeline(v7);
    const threads = clampThreads(c.zdraw_metal_pipeline_threads(v1));

    var prng = std.Random.DefaultPrng.init(0x0B16);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32) * 1.3;
    // bf16 weights + their exact f32 promotion, bf16 bias at an 8-byte offset.
    const w_bf = try allocator.alloc(u16, w_n);
    defer allocator.free(w_bf);
    const w_pro = try allocator.alloc(f32, w_n);
    defer allocator.free(w_pro);
    for (w_bf, w_pro) |*bf, *pro| {
        const bits: u32 = @bitCast(rng.floatNorm(f32) * 0.12);
        bf.* = @truncate(bits >> 16);
        pro.* = @bitCast(@as(u32, bf.*) << 16);
    }
    const bias_off: usize = 8;
    const b_bytes = try allocator.alloc(u8, @as(usize, channels) * 2 + bias_off);
    defer allocator.free(b_bytes);
    for (0..channels) |ch| {
        const bits: u32 = @bitCast(rng.floatNorm(f32) * 0.1);
        const half: u16 = @truncate(bits >> 16);
        b_bytes[bias_off + ch * 2] = @truncate(half);
        b_bytes[bias_off + ch * 2 + 1] = @truncate(half >> 8);
    }

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var wbf_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w_bf));
    defer wbf_buf.deinit();
    var wpro_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w_pro));
    defer wpro_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, b_bytes);
    defer b_buf.deinit();
    var o1_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o1_buf.deinit();
    var o7_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o7_buf.deinit();

    var worst: f64 = 0;
    inline for (.{ @as(u32, 64), @as(u32, 24) }) |ts| {
        inline for (.{ .{ v1, &o1_buf, false }, .{ v7, &o7_buf, true } }) |case| {
            var row0: u32 = 0;
            while (row0 < out_h) : (row0 += ts) {
                const up = ConvUpsampleWindowParams{
                    .channels = channels,
                    .out_height = out_h,
                    .out_width = out_w,
                    .in_height = in_h,
                    .in_width = in_w,
                    .ksize = 3,
                    .pad = 1,
                    .dtype = if (case[2]) 3 else 2,
                    .bias_dtype = 2,
                    .has_bias = 1,
                    .weight_offset = 0,
                    .bias_offset = bias_off,
                    .row0 = row0,
                    .row1 = @min(row0 + ts, out_h),
                };
                const wh = if (case[2]) wpro_buf.handle else wbf_buf.handle;
                if (mvres_chain.zdraw_metal_run_conv2d_upsample_window(queue, case[0], in_buf.handle, wh, b_buf.handle, case[1].handle, &up, threads, 0) != 0) {
                    return error.DispatchFailed;
                }
            }
        }
        const g1 = try allocator.alloc(f32, out_n);
        defer allocator.free(g1);
        c.zdraw_metal_read_buffer(o1_buf.handle, std.mem.sliceAsBytes(g1).ptr, out_n * 4);
        const g7 = try allocator.alloc(f32, out_n);
        defer allocator.free(g7);
        c.zdraw_metal_read_buffer(o7_buf.handle, std.mem.sliceAsBytes(g7).ptr, out_n * 4);
        worst = @max(worst, maxBitDiff(g1, g7));
    }
    std.debug.print("\nupsample v7 REAL-dtype gate (512ch 64->128, bf16 w/b promoted): bit|d|={d} -> {s}\n", .{
        worst, if (worst == 0) "PASS" else "FAIL",
    });
}

fn runConvUpsampleWindowGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const channels: u32 = 256;
    const in_h: u32 = 256;
    const in_w: u32 = 256;
    const out_h: u32 = in_h * 2;
    const out_w: u32 = in_w * 2;
    const ksize: u32 = 3;
    const pad: u32 = 1;
    const in_hw: usize = @as(usize, in_h) * in_w;
    const out_hw: usize = @as(usize, out_h) * out_w;
    const in_n: usize = @as(usize, channels) * in_hw;
    const out_n: usize = @as(usize, channels) * out_hw;
    const w_n: usize = @as(usize, channels) * channels * ksize * ksize;

    const conv_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d");
    defer c.zdraw_metal_release_pipeline(conv_pipe);
    const up_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window");
    defer c.zdraw_metal_release_pipeline(up_pipe);
    const conv_threads = clampThreads(c.zdraw_metal_pipeline_threads(conv_pipe));
    const up_threads = clampThreads(c.zdraw_metal_pipeline_threads(up_pipe));

    var prng = std.Random.DefaultPrng.init(0x07ADE5);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32) * 1.3;
    const weight = try allocator.alloc(f32, w_n);
    defer allocator.free(weight);
    for (weight) |*v| v.* = rng.floatNorm(f32) * 0.12;
    const bias = try allocator.alloc(f32, channels);
    defer allocator.free(bias);
    for (bias) |*v| v.* = rng.floatNorm(f32) * 0.1;

    // CPU nearest-2x upsample (out[c,R,C] = in[c,R/2,C/2]) into `high`.
    const high = try allocator.alloc(f32, out_n);
    defer allocator.free(high);
    for (0..channels) |ch| {
        for (0..out_h) |r| {
            for (0..out_w) |col| {
                high[(ch * out_h + r) * out_w + col] =
                    input[(ch * in_h + r / 2) * in_w + col / 2];
            }
        }
    }

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var high_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(high));
    defer high_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(weight));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var ref_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer ref_buf.deinit();
    var got_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer got_buf.deinit();

    // Reference: conv2d over the materialized `high` buffer.
    const cp = c.ConvParams{
        .in_ch = channels,
        .out_ch = channels,
        .height = out_h,
        .width = out_w,
        .ksize = ksize,
        .pad = pad,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    if (c.zdraw_metal_run_conv(queue, conv_pipe, high_buf.handle, w_buf.handle, b_buf.handle, ref_buf.handle, &cp, conv_threads) != 0) {
        return error.DispatchFailed;
    }
    const ref = try allocator.alloc(f32, out_n);
    defer allocator.free(ref);
    c.zdraw_metal_read_buffer(ref_buf.handle, std.mem.sliceAsBytes(ref).ptr, out_n * 4);

    // Fused: conv2d_upsample_window straight over the low-res input, two strips
    // (ragged TS=24 over out_h=64) to exercise the windowed path.
    var full_d: f64 = 0;
    var strip_d: f64 = 0;
    for ([_]u32{ out_h, 24 }) |ts| {
        var row0: u32 = 0;
        while (row0 < out_h) : (row0 += ts) {
            const row1 = @min(row0 + ts, out_h);
            const pp = ConvUpsampleWindowParams{
                .channels = channels,
                .out_height = out_h,
                .out_width = out_w,
                .in_height = in_h,
                .in_width = in_w,
                .ksize = ksize,
                .pad = pad,
                .dtype = 3,
                .bias_dtype = 3,
                .has_bias = 1,
                .weight_offset = 0,
                .bias_offset = 0,
                .row0 = row0,
                .row1 = row1,
            };
            if (mvres_chain.zdraw_metal_run_conv2d_upsample_window(queue, up_pipe, in_buf.handle, w_buf.handle, b_buf.handle, got_buf.handle, &pp, up_threads, 0) != 0) {
                return error.DispatchFailed;
            }
        }
        const got = try allocator.alloc(f32, out_n);
        defer allocator.free(got);
        c.zdraw_metal_read_buffer(got_buf.handle, std.mem.sliceAsBytes(got).ptr, out_n * 4);
        const d = maxBitDiff(ref, got);
        if (ts == out_h) full_d = d else strip_d = d;
    }

    const pass = full_d == 0 and strip_d == 0;
    std.debug.print(
        "\nconv2d upsample window: full max|d|={d} ts=24 max|d|={d} -> {s}\n",
        .{ full_d, strip_d, if (pass) "PASS" else "FAIL" },
    );
    if (!pass) return error.ConvUpsampleWindowMismatch;
}

// Exact-streaming VAE Step 3: compose the Step 1 norm split + the Step 2 conv
// window into ONE row-strip-streamed residual block and prove it is BIT-EXACT
// (max|d| == 0) against the full resblock.
//
// Oracle: the full resblock built from the SAME hand-written kernels the
// streamed path composes - full-frame vae_norm_silu -> conv2d -> vae_norm_silu
// -> conv2d -> vae_add (with conv2d skip when in_ch != out_ch). The shipping
// encode_vae_res_block routes its convs through MPSGraph by default (a different
// reduction order that no hand kernel can match bit-for-bit), so per the Step 3
// brief we match the hand-kernel resblock - which is exactly encode_vae_res_block
// under ZDRAW_VAE=raw. The streamed-vs-its-own-oracle delta must be exactly 0.
//
// Streamed: mvres_stream.run, which keeps norm1/conv1 outputs as full buffers
// (global stats + global halos) and emits the output in row-strips of height TS.
// Because the math is global, TS is a pure scheduling choice: a ragged TS=24
// (24+24+16 over H=64), TS=full (one strip), and TS=8 (many strips) must all
// give max|d|=0, including the in_ch != out_ch skip-conv case.
const VaeResShape = struct {
    in_ch: u32,
    out_ch: u32,
    height: u32 = 64,
    width: u32 = 64,
    groups: u32 = 32,
};

// Production StreamBuffers ABI (field order is the contract with
// ZdrawVaeResStreamBuffers in metal_api.m); one declaration, no drift.
const ChainStreamBuffers = mvres_chain.StreamBuffers;

// THE chain-entry unit gate: drives zdraw_metal_run_vae_res_stream_chain (the
// encoder the decode actually uses) with real buffers at a chain-like shape,
// comparing the v1 prenorm pipeline (decode-proven) against a candidate. This
// boundary had no gate, which let an in-chain-only kernel corruption slip past
// every standalone gate - same lesson as steel.
// THE dump/replay discriminator: load the live decode's captured operand set
// (ZDRAW_DUMP_DIR in mvres_stream_chain) and replay conv1 through the v1 and
// v7 prenorm kernels standalone. v1!=v7 here => real-data kernel bug; v1==v7
// => the live path hands the kernels different state.
// Upsample discriminator: replay the captured live operands (ZDRAW_DUMP_UP)
// through v1-reading-bf16 vs v7-reading-promoted-f32, full strips.
fn replayUp(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, dir: []const u8) !void {
    const meta = try loadF32s(allocator, dir, "up_meta.bin", 5);
    defer allocator.free(meta);
    const mu = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(meta));
    const channels: u32 = mu[0];
    const in_h: u32 = mu[1];
    const in_w: u32 = mu[2];
    std.debug.print("replay-up: ch={d} {d}x{d} wdtype={d} bdtype={d}\n", .{ channels, in_h, in_w, mu[3], mu[4] });
    const out_h = in_h * 2;
    const out_w = in_w * 2;
    const in_n: usize = @as(usize, channels) * in_h * in_w;
    const out_n: usize = @as(usize, channels) * out_h * out_w;
    const w_n: usize = @as(usize, channels) * channels * 9;

    const input = try loadF32s(allocator, dir, "up_input.bin", in_n);
    defer allocator.free(input);
    const w_bf16 = try loadF32s(allocator, dir, "up_w.bin", w_n / 2);
    defer allocator.free(w_bf16);
    const bias = try loadF32s(allocator, dir, "up_b.bin", channels / 2);
    defer allocator.free(bias);
    // promote the captured bf16 weights exactly as the chain does
    const w_pro = try allocator.alloc(f32, w_n);
    defer allocator.free(w_pro);
    const w_words = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(w_bf16));
    for (w_pro, w_words) |*dst, src| dst.* = @bitCast(@as(u32, src) << 16);

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var wbf_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w_bf16));
    defer wbf_buf.deinit();
    var wpro_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(w_pro));
    defer wpro_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_buf.deinit();
    var o1_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o1_buf.deinit();
    var o7_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o7_buf.deinit();

    const v1 = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window");
    defer c.zdraw_metal_release_pipeline(v1);
    const v7 = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window_v7");
    defer c.zdraw_metal_release_pipeline(v7);
    const threads = clampThreads(c.zdraw_metal_pipeline_threads(v1));

    inline for (.{ .{ v1, &o1_buf, false }, .{ v7, &o7_buf, true } }) |case| {
        var row0: u32 = 0;
        while (row0 < out_h) : (row0 += 64) {
            const up = ConvUpsampleWindowParams{
                .channels = channels,
                .out_height = out_h,
                .out_width = out_w,
                .in_height = in_h,
                .in_width = in_w,
                .ksize = 3,
                .pad = 1,
                .dtype = if (case[2]) 3 else 2,
                .bias_dtype = 2,
                .has_bias = 1,
                .weight_offset = 0,
                .bias_offset = 0,
                .row0 = row0,
                .row1 = @min(row0 + 64, out_h),
            };
            const wh = if (case[2]) wpro_buf.handle else wbf_buf.handle;
            if (mvres_chain.zdraw_metal_run_conv2d_upsample_window(queue, case[0], in_buf.handle, wh, b_buf.handle, case[1].handle, &up, threads, 0) != 0) {
                return error.DispatchFailed;
            }
        }
    }
    const g1 = try allocator.alloc(f32, out_n);
    defer allocator.free(g1);
    c.zdraw_metal_read_buffer(o1_buf.handle, std.mem.sliceAsBytes(g1).ptr, out_n * 4);
    const g7 = try allocator.alloc(f32, out_n);
    defer allocator.free(g7);
    c.zdraw_metal_read_buffer(o7_buf.handle, std.mem.sliceAsBytes(g7).ptr, out_n * 4);
    const bit_d = maxBitDiff(g1, g7);
    std.debug.print("REPLAY-UP VERDICT: v1-vs-v7 bit|d|={d} -> {s}\n", .{
        bit_d,
        if (bit_d == 0) "B: live path hands different state" else "A: real-data kernel bug",
    });
}

fn replayPrenorm(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, dir: []const u8) !void {
    const meta = try loadF32s(allocator, dir, "meta.bin", 5);
    defer allocator.free(meta);
    const mu = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(meta));
    const in_ch: u32 = mu[0];
    const out_ch: u32 = mu[1];
    const height: u32 = mu[2];
    const width: u32 = mu[3];
    const groups: u32 = mu[4];
    std.debug.print("replay: in={d} out={d} {d}x{d} groups={d}\n", .{ in_ch, out_ch, height, width, groups });
    const hw: usize = @as(usize, height) * width;
    const out_n: usize = @as(usize, out_ch) * hw;

    const input = try loadF32s(allocator, dir, "input.bin", @as(usize, in_ch) * hw);
    defer allocator.free(input);

    const n1w = try loadF32s(allocator, dir, "n1w.bin", in_ch);
    defer allocator.free(n1w);
    const n1b = try loadF32s(allocator, dir, "n1b.bin", in_ch);
    defer allocator.free(n1b);
    const c1w = try loadF32s(allocator, dir, "c1w.bin", @as(usize, out_ch) * in_ch * 9);
    defer allocator.free(c1w);
    const c1b = try loadF32s(allocator, dir, "c1b.bin", out_ch);
    defer allocator.free(c1b);

    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var st_buf = try mbuffer.Buffer.empty(device, @as(usize, groups) * 2 * 4);
    defer st_buf.deinit();
    const stats_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_stats");
    defer c.zdraw_metal_release_pipeline(stats_pipe);
    var np = std.mem.zeroes(VaeNormParams);
    np.channels = in_ch;
    np.height = height;
    np.width = width;
    np.groups = groups;
    np.dtype = 3;
    np.bias_dtype = 3;
    np.eps = 0.000001;
    if (mvres_stream.zdraw_metal_run_vae_norm_stats(queue, stats_pipe, in_buf.handle, st_buf.handle, &np, clampThreads(c.zdraw_metal_pipeline_threads(stats_pipe))) != 0) return error.DispatchFailed;
    var nw_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(n1w));
    defer nw_buf.deinit();
    var nb_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(n1b));
    defer nb_buf.deinit();
    var w_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c1w));
    defer w_buf.deinit();
    var b_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c1b));
    defer b_buf.deinit();
    var o1_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o1_buf.deinit();
    var o7_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer o7_buf.deinit();

    const pp = ConvPrenormWindowParams{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = height,
        .width = width,
        .ksize = 3,
        .pad = 1,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .groups = groups,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = height,
        .norm_dtype = 3,
        .norm_bias_dtype = 3,
        .norm_weight_offset = 0,
        .norm_bias_offset = 0,
    };
    const v1 = try compile(device, mconv_shader.conv.ptr, "conv2d_prenorm_window");
    defer c.zdraw_metal_release_pipeline(v1);
    const v7 = try compile(device, mconv_shader.conv.ptr, "conv2d_prenorm_window_v7");
    defer c.zdraw_metal_release_pipeline(v7);
    if (zdraw_metal_run_conv2d_prenorm_window(queue, v1, in_buf.handle, w_buf.handle, b_buf.handle, o1_buf.handle, st_buf.handle, nw_buf.handle, nb_buf.handle, &pp, 32) != 0) return error.DispatchFailed;
    if (zdraw_metal_run_conv2d_prenorm_window(queue, v7, in_buf.handle, w_buf.handle, b_buf.handle, o7_buf.handle, st_buf.handle, nw_buf.handle, nb_buf.handle, &pp, 32) != 0) return error.DispatchFailed;
    const g1 = try allocator.alloc(f32, out_n);
    defer allocator.free(g1);
    c.zdraw_metal_read_buffer(o1_buf.handle, std.mem.sliceAsBytes(g1).ptr, out_n * 4);
    const g7 = try allocator.alloc(f32, out_n);
    defer allocator.free(g7);
    c.zdraw_metal_read_buffer(o7_buf.handle, std.mem.sliceAsBytes(g7).ptr, out_n * 4);
    const bit_d = maxBitDiff(g1, g7);
    std.debug.print("REPLAY VERDICT: v1-vs-v7 bit|d|={d} -> {s}\n", .{
        bit_d,
        if (bit_d == 0) "B: live path hands different state" else "A: real-data kernel bug",
    });
}

fn loadF32s(allocator: std.mem.Allocator, dir: []const u8, name: []const u8, count: usize) ![]f32 {
    var pbuf: [256]u8 = undefined;
    const pz = try std.fmt.bufPrintZ(&pbuf, "{s}/{s}", .{ dir, name });
    const fh = std.c.fopen(pz.ptr, "rb") orelse return error.FileNotFound;
    defer _ = std.c.fclose(fh);
    const buf = try allocator.alloc(f32, count);
    errdefer allocator.free(buf);
    if (std.c.fread(@ptrCast(buf.ptr), 4, count, fh) != count) return error.ShortRead;
    return buf;
}

// The strict-VAE per-kernel budget (plan step 2): time every kernel class at
// the REAL 1024px production shapes/strip counts and reconstruct vae-up's
// in-chain seconds. Sum-vs-observed names the orchestration overhead share.
fn runVaeBudget(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const BShape = struct { ic: u32, oc: u32, h: u32, w: u32, n: u32, what: []const u8 };
    // 1024px ladder: mid 2 resnets @128(512), up0 3@128(512), up1 3@256(512),
    // up2 3@512(512->256 first), up3 3@1024(256->128 first); upsamples after
    // up0/up1/up2. conv counts = conv1+conv2 per resnet.
    const convs = [_]BShape{
        .{ .ic = 512, .oc = 512, .h = 128, .w = 128, .n = 10, .what = "res conv 512@128 (mid+up0)" },
        .{ .ic = 512, .oc = 512, .h = 256, .w = 256, .n = 6, .what = "res conv 512@256 (up1)" },
        .{ .ic = 512, .oc = 256, .h = 512, .w = 512, .n = 1, .what = "res conv1 512->256@512" },
        .{ .ic = 256, .oc = 256, .h = 512, .w = 512, .n = 5, .what = "res conv 256@512 (up2)" },
        .{ .ic = 256, .oc = 128, .h = 1024, .w = 1024, .n = 1, .what = "res conv1 256->128@1024" },
        .{ .ic = 128, .oc = 128, .h = 1024, .w = 1024, .n = 5, .what = "res conv 128@1024 (up3)" },
    };
    var total_ms: f64 = 0;
    std.debug.print("\n=== strict VAE budget at 1024 (standalone, v7 kernels) ===\n", .{});
    for (convs) |sh| {
        const ms = try budgetConv(io, allocator, device, queue, sh.ic, sh.oc, sh.h, sh.w, "conv2d_prenorm_window_v7");
        std.debug.print("{d:7.1} ms x{d:2} = {d:8.1} ms  {s}\n", .{ ms, sh.n, ms * f(sh.n), sh.what });
        total_ms += ms * f(sh.n);
    }
    // upsample convs (fused 2x): up0->256 (512ch), up1->512 (512ch), up2->1024 (256ch)
    const ups = [_]BShape{
        .{ .ic = 512, .oc = 512, .h = 128, .w = 128, .n = 1, .what = "upsample 512: 128->256" },
        .{ .ic = 512, .oc = 512, .h = 256, .w = 256, .n = 1, .what = "upsample 512: 256->512" },
        .{ .ic = 256, .oc = 256, .h = 512, .w = 512, .n = 1, .what = "upsample 256: 512->1024" },
    };
    for (ups) |sh| {
        const ms = try budgetUp(io, allocator, device, queue, sh.ic, sh.h, sh.w);
        std.debug.print("{d:7.1} ms x{d:2} = {d:8.1} ms  {s}\n", .{ ms, sh.n, ms, sh.what });
        total_ms += ms;
    }
    // skip 1x1s: 512->256@512, 256->128@1024
    const skips = [_]BShape{
        .{ .ic = 512, .oc = 256, .h = 512, .w = 512, .n = 1, .what = "skip 1x1 512->256@512" },
        .{ .ic = 256, .oc = 128, .h = 1024, .w = 1024, .n = 1, .what = "skip 1x1 256->128@1024" },
    };
    for (skips) |sh| {
        const ms = try budgetSkip(io, allocator, device, queue, sh.ic, sh.oc, sh.h, sh.w);
        std.debug.print("{d:7.1} ms x{d:2} = {d:8.1} ms  {s}\n", .{ ms, sh.n, ms, sh.what });
        total_ms += ms;
    }
    std.debug.print("RECONSTRUCTED conv+up+skip total: {d:.0} ms (observed in-chain vae-up ~5300 ms;\n", .{total_ms});
    std.debug.print("the difference = stats+add+dispatch+barrier overhead)\n", .{});
}

fn budgetConv(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, ic: u32, oc: u32, h: u32, w: u32, kernel: [*:0]const u8) !f64 {
    const hw: usize = @as(usize, h) * w;
    const in_n: usize = @as(usize, ic) * hw;
    const w_n: usize = @as(usize, oc) * ic * 9;
    const pipe = try compile(device, mconv_shader.conv.ptr, kernel);
    defer c.zdraw_metal_release_pipeline(pipe);
    var prng = std.Random.DefaultPrng.init(0xB0D6E7);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32);
    const wts = try allocator.alloc(f32, w_n);
    defer allocator.free(wts);
    for (wts) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const bias = try allocator.alloc(f32, oc);
    defer allocator.free(bias);
    for (bias) |*v| v.* = 0;
    const stats = try allocator.alloc(f32, 64);
    defer allocator.free(stats);
    for (0..32) |g| {
        stats[g * 2] = 0;
        stats[g * 2 + 1] = 1;
    }
    const nw = try allocator.alloc(f32, ic);
    defer allocator.free(nw);
    for (nw) |*v| v.* = 1;
    const nb = try allocator.alloc(f32, ic);
    defer allocator.free(nb);
    for (nb) |*v| v.* = 0;
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_b.deinit();
    var st_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(stats));
    defer st_b.deinit();
    var nw_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(nw));
    defer nw_b.deinit();
    var nb_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(nb));
    defer nb_b.deinit();
    var out_b = try mbuffer.Buffer.empty(device, @as(usize, oc) * hw * 4);
    defer out_b.deinit();
    const pp = ConvPrenormWindowParams{
        .in_ch = ic,
        .out_ch = oc,
        .height = h,
        .width = w,
        .ksize = 3,
        .pad = 1,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .groups = 32,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = h,
        .norm_dtype = 3,
        .norm_bias_dtype = 3,
        .norm_weight_offset = 0,
        .norm_bias_offset = 0,
    };
    // warm + 3 timed
    if (zdraw_metal_run_conv2d_prenorm_window(queue, pipe, in_b.handle, w_b.handle, b_b.handle, out_b.handle, st_b.handle, nw_b.handle, nb_b.handle, &pp, 32) != 0) return error.DispatchFailed;
    var best: f64 = 1e18;
    for (0..3) |_| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (zdraw_metal_run_conv2d_prenorm_window(queue, pipe, in_b.handle, w_b.handle, b_b.handle, out_b.handle, st_b.handle, nw_b.handle, nb_b.handle, &pp, 32) != 0) return error.DispatchFailed;
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetUp(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, ch: u32, in_h: u32, in_w: u32) !f64 {
    const out_h = in_h * 2;
    const out_w = in_w * 2;
    const in_n: usize = @as(usize, ch) * in_h * in_w;
    const w_n: usize = @as(usize, ch) * ch * 9;
    const pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_upsample_window_v7");
    defer c.zdraw_metal_release_pipeline(pipe);
    var prng = std.Random.DefaultPrng.init(0xB0D6E8);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32);
    const wts = try allocator.alloc(f32, w_n);
    defer allocator.free(wts);
    for (wts) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const bias = try allocator.alloc(f32, ch);
    defer allocator.free(bias);
    for (bias) |*v| v.* = 0;
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(bias));
    defer b_b.deinit();
    var out_b = try mbuffer.Buffer.empty(device, @as(usize, ch) * out_h * out_w * 4);
    defer out_b.deinit();
    const up = ConvUpsampleWindowParams{
        .channels = ch,
        .out_height = out_h,
        .out_width = out_w,
        .in_height = in_h,
        .in_width = in_w,
        .ksize = 3,
        .pad = 1,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = out_h,
    };
    if (mvres_chain.zdraw_metal_run_conv2d_upsample_window(queue, pipe, in_b.handle, w_b.handle, b_b.handle, out_b.handle, &up, 32, 0) != 0) return error.DispatchFailed;
    var best: f64 = 1e18;
    for (0..3) |_| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (mvres_chain.zdraw_metal_run_conv2d_upsample_window(queue, pipe, in_b.handle, w_b.handle, b_b.handle, out_b.handle, &up, 32, 0) != 0) return error.DispatchFailed;
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (dt < best) best = dt;
    }
    return best / 1e6;
}

fn budgetSkip(io: std.Io, allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque, ic: u32, oc: u32, h: u32, w: u32) !f64 {
    const hw: usize = @as(usize, h) * w;
    const pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_window");
    defer c.zdraw_metal_release_pipeline(pipe);
    var prng = std.Random.DefaultPrng.init(0xB0D6E9);
    const rng = prng.random();
    const input = try allocator.alloc(f32, @as(usize, ic) * hw);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32);
    const wts = try allocator.alloc(f32, @as(usize, oc) * ic);
    defer allocator.free(wts);
    for (wts) |*v| v.* = rng.floatNorm(f32) * 0.05;
    var in_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_b.deinit();
    var w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(wts));
    defer w_b.deinit();
    var out_b = try mbuffer.Buffer.empty(device, @as(usize, oc) * hw * 4);
    defer out_b.deinit();
    const cp = ConvWindowParams{
        .in_ch = ic,
        .out_ch = oc,
        .height = h,
        .width = w,
        .ksize = 1,
        .pad = 0,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 0,
        .weight_offset = 0,
        .bias_offset = 0,
        .row0 = 0,
        .row1 = h,
    };
    if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp, 32) != 0) return error.DispatchFailed;
    var best: f64 = 1e18;
    for (0..3) |_| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        if (mvres_stream.zdraw_metal_run_conv2d_window(queue, pipe, in_b.handle, w_b.handle, w_b.handle, out_b.handle, &cp, 32) != 0) return error.DispatchFailed;
        const dt: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
        if (dt < best) best = dt;
    }
    return best / 1e6;
}

fn runVaeResChainGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const in_ch: u32 = 512;
    const out_ch: u32 = 512;
    const groups: u32 = 32;
    const height: u32 = 64;
    const width: u32 = 1024;
    const eps: f32 = 0.000001;
    const strip_rows: u32 = 24;
    const hw: usize = @as(usize, height) * width;
    const in_n: usize = @as(usize, in_ch) * hw;
    const out_n: usize = @as(usize, out_ch) * hw;

    const stats_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_stats");
    defer c.zdraw_metal_release_pipeline(stats_pipe);
    const pre_v1 = try compile(device, mconv_shader.conv.ptr, "conv2d_prenorm_window");
    defer c.zdraw_metal_release_pipeline(pre_v1);
    const pre_v7 = try compile(device, mconv_shader.conv.ptr, "conv2d_prenorm_window_v7");
    defer c.zdraw_metal_release_pipeline(pre_v7);
    const convw_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_window");
    defer c.zdraw_metal_release_pipeline(convw_pipe);
    const addw_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_add_window");
    defer c.zdraw_metal_release_pipeline(addw_pipe);
    const stats_threads = clampThreads(c.zdraw_metal_pipeline_threads(stats_pipe));
    const conv_threads = clampThreads(c.zdraw_metal_pipeline_threads(convw_pipe));
    const add_threads = clampThreads(c.zdraw_metal_pipeline_threads(addw_pipe));

    var prng = std.Random.DefaultPrng.init(0xC4A1);
    const rng = prng.random();
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32);
    const n1w = try allocator.alloc(f32, in_ch);
    defer allocator.free(n1w);
    for (n1w) |*v| v.* = 1.0 + rng.floatNorm(f32) * 0.2;
    const n1b = try allocator.alloc(f32, in_ch);
    defer allocator.free(n1b);
    for (n1b) |*v| v.* = rng.floatNorm(f32) * 0.1;
    const c1w = try allocator.alloc(f32, @as(usize, out_ch) * in_ch * 9);
    defer allocator.free(c1w);
    for (c1w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const c1b = try allocator.alloc(f32, out_ch);
    defer allocator.free(c1b);
    for (c1b) |*v| v.* = rng.floatNorm(f32) * 0.1;
    const n2w = try allocator.alloc(f32, out_ch);
    defer allocator.free(n2w);
    for (n2w) |*v| v.* = 1.0 + rng.floatNorm(f32) * 0.2;
    const n2b = try allocator.alloc(f32, out_ch);
    defer allocator.free(n2b);
    for (n2b) |*v| v.* = rng.floatNorm(f32) * 0.1;
    const c2w = try allocator.alloc(f32, @as(usize, out_ch) * out_ch * 9);
    defer allocator.free(c2w);
    for (c2w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const c2b = try allocator.alloc(f32, out_ch);
    defer allocator.free(c2b);
    for (c2b) |*v| v.* = rng.floatNorm(f32) * 0.1;

    var n1w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(n1w));
    defer n1w_b.deinit();
    var n1b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(n1b));
    defer n1b_b.deinit();
    // Production binds weights at file offsets (no-copy wraps); replicate an
    // 8-mod-16 alignment to catch vector-load alignment assumptions.
    const c1w_off: usize = 8;
    const c1w_padded = try allocator.alloc(u8, c1w.len * 4 + c1w_off);
    defer allocator.free(c1w_padded);
    @memcpy(c1w_padded[c1w_off..], std.mem.sliceAsBytes(c1w));
    var c1w_b = try mbuffer.Buffer.fromBytes(device, c1w_padded);
    defer c1w_b.deinit();
    var c1b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c1b));
    defer c1b_b.deinit();
    var n2w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(n2w));
    defer n2w_b.deinit();
    var n2b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(n2b));
    defer n2b_b.deinit();
    var c2w_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c2w));
    defer c2w_b.deinit();
    var c2b_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c2b));
    defer c2b_b.deinit();
    var conv1_out = try mbuffer.Buffer.empty(device, out_n * 4);
    defer conv1_out.deinit();
    var stats1 = try mbuffer.Buffer.empty(device, @as(usize, groups) * 2 * 4);
    defer stats1.deinit();
    var stats2 = try mbuffer.Buffer.empty(device, @as(usize, groups) * 2 * 4);
    defer stats2.deinit();
    var slot_a = try mbuffer.Buffer.empty(device, in_n * 4);
    defer slot_a.deinit();
    var slot_b = try mbuffer.Buffer.empty(device, out_n * 4);
    defer slot_b.deinit();
    const out_a = try allocator.alloc(f32, out_n);
    defer allocator.free(out_a);
    const out_b = try allocator.alloc(f32, out_n);
    defer allocator.free(out_b);

    const ps = mvres_param.ResParams{
        .norm1 = .{
            .channels = in_ch,
            .height = height,
            .width = width,
            .groups = groups,
            .dtype = 3,
            .bias_dtype = 3,
            .eps = eps,
            .weight_offset = 0,
            .bias_offset = 0,
        },
        .conv1 = .{
            .in_ch = in_ch,
            .out_ch = out_ch,
            .height = height,
            .width = width,
            .ksize = 3,
            .pad = 1,
            .dtype = 3,
            .bias_dtype = 3,
            .has_bias = 1,
            .weight_offset = c1w_off,
            .bias_offset = 0,
        },
        .norm2 = .{
            .channels = out_ch,
            .height = height,
            .width = width,
            .groups = groups,
            .dtype = 3,
            .bias_dtype = 3,
            .eps = eps,
            .weight_offset = 0,
            .bias_offset = 0,
        },
        .conv2 = .{
            .in_ch = out_ch,
            .out_ch = out_ch,
            .height = height,
            .width = width,
            .ksize = 3,
            .pad = 1,
            .dtype = 3,
            .bias_dtype = 3,
            .has_bias = 1,
            .weight_offset = 0,
            .bias_offset = 0,
        },
        .skip = .{
            .in_ch = in_ch,
            .out_ch = out_ch,
            .height = height,
            .width = width,
            .ksize = 1,
            .pad = 0,
            .dtype = 3,
            .bias_dtype = 3,
            .has_bias = 0,
            .weight_offset = 0,
            .bias_offset = 0,
        },
        .has_skip = 0,
        .out_count = @intCast(out_n),
    };

    // Three chained blocks with the REAL pool ping-pong: block0 reads the
    // uploaded input and writes slotB; block1 reads slotB, writes slotA (which
    // held the now-dead input); block2 reads slotA, writes slotB. conv1_out and
    // stats are shared scratch across all blocks - exactly dispatchGroup's
    // buffer plumbing.
    inline for (.{ .{ pre_v1, &out_a }, .{ pre_v7, &out_b } }) |case| {
        c.zdraw_metal_write_buffer(slot_a.handle, std.mem.sliceAsBytes(input).ptr, in_n * 4);
        const slotw = [_]ChainStreamBuffers{
            mkChainBufs(slot_a.handle, slot_b.handle, conv1_out.handle, stats1.handle, stats2.handle, n1w_b.handle, n1b_b.handle, c1w_b.handle, c1b_b.handle, n2w_b.handle, n2b_b.handle, c2w_b.handle, c2b_b.handle),
            mkChainBufs(slot_b.handle, slot_a.handle, conv1_out.handle, stats1.handle, stats2.handle, n1w_b.handle, n1b_b.handle, c1w_b.handle, c1b_b.handle, n2w_b.handle, n2b_b.handle, c2w_b.handle, c2b_b.handle),
            mkChainBufs(slot_a.handle, slot_b.handle, conv1_out.handle, stats1.handle, stats2.handle, n1w_b.handle, n1b_b.handle, c1w_b.handle, c1b_b.handle, n2w_b.handle, n2b_b.handle, c2w_b.handle, c2b_b.handle),
        };
        const pa = [_]mvres_param.ResParams{ ps, ps, ps };
        if (mvres_chain.zdraw_metal_run_vae_res_stream_chain(
            queue,
            stats_pipe,
            case[0],
            convw_pipe,
            addw_pipe,
            stats_pipe,
            null,
            convw_pipe,
            null,
            null,
            &slotw,
            &pa,
            3,
            strip_rows,
            stats_threads,
            conv_threads,
            add_threads,
            0,
            0,
            null,
        ) != 0) return error.DispatchFailed;
        c.zdraw_metal_read_buffer(slot_b.handle, std.mem.sliceAsBytes(case[1].*).ptr, out_n * 4);
    }
    const bit_d = maxBitDiff(out_a, out_b);

    // Skip-block case: in 512 -> out 256 with has_skip=1, and the skip output
    // buffer ALIASING conv1_out exactly as the production pool does.
    var skipw = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c1w[0 .. 256 * 512]));
    defer skipw.deinit();
    var ps2 = ps;
    ps2.conv1.out_ch = 256;
    ps2.norm2.channels = 256;
    ps2.conv2.in_ch = 256;
    ps2.conv2.out_ch = 256;
    ps2.skip = .{
        .in_ch = 512,
        .out_ch = 256,
        .height = height,
        .width = width,
        .ksize = 1,
        .pad = 0,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 0,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    ps2.has_skip = 1;
    ps2.out_count = @intCast(@as(usize, 256) * hw);
    var skip_d: f64 = -1;
    inline for (.{ .{ pre_v1, out_a }, .{ pre_v7, out_b } }) |case| {
        c.zdraw_metal_write_buffer(slot_a.handle, std.mem.sliceAsBytes(input).ptr, in_n * 4);
        var bufs2 = mkChainBufs(slot_a.handle, slot_b.handle, conv1_out.handle, stats1.handle, stats2.handle, n1w_b.handle, n1b_b.handle, c1w_b.handle, c1b_b.handle, n2w_b.handle, n2b_b.handle, c2w_b.handle, c2b_b.handle);
        bufs2.skip = conv1_out.handle; // the production aliasing
        bufs2.skip_w = skipw.handle;
        const slotw2 = [_]ChainStreamBuffers{bufs2};
        const pa2 = [_]mvres_param.ResParams{ps2};
        if (mvres_chain.zdraw_metal_run_vae_res_stream_chain(
            queue,
            stats_pipe,
            case[0],
            convw_pipe,
            addw_pipe,
            stats_pipe,
            null,
            convw_pipe,
            null,
            null,
            &slotw2,
            &pa2,
            1,
            strip_rows,
            stats_threads,
            conv_threads,
            add_threads,
            0,
            0,
            null,
        ) != 0) return error.DispatchFailed;
        c.zdraw_metal_read_buffer(slot_b.handle, std.mem.sliceAsBytes(case[1]).ptr, @as(usize, 256) * hw * 4);
    }
    skip_d = maxBitDiff(out_a[0 .. 256 * hw], out_b[0 .. 256 * hw]);

    // Real-dtype case: the production VAE stores conv weights as BF16
    // (dtype=2). v1 reads bf16 via read_value; v7's f32-only contract is
    // satisfied by promoting the SAME values to f32 (exact widening) — the
    // production ZDRAW_VAE_V7 path. This is the case the synthetic-f32 gate
    // missed: routing v7 at dtype=2 without promotion corrupts.
    const c1w_bf = try allocator.alloc(u16, c1w.len);
    defer allocator.free(c1w_bf);
    const c1w_pro = try allocator.alloc(f32, c1w.len);
    defer allocator.free(c1w_pro);
    for (c1w, c1w_bf, c1w_pro) |v, *bf, *pro| {
        const bits: u32 = @bitCast(v);
        bf.* = @truncate(bits >> 16); // RZ truncation is fine for test data
        pro.* = @bitCast(@as(u32, bf.*) << 16);
    }
    const c2w_bf = try allocator.alloc(u16, c2w.len);
    defer allocator.free(c2w_bf);
    const c2w_pro = try allocator.alloc(f32, c2w.len);
    defer allocator.free(c2w_pro);
    for (c2w, c2w_bf, c2w_pro) |v, *bf, *pro| {
        const bits: u32 = @bitCast(v);
        bf.* = @truncate(bits >> 16);
        pro.* = @bitCast(@as(u32, bf.*) << 16);
    }
    var c1wbf_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c1w_bf));
    defer c1wbf_b.deinit();
    var c2wbf_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c2w_bf));
    defer c2wbf_b.deinit();
    var c1wpro_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c1w_pro));
    defer c1wpro_b.deinit();
    var c2wpro_b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(c2w_pro));
    defer c2wpro_b.deinit();
    var ps_bf = ps;
    ps_bf.conv1.dtype = 2;
    ps_bf.conv2.dtype = 2;
    ps_bf.conv1.weight_offset = 0; // bf16/promoted buffers are unpadded
    ps_bf.conv2.weight_offset = 0;
    var bf_d: f64 = -1;
    inline for (.{
        .{ pre_v1, out_a, false },
        .{ pre_v7, out_b, true },
    }) |case| {
        c.zdraw_metal_write_buffer(slot_a.handle, std.mem.sliceAsBytes(input).ptr, in_n * 4);
        var bb = mkChainBufs(slot_a.handle, slot_b.handle, conv1_out.handle, stats1.handle, stats2.handle, n1w_b.handle, n1b_b.handle, c1wbf_b.handle, c1b_b.handle, n2w_b.handle, n2b_b.handle, c2wbf_b.handle, c2b_b.handle);
        var pp_case = ps_bf;
        if (case[2]) {
            bb.conv1_w = c1wpro_b.handle;
            bb.conv2_w = c2wpro_b.handle;
            pp_case.conv1.dtype = 3;
            pp_case.conv2.dtype = 3;
        }
        const sw = [_]ChainStreamBuffers{bb};
        const pa3 = [_]mvres_param.ResParams{pp_case};
        if (mvres_chain.zdraw_metal_run_vae_res_stream_chain(
            queue,
            stats_pipe,
            case[0],
            convw_pipe,
            addw_pipe,
            stats_pipe,
            null,
            convw_pipe,
            null,
            null,
            &sw,
            &pa3,
            1,
            strip_rows,
            stats_threads,
            conv_threads,
            add_threads,
            0,
            0,
            null,
        ) != 0) return error.DispatchFailed;
        c.zdraw_metal_read_buffer(slot_b.handle, std.mem.sliceAsBytes(case[1]).ptr, out_n * 4);
    }
    bf_d = maxBitDiff(out_a, out_b);

    std.debug.print("\nvae res CHAIN gate (v1 vs v7): pingpong bit|d|={d} | skip-alias bit|d|={d} | bf16-promoted bit|d|={d} -> {s}\n", .{
        bit_d, skip_d, bf_d, if (bit_d == 0 and skip_d == 0 and bf_d == 0) "PASS" else "FAIL",
    });
}

fn mkChainBufs(
    input: *anyopaque,
    output: *anyopaque,
    conv1_out: *anyopaque,
    stats1: *anyopaque,
    stats2: *anyopaque,
    n1w: *anyopaque,
    n1b: *anyopaque,
    c1w: *anyopaque,
    c1b: *anyopaque,
    n2w: *anyopaque,
    n2b: *anyopaque,
    c2w: *anyopaque,
    c2b: *anyopaque,
) ChainStreamBuffers {
    return .{
        .input = input,
        .output = output,
        .conv1_out = conv1_out,
        .stats1 = stats1,
        .stats2 = stats2,
        .skip = output,
        .norm1_w = n1w,
        .norm1_b = n1b,
        .conv1_w = c1w,
        .conv1_b = c1b,
        .norm2_w = n2w,
        .norm2_b = n2b,
        .conv2_w = c2w,
        .conv2_b = c2b,
        .skip_w = c1w,
        .skip_b = c1b,
        .norm = conv1_out,
    };
}

fn runVaeResStreamGate(allocator: std.mem.Allocator, device: *anyopaque, queue: *anyopaque) !void {
    const eps: f32 = 0.000001; // matches src/vres.zig VAE GroupNorm eps

    // Compile every kernel once; both the oracle and the streamed path reuse them.
    const norm_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_silu");
    defer c.zdraw_metal_release_pipeline(norm_pipe);
    const conv_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d");
    defer c.zdraw_metal_release_pipeline(conv_pipe);
    const add_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_add");
    defer c.zdraw_metal_release_pipeline(add_pipe);
    const stats_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_stats");
    defer c.zdraw_metal_release_pipeline(stats_pipe);
    const apply_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_norm_apply_silu_window");
    defer c.zdraw_metal_release_pipeline(apply_pipe);
    const win_pipe = try compile(device, mconv_shader.conv.ptr, "conv2d_window");
    defer c.zdraw_metal_release_pipeline(win_pipe);
    const addw_pipe = try compile(device, vnorm_shader.vnorm.ptr, "vae_add_window");
    defer c.zdraw_metal_release_pipeline(addw_pipe);

    const norm_threads = clampThreads(c.zdraw_metal_pipeline_threads(norm_pipe));
    const conv_threads = clampThreads(c.zdraw_metal_pipeline_threads(conv_pipe));
    const add_threads = clampThreads(c.zdraw_metal_pipeline_threads(add_pipe));
    const stats_threads = clampThreads(c.zdraw_metal_pipeline_threads(stats_pipe));
    const apply_threads = clampThreads(c.zdraw_metal_pipeline_threads(apply_pipe));
    const win_threads = clampThreads(c.zdraw_metal_pipeline_threads(win_pipe));
    const addw_threads = clampThreads(c.zdraw_metal_pipeline_threads(addw_pipe));

    const ctx = mvres_stream.Ctx{
        .queue = queue,
        .stats_pipe = stats_pipe,
        .apply_pipe = apply_pipe,
        .conv_pipe = win_pipe,
        .add_pipe = addw_pipe,
        .stats_threads = stats_threads,
        .apply_threads = apply_threads,
        .conv_threads = win_threads,
        .add_threads = addw_threads,
    };
    const oracle = OracleCtx{
        .queue = queue,
        .norm_pipe = norm_pipe,
        .conv_pipe = conv_pipe,
        .add_pipe = add_pipe,
        .norm_threads = norm_threads,
        .conv_threads = conv_threads,
        .add_threads = add_threads,
    };

    // Representative block: in_ch=out_ch=256 (no skip). H=W=64, groups=32, k=3.
    const plain = VaeResShape{ .in_ch = 256, .out_ch = 256 };
    const plain_max = try resStreamCase(allocator, device, ctx, oracle, plain, eps, 24, 0x3E5);
    // Same shape, single strip (TS=H) and many strips (TS=8): pure scheduling.
    const full_max = try resStreamCase(allocator, device, ctx, oracle, plain, eps, plain.height, 0x3E5);
    const many_max = try resStreamCase(allocator, device, ctx, oracle, plain, eps, 8, 0x3E5);

    // in_ch != out_ch (512 -> 256): exercises the 1x1 skip conv, ragged TS=24.
    const skip = VaeResShape{ .in_ch = 512, .out_ch = 256 };
    const skip_max = try resStreamCase(allocator, device, ctx, oracle, skip, eps, 24, 0x5C1);

    const pass = plain_max == 0 and full_max == 0 and many_max == 0 and skip_max == 0;
    std.debug.print(
        "\nvae resblock stream: ts=24 max|d|={d} | ts=8 max|d|={d} | ts=full max|d|={d} | skip-case max|d|={d} -> {s}\n",
        .{ plain_max, many_max, full_max, skip_max, if (pass) "PASS" else "FAIL" },
    );
    if (!pass) return error.VaeResStreamMismatch;
}

// Queue + pipelines/threads for the hand-kernel oracle resblock.
const OracleCtx = struct {
    queue: *anyopaque,
    norm_pipe: *anyopaque,
    conv_pipe: *anyopaque,
    add_pipe: *anyopaque,
    norm_threads: usize,
    conv_threads: usize,
    add_threads: usize,
};

// Build the oracle (full resblock) and the streamed output for one shape/TS,
// returning max|streamed - oracle| over the whole frame (must be 0).
fn resStreamCase(
    allocator: std.mem.Allocator,
    device: *anyopaque,
    ctx: mvres_stream.Ctx,
    oracle: OracleCtx,
    shape: VaeResShape,
    eps: f32,
    strip_rows: u32,
    seed: u64,
) !f64 {
    const hw: usize = @as(usize, shape.height) * shape.width;
    const in_n: usize = @as(usize, shape.in_ch) * hw;
    const out_n: usize = @as(usize, shape.out_ch) * hw;
    const has_skip = shape.in_ch != shape.out_ch;

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // Inputs and exact f32 weights/biases for every stage.
    const input = try allocator.alloc(f32, in_n);
    defer allocator.free(input);
    for (input) |*v| v.* = rng.floatNorm(f32) * 1.4;
    const norm1_w = try allocator.alloc(f32, shape.in_ch);
    defer allocator.free(norm1_w);
    const norm1_b = try allocator.alloc(f32, shape.in_ch);
    defer allocator.free(norm1_b);
    const conv1_w = try allocator.alloc(f32, @as(usize, shape.out_ch) * shape.in_ch * 9);
    defer allocator.free(conv1_w);
    const conv1_b = try allocator.alloc(f32, shape.out_ch);
    defer allocator.free(conv1_b);
    const norm2_w = try allocator.alloc(f32, shape.out_ch);
    defer allocator.free(norm2_w);
    const norm2_b = try allocator.alloc(f32, shape.out_ch);
    defer allocator.free(norm2_b);
    const conv2_w = try allocator.alloc(f32, @as(usize, shape.out_ch) * shape.out_ch * 9);
    defer allocator.free(conv2_w);
    const conv2_b = try allocator.alloc(f32, shape.out_ch);
    defer allocator.free(conv2_b);
    const skip_w = try allocator.alloc(f32, @as(usize, shape.out_ch) * shape.in_ch);
    defer allocator.free(skip_w);
    const skip_b = try allocator.alloc(f32, shape.out_ch);
    defer allocator.free(skip_b);
    for (norm1_w) |*v| v.* = rng.floatNorm(f32) * 0.3 + 1.0;
    for (norm1_b) |*v| v.* = rng.floatNorm(f32) * 0.2;
    for (conv1_w) |*v| v.* = rng.floatNorm(f32) * 0.12;
    for (conv1_b) |*v| v.* = rng.floatNorm(f32) * 0.1;
    for (norm2_w) |*v| v.* = rng.floatNorm(f32) * 0.3 + 1.0;
    for (norm2_b) |*v| v.* = rng.floatNorm(f32) * 0.2;
    for (conv2_w) |*v| v.* = rng.floatNorm(f32) * 0.12;
    for (conv2_b) |*v| v.* = rng.floatNorm(f32) * 0.1;
    for (skip_w) |*v| v.* = rng.floatNorm(f32) * 0.2;
    for (skip_b) |*v| v.* = rng.floatNorm(f32) * 0.1;

    // Device buffers: input, weights/biases (offset 0, dtype 3), and scratch.
    var in_buf = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var n1w = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm1_w));
    defer n1w.deinit();
    var n1b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm1_b));
    defer n1b.deinit();
    var c1w = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(conv1_w));
    defer c1w.deinit();
    var c1b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(conv1_b));
    defer c1b.deinit();
    var n2w = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm2_w));
    defer n2w.deinit();
    var n2b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(norm2_b));
    defer n2b.deinit();
    var c2w = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(conv2_w));
    defer c2w.deinit();
    var c2b = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(conv2_b));
    defer c2b.deinit();
    var skw = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(skip_w));
    defer skw.deinit();
    var skb = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(skip_b));
    defer skb.deinit();

    // --- Oracle: full resblock from the hand-written kernels. ---
    const ref = try refResBlock(allocator, device, oracle, shape, eps, .{
        .input = in_buf.handle,
        .norm1_w = n1w.handle,
        .norm1_b = n1b.handle,
        .conv1_w = c1w.handle,
        .conv1_b = c1b.handle,
        .norm2_w = n2w.handle,
        .norm2_b = n2b.handle,
        .conv2_w = c2w.handle,
        .conv2_b = c2b.handle,
        .skip_w = if (has_skip) skw.handle else null,
        .skip_b = if (has_skip) skb.handle else null,
    });
    defer allocator.free(ref);

    // --- Streamed: row-strip resblock. ---
    var out_buf = try mbuffer.Buffer.empty(device, out_n * 4);
    defer out_buf.deinit();
    try mvres_stream.run(device, ctx, .{
        .input = in_buf.handle,
        .output = out_buf.handle,
        .norm1_w = n1w.handle,
        .norm1_b = n1b.handle,
        .conv1_w = c1w.handle,
        .conv1_b = c1b.handle,
        .norm2_w = n2w.handle,
        .norm2_b = n2b.handle,
        .conv2_w = c2w.handle,
        .conv2_b = c2b.handle,
        .skip_w = if (has_skip) skw.handle else null,
        .skip_b = if (has_skip) skb.handle else null,
    }, .{
        .in_ch = shape.in_ch,
        .out_ch = shape.out_ch,
        .height = shape.height,
        .width = shape.width,
        .groups = shape.groups,
        .eps = eps,
    }, strip_rows);

    const got = try allocator.alloc(f32, out_n);
    defer allocator.free(got);
    c.zdraw_metal_read_buffer(out_buf.handle, std.mem.sliceAsBytes(got).ptr, out_n * 4);
    return maxBitDiff(ref, got);
}

// Device handles for one resblock's input + weights/biases.
const ResHandles = struct {
    input: *anyopaque,
    norm1_w: *anyopaque,
    norm1_b: *anyopaque,
    conv1_w: *anyopaque,
    conv1_b: *anyopaque,
    norm2_w: *anyopaque,
    norm2_b: *anyopaque,
    conv2_w: *anyopaque,
    conv2_b: *anyopaque,
    skip_w: ?*anyopaque,
    skip_b: ?*anyopaque,
};

// Full resblock via the hand kernels: norm1+SiLU -> conv1 -> norm2+SiLU ->
// conv2 -> + (skip conv or identity). Returns the result read back to the host.
fn refResBlock(
    allocator: std.mem.Allocator,
    device: *anyopaque,
    oracle: OracleCtx,
    shape: VaeResShape,
    eps: f32,
    h: ResHandles,
) ![]f32 {
    const hw: usize = @as(usize, shape.height) * shape.width;
    const in_n: usize = @as(usize, shape.in_ch) * hw;
    const out_n: usize = @as(usize, shape.out_ch) * hw;

    var n1 = try mbuffer.Buffer.empty(device, in_n * 4);
    defer n1.deinit();
    var work = try mbuffer.Buffer.empty(device, out_n * 4);
    defer work.deinit();
    var out = try mbuffer.Buffer.empty(device, out_n * 4);
    defer out.deinit();

    const np1 = VaeNormParams{
        .channels = shape.in_ch,
        .height = shape.height,
        .width = shape.width,
        .groups = shape.groups,
        .dtype = 3,
        .bias_dtype = 3,
        .eps = eps,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    const np2 = VaeNormParams{
        .channels = shape.out_ch,
        .height = shape.height,
        .width = shape.width,
        .groups = shape.groups,
        .dtype = 3,
        .bias_dtype = 3,
        .eps = eps,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    const cp1 = c.ConvParams{
        .in_ch = shape.in_ch,
        .out_ch = shape.out_ch,
        .height = shape.height,
        .width = shape.width,
        .ksize = 3,
        .pad = 1,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    const cp2 = c.ConvParams{
        .in_ch = shape.out_ch,
        .out_ch = shape.out_ch,
        .height = shape.height,
        .width = shape.width,
        .ksize = 3,
        .pad = 1,
        .dtype = 3,
        .bias_dtype = 3,
        .has_bias = 1,
        .weight_offset = 0,
        .bias_offset = 0,
    };

    // norm1+SiLU(input) -> n1
    if (zdraw_metal_run_vae_norm(oracle.queue, oracle.norm_pipe, h.input, h.norm1_w, h.norm1_b, n1.handle, &np1, oracle.norm_threads) != 0) {
        return error.DispatchFailed;
    }
    // conv1(n1) -> work
    if (c.zdraw_metal_run_conv(oracle.queue, oracle.conv_pipe, n1.handle, h.conv1_w, h.conv1_b, work.handle, &cp1, oracle.conv_threads) != 0) {
        return error.DispatchFailed;
    }
    // norm2+SiLU(work) -> work (in place, like vres)
    if (zdraw_metal_run_vae_norm(oracle.queue, oracle.norm_pipe, work.handle, h.norm2_w, h.norm2_b, work.handle, &np2, oracle.norm_threads) != 0) {
        return error.DispatchFailed;
    }
    // conv2(work) -> out
    if (c.zdraw_metal_run_conv(oracle.queue, oracle.conv_pipe, work.handle, h.conv2_w, h.conv2_b, out.handle, &cp2, oracle.conv_threads) != 0) {
        return error.DispatchFailed;
    }
    // residual: skip conv of input (in_ch != out_ch) or identity.
    var residual: *anyopaque = h.input;
    var skip_buf: ?mbuffer.Buffer = null;
    defer if (skip_buf) |*b| b.deinit();
    if (h.skip_w) |skip_w| {
        skip_buf = try mbuffer.Buffer.empty(device, out_n * 4);
        const sp = c.ConvParams{
            .in_ch = shape.in_ch,
            .out_ch = shape.out_ch,
            .height = shape.height,
            .width = shape.width,
            .ksize = 1,
            .pad = 0,
            .dtype = 3,
            .bias_dtype = 3,
            .has_bias = 1,
            .weight_offset = 0,
            .bias_offset = 0,
        };
        if (c.zdraw_metal_run_conv(oracle.queue, oracle.conv_pipe, h.input, skip_w, h.skip_b.?, skip_buf.?.handle, &sp, oracle.conv_threads) != 0) {
            return error.DispatchFailed;
        }
        residual = skip_buf.?.handle;
    }
    if (zdraw_metal_run_vae_add(oracle.queue, oracle.add_pipe, out.handle, residual, @intCast(out_n), oracle.add_threads) != 0) {
        return error.DispatchFailed;
    }

    const ref = try allocator.alloc(f32, out_n);
    c.zdraw_metal_read_buffer(out.handle, std.mem.sliceAsBytes(ref).ptr, out_n * 4);
    return ref;
}

fn convWindowParams(cp: c.ConvParams, row0: u32, row1: u32) ConvWindowParams {
    return .{
        .in_ch = cp.in_ch,
        .out_ch = cp.out_ch,
        .height = cp.height,
        .width = cp.width,
        .ksize = cp.ksize,
        .pad = cp.pad,
        .dtype = cp.dtype,
        .bias_dtype = cp.bias_dtype,
        .has_bias = cp.has_bias,
        .weight_offset = cp.weight_offset,
        .bias_offset = cp.bias_offset,
        .row0 = row0,
        .row1 = row1,
    };
}

fn windowParams(np: VaeNormParams, row0: u32, row1: u32, col0: u32, col1: u32) VaeNormWindowParams {
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
        .col0 = col0,
        .col1 = col1,
    };
}

fn maxBitDiff(a: []const f32, b: []const f32) f64 {
    var worst: f64 = 0;
    for (a, b) |av, bv| worst = @max(worst, @abs(@as(f64, av) - @as(f64, bv)));
    return worst;
}

fn runChainProbe(device: *anyopaque, queue: *anyopaque) void {
    const ops: c_int = 240;
    const dep = zdraw_metal_chain_probe(device, queue, 0, ops);
    const ind = zdraw_metal_chain_probe(device, queue, 4, ops);
    if (dep == 0 or ind == 0) {
        std.debug.print("\nchain probe: unavailable\n", .{});
        return;
    }
    const per_dep = @as(f64, @floatFromInt(dep)) / 240.0 / 1000.0;
    const per_ind = @as(f64, @floatFromInt(ind)) / 240.0 / 1000.0;
    std.debug.print(
        "\nchain probe (240 ops, 288x3840x3840 direct-f16): small-bufs {d:.0} us/op, big-offset {d:.0} us/op, ratio {d:.2}\n",
        .{ per_dep, per_ind, per_ind / per_dep },
    );
}

extern fn zdraw_metal_frag_map_probe(device: *anyopaque, queue: *anyopaque, out: [*]f32) c_int;

fn runFragMap(device: *anyopaque, queue: *anyopaque) void {
    var out: [64]f32 = undefined;
    if (zdraw_metal_frag_map_probe(device, queue, &out) != 0) {
        std.debug.print("frag map: unavailable\n", .{});
        return;
    }
    std.debug.print("frag map (thread: e0,e1 as row*8+col):\n", .{});
    for (0..32) |tid| {
        const a: u32 = @intFromFloat(out[tid * 2]);
        const b: u32 = @intFromFloat(out[tid * 2 + 1]);
        std.debug.print("  t{d:>2}: ({d},{d}) ({d},{d})\n", .{ tid, a / 8, a % 8, b / 8, b % 8 });
    }
}

// Klein attention headroom at the 1024 shape (24 heads x 4608 tokens x d128,
// 261 GFLOP per call): the resident MFA path as dispatched by the single
// block, in three cuts, wall and GPU-active per call. Interleave with
// tools/mlx_kernel_race.py (MLX SDPA, same shape) on the same box; the GPU
// throttles ~25% within a minute, so only back-to-back pairs count.
// Gated by ZDRAW_KLEIN_ATTNBENCH=1.
fn kleinAttnBench(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    runs: u32,
) !void {
    const tokens: usize = 4608;
    const heads: usize = 24;
    const dim: usize = 128;
    const hidden = heads * dim;
    const n = tokens * hidden;
    const data = try allocator.alloc(f32, n);
    defer allocator.free(data);
    var prng = std.Random.DefaultPrng.init(0xa77e);
    const rng = prng.random();
    var q = try fillNormal(device, data, rng, 0.5);
    defer q.deinit();
    var k = try fillNormal(device, data, rng, 0.5);
    defer k.deinit();
    var v = try fillNormal(device, data, rng, 0.5);
    defer v.deinit();
    var o = try mbuffer.Buffer.empty(device, n * 4);
    defer o.deinit();
    const params = metal_c.AttnParams{
        .tokens = @intCast(tokens),
        .heads = @intCast(heads),
        .kv_heads = @intCast(heads),
        .head_dim = @intCast(dim),
        .causal = 0,
    };
    const flops: f64 = 4.0 * f(tokens) * f(tokens) * f(dim) * f(heads);
    std.debug.print(
        "\nKlein attention headroom ({d} runs; 24x4608x128, {d:.0} GFLOP/call):\n",
        .{ runs, flops / 1e9 },
    );
    const Cut = struct { label: []const u8, mode: u8 };
    const cuts = [_]Cut{
        .{ .label = "full: 3 converts + MFA + un-permute", .mode = 0 },
        .{ .label = "hm q/k: v convert + MFA + un-permute", .mode = 1 },
        .{ .label = "hm q/k, O kept: v convert + MFA    ", .mode = 2 },
    };
    // One full run first so the head-major q/k/v scratch (slots 0/1/2) holds
    // converted data for the hm cuts and for the steel arm.
    try attnOnce(queue, &q, &k, &v, &o, &params, 0, device, tokens, heads, dim);
    // Reference O (token-major f32) from the MFA path for the steel arm's drift.
    const o_ref = try allocator.alloc(f32, n);
    defer allocator.free(o_ref);
    c.zdraw_metal_read_buffer(o.handle, std.mem.sliceAsBytes(o_ref).ptr, n * 4);
    var o_steel = try mbuffer.Buffer.empty(device, n * 2);
    defer o_steel.deinit();
    for (cuts) |cut| {
        var wall: [256]u64 = undefined;
        var gpu: [256]u64 = undefined;
        const count = @min(runs, 256);
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const g0 = metal_c.zdraw_metal_gpu_ns();
            const t0 = std.Io.Timestamp.now(io, .awake);
            try attnOnce(queue, &q, &k, &v, &o, &params, cut.mode, device, tokens, heads, dim);
            wall[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            gpu[i] = metal_c.zdraw_metal_gpu_ns() - g0;
        }
        std.mem.sort(u64, wall[0..count], {}, std.sort.asc(u64));
        std.mem.sort(u64, gpu[0..count], {}, std.sort.asc(u64));
        const w_ms = f(wall[count / 2]) / 1e6;
        const g_ms = f(gpu[count / 2]) / 1e6;
        std.debug.print(
            "  {s}: wall {d:>6.2} ms  gpu-active {d:>6.2} ms  {d:>5.2} TFLOP/s (gpu)\n",
            .{ cut.label, w_ms, g_ms, flops / g_ms / 1e9 },
        );
    }
    // MLX steel attention over the same head-major inputs (slots 0/1/2), O
    // head-major half; drift against the MFA token-major f32 output.
    {
        const bytes = tokens * heads * dim * 2;
        const qhm = try hmSlot(device, 0, bytes);
        const khm = try hmSlot(device, 1, bytes);
        const vhm = try hmSlot(device, 2, bytes);
        var wall: [256]u64 = undefined;
        var gpu: [256]u64 = undefined;
        const count = @min(runs, 256);
        var ok = true;
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const g0 = metal_c.zdraw_metal_gpu_ns();
            const t0 = std.Io.Timestamp.now(io, .awake);
            const bt = metal_c.zdraw_metal_batch_begin(queue) orelse
                return error.MetalDispatchFailed;
            const os = o_steel.handle;
            const rc = metal_c.zdraw_metal_run_attention_steel_hm_enc(bt, qhm, khm, vhm, os, &params);
            if (metal_c.zdraw_metal_batch_end(bt) != 0 or rc != 0) {
                ok = false;
                break;
            }
            wall[i] = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            gpu[i] = metal_c.zdraw_metal_gpu_ns() - g0;
        }
        if (!ok) {
            std.debug.print("  steel attention: unavailable (no steel_attn_h128 in the metallib)\n", .{});
            return;
        }
        std.mem.sort(u64, wall[0..count], {}, std.sort.asc(u64));
        std.mem.sort(u64, gpu[0..count], {}, std.sort.asc(u64));
        const w_ms = f(wall[count / 2]) / 1e6;
        const g_ms = f(gpu[count / 2]) / 1e6;
        const oh = try allocator.alloc(f16, n);
        defer allocator.free(oh);
        c.zdraw_metal_read_buffer(o_steel.handle, std.mem.sliceAsBytes(oh).ptr, n * 2);
        var worst: f64 = 0;
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (0..tokens) |t| {
            for (0..heads) |h| {
                for (0..dim) |d| {
                    const r: f64 = o_ref[t * hidden + h * dim + d];
                    const s: f64 = @as(f32, @floatCast(oh[(h * tokens + t) * dim + d]));
                    const diff = @abs(r - s);
                    if (!std.math.isFinite(s)) {
                        worst = std.math.inf(f64);
                    } else if (diff > worst) worst = diff;
                    dot += r * s;
                    na += r * r;
                    nb += s * s;
                }
            }
        }
        const cosine = dot / @sqrt(na * nb);
        std.debug.print(
            "  steel attention (MLX kernel, hm in/out): wall {d:>6.2} ms  gpu-active {d:>6.2} ms" ++
                "  {d:>5.2} TFLOP/s  max|d| vs MFA {d:.4}  cos {d:.6}\n",
            .{ w_ms, g_ms, flops / g_ms / 1e9, worst, cosine },
        );
    }
}

fn fillNormal(device: *anyopaque, scratch: []f32, rng: std.Random, scale: f32) !mbuffer.Buffer {
    for (scratch) |*x| x.* = rng.floatNorm(f32) * scale;
    return mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(scratch));
}

fn attnOnce(
    queue: *anyopaque,
    q: *mbuffer.Buffer,
    k: *mbuffer.Buffer,
    v: *mbuffer.Buffer,
    o: *mbuffer.Buffer,
    params: *const metal_c.AttnParams,
    mode: u8,
    device: *anyopaque,
    tokens: usize,
    heads: usize,
    dim: usize,
) !void {
    const bt = metal_c.zdraw_metal_batch_begin(queue) orelse return error.MetalDispatchFailed;
    var rc: c_int = 0;
    if (mode == 0) {
        const qh = q.handle;
        const kh = k.handle;
        rc = metal_c.zdraw_metal_run_attention_mfa_enc(bt, qh, kh, v.handle, o.handle, params, 0, 0);
    } else {
        const bytes = tokens * heads * dim * 2;
        const qhm = try hmSlot(device, 0, bytes);
        const khm = try hmSlot(device, 1, bytes);
        const keep: c_int = if (mode == 2) 1 else 0;
        const vh = v.handle;
        rc = metal_c.zdraw_metal_run_attention_mfa_hm_enc(bt, qhm, khm, vh, o.handle, params, 0, 0, keep);
    }
    if (metal_c.zdraw_metal_batch_end(bt) != 0 or rc != 0) return error.MetalDispatchFailed;
}

/// The bench's metallib: ZDRAW_STEEL_LIB, else the one `zig build` installs
/// beside the binary (run from the repo root), else the legacy /tmp path.
extern fn _NSGetExecutablePath(buf: [*]u8, size: *u32) c_int;
var steel_lib_buf: [4096]u8 = undefined;

fn steelLibPath() [*:0]const u8 {
    if (std.c.getenv("ZDRAW_STEEL_LIB")) |p| return p;
    // lib/steel.metallib beside the bench binary (zig-out/bin/../lib), as the
    // engine resolves it; the cwd under `zig build gemmbench` is not the root.
    var exe: [4096]u8 = undefined;
    var size: u32 = exe.len;
    if (_NSGetExecutablePath(&exe, &size) == 0) {
        const path = std.mem.sliceTo(&exe, 0);
        if (std.mem.lastIndexOfScalar(u8, path, '/')) |slash| {
            const dir = path[0..slash];
            const fallback: [*:0]const u8 = "/tmp/steel_gemm.metallib";
            const candidate = std.fmt.bufPrintZ(&steel_lib_buf, "{s}/../lib/steel.metallib", .{dir}) catch
                return fallback;
            if (std.c.access(candidate.ptr, std.c.F_OK) == 0) return candidate.ptr;
        }
    }
    const beside: [*:0]const u8 = "zig-out/lib/steel.metallib";
    if (std.c.access(beside, std.c.F_OK) == 0) return beside;
    return "/tmp/steel_gemm.metallib";
}

fn hmSlot(device: *anyopaque, slot: c_int, bytes: usize) !*anyopaque {
    return metal_c.zdraw_metal_attn_hm_scratch(device, slot, bytes) orelse error.MetalDispatchFailed;
}
