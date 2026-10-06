//! Localisation instruments for the resident Klein transformer, all behind
//! ZDRAW_KLEIN_DUMP_BLOCKS=<dir> and limited to the first two forwards of a
//! process. Each point flushes the batch, reads the activation back and
//! appends one line (hash, size, rms for f32 data) to <dir>/blocks.txt, so a
//! wrong render is pinned to the first point whose output differs from a
//! good run's. The 6 Oct 2026 base-model diagnosis ran on these lines.
const std = @import("std");
const metal_c = @import("../metal/metal_c.zig");
const tensor = @import("../pack/tensor.zig");
const zflux2 = @import("zflux2.zig");
const res = @import("zflux2_resident.zig");

const Ctx = res.Ctx;

pub fn enabled(ctx: *const Ctx) bool {
    return std.c.getenv("ZDRAW_KLEIN_DUMP_BLOCKS") != null and ctx.dump_forwards < 2;
}

/// A whole pooled buffer: hash of every byte it holds.
pub fn point(ctx: *Ctx, label: []const u8, idx: usize, handle: *anyopaque) !void {
    if (!enabled(ctx)) return;
    try ctx.flush();
    const n = metal_c.zdraw_metal_buffer_length(handle);
    const bytes = try readBack(ctx, handle, n);
    defer ctx.allocator.free(bytes);
    try append(ctx, label, idx, n, std.hash.Wyhash.hash(0, bytes), null);
}

/// An f32 activation of `count` elements: hash plus rms.
pub fn activation(
    ctx: *Ctx,
    label: []const u8,
    idx: usize,
    handle: *anyopaque,
    count: usize,
) !void {
    if (!enabled(ctx)) return;
    try ctx.flush();
    const bytes = try readBack(ctx, handle, count * 4);
    defer ctx.allocator.free(bytes);
    const vals = std.mem.bytesAsSlice(f32, bytes);
    var acc: f64 = 0;
    for (vals) |x| acc += @as(f64, x) * @as(f64, x);
    const rms = @sqrt(acc / @as(f64, @floatFromInt(count)));
    try append(ctx, label, idx, count * 4, std.hash.Wyhash.hash(0, bytes), rms);
}

/// A weight as bound for the GPU, beside the CPU view it came from: the two
/// hashes agree for a healthy bind, and the view's hash names the bytes the
/// model really multiplies by (the base-model defect showed here as a view
/// hash that matched another checkpoint's pack).
pub fn weight(ctx: *Ctx, label: []const u8, handle: *anyopaque, view: tensor.View) !void {
    if (!enabled(ctx)) return;
    try ctx.flush();
    const n = metal_c.zdraw_metal_buffer_length(handle);
    const bytes = try readBack(ctx, handle, n);
    defer ctx.allocator.free(bytes);
    const off: usize = if (view.source) |src| src.offset else 0;
    const gpu_hash = std.hash.Wyhash.hash(0, bytes);
    const cpu_hash = std.hash.Wyhash.hash(0, view.bytes);
    var line: [256]u8 = undefined;
    const text = try std.fmt.bufPrint(
        &line,
        "fwd{d} {s} gpu_bytes={d} gpu_hash={x:0>16} cpu_dtype={s} cpu_bytes={d} " ++
            "cpu_hash={x:0>16} src_off={d}\n",
        .{
            ctx.dump_forwards,    label,          n,        gpu_hash,
            @tagName(view.dtype), view.bytes.len, cpu_hash, off,
        },
    );
    try write(text);
}

/// The whole mapped sidecar as the CPU sees it.
pub fn sidecar(ctx: *Ctx, bytes: []const u8) !void {
    if (!enabled(ctx)) return;
    var line: [128]u8 = undefined;
    const h = std.hash.Wyhash.hash(0, bytes);
    const fmt = "fwd{d} sidecar bytes={d} cpu_hash={x:0>16}\n";
    const text = try std.fmt.bufPrint(&line, fmt, .{ ctx.dump_forwards, bytes.len, h });
    try write(text);
}

/// The forward's inputs in one call: the uploaded latents, the mapped
/// sidecar and the two embedder weights as bound.
pub fn inputs(
    ctx: *Ctx,
    lat_h: *anyopaque,
    img_len: usize,
    wx: *anyopaque,
    loaded: *const zflux2.Loaded,
) !void {
    if (!enabled(ctx)) return;
    try activation(ctx, "latents", 0, lat_h, img_len * 128);
    try sidecar(ctx, ctx.wbind.sidecarBytes());
    try weight(ctx, "w_x_embed", wx, loaded.globals.x_embed);
    const wc = try ctx.weight(loaded.globals.context_embed);
    try weight(ctx, "w_ctx_embed", wc.handle, loaded.globals.context_embed);
}

/// Both embedder outputs.
pub fn embedders(ctx: *Ctx, img: *anyopaque, txt: *anyopaque, img_len: usize, hidden: usize) !void {
    if (!enabled(ctx)) return;
    try activation(ctx, "embed_img", 0, img, img_len * hidden);
    try activation(ctx, "embed_txt", 0, txt, zflux2.txt_len * hidden);
}

fn readBack(ctx: *Ctx, handle: *anyopaque, n: usize) ![]u8 {
    const bytes = try ctx.allocator.alloc(u8, n);
    errdefer ctx.allocator.free(bytes);
    const rc = metal_c.zdraw_metal_read_buffer_any(ctx.attn.queue, handle, bytes.ptr, n);
    if (rc != 0) return error.MetalDispatchFailed;
    return bytes;
}

fn append(ctx: *Ctx, label: []const u8, idx: usize, bytes: usize, hash: u64, rms: ?f64) !void {
    var line: [192]u8 = undefined;
    const f = ctx.dump_forwards;
    const head = "fwd{d} {s}{d} bytes={d} hash={x:0>16}";
    const text = if (rms) |r|
        try std.fmt.bufPrint(&line, head ++ " rms={d:.6}\n", .{ f, label, idx, bytes, hash, r })
    else
        try std.fmt.bufPrint(&line, head ++ "\n", .{ f, label, idx, bytes, hash });
    try write(text);
}

fn write(text: []const u8) !void {
    const dir = std.mem.span(std.c.getenv("ZDRAW_KLEIN_DUMP_BLOCKS") orelse return);
    var pbuf: [512]u8 = undefined;
    const path = try std.fmt.bufPrintZ(&pbuf, "{s}/blocks.txt", .{dir});
    const file = std.c.fopen(path.ptr, "a") orelse return;
    defer _ = std.c.fclose(file);
    _ = std.c.fwrite(text.ptr, 1, text.len, file);
}
