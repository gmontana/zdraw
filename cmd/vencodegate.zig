//! Gate for the CPU VAE encoder (src/vencode.zig) against the diffusers
//! oracle (tools/quality/vae_encode_oracle.py): reads the oracle's input
//! tensor (f32 [3, H, W] in [-1, 1]), runs `vencode.mean`, writes the
//! [32, H/8, W/8] mean as raw f32 for the oracle's --compare.
//!   zig build vencodegate -- <weights dir> <input.bin> <H> <W> <out.bin>
const std = @import("std");
const mattn = @import("zdraw").mattn;
const mconv = @import("zdraw").mconv;
const mlinear = @import("zdraw").mlinear;
const tensor_file = @import("zdraw").tensor_file;
const vencode = @import("zdraw").vencode;
const vviews = @import("zdraw").vviews;

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    _ = iter.next();
    const weights = iter.next() orelse return usage();
    const input = iter.next() orelse return usage();
    const h = try std.fmt.parseInt(usize, iter.next() orelse return usage(), 10);
    const w = try std.fmt.parseInt(usize, iter.next() orelse return usage(), 10);
    const out = iter.next() orelse return usage();
    run(init.io, init.gpa, weights, input, h, w, out) catch |err| {
        std.debug.print("vencodegate: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn usage() void {
    std.debug.print("usage: vencodegate <weights dir> <input.bin> <H> <W> <out.bin>\n", .{});
}

fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    weights: []const u8,
    input: []const u8,
    h: usize,
    w: usize,
    out: []const u8,
) !void {
    const rgb = try readF32(io, allocator, input, 3 * h * w);
    defer allocator.free(rgb);
    const vae_path = try std.fmt.allocPrint(allocator, "{s}/vae/diffusion_pytorch_model.safetensors", .{weights});
    defer allocator.free(vae_path);
    var vae = try tensor_file.open(io, allocator, vae_path);
    defer vae.deinit(io, allocator);
    const views = try vviews.loadEncoder(&vae);
    // ZDRAW_VENCODE_GPU=0 keeps the CPU reference tier; default: the
    // decoder's Metal contexts (conv, attention, linear).
    const gpu = std.c.getenv("ZDRAW_VENCODE_GPU") == null or std.c.getenv("ZDRAW_VENCODE_GPU").?[0] != '0';
    var lin: ?mlinear.Context = if (gpu) mlinear.Context.init() catch null else null;
    defer if (lin) |*m| m.deinit();
    var attn: ?mattn.Context = if (gpu) mattn.Context.init() catch null else null;
    defer if (attn) |*a| a.deinit();
    var cv: ?mconv.Context = if (gpu) mconv.Context.init() catch null else null;
    defer if (cv) |*c| c.deinit();
    const lp: ?*mlinear.Context = if (lin) |*m| m else null;
    const ap: ?*mattn.Context = if (attn) |*a| a else null;
    const cp: ?*mconv.Context = if (cv) |*c| c else null;
    std.debug.print("vencodegate: metal conv {any} attn {any} linear {any}\n", .{ cp != null, ap != null, lp != null });
    const started = std.Io.Timestamp.now(io, .awake);
    const mean = try vencode.mean(allocator, lp, cp, ap, rgb, views, .{ .height = h, .width = w });
    defer allocator.free(mean);
    const ms = @divTrunc(started.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds(), std.time.ns_per_ms);
    const file = try std.Io.Dir.cwd().createFile(io, out, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(mean));
    std.debug.print("vencodegate: wrote {s} ({d} floats) in {d} ms\n", .{ out, mean.len, ms });
}

fn readF32(io: std.Io, allocator: std.mem.Allocator, path: []const u8, count: usize) ![]f32 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);
    var rbuf: [1 << 16]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    try reader.interface.readSliceAll(std.mem.sliceAsBytes(values));
    return values;
}
