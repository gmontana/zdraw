//! Dev-only `.zpack` sidecar builder.

const std = @import("std");
const packer = @import("zdraw").zimage_packer;

const zpack_families = @import("zdraw").zpack_families;
const zpack_kinds = @import("zdraw").zpack_kinds;
pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    packer.run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("zpackbuild: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn parse(iter: *std.process.Args.Iterator) !packer.Options {
    _ = iter.next();
    var out = packer.Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) out.weights = try need(iter) else if (std.mem.eql(u8, arg, "--out")) {
            out.out = try need(iter);
        } else if (std.mem.eql(u8, arg, "--last")) {
            out.last = try usizeArg(iter);
        } else if (std.mem.eql(u8, arg, "--group")) {
            out.group = try usizeArg(iter);
        } else if (std.mem.eql(u8, arg, "--w6-down")) {
            out.w6_down = true;
        } else if (std.mem.eql(u8, arg, "--w16-refiners")) {
            out.w16_refiners = true;
        } else if (std.mem.eql(u8, arg, "--bits")) {
            out.bits = try usizeArg(iter);
            if (out.bits != 6 and out.bits != 8 and out.bits != 16) return error.UnknownOption;
        } else if (std.mem.eql(u8, arg, "--kinds")) {
            out.kinds = try zpack_kinds.parse(try need(iter));
        } else if (std.mem.eql(u8, arg, "--families")) {
            out.families = try zpack_families.parse(try need(iter));
        } else if (std.mem.eql(u8, arg, "--lora")) {
            out.lora = try need(iter);
        } else if (std.mem.eql(u8, arg, "--lora-scale")) {
            const raw = try need(iter);
            out.lora_scale = std.fmt.parseFloat(f32, raw) catch return error.UnknownOption;
        } else return error.UnknownOption;
    }
    return out;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse return error.MissingValue;
}

fn usizeArg(iter: *std.process.Args.Iterator) !usize {
    return std.fmt.parseInt(usize, try need(iter), 10);
}

fn usage() void {
    std.debug.print(
        \\zdraw zpackbuild
        \\  zig build zpackbuild -- --weights path/to/Z-Image-Turbo [--last 8]
        \\      [--bits 6|8|16] [--kinds down] [--families main]
        \\      [--w6-down] [--w16-refiners] [--out runs/file.zpack]
        \\      [--lora adapter.safetensors --lora-scale 0.8]
        \\
    , .{});
}
