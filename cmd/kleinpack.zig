//! Dev-only FLUX.2 Klein W16/W6 sidecar builder (CLI over src/klein_packer.zig).

const std = @import("std");
const packer = @import("zdraw").klein_packer;
const klein_bitmap = @import("zdraw").klein_bitmap;
const qwen_pack = @import("zdraw").qwen_pack;

const zflux2 = @import("zdraw").zflux2;
pub fn main(init: std.process.Init) !void {
    var out_buf: [256]u8 = undefined;
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    var opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    if (opts.out.len == 0) {
        opts.out = try std.fmt.bufPrint(&out_buf, "runs/{s}", .{opts.cfg.pack_name});
    }
    if (opts.text_bits != 0 and opts.text_out.len == 0) {
        opts.text_out = "runs/" ++ qwen_pack.pack_name;
    }
    // --bits map:FILE: the JSON map lives for the whole pack run.
    var map: ?klein_bitmap.Map = null;
    defer if (map) |*m| m.deinit();
    if (map_path) |path| {
        const limit: std.Io.Limit = .limited(1 << 20);
        const text = std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, limit) catch |err| {
            std.debug.print("kleinpack: cannot read {s}: {s}\n", .{ path, @errorName(err) });
            std.process.exit(1);
        };
        defer init.gpa.free(text);
        map = klein_bitmap.Map.parse(init.gpa, text) catch |err| {
            std.debug.print("kleinpack: bad bit map {s}: {s}\n", .{ path, @errorName(err) });
            std.process.exit(1);
        };
        opts.bit_map = &map.?;
    }
    packer.run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("kleinpack: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

/// The path from `--bits map:FILE`, remembered for main (the Options hold a
/// pointer to the parsed map, which must outlive the pack run).
var map_path: ?[]const u8 = null;

// Variant tags, not just sizes: the output filename comes from cfg.pack_name,
// so packing a base checkpoint under a distilled tag would overwrite the
// distilled sidecar.
fn sizeConfig(value: []const u8) !zflux2.Config {
    if (std.mem.eql(u8, value, "4b")) return zflux2.Config.klein_4b;
    if (std.mem.eql(u8, value, "9b")) return zflux2.Config.klein_9b;
    if (std.mem.eql(u8, value, "base4b")) return zflux2.Config.klein_base_4b;
    if (std.mem.eql(u8, value, "base9b")) return zflux2.Config.klein_base_9b;
    if (std.mem.eql(u8, value, "9bkv")) return zflux2.Config.klein_9b_kv;
    return error.BadSize;
}

fn parse(iter: *std.process.Args.Iterator) !packer.Options {
    _ = iter.next();
    var out = packer.Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) {
            out.weights = try need(iter);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out.out = try need(iter);
        } else if (std.mem.eql(u8, arg, "--size")) {
            const value = try need(iter);
            out.cfg = try sizeConfig(value);
        } else if (std.mem.eql(u8, arg, "--lora")) {
            out.lora = try need(iter);
        } else if (std.mem.eql(u8, arg, "--lora-scale")) {
            const value = try need(iter);
            out.lora_scale = std.fmt.parseFloat(f32, value) catch return error.BadScale;
        } else if (std.mem.eql(u8, arg, "--globals")) {
            out.globals = true;
        } else if (std.mem.eql(u8, arg, "--text-bits")) {
            out.text_bits = std.fmt.parseInt(usize, try need(iter), 10) catch return error.BadBits;
        } else if (std.mem.eql(u8, arg, "--text-out")) {
            out.text_out = try need(iter);
        } else if (std.mem.eql(u8, arg, "--text-only")) {
            out.text_only = true;
        } else if (std.mem.eql(u8, arg, "--bits")) {
            const value = try need(iter);
            if (std.mem.eql(u8, value, "mixed")) {
                out.mixed = true;
            } else if (std.mem.startsWith(u8, value, "map:")) {
                map_path = value["map:".len..];
                if (map_path.?.len == 0) return error.BadBits;
            } else if (std.mem.eql(u8, value, "mixed3")) {
                out.mixed = true;
                out.mixed_w3 = true;
            } else {
                out.bits = std.fmt.parseInt(usize, value, 10) catch return error.BadBits;
                const ok = out.bits == 2 or out.bits == 4 or out.bits == 6 or out.bits == 16;
                if (!ok) return error.BadBits;
            }
        } else if (std.mem.eql(u8, arg, "--w6-scope")) {
            const value = try need(iter);
            out.scope = std.meta.stringToEnum(packer.W6Scope, value) orelse return error.BadScope;
        } else if (std.mem.eql(u8, arg, "--help")) {
            usage();
            std.process.exit(0);
        } else return error.BadArg;
    }
    return out;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn usage() void {
    std.debug.print(
        \\zdraw kleinpack
        \\  zig build kleinpack -- --weights path/to/FLUX.2-Klein
        \\      [--size 4b|9b|base4b|base9b|9bkv]
        \\      [--out runs/<per-variant default>] [--globals]
        \\      [--bits 2|4|6|16|mixed|mixed3|map:FILE]
        \\        (map:FILE is a JSON per-class or per-block width map, src/klein_bitmap.zig)
        \\      [--text-bits 4 [--text-out runs/zdraw-klein-text-w4.zpack] [--text-only]]
        \\      [--w6-scope singles_out|singles|singles_dff|all]
        \\      [--lora adapter.safetensors] [--lora-scale 1.0]
        \\
    , .{});
}
