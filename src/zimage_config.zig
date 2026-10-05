//! Typed Z-Image-Turbo config values used by the pipeline.

const std = @import("std");

pub const Config = struct {
    text: Text,
    transformer: Transformer,
    vae: Vae,
    scheduler: Scheduler,
};

pub const Text = struct {
    hidden_size: u32,
    intermediate_size: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    rms_norm_eps: f64,
    rope_theta: f64,
};

pub const Transformer = struct {
    dim: u32,
    layers: u32,
    refiner_layers: u32,
    heads: u32,
    kv_heads: u32,
    cap_feat_dim: u32,
    in_channels: u32,
    axes_dims: [3]u32,
    axes_lens: [3]u32,
    norm_eps: f64,
    rope_theta: f64,
    t_scale: f64,
    qk_norm: bool,
};

pub const Vae = struct {
    latent_channels: u32,
    scaling_factor: f64,
    shift_factor: f64,
};

pub const Scheduler = struct {
    train_steps: u32,
    shift: f64,
};

pub const Error = error{
    ConfigTooLarge,
    InvalidConfig,
};

pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !Config {
    return .{
        .text = try loadText(io, allocator, root),
        .transformer = try loadTransformer(io, allocator, root),
        .vae = try loadVae(io, allocator, root),
        .scheduler = try loadScheduler(io, allocator, root),
    };
}

fn loadText(io: std.Io, allocator: std.mem.Allocator, root: []const u8) !Text {
    var json = try readJson(io, allocator, root, "text_encoder/config.json");
    defer json.deinit();
    const object = try rootObject(json.value);
    return .{
        .hidden_size = try getU32(object, "hidden_size"),
        .intermediate_size = try getU32(object, "intermediate_size"),
        .layers = try getU32(object, "num_hidden_layers"),
        .heads = try getU32(object, "num_attention_heads"),
        .kv_heads = try getU32(object, "num_key_value_heads"),
        .head_dim = try getU32(object, "head_dim"),
        .vocab_size = try getU32(object, "vocab_size"),
        .rms_norm_eps = try getF64(object, "rms_norm_eps"),
        .rope_theta = try getF64(object, "rope_theta"),
    };
}

fn loadTransformer(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !Transformer {
    var json = try readJson(io, allocator, root, "transformer/config.json");
    defer json.deinit();
    const object = try rootObject(json.value);
    return .{
        .dim = try getU32(object, "dim"),
        .layers = try getU32(object, "n_layers"),
        .refiner_layers = try getU32(object, "n_refiner_layers"),
        .heads = try getU32(object, "n_heads"),
        .kv_heads = try getU32(object, "n_kv_heads"),
        .cap_feat_dim = try getU32(object, "cap_feat_dim"),
        .in_channels = try getU32(object, "in_channels"),
        .axes_dims = try getArray3(object, "axes_dims"),
        .axes_lens = try getArray3(object, "axes_lens"),
        .norm_eps = try getF64(object, "norm_eps"),
        .rope_theta = try getF64(object, "rope_theta"),
        .t_scale = try getF64(object, "t_scale"),
        .qk_norm = try getBool(object, "qk_norm"),
    };
}

fn loadVae(io: std.Io, allocator: std.mem.Allocator, root: []const u8) !Vae {
    var json = try readJson(io, allocator, root, "vae/config.json");
    defer json.deinit();
    const object = try rootObject(json.value);
    return .{
        .latent_channels = try getU32(object, "latent_channels"),
        .scaling_factor = try getF64(object, "scaling_factor"),
        .shift_factor = try getF64(object, "shift_factor"),
    };
}

fn loadScheduler(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !Scheduler {
    var json = try readJson(io, allocator, root, "scheduler/scheduler_config.json");
    defer json.deinit();
    const object = try rootObject(json.value);
    return .{
        .train_steps = try getU32(object, "num_train_timesteps"),
        .shift = try getF64(object, "shift"),
    };
}

fn readJson(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    name: []const u8,
) !std.json.Parsed(std.json.Value) {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, name });
    defer allocator.free(path);
    const bytes = try readFile(io, allocator, path);
    defer allocator.free(bytes);
    return std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
    });
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > 1024 * 1024) return error.ConfigTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

fn rootObject(value: std.json.Value) !std.json.ObjectMap {
    if (value != .object) return error.InvalidConfig;
    return value.object;
}

fn getU32(object: std.json.ObjectMap, key: []const u8) !u32 {
    const value = object.get(key) orelse return error.InvalidConfig;
    if (value != .integer or value.integer < 0) return error.InvalidConfig;
    return @intCast(value.integer);
}

fn getF64(object: std.json.ObjectMap, key: []const u8) !f64 {
    const value = object.get(key) orelse return error.InvalidConfig;
    return switch (value) {
        .float => value.float,
        .integer => @floatFromInt(value.integer),
        else => error.InvalidConfig,
    };
}

fn getBool(object: std.json.ObjectMap, key: []const u8) !bool {
    const value = object.get(key) orelse return error.InvalidConfig;
    if (value != .bool) return error.InvalidConfig;
    return value.bool;
}

fn getArray3(object: std.json.ObjectMap, key: []const u8) ![3]u32 {
    const value = object.get(key) orelse return error.InvalidConfig;
    if (value != .array or value.array.items.len != 3) return error.InvalidConfig;
    return .{
        try readArrayU32(value.array.items[0]),
        try readArrayU32(value.array.items[1]),
        try readArrayU32(value.array.items[2]),
    };
}

fn readArrayU32(value: std.json.Value) !u32 {
    if (value != .integer or value.integer < 0) return error.InvalidConfig;
    return @intCast(value.integer);
}
