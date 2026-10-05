//! Image stage of the safety filter: a ViT-base/16 image classifier
//! (Falconsai/nsfw_image_detection, Apache-2.0; two classes, "normal" and
//! "nsfw") over the decoded pixels, read straight from its Hugging Face
//! safetensors. The GEMMs go through the engine's linear path (Metal when a
//! context is given, CPU otherwise); LayerNorm, GELU, softmax and the
//! head_dim-64 attention run on the CPU (seq 197: well under 1 GFLOP).
const std = @import("std");
const mlinear = @import("mlinear.zig");
const tensor = @import("tensor.zig");
const tensor_file = @import("tensor_file.zig");
const vproj = @import("vproj.zig");

pub const Config = struct {
    hidden: usize = 768,
    layers: usize = 12,
    heads: usize = 12,
    intermediate: usize = 3072,
    image: usize = 224,
    patch: usize = 16,
    eps: f32 = 1e-12,
    nsfw_index: usize = 1,
    mean: [3]f32 = .{ 0.5, 0.5, 0.5 },
    std: [3]f32 = .{ 0.5, 0.5, 0.5 },
};

/// The classifier's weights (mapped) and configuration.
pub const Model = struct {
    mapped: tensor_file.Mapped,
    cfg: Config,

    pub fn deinit(self: *Model, io: std.Io, allocator: std.mem.Allocator) void {
        self.mapped.deinit(io, allocator);
        self.* = undefined;
    }
};

const HfConfig = struct {
    hidden_size: ?usize = null,
    num_hidden_layers: ?usize = null,
    num_attention_heads: ?usize = null,
    intermediate_size: ?usize = null,
    image_size: ?usize = null,
    patch_size: ?usize = null,
    layer_norm_eps: ?f32 = null,
    id2label: ?std.json.ArrayHashMap([]const u8) = null,
};

/// Open `<dir>/model.safetensors`, with `<dir>/config.json` overriding the
/// ViT-base defaults when present (hidden size, layers, heads, the index of
/// the "nsfw" label).
pub fn open(io: std.Io, allocator: std.mem.Allocator, dir: []const u8) !Model {
    var cfg = Config{};
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
    defer allocator.free(cfg_path);
    if (readSmall(io, allocator, cfg_path, 1 << 20)) |bytes| {
        defer allocator.free(bytes);
        applyConfig(allocator, bytes, &cfg) catch return error.InvalidConfig;
    } else |_| {}
    const model_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir});
    defer allocator.free(model_path);
    const mapped = try tensor_file.open(io, allocator, model_path);
    return .{ .mapped = mapped, .cfg = cfg };
}

fn applyConfig(allocator: std.mem.Allocator, bytes: []const u8, cfg: *Config) !void {
    var parsed = try std.json.parseFromSlice(HfConfig, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const h = parsed.value;
    if (h.hidden_size) |v| cfg.hidden = v;
    if (h.num_hidden_layers) |v| cfg.layers = v;
    if (h.num_attention_heads) |v| cfg.heads = v;
    if (h.intermediate_size) |v| cfg.intermediate = v;
    if (h.image_size) |v| cfg.image = v;
    if (h.patch_size) |v| cfg.patch = v;
    if (h.layer_norm_eps) |v| cfg.eps = v;
    if (h.id2label) |labels| {
        var it = labels.map.iterator();
        while (it.next()) |entry| {
            if (std.ascii.eqlIgnoreCase(entry.value_ptr.*, "nsfw")) {
                cfg.nsfw_index = std.fmt.parseInt(usize, entry.key_ptr.*, 10) catch cfg.nsfw_index;
            }
        }
    }
}

fn readSmall(io: std.Io, allocator: std.mem.Allocator, path: []const u8, max: usize) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > max) return error.FileTooLarge;
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(buf);
    var reader_buf: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &reader_buf);
    try reader.interface.readSliceAll(buf);
    return buf;
}

/// p(nsfw) for an RGB8 image of any size (resized to the model's input).
pub fn score(
    allocator: std.mem.Allocator,
    model: *const Model,
    metal: ?*mlinear.Context,
    pixels: []const u8,
    width: usize,
    height: usize,
) !f32 {
    const cfg = model.cfg;
    const side = cfg.image / cfg.patch;
    const patches = side * side;
    const tokens = patches + 1;
    const hidden = cfg.hidden;
    const patch_dim = 3 * cfg.patch * cfg.patch;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    // Patches in the conv weight's flattening order: channel, y, x.
    const patch_in = try a.alloc(f32, patches * patch_dim);
    patchify(cfg, pixels, width, height, patch_in);
    const x = try a.alloc(f32, tokens * hidden);
    const embed_raw = try need(model, "vit.embeddings.patch_embeddings.projection.weight");
    const embed_w = try view2d(a, embed_raw, hidden, patch_dim);
    const embed_b = try need(model, "vit.embeddings.patch_embeddings.projection.bias");
    try vproj.run(metal, x[hidden..], patch_in, embed_w, embed_b, patches);
    try copyView(try need(model, "vit.embeddings.cls_token"), x[0..hidden]);
    const pos = try need(model, "vit.embeddings.position_embeddings");
    for (0..tokens * hidden) |i| x[i] += pos.atF32Unchecked(i);

    var sc = Scratch{
        .h = try a.alloc(f32, tokens * hidden),
        .q = try a.alloc(f32, tokens * hidden),
        .k = try a.alloc(f32, tokens * hidden),
        .v = try a.alloc(f32, tokens * hidden),
        .ctx = try a.alloc(f32, tokens * hidden),
        .mid = try a.alloc(f32, tokens * cfg.intermediate),
        .scores = try a.alloc(f32, tokens),
    };
    for (0..cfg.layers) |layer| try encoderLayer(model, metal, layer, x, &sc, tokens, cfg);
    // Final LayerNorm on the CLS token, the two-way classifier, softmax.
    const cls = x[0..hidden];
    const normed = try a.alloc(f32, hidden);
    const ln_w = try need(model, "vit.layernorm.weight");
    const ln_b = try need(model, "vit.layernorm.bias");
    try normRow(cls, normed, ln_w, ln_b, cfg.eps);
    const head_w = try need(model, "classifier.weight");
    const head_b = try need(model, "classifier.bias");
    const classes = head_w.shape[0];
    const logits = try a.alloc(f32, classes);
    try vproj.run(null, logits, normed, head_w, head_b, 1);
    softmax(logits);
    return logits[@min(cfg.nsfw_index, classes - 1)];
}

const Scratch = struct {
    h: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    ctx: []f32,
    mid: []f32,
    scores: []f32,
};

/// One pre-LN transformer block, in place on x [tokens, hidden].
fn encoderLayer(
    model: *const Model,
    metal: ?*mlinear.Context,
    layer: usize,
    x: []f32,
    s: *Scratch,
    tokens: usize,
    cfg: Config,
) !void {
    const hidden = cfg.hidden;
    var name_buf: [128]u8 = undefined;
    try layerNorm(model, layer, "layernorm_before", x, s.h, tokens, hidden, cfg.eps);
    try dense(model, metal, &name_buf, layer, "attention.attention.query", s.h, s.q, tokens);
    try dense(model, metal, &name_buf, layer, "attention.attention.key", s.h, s.k, tokens);
    try dense(model, metal, &name_buf, layer, "attention.attention.value", s.h, s.v, tokens);
    attention(s.q, s.k, s.v, s.ctx, s.scores, tokens, hidden, cfg.heads);
    try dense(model, metal, &name_buf, layer, "attention.output.dense", s.ctx, s.h, tokens);
    for (x, s.h) |*xi, hi| xi.* += hi;
    try layerNorm(model, layer, "layernorm_after", x, s.h, tokens, hidden, cfg.eps);
    try dense(model, metal, &name_buf, layer, "intermediate.dense", s.h, s.mid, tokens);
    for (s.mid) |*m| m.* = gelu(m.*);
    try dense(model, metal, &name_buf, layer, "output.dense", s.mid, s.h, tokens);
    for (x, s.h) |*xi, hi| xi.* += hi;
}

fn need(model: *const Model, name: []const u8) !tensor.View {
    return (try model.mapped.view(name)) orelse error.MissingTensor;
}

/// The same bytes seen as a [rows, cols] matrix (the patch-embed conv weight);
/// the shape lives in the arena (a view keeps a slice, never a temporary).
fn view2d(a: std.mem.Allocator, v: tensor.View, rows: usize, cols: usize) !tensor.View {
    if (try v.elems() != rows * cols) return error.InvalidShape;
    const shape = try a.alloc(usize, 2);
    shape[0] = rows;
    shape[1] = cols;
    var out = v;
    out.shape = shape;
    return out;
}

fn copyView(v: tensor.View, out: []f32) !void {
    if (try v.elems() != out.len) return error.InvalidShape;
    for (out, 0..) |*o, i| o.* = v.atF32Unchecked(i);
}

fn dense(
    model: *const Model,
    metal: ?*mlinear.Context,
    name_buf: []u8,
    layer: usize,
    comptime leaf: []const u8,
    input: []const f32,
    out: []f32,
    tokens: usize,
) !void {
    const w_fmt = "vit.encoder.layer.{d}." ++ leaf ++ ".weight";
    const w_name = try std.fmt.bufPrint(name_buf, w_fmt, .{layer});
    const w = try need(model, w_name);
    const b_fmt = "vit.encoder.layer.{d}." ++ leaf ++ ".bias";
    const b_name = try std.fmt.bufPrint(name_buf, b_fmt, .{layer});
    const b = try need(model, b_name);
    try vproj.run(metal, out, input, w, b, tokens);
}

fn layerNorm(
    model: *const Model,
    layer: usize,
    comptime leaf: []const u8,
    x: []const f32,
    out: []f32,
    tokens: usize,
    hidden: usize,
    eps: f32,
) !void {
    var name_buf: [128]u8 = undefined;
    const w_fmt = "vit.encoder.layer.{d}." ++ leaf ++ ".weight";
    const w = try need(model, try std.fmt.bufPrint(&name_buf, w_fmt, .{layer}));
    var b_buf: [128]u8 = undefined;
    const b_fmt = "vit.encoder.layer.{d}." ++ leaf ++ ".bias";
    const b = try need(model, try std.fmt.bufPrint(&b_buf, b_fmt, .{layer}));
    for (0..tokens) |t| {
        try normRow(x[t * hidden ..][0..hidden], out[t * hidden ..][0..hidden], w, b, eps);
    }
}

fn normRow(row: []const f32, out: []f32, w: tensor.View, b: tensor.View, eps: f32) !void {
    var mean: f64 = 0;
    for (row) |v| mean += v;
    mean /= @floatFromInt(row.len);
    var variance: f64 = 0;
    for (row) |v| variance += (v - mean) * (v - mean);
    variance /= @floatFromInt(row.len);
    const inv = 1.0 / @sqrt(variance + eps);
    for (out, row, 0..) |*o, v, i| {
        o.* = @floatCast((v - mean) * inv * w.atF32Unchecked(i) + b.atF32Unchecked(i));
    }
}

/// Multi-head self-attention over token-major [tokens, hidden] q/k/v.
fn attention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,
    scores: []f32,
    tokens: usize,
    hidden: usize,
    heads: usize,
) void {
    const dim = hidden / heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    @memset(out, 0);
    for (0..heads) |hd| {
        const off = hd * dim;
        for (0..tokens) |i| {
            const qi = q[i * hidden + off ..][0..dim];
            for (0..tokens) |j| {
                const kj = k[j * hidden + off ..][0..dim];
                var s: f32 = 0;
                for (qi, kj) |a, b| s += a * b;
                scores[j] = s * scale;
            }
            softmax(scores[0..tokens]);
            const oi = out[i * hidden + off ..][0..dim];
            for (0..tokens) |j| {
                const vj = v[j * hidden + off ..][0..dim];
                const p = scores[j];
                for (oi, vj) |*o, vv| o.* += p * vv;
            }
        }
    }
}

fn softmax(x: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (x) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// Exact (erf) GELU, erf by Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7).
fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x / std.math.sqrt2));
}

fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0) -1 else 1;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const inner = 1.421413741 + t * (-1.453152027 + t * 1.061405429);
    const poly = t * (0.254829592 + t * (-0.284496736 + t * inner));
    return sign * (1.0 - poly * @exp(-ax * ax));
}

/// Bilinear resize to the model input, normalise, and flatten each patch in
/// channel-major (c, y, x) order into `out` [patches, 3 * patch * patch].
fn patchify(cfg: Config, pixels: []const u8, width: usize, height: usize, out: []f32) void {
    const n = cfg.image;
    const side = n / cfg.patch;
    for (0..n) |y| {
        const sy = (@as(f32, @floatFromInt(y)) + 0.5) * @as(f32, @floatFromInt(height)) /
            @as(f32, @floatFromInt(n)) - 0.5;
        for (0..n) |x| {
            const sx = (@as(f32, @floatFromInt(x)) + 0.5) * @as(f32, @floatFromInt(width)) /
                @as(f32, @floatFromInt(n)) - 0.5;
            const py = y / cfg.patch;
            const px = x / cfg.patch;
            const patch = py * side + px;
            const iy = y % cfg.patch;
            const ix = x % cfg.patch;
            for (0..3) |c| {
                const value = sample(pixels, width, height, c, sx, sy) / 255.0;
                const norm = (value - cfg.mean[c]) / cfg.std[c];
                const slot = c * cfg.patch * cfg.patch + iy * cfg.patch + ix;
                out[patch * 3 * cfg.patch * cfg.patch + slot] = norm;
            }
        }
    }
}

fn sample(pixels: []const u8, width: usize, height: usize, c: usize, sx: f32, sy: f32) f32 {
    const fx = @max(sx, 0);
    const fy = @max(sy, 0);
    const x0: usize = @min(@as(usize, @intFromFloat(fx)), width - 1);
    const y0: usize = @min(@as(usize, @intFromFloat(fy)), height - 1);
    const x1 = @min(x0 + 1, width - 1);
    const y1 = @min(y0 + 1, height - 1);
    const tx = fx - @as(f32, @floatFromInt(x0));
    const ty = fy - @as(f32, @floatFromInt(y0));
    const p = struct {
        fn at(px: []const u8, w: usize, x: usize, y: usize, ch: usize) f32 {
            return @floatFromInt(px[(y * w + x) * 3 + ch]);
        }
    };
    const top = p.at(pixels, width, x0, y0, c) * (1 - tx) +
        p.at(pixels, width, x1, y0, c) * tx;
    const bottom = p.at(pixels, width, x0, y1, c) * (1 - tx) +
        p.at(pixels, width, x1, y1, c) * tx;
    return top * (1 - ty) + bottom * ty;
}

test "softmax sums to one and gelu is exact at zero" {
    var x = [_]f32{ 1, 2, 3 };
    softmax(&x);
    try std.testing.expectApproxEqAbs(@as(f32, 1), x[0] + x[1] + x[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), gelu(0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), gelu(1), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5205), erf(0.5), 1e-4);
}

test "patchify lays a 32x32 image out as four 16x16 patches, channel-major" {
    const cfg = Config{ .image = 32, .patch = 16 };
    var pixels: [32 * 32 * 3]u8 = undefined;
    for (&pixels, 0..) |*p, i| p.* = @intCast((i / 3) % 256);
    var out: [4 * 768]f32 = undefined;
    patchify(cfg, &pixels, 32, 32, &out);
    // Patch 0, channel 0, pixel (0,0): value 0 -> (0/255 - 0.5) / 0.5 = -1.
    try std.testing.expectApproxEqAbs(@as(f32, -1), out[0], 1e-6);
    // Patch 1 starts at x = 16: pixel index 16 -> 16/255.
    try std.testing.expectApproxEqAbs(@as(f32, (16.0 / 255.0 - 0.5) / 0.5), out[768], 1e-4);
}
