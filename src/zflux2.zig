//! FLUX.2 Klein model definition: config + weight views.
//!
//! Maps transformer tensors in the HF snapshot into zdraw views. The model is bias-free; all tensors are
//! BF16 in the shipped checkpoints.

const std = @import("std");

const tensor = @import("tensor.zig");
const zconfig = @import("zimage_config.zig");
const tensor_file = @import("tensor_file.zig");

/// Transformer dims for the two released Klein sizes. The architecture is
/// identical; only dims change (verified against both HF configs).
pub const Config = struct {
    hidden: u32, // num_attention_heads * 128
    heads: u32,
    head_dim: u32 = 128,
    double_layers: u32,
    single_layers: u32,
    joint_dim: u32, // 3 x text-encoder hidden (layer-concat conditioning)
    ffn_inner: u32, // mlp_ratio 3.0 -> 3 * hidden
    in_channels: u32 = 128, // 32 latent ch x 2x2 VAE patching
    rope_theta: f32 = 2000.0,
    // axes_dims_rope = [32,32,32,32] on both sizes
    text_inner: u32, // Qwen3 encoder intermediate_size (explicit per size)
    pack_name: []const u8, // W16 sidecar filename (4B keeps the legacy name)

    pub const klein_4b: Config = .{
        .hidden = 3072,
        .heads = 24,
        .double_layers = 5,
        .single_layers = 20,
        .joint_dim = 7680,
        .ffn_inner = 9216,
        .text_inner = 9728, // Qwen3-4B
        .pack_name = "zdraw-klein-w16.zpack",
    };

    pub const klein_9b: Config = .{
        .hidden = 4096,
        .heads = 32,
        .double_layers = 8,
        .single_layers = 24,
        .joint_dim = 12288,
        .ffn_inner = 12288,
        .text_inner = 12288, // Qwen3-8B
        .pack_name = "zdraw-klein9b-w16.zpack",
    };

    // Base and kv variants share their distilled sibling's dims (derived
    // here so the two can never drift) but MUST NOT share its sidecar
    // filename: the packer writes runs/<pack_name> and the runtime falls
    // back to runs/<pack_name>, and swapLoaded validates only shape and
    // dtype, which are identical across variants. A shared name would let a
    // base run silently load distilled weights.
    fn withPack(base: Config, name: []const u8) Config {
        var out = base;
        out.pack_name = name;
        return out;
    }

    pub const klein_base_4b = withPack(klein_4b, "zdraw-klein-base4b-w16.zpack");
    pub const klein_base_9b = withPack(klein_9b, "zdraw-klein-base9b-w16.zpack");
    pub const klein_9b_kv = withPack(klein_9b, "zdraw-klein9bkv-w16.zpack");
};

/// Family-wide pipeline constants (identical for every released Klein size;
/// asserted where kernels specialize on them).
pub const head_dim: usize = 128;
pub const txt_len: usize = 512; // padded prompt length the DiT consumes
pub const ln_eps: f32 = 1e-6;
pub const rms_eps: f32 = 1e-6;

// ── Shared block layout ──────────────────────────────────────────────────
// The CPU/oracle backend (zflux2_dit) and the resident backend
// (zflux2_resident) are two hand-written implementations of the SAME block
// math. Every layout fact they must agree on lives here so a divergence is a
// compile error or an obvious one-line diff, never a silent oracle failure.

/// adaLN modulation layout: linear(SiLU(temb)) emits per-set vectors in this
/// component order, each `hidden` wide. Both execution backends index by name.
pub const ModOff = struct {
    pub const shift: usize = 0;
    pub const scale: usize = 1;
    pub const gate: usize = 2;
};

/// Modulation set indices. A double block packs two sets (msa then mlp =
/// 6*hidden); the single block and final layer carry one.
pub const mod_msa: usize = 0;
pub const mod_mlp: usize = 1;

/// Element offset of modulation component `comp` (ModOff.*) in set `set`
/// within a packed mod vector. e.g. the mlp gate sits at (1*3+2)*hidden.
pub fn modOffset(set: usize, comp: usize, hidden: usize) usize {
    return (set * 3 + comp) * hidden;
}

/// Joint attention concatenates the streams TEXT-FIRST: tokens [0, txt_len)
/// are text, [txt_len, txt_len+img_len) are image. The double-block qkv
/// concat, the pre-single merge, and the final image-slice all assume this.
pub const cat_txt_first = true;

/// Single-stream fused input GEMM (`Single.qkv_mlp`) row layout: q, k, v each
/// `hidden` rows, then the swiglu gate+up block (`2*ffn_inner` rows). These
/// are the row INDICES (multiply by hidden); mlp begins after `qkv` rows.
pub const SingleRows = struct {
    pub const q: usize = 0;
    pub const k: usize = 1;
    pub const v: usize = 2;
    pub const qkv: usize = 3; // rows used by q+k+v; the mlp block starts here
};

/// Final adaLN (`Globals.norm_out`) emits its pair SCALE-FIRST (FLUX order):
/// element [0, hidden) is scale, [hidden, 2*hidden) is shift — the OPPOSITE of
/// the per-block [shift, scale, gate] order above. Both backends rely on this.
pub const final_scale_first = true;

/// Qwen3 text-encoder dims per Klein size (the conditioning encoder).
pub fn textConfig(cfg: Config) zconfig.Text {
    return .{
        .hidden_size = cfg.joint_dim / 3,
        .intermediate_size = cfg.text_inner,
        .layers = 36,
        .heads = 32,
        .kv_heads = 8,
        .head_dim = 128,
        .vocab_size = 151936,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1e6,
    };
}

/// One double-stream block (`transformer_blocks.N`): separate img/txt qkv,
/// joint attention, per-head RMS qk norms, swiglu-class FFNs with fused
/// gate+up (`linear_in` rows = 2 * ffn_inner).
pub const Double = struct {
    to_q: tensor.View,
    to_k: tensor.View,
    to_v: tensor.View,
    add_q: tensor.View,
    add_k: tensor.View,
    add_v: tensor.View,
    norm_q: tensor.View,
    norm_k: tensor.View,
    norm_added_q: tensor.View,
    norm_added_k: tensor.View,
    to_out: tensor.View,
    to_add_out: tensor.View,
    ff_in: tensor.View,
    ff_out: tensor.View,
    ffc_in: tensor.View,
    ffc_out: tensor.View,
};

/// One single-stream block (`single_transformer_blocks.N`): one fused input
/// GEMM (qkv + mlp-in) and one fused output GEMM (attn + mlp-out).
pub const Single = struct {
    qkv_mlp: tensor.View, // [3*hidden + 2*ffn_inner, hidden]
    out: tensor.View, // [hidden, hidden + ffn_inner]
    norm_q: tensor.View,
    norm_k: tensor.View,
};

pub const Globals = struct {
    x_embed: tensor.View, // [hidden, 128]
    context_embed: tensor.View, // [hidden, joint_dim]
    mod_img: tensor.View, // [6*hidden, hidden]
    mod_txt: tensor.View, // [6*hidden, hidden]
    mod_single: tensor.View, // [3*hidden, hidden]
    time_in_1: tensor.View, // [hidden, 256]
    time_in_2: tensor.View, // [hidden, hidden]
    norm_out: tensor.View, // [2*hidden, hidden]
    proj_out: tensor.View, // [128, hidden]
};

/// Mapped safetensors shards of one component dir (1 file for the 4B
/// transformer, 2 for the 9B). Name lookup tries each shard.
pub const Files = struct {
    items: []tensor_file.Mapped,

    pub fn deinit(self: *Files, io: std.Io, allocator: std.mem.Allocator) void {
        for (self.items) |*m| m.deinit(io, allocator);
        allocator.free(self.items);
        self.* = undefined;
    }

    pub fn view(self: *const Files, name: []const u8) !?tensor.View {
        for (self.items) |*m| {
            if (try m.view(name)) |v| return v;
        }
        return null;
    }
};

/// Open every *.safetensors file directly inside `dir`.
pub fn openDir(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !Files {
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }
    if (names.items.len == 0) return error.MissingTensor;
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    var items = try allocator.alloc(tensor_file.Mapped, names.items.len);
    var opened: usize = 0;
    errdefer {
        for (items[0..opened]) |*m| m.deinit(io, allocator);
        allocator.free(items);
    }
    for (names.items, 0..) |n, i| {
        var buf: [1024]u8 = undefined;
        const full = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir_path, n });
        items[i] = try tensor_file.open(io, allocator, full);
        opened += 1;
    }
    return .{ .items = items };
}

pub const Loaded = struct {
    files: Files,
    cfg: Config,
    globals: Globals,
    doubles: []Double,
    singles: []Single,

    /// Shape storage of a sidecar-only load (zflux2_pack.loadFromSidecar);
    /// empty when the views borrow their shapes from the mapped shard.
    raw_shapes: []usize = &.{},

    pub fn deinit(self: *Loaded, io: std.Io, allocator: std.mem.Allocator) void {
        allocator.free(self.doubles);
        allocator.free(self.singles);
        if (self.raw_shapes.len != 0) allocator.free(self.raw_shapes);
        self.files.deinit(io, allocator);
        self.* = undefined;
    }
};

fn need(files: *const Files, name: []const u8) !tensor.View {
    const v = try files.view(name);
    return v orelse {
        std.debug.print("zflux2: missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
}

fn blockView(
    files: *const Files,
    comptime fmt: []const u8,
    idx: usize,
) !tensor.View {
    var buf: [128]u8 = undefined;
    return need(files, try std.fmt.bufPrint(&buf, fmt, .{idx}));
}

/// Open the transformer component dir (any shard count) and map every
/// weight (e.g. <snapshot>/transformer).
pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    cfg: Config,
) !Loaded {
    var file = try openDir(io, allocator, dir_path);
    errdefer file.deinit(io, allocator);

    const globals: Globals = .{
        .x_embed = try need(&file, "x_embedder.weight"),
        .context_embed = try need(&file, "context_embedder.weight"),
        .mod_img = try need(&file, "double_stream_modulation_img.linear.weight"),
        .mod_txt = try need(&file, "double_stream_modulation_txt.linear.weight"),
        .mod_single = try need(&file, "single_stream_modulation.linear.weight"),
        .time_in_1 = try need(&file, "time_guidance_embed.timestep_embedder.linear_1.weight"),
        .time_in_2 = try need(&file, "time_guidance_embed.timestep_embedder.linear_2.weight"),
        .norm_out = try need(&file, "norm_out.linear.weight"),
        .proj_out = try need(&file, "proj_out.weight"),
    };

    const doubles = try allocator.alloc(Double, cfg.double_layers);
    errdefer allocator.free(doubles);
    for (doubles, 0..) |*d, i| {
        d.* = .{
            .to_q = try blockView(&file, "transformer_blocks.{d}.attn.to_q.weight", i),
            .to_k = try blockView(&file, "transformer_blocks.{d}.attn.to_k.weight", i),
            .to_v = try blockView(&file, "transformer_blocks.{d}.attn.to_v.weight", i),
            .add_q = try blockView(&file, "transformer_blocks.{d}.attn.add_q_proj.weight", i),
            .add_k = try blockView(&file, "transformer_blocks.{d}.attn.add_k_proj.weight", i),
            .add_v = try blockView(&file, "transformer_blocks.{d}.attn.add_v_proj.weight", i),
            .norm_q = try blockView(&file, "transformer_blocks.{d}.attn.norm_q.weight", i),
            .norm_k = try blockView(&file, "transformer_blocks.{d}.attn.norm_k.weight", i),
            .norm_added_q = try blockView(&file, "transformer_blocks.{d}.attn.norm_added_q.weight", i),
            .norm_added_k = try blockView(&file, "transformer_blocks.{d}.attn.norm_added_k.weight", i),
            .to_out = try blockView(&file, "transformer_blocks.{d}.attn.to_out.0.weight", i),
            .to_add_out = try blockView(&file, "transformer_blocks.{d}.attn.to_add_out.weight", i),
            .ff_in = try blockView(&file, "transformer_blocks.{d}.ff.linear_in.weight", i),
            .ff_out = try blockView(&file, "transformer_blocks.{d}.ff.linear_out.weight", i),
            .ffc_in = try blockView(&file, "transformer_blocks.{d}.ff_context.linear_in.weight", i),
            .ffc_out = try blockView(&file, "transformer_blocks.{d}.ff_context.linear_out.weight", i),
        };
    }

    const singles = try allocator.alloc(Single, cfg.single_layers);
    errdefer allocator.free(singles);
    for (singles, 0..) |*s, i| {
        s.* = .{
            .qkv_mlp = try blockView(&file, "single_transformer_blocks.{d}.attn.to_qkv_mlp_proj.weight", i),
            .out = try blockView(&file, "single_transformer_blocks.{d}.attn.to_out.weight", i),
            .norm_q = try blockView(&file, "single_transformer_blocks.{d}.attn.norm_q.weight", i),
            .norm_k = try blockView(&file, "single_transformer_blocks.{d}.attn.norm_k.weight", i),
        };
    }

    return .{
        .files = file,
        .cfg = cfg,
        .globals = globals,
        .doubles = doubles,
        .singles = singles,
    };
}

/// Shape sanity for the mapped views: catches transposed or mis-sized
/// checkpoints before any compute exists.
pub fn check(self: *const Loaded) !void {
    const h = self.cfg.hidden;
    const inner = self.cfg.ffn_inner;
    try expectShape(self.globals.x_embed, h, self.cfg.in_channels);
    try expectShape(self.globals.context_embed, h, self.cfg.joint_dim);
    try expectShape(self.globals.mod_img, 6 * h, h);
    try expectShape(self.globals.mod_single, 3 * h, h);
    try expectShape(self.globals.proj_out, self.cfg.in_channels, h);
    for (self.doubles) |d| {
        try expectShape(d.to_q, h, h);
        try expectShape(d.ff_in, 2 * inner, h);
        try expectShape(d.ff_out, h, inner);
    }
    for (self.singles) |s| {
        try expectShape(s.qkv_mlp, 3 * h + 2 * inner, h);
        try expectShape(s.out, h, h + inner);
    }
}

fn expectShape(v: tensor.View, rows: usize, cols: usize) !void {
    if (v.shape.len != 2 or v.shape[0] != rows or v.shape[1] != cols) {
        std.debug.print(
            "zflux2: shape mismatch: got {any}, want [{d},{d}]\n",
            .{ v.shape, rows, cols },
        );
        return error.ShapeMismatch;
    }
}

test "variant configs match the published checkpoint architecture" {
    // Verified against transformer/config.json + text_encoder/config.json of
    // black-forest-labs/FLUX.2-klein-{4B,9B} (2026-08-06): hidden is
    // heads*128, mlp_ratio 3.0, and the Qwen3 conditioning encoder is
    // joint_dim/3 wide. Guards new variants against copy-paste dims.
    for ([_]Config{ Config.klein_4b, Config.klein_9b }) |cfg| {
        try std.testing.expectEqual(cfg.heads * head_dim, cfg.hidden);
        try std.testing.expectEqual(cfg.hidden * 3, cfg.ffn_inner);
        try std.testing.expectEqual(cfg.joint_dim / 3, textConfig(cfg).hidden_size);
        try std.testing.expect(cfg.pack_name.len > 0);
    }
    try std.testing.expectEqual(@as(u32, 9728), textConfig(Config.klein_4b).intermediate_size);
    try std.testing.expectEqual(@as(u32, 12288), textConfig(Config.klein_9b).intermediate_size);
    try std.testing.expect(!std.mem.eql(
        u8,
        Config.klein_4b.pack_name,
        Config.klein_9b.pack_name,
    ));
}
