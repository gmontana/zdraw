const std = @import("std");

const env = @import("env.zig");
const kinds = @import("model_kind.zig");
const mattn = @import("mattn.zig");
const mconv = @import("mconv.zig");
const mlinear = @import("mlinear.zig");
const progress = @import("progress.zig");
const shards = @import("shards.zig");
const tensor_file = @import("tensor_file.zig");
const weights = @import("weights.zig");
const zconfig = @import("zimage_config.zig");
const zimage = @import("zimage.zig");
const zpack_file = @import("zpack_file.zig");
const zrope = @import("zrope.zig");
const zw16 = @import("zw16.zig");
const ztx = @import("ztx.zig");

pub const ZpackSidecar = zpack_file.Mapped;

pub fn validate(
    io: std.Io,
    allocator: std.mem.Allocator,
    kind: kinds.ModelKind,
    root: []const u8,
) !void {
    const stage = try progress.begin(io, allocator, "checking model files");
    try weights.validate(io, allocator, kind, root);
    try progress.done(io, allocator, stage);
}

pub fn loadMeta(io: std.Io, allocator: std.mem.Allocator, root: []const u8) !zimage.Loaded {
    const stage = try progress.begin(io, allocator, "loading model metadata");
    const loaded = try zimage.load(io, allocator, root);
    try progress.done(io, allocator, stage);
    return loaded;
}

pub fn loadTx(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    loaded: *const zimage.Loaded,
) !ztx.Loaded {
    const stage = try progress.begin(io, allocator, "loading transformer");
    const tx = try ztx.load(
        io,
        allocator,
        root,
        loaded.config.transformer,
        loaded.indexes.transformer,
    );
    try progress.done(io, allocator, stage);
    return tx;
}

pub fn loadRope(
    io: std.Io,
    allocator: std.mem.Allocator,
    cfg: zconfig.Transformer,
) !zrope.Cache {
    const stage = try progress.begin(io, allocator, "preparing rope");
    const rope = try zrope.Cache.init(allocator, .{
        .dims = .{ cfg.axes_dims[0], cfg.axes_dims[1], cfg.axes_dims[2] },
        .lens = .{ cfg.axes_lens[0], cfg.axes_lens[1], cfg.axes_lens[2] },
        .theta = @floatCast(cfg.rope_theta),
    });
    try progress.done(io, allocator, stage);
    return rope;
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, root: []const u8) !tensor_file.Mapped {
    const stage = try progress.begin(io, allocator, "loading VAE decoder");
    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/vae/diffusion_pytorch_model.safetensors",
        .{root},
    );
    defer allocator.free(path);
    const vae = try tensor_file.open(io, allocator, path);
    try progress.done(io, allocator, stage);
    return vae;
}

pub fn textRoot(allocator: std.mem.Allocator, root: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/text_encoder", .{root});
}

pub fn loadText(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    loaded: *const zimage.Loaded,
) !shards.Store {
    const stage = try progress.begin(io, allocator, "loading text encoder");
    const path = try textRoot(allocator, root);
    defer allocator.free(path);
    const store = try shards.open(io, allocator, path, loaded.indexes.text);
    try progress.done(io, allocator, stage);
    return store;
}

pub fn initLinear(io: std.Io, allocator: std.mem.Allocator) !?mlinear.Context {
    const ctx = mlinear.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => {
            try progress.event(io, allocator, "Metal runtime linear unavailable");
            return null;
        },
        else => return err,
    };
    try progress.event(io, allocator, "Metal runtime linear ready");
    return ctx;
}

pub fn loadZpack(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    linear: *?mlinear.Context,
) !?ZpackSidecar {
    var buf: [1024]u8 = undefined;
    const path = if (std.c.getenv("ZDRAW_ZPACK")) |raw|
        std.mem.span(raw)
    else
        std.fmt.bufPrint(&buf, "{s}/zdraw-w16.zpack", .{root}) catch return null;
    if (path.len == 0) return null;
    if (std.c.getenv("ZDRAW_ZPACK") == null) {
        std.Io.Dir.cwd().access(io, path, .{}) catch |err| {
            // A missing default sidecar (e.g. a dangling symlink) silently
            // degrades every W16-substituted GEMM to the generic tier (~13%
            // slower at 1024); say so when substitution is on, name the
            // one-time build, and fail under ZDRAW_REQUIRE_ZPACK.
            if (zw16.enabled()) {
                var msg: [1400]u8 = undefined;
                const text = std.fmt.bufPrint(
                    &msg,
                    "W16 sidecar missing at {s}; running the slower generic GEMM tier. " ++
                        "Build it once with: zig build zpackbuild -- --weights <weights> " ++
                        "--bits 16 --kinds all --families all --last 30 --out {s}",
                    .{ path, path },
                ) catch "W16 sidecar missing; see README (zig build zpackbuild)";
                try progress.event(io, allocator, text);
                if (env.flag("ZDRAW_REQUIRE_ZPACK", false)) return err;
            }
            return null;
        };
    }
    // An adapted (LoRA-merged) sidecar carries a `<pack>.lora.json` note.
    // Strict's exact GEMMs read the checkpoint and never substitute from the
    // sidecar, so the adapter would silently not apply: refuse instead
    // (ledger zimage-lora-bake-20260903).
    if (std.c.getenv("ZDRAW_GEMM")) |gemm| {
        if (std.mem.eql(u8, std.mem.span(gemm), "exact")) {
            var note_buf: [1100]u8 = undefined;
            const note = std.fmt.bufPrint(&note_buf, "{s}.lora.json", .{path}) catch path;
            if (std.Io.Dir.cwd().access(io, note, .{})) |_| {
                try progress.event(
                    io,
                    allocator,
                    "the selected sidecar is LoRA-adapted, but --profile strict renders " ++
                        "the base checkpoint (exact GEMMs never read the sidecar); " ++
                        "use --profile product for adapted packs",
                );
                return error.AdaptedSidecarUnderStrict;
            } else |_| {}
        }
    }
    const stage = try progress.begin(io, allocator, "loading zpack sidecar");
    var sidecar = try zpack_file.open(io, path);
    errdefer sidecar.deinit(io);
    if (linear.*) |*ctx| ctx.buffers.setSidecar(sidecar.bytes());
    try progress.done(io, allocator, stage);
    return sidecar;
}

pub fn initAttn(io: std.Io, allocator: std.mem.Allocator) !?mattn.Context {
    const ctx = mattn.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => {
            try progress.event(io, allocator, "Metal runtime attention unavailable");
            return null;
        },
        else => return err,
    };
    try progress.event(io, allocator, "Metal runtime attention ready");
    return ctx;
}

pub fn initVae(io: std.Io, allocator: std.mem.Allocator) !?mconv.Context {
    const ctx = mconv.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => {
            try progress.event(io, allocator, "Metal VAE conv unavailable");
            return null;
        },
        else => return err,
    };
    try progress.event(io, allocator, "Metal VAE conv ready");
    return ctx;
}
