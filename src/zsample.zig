//! Z-Image sampling up to final latents.
//!
//! The caller owns VAE decode and PNG output. This file only makes the latent
//! tensor that the decoder consumes.

const std = @import("std");

const env = @import("env.zig");
const scheduler = @import("scheduler.zig");
const progress = @import("progress.zig");
const zdenoise = @import("zdenoise.zig");
const zconfig = @import("zimage_config.zig");
const zimage = @import("zimage.zig");
const zlatent = @import("zlatent.zig");
const zpatch = @import("zpatch.zig");
const zrope = @import("zrope.zig");
const ztext = @import("zimage_text.zig");
const ztrace = @import("ztrace_capture.zig");
const ztx = @import("ztx.zig");

pub const Request = struct {
    root: []const u8,
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
};

pub const Prepared = struct {
    tx: *const ztx.Loaded,
    rope: zrope.Cache,
    text: ztext.Prepared,
    denoise: zdenoise.Prepared,
};

pub const Latents = struct {
    values: []f32,
    shape: zpatch.Shape,

    pub fn deinit(self: *Latents, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        self.* = undefined;
    }
};

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    loaded: *const zimage.Loaded,
    request: Request,
) !Latents {
    const text_stage = try progress.begin(io, allocator, "encoding prompt");
    var text = try ztext.encode(
        io,
        allocator,
        request.root,
        loaded.config.text,
        loaded.tokens,
        loaded.indexes.text,
        request.prompt,
    );
    defer text.deinit(allocator);
    try progress.done(io, allocator, text_stage);

    const tx_stage = try progress.begin(io, allocator, "loading transformer");
    var tx = try ztx.load(
        io,
        allocator,
        request.root,
        loaded.config.transformer,
        loaded.indexes.transformer,
    );
    defer tx.deinit(io, allocator);
    try progress.done(io, allocator, tx_stage);

    const prep = try progress.begin(io, allocator, "preparing schedule");
    var rope = try makeRope(allocator, loaded.config.transformer);
    defer rope.deinit(allocator);
    var schedule = try scheduler.makeZImage(allocator, request.steps);
    defer schedule.deinit(allocator);
    try progress.done(io, allocator, prep);

    const cfg = loaded.config.transformer;
    return try sample(io, allocator, text, &tx, rope, schedule, cfg, request, null);
}

pub fn runPrepared(
    io: std.Io,
    allocator: std.mem.Allocator,
    loaded: *const zimage.Loaded,
    prepared: Prepared,
    request: Request,
) !Latents {
    var text = try encodeText(io, allocator, loaded, prepared, request.prompt);
    defer text.deinit(allocator);
    return runPreparedText(io, allocator, loaded, prepared, request, text);
}

// Encode the prompt only; the caller owns (and may cache) the result.
pub fn encodeText(
    io: std.Io,
    allocator: std.mem.Allocator,
    loaded: *const zimage.Loaded,
    prepared: Prepared,
    prompt: []const u8,
) !ztext.Encoded {
    const text_stage = try progress.begin(io, allocator, "encoding prompt");
    const text = try ztext.encodePrepared(
        io,
        allocator,
        prepared.text,
        loaded.config.text,
        loaded.tokens,
        loaded.indexes.text,
        prompt,
    );
    try progress.done(io, allocator, text_stage);
    if (relText()) {
        if (prepared.text.metal) |metal| metal.buffers.clearSources();
    }
    return text;
}

// Sample with an already-encoded prompt (borrowed, not consumed).
pub fn runPreparedText(
    io: std.Io,
    allocator: std.mem.Allocator,
    loaded: *const zimage.Loaded,
    prepared: Prepared,
    request: Request,
    text: ztext.Encoded,
) !Latents {
    const prep = try progress.begin(io, allocator, "preparing schedule");
    var schedule = try scheduler.makeZImage(allocator, request.steps);
    defer schedule.deinit(allocator);
    try progress.done(io, allocator, prep);

    return try sample(
        io,
        allocator,
        text,
        prepared.tx,
        prepared.rope,
        schedule,
        loaded.config.transformer,
        request,
        prepared.denoise,
    );
}

fn sample(
    io: std.Io,
    allocator: std.mem.Allocator,
    text: ztext.Encoded,
    tx: *const ztx.Loaded,
    rope: zrope.Cache,
    schedule: scheduler.Schedule,
    cfg: zconfig.Transformer,
    request: Request,
    denoise: ?zdenoise.Prepared,
) !Latents {
    const shape = try zlatent.shape(request.width, request.height, cfg);
    const values = try allocator.alloc(f32, zlatent.len(shape));
    errdefer allocator.free(values);
    try zdenoise.initLatents(io, values, request.seed);
    var trace = try ztrace.Capture.fromEnv(io, allocator, .{
        .prompt = request.prompt,
        .width = request.width,
        .height = request.height,
        .steps = request.steps,
        .seed = request.seed,
        .layers = tx.layers.items.len,
    });
    defer if (trace) |*capture| capture.deinit();
    const req = zdenoise.Request{
        .cap = text.embeds,
        .shape = shape,
        .schedule = schedule,
        .tx = tx,
        .cfg = cfg,
        .rope = rope,
        .trace_capture = if (trace) |*capture| capture else null,
    };
    if (denoise) |prepared| {
        try zdenoise.runPrepared(io, allocator, values, req, prepared);
    } else {
        try zdenoise.run(io, allocator, values, req);
    }
    return .{ .values = values, .shape = shape };
}

fn makeRope(allocator: std.mem.Allocator, cfg: zconfig.Transformer) !zrope.Cache {
    return zrope.Cache.init(allocator, .{
        .dims = .{ cfg.axes_dims[0], cfg.axes_dims[1], cfg.axes_dims[2] },
        .lens = .{ cfg.axes_lens[0], cfg.axes_lens[1], cfg.axes_lens[2] },
        .theta = @floatCast(cfg.rope_theta),
    });
}

fn relText() bool {
    return env.flag("ZDRAW_RELEASE_TEXT_WEIGHTS", false);
}

test "latent result owns values" {
    var latents = Latents{
        .values = try std.testing.allocator.alloc(f32, 4),
        .shape = .{ .channels = 1, .frames = 1, .height = 2, .width = 2, .patch = 2, .f_patch = 1 },
    };
    latents.deinit(std.testing.allocator);
}
