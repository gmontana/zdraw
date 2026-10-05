const std = @import("std");

const env = @import("env.zig");
const genlock = @import("genlock.zig");
const kinds = @import("model_kind.zig");
const mattn = @import("mattn.zig");
const mconv = @import("mconv.zig");
const metrics = @import("metrics.zig");
const mlend = @import("mlend.zig");
const mlinear = @import("mlinear.zig");
const mvres_stream_chain = @import("mvres_stream_chain.zig");
const progress = @import("progress.zig");
const rtload = @import("runtime_load.zig");
const tae = @import("tae.zig");
const profile = @import("profile.zig");
const shards = @import("shards.zig");
const tensor_file = @import("tensor_file.zig");
const vdecode = @import("vdecode.zig");
const vrgb = @import("vrgb.zig");
const vviews = @import("vviews.zig");
const zimage = @import("zimage.zig");
const zpack_trace = @import("zpack_trace.zig");
const zrope = @import("zrope.zig");
const zsample = @import("zsample.zig");
const zstep = @import("zstep.zig");
const ztext = @import("zimage_text.zig");
const ztx = @import("ztx.zig");

const Allocator = std.mem.Allocator;
const ModelKind = kinds.ModelKind;

pub const Request = struct {
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
};

pub const Result = struct {
    pixels: []u8,
    width: u32,
    height: u32,
};

pub const Runtime = struct {
    root: []u8,
    loaded: zimage.Loaded,
    tx: ztx.Loaded,
    rope: zrope.Cache,
    text_store: shards.Store,
    linear: ?mlinear.Context,
    attn: ?mattn.Context,
    vae: tensor_file.Mapped,
    vae_views: vviews.Views,
    vae_metal: ?mconv.Context,
    // Persistent streamed-VAE ctx (lazy): its GPU pool is reused across
    // decodes so per-decode buffer churn cannot accrue wiring.
    vae_stream: ?mvres_stream_chain.Ctx = null,
    zpack: ?rtload.ZpackSidecar,
    // Same prompt -> same embedding; re-rolls skip the ~0.5 s text encode.
    memo_prompt: ?[]u8 = null,
    memo_text: ?ztext.Encoded = null,

    pub fn init(io: std.Io, allocator: Allocator, kind: ModelKind, root: []const u8) !Runtime {
        if (kind != .z_image_turbo) return error.UnsupportedInference;
        profile.apply(io, allocator);
        const owned_root = try allocator.dupe(u8, root);
        errdefer allocator.free(owned_root);
        try rtload.validate(io, allocator, kind, owned_root);
        var loaded = try rtload.loadMeta(io, allocator, owned_root);
        errdefer loaded.deinit(allocator);
        var tx = try rtload.loadTx(io, allocator, owned_root, &loaded);
        errdefer tx.deinit(io, allocator);
        var rope = try rtload.loadRope(io, allocator, loaded.config.transformer);
        errdefer rope.deinit(allocator);
        var text_store = try rtload.loadText(io, allocator, owned_root, &loaded);
        errdefer text_store.deinit(io, allocator);
        var linear_state = try initLinearState(io, allocator, owned_root);
        errdefer linear_state.deinit(io);
        var attn = try rtload.initAttn(io, allocator);
        errdefer if (attn) |*ctx| ctx.deinit();
        var vae = try rtload.loadVae(io, allocator, owned_root);
        errdefer vae.deinit(io, allocator);
        const views = try vviews.load(&vae);
        var metal = try rtload.initVae(io, allocator);
        errdefer if (metal) |*ctx| ctx.deinit();
        metrics.memtrace("load");
        return .{
            .root = owned_root,
            .loaded = loaded,
            .tx = tx,
            .rope = rope,
            .text_store = text_store,
            .linear = linear_state.linear,
            .attn = attn,
            .vae = vae,
            .vae_views = views,
            .vae_metal = metal,
            .zpack = linear_state.zpack,
        };
    }

    pub fn deinit(self: *Runtime, io: std.Io, allocator: std.mem.Allocator) void {
        zstep.clearCache(allocator);
        if (self.memo_text) |*text| text.deinit(allocator);
        if (self.memo_prompt) |prompt| allocator.free(prompt);
        if (self.vae_stream) |*s| s.deinit();
        if (self.vae_metal) |*ctx| ctx.deinit();
        if (self.attn) |*ctx| ctx.deinit();
        if (self.linear) |*ctx| ctx.deinit();
        if (self.zpack) |*sidecar| sidecar.deinit(io);
        self.vae.deinit(io, allocator);
        self.text_store.deinit(io, allocator);
        self.rope.deinit(allocator);
        self.tx.deinit(io, allocator);
        self.loaded.deinit(allocator);
        allocator.free(self.root);
        self.* = undefined;
    }

    pub fn generate(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
    ) !Result {
        if (request.prompt.len == 0) return error.EmptyPrompt;
        if (request.width == 0 or request.height == 0) return error.InvalidImageSize;
        // Concurrent renders corrupt each other (genlock.zig); serialize.
        genlock.acquire();
        defer genlock.release();
        // ZDRAW_METRICS dumps global Metal counters (command buffers, GPU
        // readbacks/waits, dispatches) after the run — the resident-path
        // visibility the trace tool can't give (its probe forces the slow
        // path). Cumulative counts; no reset (reset would orphan stacked
        // weights the zpack sidecar registered at load).
        const want_metrics = env.flag("ZDRAW_METRICS", false);
        var latents = try self.sample(io, allocator, request);
        defer latents.deinit(allocator);
        const result = try self.decode(io, allocator, request, latents);
        if (want_metrics) try metrics.report(io, allocator);
        if (metrics.memtraceEnabled()) zpack_trace.report();
        return result;
    }

    fn sample(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
    ) !zsample.Latents {
        const stage = try progress.begin(io, allocator, "sampling latents");
        const prepared = zsample.Prepared{
            .tx = &self.tx,
            .rope = self.rope,
            .text = .{
                .metal = self.linearPtr(),
                .attn = self.attnPtr(),
                .store = &self.text_store,
            },
            .denoise = .{ .metal = self.linearPtr(), .attn = self.attnPtr() },
        };
        const text = try self.memoText(io, allocator, prepared, request.prompt);
        metrics.memtrace("after-encode");
        const latents = try zsample.runPreparedText(io, allocator, &self.loaded, prepared, .{
            .root = self.root,
            .prompt = request.prompt,
            .width = request.width,
            .height = request.height,
            .steps = request.steps,
            .seed = request.seed,
        }, text);
        try progress.done(io, allocator, stage);
        return latents;
    }

    fn memoText(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        prepared: zsample.Prepared,
        prompt: []const u8,
    ) !ztext.Encoded {
        if (self.memo_prompt) |cached| {
            if (std.mem.eql(u8, cached, prompt)) return self.memo_text.?;
        }
        const text = try zsample.encodeText(io, allocator, &self.loaded, prepared, prompt);
        if (self.memo_text) |*old| old.deinit(allocator);
        if (self.memo_prompt) |old| allocator.free(old);
        self.memo_prompt = try allocator.dupe(u8, prompt);
        self.memo_text = text;
        return text;
    }

    fn linearPtr(self: *Runtime) ?*mlinear.Context {
        return if (self.linear) |*ctx| ctx else null;
    }

    fn attnPtr(self: *Runtime) ?*mattn.Context {
        return if (self.attn) |*ctx| ctx else null;
    }

    fn decode(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
        latents: zsample.Latents,
    ) !Result {
        const float_len = try rgbLen(request.width, request.height);
        const sample_out = try allocator.alloc(f32, float_len);
        defer allocator.free(sample_out);
        const stage = try progress.begin(io, allocator, "decoding image");
        const metal = if (self.vae_metal) |*ctx| ctx else null;
        const linear = self.linearPtr();
        const attn = self.attnPtr();
        const views = self.vae_views;
        metrics.memtrace("pre-vae");
        if (!try tae.decodeIfEnabled(io, allocator, metal, sample_out, latents.values, .{
            .height = latents.shape.height,
            .width = latents.shape.width,
        })) {
            const stream = try vdecode.ensureStream(metal, &self.vae_stream);
            // The denoise chain is idle during the decode; its gateup slot
            // hosts the VAE's conv1_out (product f16 only: strict's f32
            // request exceeds the slot and keeps its own buffer). Pinned so
            // no grow can free it under the borrower.
            const offer: ?mlend.Offer = if (linear) |l| mlend.chainOffer(&l.pool) else null;
            if (linear) |l| l.pool.pin(.gateup);
            defer if (linear) |l| l.pool.unpin(.gateup);
            try vdecode.run(
                metal,
                linear,
                attn,
                stream,
                allocator,
                sample_out,
                latents.values,
                views,
                .{ .height = latents.shape.height, .width = latents.shape.width },
                offer,
            );
        }
        metrics.memtrace("post-vae");
        try progress.done(io, allocator, stage);
        const pixels = try allocator.alloc(u8, float_len);
        errdefer allocator.free(pixels);
        try vrgb.run(pixels, sample_out, request.width, request.height);
        metrics.memtrace("post-rgb");
        return .{ .pixels = pixels, .width = request.width, .height = request.height };
    }
};

const LinearState = struct {
    linear: ?mlinear.Context,
    zpack: ?rtload.ZpackSidecar,

    fn deinit(self: *LinearState, io: std.Io) void {
        if (self.linear) |*ctx| ctx.deinit();
        if (self.zpack) |*sidecar| sidecar.deinit(io);
    }
};

fn initLinearState(io: std.Io, allocator: Allocator, root: []const u8) !LinearState {
    var linear = try rtload.initLinear(io, allocator);
    errdefer if (linear) |*ctx| ctx.deinit();
    var zpack = try rtload.loadZpack(io, allocator, root, &linear);
    errdefer if (zpack) |*sidecar| sidecar.deinit(io);
    return .{ .linear = linear, .zpack = zpack };
}

fn rgbLen(width: u32, height: u32) !usize {
    const pixels = std.math.mul(usize, @intCast(width), @intCast(height)) catch {
        return error.InvalidImageSize;
    };
    return std.math.mul(usize, pixels, 3) catch error.InvalidImageSize;
}
