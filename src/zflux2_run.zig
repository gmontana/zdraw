//! FLUX.2 Klein runtime: weights, tokenizer, encoder shards, VAE, and Metal
//! contexts loaded once; `generate` runs per request. Holding a Runtime
//! across requests is what makes warm timing, prompt-embed memoization, and
//! batched seeds natural (the god-function `generateKlein` used to reload
//! everything every call).
//!
//! Two attention contexts are intentional, not redundant: `attn` (default
//! kernel) serves the GQA Qwen3 encoder, the CPU reference DiT, and the VAE
//! mid-attention; the resident denoise path owns its own block16 context
//! inside `res`. They have different kernel requirements.

const std = @import("std");
const model_kind = @import("model_kind.zig");

const genlock = @import("genlock.zig");
const init_image = @import("init_image.zig");
const seed_field = @import("seed_field.zig");

const tokenizer = @import("tokenizer.zig");
const weight_index = @import("weight_index.zig");
const shards = @import("shards.zig");
const tensor_file = @import("tensor_file.zig");
const zconfig = @import("zimage_config.zig");
const util = @import("session_util.zig");

const zflux2 = @import("zflux2.zig");
const zdenoise = @import("zdenoise.zig");
const zflux2_dit = @import("zflux2_dit.zig");
const zflux2_schedule = @import("zflux2_schedule.zig");
const zflux2_vae = @import("zflux2_vae.zig");
const zflux2_pack = @import("zflux2_pack.zig");
const zflux2_res = @import("zflux2_resident.zig");
const zpack_file = @import("zpack_file.zig");

const qenc = @import("qwen_encoder.zig");
const qwen_pack = @import("qwen_pack.zig");
const qres = @import("qwen_resident.zig");
const qscratch = @import("qwen_scratch.zig");

const mlinear = @import("mlinear.zig");
const mattn = @import("mattn.zig");
const mconv = @import("mconv.zig");
const vae_mode = @import("vae_mode.zig");
const vviews = @import("vviews.zig");
const vdecode = @import("vdecode.zig");
const mvres_stream_chain = @import("mvres_stream_chain.zig");
const vrgb = @import("vrgb.zig");
const preview = @import("preview.zig");
const progress = @import("progress.zig");
const metrics = @import("metrics.zig");

const stacked_layers = [_]usize{ 9, 18, 27 };
const rope_theta: f64 = 2000.0;
const max_prompt_tokens: usize = 512;
const max_image_tokens: usize = model_kind.klein_max_image_tokens;

pub const Request = struct {
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    /// Classifier-free guidance scale. 1.0 = single pass (every distilled
    /// Klein); >1.0 runs the base-model CFG pair per step.
    guidance: f32 = 1.0,
    /// img2img: the image whose latent the sampler starts from (empty =
    /// text to image) and how far back it starts (1 = ignore the image).
    init_image: []const u8 = "",
    strength: f32 = 1.0,
    /// "Fix a part": a mask image (white = redraw) over the init image; the
    /// rest is held to the original at every step.
    mask: []const u8 = "",
    /// Instruction editing: a reference image whose tokens ride along in the
    /// sequence (Klein's edit mode); the prompt is the instruction.
    ref_image: []const u8 = "",
    /// Wander: start from a blend of four corner seeds at (u, v).
    field: seed_field.Field = .{},
};

/// diffusers' Flux2KleinPipeline encodes the EMPTY prompt as the
/// unconditional conditioning for the base models' CFG (negative_prompt
/// defaults to ""; it goes through the same chat template and 512 padding),
/// and tools/diffusers_bench.py is the gate's reference, so the engine
/// matches it. There is no negative-prompt input on this family.
const neg_prompt = "";

pub const Result = struct {
    pixels: []u8,
    width: u32,
    height: u32,
};

pub const Runtime = struct {
    alloc: std.mem.Allocator, // persistent (Runtime lifetime) for the memo
    cfg: zflux2.Config,
    text: zconfig.Text,
    // Prompt-embed memoization: rerolls and batched seeds reuse the Qwen3
    // encode (the ~13% encode bucket) when the prompt is unchanged.
    memo_hash: u64 = 0,
    memo_prompt: ?[]u8 = null, // owned copy; verified on hit (hash is pre-filter)
    memo_embeds: ?[]f32 = null,
    // The CFG unconditional conditioning is a fixed prompt, so it lives in
    // its own slot instead of evicting the positive memo every step.
    neg_embeds: ?[]f32 = null,
    // Reference-image memoization (instruction editing): the app edits one
    // photo many times and the VAE encode is ~8 s at 1024. Keyed on the
    // file's path, size and mtime plus the render size (ZDRAW_REF_CACHE=0
    // disables). A hit returns the same bytes a fresh encode would.
    ref_hash: u64 = 0,
    ref_key: ?[]u8 = null,
    ref_tokens: ?[]f32 = null,
    tokens: tokenizer.Loaded,
    te_index: weight_index.Index,
    te_store: shards.Store,
    loaded: zflux2.Loaded,
    zpack: ?zpack_file.Mapped,
    /// The sidecar's tier (16, 6 or 4 bits per weight); 16 without a sidecar.
    pack_bits: u8 = 16,
    /// The text pack (4-bit encoder linears) and its by-name table; the
    /// text_encoder shards may be absent when it is present.
    text_pack: ?zpack_file.Mapped = null,
    text_table: ?*qwen_pack.Table = null,
    /// The encoder's tier: 4 with a text pack, 16 from the bf16 shards.
    text_bits: u8 = 16,
    vae: tensor_file.Mapped,
    views: vviews.Views,
    lin: ?mlinear.Context,
    attn: mattn.Context,
    conv: ?mconv.Context,
    res: ?zflux2_res.Ctx,
    // Resident text encoder (ZDRAW_KLEIN_TE_RES): borrows the mapped te
    // shards, so it is destroyed before te_store in deinit.
    te_res: ?qres.Ctx,
    // Persistent streamed/pooled VAE context (created lazily on first decode,
    // reused across decodes so the pool's GPU wiring is the high-water mark, not
    // a per-decode accrual). Routes the FLUX.2 VAE through the resident exact-
    // streaming path instead of the untiled path — the untiled path wires the
    // full-resolution upsample buffers and never unwires them (+11 GB phys at
    // 1024px). Streamed output is bit-equal to untiled (max|d|=1, product band).
    vae_stream: ?mvres_stream_chain.Ctx = null,

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        cfg: zflux2.Config,
        weights_dir: []const u8,
    ) !Runtime {
        // Klein's f16 streamed VAE tier is validated against the f32 streamed
        // baseline at <=1 LSB drift on the 1024px scorecard prompt, and cuts
        // decode GPU time by ~4.4s. Explicit ZDRAW_VAE* env values still win.
        vae_mode.applyEnv(.product, false);
        const text = zflux2.textConfig(cfg);

        const tok = try tokenizer.load(io, allocator, weights_dir);
        errdefer {
            var t = tok;
            t.deinit(allocator);
        }

        var text_pack = try findTextPack(io, allocator, weights_dir);
        errdefer if (text_pack) |*p| p.deinit(io);
        const text_table: ?*qwen_pack.Table = if (text_pack) |*p|
            try qwen_pack.create(allocator, p.bytes())
        else
            null;
        errdefer if (text_table) |t| qwen_pack.destroy(allocator, t);
        var te = try loadText(io, allocator, weights_dir, text_table);
        errdefer te.deinit(io, allocator);

        const tx_dir = try std.fmt.allocPrint(allocator, "{s}/transformer", .{weights_dir});
        defer allocator.free(tx_dir);
        var zpack = try findPack(io, allocator, weights_dir, cfg);
        errdefer if (zpack) |*sidecar| sidecar.deinit(io);
        var loaded = try loadWeights(io, allocator, tx_dir, cfg, zpack);
        errdefer loaded.deinit(io, allocator);
        const pack_bits: u8 = if (zpack) |*sidecar|
            try zflux2_pack.sidecarBits(sidecar.bytes())
        else
            16;

        const vae_path = try std.fmt.allocPrint(
            allocator,
            "{s}/vae/diffusion_pytorch_model.safetensors",
            .{weights_dir},
        );
        defer allocator.free(vae_path);
        var vae = try tensor_file.open(io, allocator, vae_path);
        errdefer vae.deinit(io, allocator);
        const views = try vviews.load(&vae);

        var lin = mlinear.Context.init() catch null;
        errdefer if (lin) |*m| m.deinit();
        var attn = try mattn.Context.init();
        errdefer attn.deinit();
        var conv = mconv.Context.init() catch null;
        errdefer if (conv) |*c| c.deinit();

        const res: ?zflux2_res.Ctx = if (util.envFlag("ZDRAW_KLEIN_RES", true))
            try zflux2_res.Ctx.init(allocator)
        else
            null;
        errdefer if (res) |*r| {
            var rc = r.*;
            rc.deinit();
        };
        // Default ON since 2026-08-07: readbacks 175 to 14, encode lap 3350
        // to 389 ms same box/session, conditioning cos 1.000000 vs per-op,
        // klein_gate quick 2/2. ZDRAW_KLEIN_TE_RES=0 restores the per-op
        // oracle route.
        const te_res: ?qres.Ctx = if (util.envFlag("ZDRAW_KLEIN_TE_RES", true))
            try qres.Ctx.init(allocator)
        else
            null;
        metrics.memtrace("klein-init");

        return .{
            .alloc = allocator,
            .cfg = cfg,
            .text = text,
            .tokens = tok,
            .te_index = te.index,
            .te_store = te.store,
            .loaded = loaded,
            .zpack = zpack,
            .pack_bits = pack_bits,
            .text_pack = text_pack,
            .text_table = text_table,
            .text_bits = if (text_table != null) qwen_pack.bits else 16,
            .vae = vae,
            .views = views,
            .lin = lin,
            .attn = attn,
            .conv = conv,
            .res = res,
            .te_res = te_res,
        };
    }

    pub fn deinit(self: *Runtime, io: std.Io, allocator: std.mem.Allocator) void {
        if (self.memo_embeds) |e| self.alloc.free(e);
        if (self.neg_embeds) |e| self.alloc.free(e);
        if (self.memo_prompt) |pr| self.alloc.free(pr);
        if (self.ref_tokens) |z| self.alloc.free(z);
        if (self.ref_key) |k| self.alloc.free(k);
        if (self.te_res) |*t| t.deinit();
        if (self.res) |*r| r.deinit();
        if (self.vae_stream) |*s| s.deinit();
        if (self.conv) |*c| c.deinit();
        self.attn.deinit();
        if (self.lin) |*m| m.deinit();
        self.vae.deinit(io, allocator);
        self.loaded.deinit(io, allocator);
        if (self.zpack) |*sidecar| sidecar.deinit(io);
        self.te_store.deinit(io, allocator);
        self.te_index.deinit(allocator);
        if (self.text_table) |t| qwen_pack.destroy(allocator, t);
        if (self.text_pack) |*p| p.deinit(io);
        self.tokens.deinit(allocator);
        self.* = undefined;
    }

    const Text = struct {
        index: weight_index.Index,
        store: shards.Store,

        fn deinit(self: *Text, io: std.Io, allocator: std.mem.Allocator) void {
            self.store.deinit(io, allocator);
            self.index.deinit(allocator);
        }
    };

    /// The encoder's weights: the bf16 shards, with the text pack's views
    /// answering first when there is one; the text pack alone when the
    /// text_encoder directory carries no shard index.
    fn loadText(
        io: std.Io,
        allocator: std.mem.Allocator,
        weights_dir: []const u8,
        table: ?*qwen_pack.Table,
    ) !Text {
        const index_path = try std.fmt.allocPrint(
            allocator,
            "{s}/text_encoder/model.safetensors.index.json",
            .{weights_dir},
        );
        defer allocator.free(index_path);
        const te_root = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{weights_dir});
        defer allocator.free(te_root);
        if (weight_index.read(io, allocator, index_path)) |index| {
            var idx = index;
            errdefer idx.deinit(allocator);
            var store = try shards.open(io, allocator, te_root, idx);
            if (table) |t| store.overrides = &t.overrides;
            return .{ .index = idx, .store = store };
        } else |err| switch (err) {
            error.FileNotFound => {
                const t = table orelse return error.MissingFile;
                const msg = "no text_encoder shards; loading the text pack";
                try progress.event(io, allocator, msg);
                return .{
                    .index = try weight_index.empty(allocator),
                    .store = try shards.packOnly(allocator, &t.overrides),
                };
            },
            else => return err,
        }
    }

    /// ZDRAW_KLEIN_TEXT_ZPACK, else the text pack beside the weights or under
    /// runs/; none is the bf16 shards.
    fn findTextPack(
        io: std.Io,
        allocator: std.mem.Allocator,
        root: []const u8,
    ) !?zpack_file.Mapped {
        const label = "loading Klein text pack";
        if (std.c.getenv("ZDRAW_KLEIN_TEXT_ZPACK")) |raw| {
            const path = std.mem.span(raw);
            if (path.len == 0) return null;
            return try openPack(io, allocator, path, label);
        }
        var root_buf: [1024]u8 = undefined;
        const beside = std.fmt.bufPrint(
            &root_buf,
            "{s}/{s}",
            .{ root, qwen_pack.pack_name },
        ) catch null;
        const candidates = [_]?[]const u8{ beside, "runs/" ++ qwen_pack.pack_name };
        for (candidates) |maybe| {
            const path = maybe orelse continue;
            std.Io.Dir.cwd().access(io, path, .{}) catch continue;
            return try openPack(io, allocator, path, label);
        }
        return null;
    }

    /// The transformer weights: from the shard directory with the sidecar
    /// swapped in, or, when the shard is absent and the sidecar carries the
    /// globals (`kleinpack --globals`), from the sidecar alone.
    fn loadWeights(
        io: std.Io,
        allocator: std.mem.Allocator,
        tx_dir: []const u8,
        cfg: zflux2.Config,
        zpack: ?zpack_file.Mapped,
    ) !zflux2.Loaded {
        if (zflux2.load(io, allocator, tx_dir, cfg)) |shard| {
            var loaded = shard;
            errdefer loaded.deinit(io, allocator);
            if (zpack) |*sidecar| {
                try zflux2_pack.swapLoaded(sidecar.bytes(), &loaded);
            } else if (!zflux2_pack.allowUnpacked()) {
                return error.MissingPackedSidecar;
            } else {
                const msg = "Klein W16 sidecar missing; using high-memory unpacked weights";
                try progress.event(io, allocator, msg);
            }
            return loaded;
        } else |err| switch (err) {
            // No transformer directory, or one without a shard in it.
            error.FileNotFound, error.MissingTensor => {
                const sidecar = zpack orelse return err;
                if (!zflux2_pack.hasRaw(sidecar.bytes())) return err;
                const msg = "no transformer shard; loading the sidecar's globals";
                try progress.event(io, allocator, msg);
                return zflux2_pack.loadFromSidecar(allocator, sidecar.bytes(), cfg);
            },
            else => return err,
        }
    }

    fn findPack(
        io: std.Io,
        allocator: std.mem.Allocator,
        root: []const u8,
        cfg: zflux2.Config,
    ) !?zpack_file.Mapped {
        const env_path = std.c.getenv("ZDRAW_KLEIN_ZPACK");
        if (env_path) |raw| {
            const path = std.mem.span(raw);
            if (path.len == 0) return null;
            return try openPack(io, allocator, path, "loading Klein zpack sidecar");
        }

        var root_buf: [1024]u8 = undefined;
        var candidates: [2][]const u8 = undefined;
        var count: usize = 0;
        var local_buf: [256]u8 = undefined;
        if (std.fmt.bufPrint(&root_buf, "{s}/{s}", .{ root, cfg.pack_name })) |path| {
            candidates[count] = path;
            count += 1;
        } else |_| {}
        candidates[count] = try std.fmt.bufPrint(&local_buf, "runs/{s}", .{cfg.pack_name});
        count += 1;

        for (candidates[0..count]) |path| {
            std.Io.Dir.cwd().access(io, path, .{}) catch continue;
            return try openPack(io, allocator, path, "loading Klein zpack sidecar");
        }
        return null;
    }

    fn openPack(
        io: std.Io,
        allocator: std.mem.Allocator,
        path: []const u8,
        label: []const u8,
    ) !zpack_file.Mapped {
        const stage = try progress.begin(io, allocator, label);
        const sidecar = try zpack_file.open(io, path);
        try progress.done(io, allocator, stage);
        return sidecar;
    }

    fn linPtr(self: *Runtime) ?*mlinear.Context {
        return if (self.lin) |*m| m else null;
    }

    /// Memoized conditioning: returns a slice owned by the Runtime (valid
    /// until the next distinct prompt or deinit). Same prompt -> cached.
    fn embedsFor(
        self: *Runtime,
        io: std.Io,
        prompt: []const u8,
    ) ![]const f32 {
        const h = std.hash.Wyhash.hash(0, prompt);
        if (self.memo_embeds) |e| {
            // Hash pre-filters; byte-equality is the real key (no hash-collision
            // footgun).
            if (h == self.memo_hash and self.memo_prompt != null and
                std.mem.eql(u8, self.memo_prompt.?, prompt)) return e;
            self.alloc.free(e);
            self.memo_embeds = null;
            if (self.memo_prompt) |pr| self.alloc.free(pr);
            self.memo_prompt = null;
        }
        const e = try self.encodeText(io, self.alloc, prompt);
        errdefer self.alloc.free(e);
        const pr = try self.alloc.dupe(u8, prompt);
        self.memo_embeds = e;
        self.memo_prompt = pr;
        self.memo_hash = h;
        return e;
    }

    /// Unconditional conditioning for CFG, encoded once per runtime.
    fn negEmbedsFor(self: *Runtime, io: std.Io) ![]const f32 {
        if (self.neg_embeds) |e| return e;
        const e = try self.encodeText(io, self.alloc, neg_prompt);
        self.neg_embeds = e;
        return e;
    }

    /// Qwen3 stacked-tap conditioning for one prompt. Caller frees the slice.
    fn encodeText(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        prompt: []const u8,
    ) ![]f32 {
        var ids = try self.tokens.encodeKlein(allocator, prompt, max_prompt_tokens);
        defer ids.deinit(allocator);
        const hidden: usize = self.text.hidden_size;
        const embeds = try allocator.alloc(f32, max_prompt_tokens * stacked_layers.len * hidden);
        errdefer allocator.free(embeds);
        // The references (diffusers, mflux) pad to 512 AND pass the padding
        // mask; without it the padded rows - which the DiT consumes unmasked -
        // diverge from the reference. Right padding keeps real rows unchanged.
        var valid: usize = 0;
        for (ids.mask) |m| valid += @intFromBool(m != 0);
        if (self.te_res != null) {
            // Resident route (ZDRAW_KLEIN_TE_RES): one batched command buffer
            // for the whole stacked encode, no CPU layer scratch. The per-op
            // route below stays the oracle; TE_BATCH/TE_PROJ govern only it.
            try self.encodeResident(io, allocator, embeds, ids.ids, valid);
        } else {
            const cfg = qenc.attnConfig(self.text, max_prompt_tokens);
            var scratch = try qscratch.init(allocator, cfg, self.text.intermediate_size);
            defer scratch.deinit(allocator);
            defer if (self.lin) |*m| m.clearWCache();
            // Back on by default: the divergence blamed on this route in the
            // 2026-08-06 investigation was the AttnParams ABI mismatch (the C
            // struct was never widened for the padding limit, so the kernel read
            // it as garbage). With that fixed both routes match the diffusers
            // reference at cos 0.99984 on real rows and each other at 0.999875.
            const batch = util.envFlag("ZDRAW_KLEIN_TE_BATCH", true);
            // Bisect knob for the divergence: force the projections onto the
            // other route than the FFN. Unset = both follow TE_BATCH.
            const proj: ?bool = if (std.c.getenv("ZDRAW_KLEIN_TE_PROJ")) |raw|
                raw[0] == '1'
            else
                null;
            try qenc.run(io, allocator, self.linPtr(), &self.attn, embeds, ids.ids, &self.te_store, self.te_index, self.text, .{
                .stacked = &stacked_layers,
            }, &scratch, batch, proj, valid);
        }
        // Debug: dump the conditioning tensor for A/B path comparisons.
        if (std.c.getenv("ZDRAW_KLEIN_TE_DUMP")) |dir| {
            try dumpStepF32(allocator, std.mem.span(dir), "embeds", 0, embeds);
            // Tokenization is the cheapest explanation for a conditioning
            // mismatch, so the ids ship with the tensor.
            const as_f32 = try allocator.alloc(f32, valid);
            defer allocator.free(as_f32);
            for (as_f32, ids.ids[0..valid]) |*d, t| d.* = @floatFromInt(t);
            try dumpStepF32(allocator, std.mem.span(dir), "ids", 0, as_f32);
        }
        return embeds;
    }

    fn encodeResident(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        embeds: []f32,
        ids: []const u32,
        valid: usize,
    ) !void {
        const t = &self.te_res.?;
        try t.encode(
            io,
            allocator,
            embeds,
            ids,
            &self.te_store,
            self.te_index,
            self.text,
            &stacked_layers,
            valid,
        );
    }

    /// Shared request geometry for the single and batched generate paths.
    const Geom = struct {
        gw: u32,
        gh: u32,
        img_tokens: usize,
        latent_len: usize,
        sample_len: usize,
    };

    fn geomFor(request: Request) !Geom {
        // Both sides in multiples of 32: the 2x2 patchify needs an even
        // latent grid on each axis (a 16-multiple that is odd in grid units
        // dies inside the kernels instead of here).
        if (request.width % 32 != 0 or request.height % 32 != 0) return error.InvalidImageSize;
        const gw = request.width / 16;
        const gh = request.height / 16;
        const img_tokens = std.math.mul(usize, @as(usize, gw), @as(usize, gh)) catch return error.InvalidImageSize;
        if (img_tokens > max_image_tokens) return error.InvalidImageSize;
        const latent_len = std.math.mul(usize, img_tokens, 128) catch return error.InvalidImageSize;
        const pixel_count = std.math.mul(usize, @as(usize, request.width), @as(usize, request.height)) catch return error.InvalidImageSize;
        const sample_len = std.math.mul(usize, pixel_count, 3) catch return error.InvalidImageSize;
        return .{ .gw = gw, .gh = gh, .img_tokens = img_tokens, .latent_len = latent_len, .sample_len = sample_len };
    }

    /// Decode one image's latents through the shared streamed-VAE path into a
    /// linear RGB sample buffer. Caller owns the returned slice (and runs the
    /// vrgb quantization, which sits outside/inside the vae metrics lap
    /// depending on the caller's historical boundary).
    fn decodeSample(
        self: *Runtime,
        allocator: std.mem.Allocator,
        latents: []const f32,
        g: Geom,
    ) ![]f32 {
        metrics.memtrace("klein-pre-vae");
        const prepared = try zflux2_vae.prepare(allocator, latents, g.gh, g.gw, &self.vae);
        defer allocator.free(prepared);
        const cp: ?*mconv.Context = if (self.conv) |*c| c else null;
        // Lazily build the persistent streamed-VAE ctx. Explicit
        // ZDRAW_VAE_STREAM=0 still routes through the untiled fallback.
        const vstream = try vdecode.ensureStream(cp, &self.vae_stream);
        const sample = try allocator.alloc(f32, g.sample_len);
        errdefer allocator.free(sample);
        // The resident DiT pool is idle for the whole decode (the denoise
        // waited on its last batch), so it lends the mid-attention its
        // scratch instead of the chain pool growing a second set.
        const offer = if (self.res) |*rc| rc.offer() else null;
        defer if (self.res) |*rc| rc.endOffer();
        try vdecode.run(cp, self.linPtr(), &self.attn, vstream, allocator, sample, prepared, self.views, .{
            .height = g.gh * 2,
            .width = g.gw * 2,
        }, offer);
        metrics.memtrace("klein-post-vae");
        return sample;
    }

    /// Classifier-free guidance: the unconditional pass over the full
    /// [image ; reference] sequence (diffusers concatenates the reference
    /// latents in both branches), combined into the image part of `v`.
    fn guided(
        self: *Runtime,
        allocator: std.mem.Allocator,
        v: []f32,
        v_neg: []f32,
        x_full: []const f32,
        neg_embeds: []const f32,
        rope: zflux2_dit.Rope,
        t: f32,
        guidance: f32,
    ) !void {
        if (self.res) |*rc| {
            try zflux2_res.forward(rc, v_neg, x_full, neg_embeds, &self.loaded, rope, t, .{});
        } else {
            const lin = self.linPtr();
            const at = &self.attn;
            const ld = &self.loaded;
            try zflux2_dit.forward(allocator, lin, at, v_neg, x_full, neg_embeds, ld, rope, t);
        }
        for (v, v_neg[0..v.len]) |*vi, nvi| vi.* = nvi + guidance * (vi.* - nvi);
    }

    /// The transformer's sequence for one render: the rope table and the
    /// [image ; reference] latent/velocity buffers (reference absent = the
    /// image alone).
    const Seq = struct {
        rope: zflux2_dit.Rope,
        xf: []f32,
        vf: []f32,
        ref: ?[]f32,

        fn deinit(self: *Seq, allocator: std.mem.Allocator) void {
            self.rope.deinit(allocator);
            allocator.free(self.xf);
            allocator.free(self.vf);
            if (self.ref) |r| allocator.free(r);
        }
    };

    fn sequence(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
        g: Geom,
        latent_len: usize,
    ) !Seq {
        // The reference encode (or its memo hit) gets its own lap: ~8 s at
        // 1024 on a miss. The enclosing klein-denoise lap still spans it.
        const t0 = metrics.now();
        const ref = try self.encodeRef(io, allocator, request, g);
        errdefer if (ref) |r| allocator.free(r);
        _ = metrics.lap("klein-ref", t0);
        const ref_tokens: usize = if (ref) |r| r.len / 128 else 0;
        const rh: usize = if (ref != null) g.gh else 0;
        const rw: usize = if (ref != null) g.gw else 0;
        var rope = try zflux2_dit.buildRopeRef(allocator, g.gh, g.gw, rh, rw, rope_theta);
        errdefer rope.deinit(allocator);
        const xf = try allocator.alloc(f32, latent_len + ref_tokens * 128);
        errdefer allocator.free(xf);
        const vf = try allocator.alloc(f32, xf.len);
        errdefer allocator.free(vf);
        if (ref) |r| @memcpy(xf[latent_len..], r);
        return .{ .rope = rope, .xf = xf, .vf = vf, .ref = ref };
    }

    /// The reference image (instruction editing) as packed DiT-space tokens
    /// at the render's grid, or null. Caller frees.
    fn encodeRef(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
        g: Geom,
    ) !?[]f32 {
        if (request.ref_image.len == 0) return null;
        const w: usize = request.width;
        const h: usize = request.height;
        const key = try refKey(io, allocator, request.ref_image, w, h);
        defer allocator.free(key);
        if (self.refCached(key)) |z| return try allocator.dupe(f32, z);
        const stage = try progress.begin(io, allocator, "encoding reference");
        const rgb = try init_image.loadRgb(allocator, request.ref_image, w, h);
        defer allocator.free(rgb);
        const cp: ?*mconv.Context = if (self.conv) |*c| c else null;
        const lp = self.linPtr();
        const z = try init_image.encodePacked(allocator, lp, cp, &self.attn, &self.vae, rgb, w, h);
        errdefer allocator.free(z);
        if (z.len != g.latent_len) return error.InvalidImageSize;
        try progress.done(io, allocator, stage);
        try self.refStore(key, z);
        return z;
    }

    /// The memo key: path, size and mtime of the photo, and the render size.
    /// A missing or unreadable file keys on the path alone (the encode then
    /// fails loudly as before).
    fn refKey(
        io: std.Io,
        allocator: std.mem.Allocator,
        path: []const u8,
        w: usize,
        h: usize,
    ) ![]u8 {
        var size: u64 = 0;
        var mtime: i96 = 0;
        if (std.Io.Dir.cwd().openFile(io, path, .{})) |file| {
            defer file.close(io);
            if (file.stat(io)) |st| {
                size = st.size;
                mtime = st.mtime.nanoseconds;
            } else |_| {}
        } else |_| {}
        return std.fmt.allocPrint(allocator, "{s}|{d}|{d}|{d}|{d}", .{ path, size, mtime, w, h });
    }

    fn refCached(self: *Runtime, key: []const u8) ?[]const f32 {
        if (!util.envFlag("ZDRAW_REF_CACHE", true)) return null;
        const z = self.ref_tokens orelse return null;
        const k = self.ref_key orelse return null;
        if (std.hash.Wyhash.hash(0, key) != self.ref_hash) return null;
        if (!std.mem.eql(u8, k, key)) return null;
        return z;
    }

    fn refStore(self: *Runtime, key: []const u8, z: []const f32) !void {
        if (!util.envFlag("ZDRAW_REF_CACHE", true)) return;
        if (self.ref_tokens) |old| self.alloc.free(old);
        self.ref_tokens = null;
        if (self.ref_key) |old| self.alloc.free(old);
        self.ref_key = null;
        self.ref_tokens = try self.alloc.dupe(f32, z);
        self.ref_key = try self.alloc.dupe(u8, key);
        self.ref_hash = std.hash.Wyhash.hash(0, key);
    }

    /// What img2img leaves behind for the loop: the start step, and for a
    /// masked edit the original latent, the untouched noise and the mask.
    const Init = struct {
        first_step: usize = 0,
        z0: ?[]f32 = null,
        noise0: ?[]f32 = null,
        mask: ?[]f32 = null,

        fn deinit(self: *Init, allocator: std.mem.Allocator) void {
            if (self.z0) |b| allocator.free(b);
            if (self.noise0) |b| allocator.free(b);
            if (self.mask) |b| allocator.free(b);
        }

        fn repaint(self: Init, x: []f32, sigma_next: f32) void {
            if (self.mask) |m| init_image.repaint(x, self.z0.?, self.noise0.?, m, sigma_next);
        }
    };

    /// img2img start: encode the init image on the runtime's contexts, mix
    /// it with the seeded noise at the start step's sigma.
    fn applyInit(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
        x: []f32,
        sched: *zflux2_schedule.Schedule,
        g: Geom,
    ) !Init {
        if (request.init_image.len == 0) return .{};
        const stage = try progress.begin(io, allocator, "encoding init image");
        const w: usize = request.width;
        const h: usize = request.height;
        const rgb = try init_image.loadRgb(allocator, request.init_image, w, h);
        defer allocator.free(rgb);
        const cp: ?*mconv.Context = if (self.conv) |*c| c else null;
        const lp = self.linPtr();
        const z0 = try init_image.encodePacked(allocator, lp, cp, &self.attn, &self.vae, rgb, w, h);
        errdefer allocator.free(z0);
        if (z0.len != g.latent_len) return error.InvalidImageSize;
        // Strength is the noise fraction the sampler starts from: the whole
        // schedule is squeezed into [strength, 0], so 0.2 is a gentle pass
        // and 1 ignores the image. 0 returns the image (no steps).
        const strength = std.math.clamp(request.strength, 0.0, 1.0);
        var out = Init{ .first_step = if (strength <= 0.0) request.steps else 0 };
        if (strength > 0.0 and strength < 1.0) zflux2_schedule.scaleTo(sched, strength);
        if (request.mask.len > 0) {
            out.noise0 = try allocator.dupe(f32, x);
            out.mask = try init_image.loadMask(allocator, request.mask, w, h);
            out.z0 = z0;
        } else {
            defer allocator.free(z0);
            init_image.mix(x, z0, if (strength <= 0.0) 0.0 else sched.sigmas[0]);
            try progress.done(io, allocator, stage);
            return out;
        }
        init_image.mix(x, z0, if (strength <= 0.0) 0.0 else sched.sigmas[0]);
        try progress.done(io, allocator, stage);
        return out;
    }

    /// One phase boundary: the wall lap, the Metal counter delta and the
    /// MEMTRACE stage, in the order the laps have always been taken.
    fn phaseEnd(name: []const u8, lap: u64, mc: *metrics.Counters) u64 {
        const next = metrics.lap(name, lap);
        metrics.recordMetal(name, mc.*);
        mc.* = metrics.snapshot();
        metrics.memtrace(name);
        return next;
    }

    pub fn generate(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
    ) !Result {
        const g = try geomFor(request);
        // Concurrent renders corrupt each other (genlock.zig); serialize.
        genlock.acquire();
        defer genlock.release();

        var lap = metrics.now();
        // Per-phase Metal counters (dispatches, GPU-active ms) beside the wall
        // laps: the difference is the host-side / GPU-idle time inside a phase.
        var mc = metrics.snapshot();
        const tok_stage = try progress.begin(io, allocator, "encoding prompt");
        const embeds = try self.embedsFor(io, request.prompt);
        try progress.done(io, allocator, tok_stage);
        lap = phaseEnd("klein-encode", lap, &mc);

        // Seeded gaussian latents in the packed token space (zdraw's own RNG;
        // seeds are zdraw-stable, not diffusers-compatible). ZDRAW_LATENT_IN
        // injects a shared x0 instead (single-seed renders only).
        const x = try allocator.alloc(f32, g.latent_len);
        defer allocator.free(x);
        try zdenoise.initLatents(io, x, request.seed);
        if (request.field.active()) try seed_field.fill(allocator, x, request.field);

        // Instruction editing: the reference's tokens follow the image's in
        // one buffer; the transformer sees both, the sampler updates only the
        // image half.
        var seq = try self.sequence(io, allocator, request, g, x.len);
        defer seq.deinit(allocator);
        const rope = seq.rope;
        var sched = try zflux2_schedule.make(allocator, g.img_tokens, request.steps);
        defer sched.deinit(allocator);

        // img2img: the image's latent, noised to the strength (a squeezed schedule).
        var start = try self.applyInit(io, allocator, request, x, &sched, g);
        defer start.deinit(allocator);
        const first_step = start.first_step;

        const denoise = try progress.begin(io, allocator, "sampling latents");
        const v = try allocator.alloc(f32, x.len);
        defer allocator.free(v);
        // Diagnostic only: dump resident/non-resident step tensors for drift gates.
        const dump_steps = std.c.getenv("ZDRAW_KLEIN_DUMP_STEPS");
        // A3 (ZDRAW_KLEIN_XRES, default off): latents stay device-resident
        // across steps — the Euler update runs on-GPU at the start of the next
        // forward, removing the per-step v readback / CPU axpy / x re-upload.
        // Disabled under DUMP_STEPS (the drift instrument needs CPU tensors).
        // CFG (base variants): a second forward on the unconditional
        // conditioning per step, combined before the Euler update. XRES keeps
        // v on-device, so it cannot host the CPU-side combine.
        const cfg_on = request.guidance > 1.0;
        const neg_embeds = if (cfg_on) try self.negEmbedsFor(io) else embeds;
        const v_neg: []f32 = if (cfg_on) try allocator.alloc(f32, seq.vf.len) else &.{};
        defer if (cfg_on) allocator.free(v_neg);
        const xres = self.res != null and dump_steps == null and !cfg_on and seq.ref == null and
            first_step == 0 and util.envFlag("ZDRAW_KLEIN_XRES", false);
        for (first_step..request.steps) |step| {
            if (self.res) |*rc| {
                if (xres) {
                    const dt_in: ?f32 = if (step > 0) try zflux2_schedule.delta(sched, step - 1) else null;
                    try zflux2_res.forward(rc, v, x, embeds, &self.loaded, rope, sched.timesteps[step], .{ .dt_in = dt_in, .read_v = false });
                    try progress.step(io, allocator, step + 1, request.steps);
                    continue;
                }
                @memcpy(seq.xf[0..x.len], x);
                const ts = sched.timesteps[step];
                try zflux2_res.forward(rc, seq.vf, seq.xf, embeds, &self.loaded, rope, ts, .{});
                @memcpy(v, seq.vf[0..x.len]);
            } else {
                @memcpy(seq.xf[0..x.len], x);
                const lin = self.linPtr();
                const ts = sched.timesteps[step];
                try zflux2_dit.forward(allocator, lin, &self.attn, seq.vf, seq.xf, embeds, &self.loaded, rope, ts);
                @memcpy(v, seq.vf[0..x.len]);
            }
            if (cfg_on) {
                // seq.xf still holds this step's [image ; reference] input.
                const t = sched.timesteps[step];
                try self.guided(allocator, v, v_neg, seq.xf, neg_embeds, rope, t, request.guidance);
            }
            if (dump_steps) |dir| try dumpStepF32(allocator, std.mem.span(dir), "v", step, v);
            const dt = try zflux2_schedule.delta(sched, step);
            for (x, v) |*xi, vi| xi.* += dt * vi;
            start.repaint(x, sched.sigmas[step + 1]);
            if (dump_steps) |dir| try dumpStepF32(allocator, std.mem.span(dir), "x", step, x);
            const pd = preview.dir();
            const sn = sched.sigmas[step + 1];
            const n: usize = request.steps;
            try preview.write(io, allocator, pd, x, v, sn, g.gh, g.gw, step + 1, n, 0);
            try progress.step(io, allocator, step + 1, request.steps);
            if (step == first_step) metrics.memtrace("klein-step-first");
            if (step + 1 == request.steps) metrics.memtrace("klein-step-last");
        }
        if (xres) {
            const rc = &self.res.?;
            try zflux2_res.finishEuler(rc, x, try zflux2_schedule.delta(sched, request.steps - 1));
        }
        try progress.done(io, allocator, denoise);
        lap = phaseEnd("klein-denoise", lap, &mc);
        if (self.res) |*rc| rc.reportTrace();

        const vae_stage = try progress.begin(io, allocator, "decoding image");
        const sample = try self.decodeSample(allocator, x, g);
        defer allocator.free(sample);
        try progress.done(io, allocator, vae_stage);
        _ = phaseEnd("klein-vae", lap, &mc);

        const pixels = try allocator.alloc(u8, sample.len);
        errdefer allocator.free(pixels);
        try vrgb.run(pixels, sample, request.width, request.height);
        return .{ .pixels = pixels, .width = request.width, .height = request.height };
    }

    /// Batched multi-seed generation: one wider denoise for all seeds (weights
    /// fetched once per layer for the whole batch), then a serial VAE decode
    /// per image. The prompt is shared (one Qwen3 encode). Falls back to
    /// serial generates when the resident path is unavailable or guidance > 1
    /// (forwardBatch has no CFG pass); both activation modes batch. Caller
    /// owns the returned slice and each pixels.
    pub fn generateMulti(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
        seeds: []const u64,
    ) ![]Result {
        if (seeds.len == 0) return error.InvalidImageSize;
        const results = try allocator.alloc(Result, seeds.len);
        errdefer allocator.free(results);
        // forwardBatch has no CFG pass, so a guided batch would silently
        // skip CFG; guidance > 1 stays serial. Both activation modes batch.
        const batch_ok = self.res != null and seeds.len > 1 and
            request.guidance <= 1.0 and request.ref_image.len == 0 and
            std.c.getenv("ZDRAW_KLEIN_DUMP_STEPS") == null;
        if (!batch_ok) {
            if (seeds.len > 1) try progress.event(io, allocator, "seeds run serially (batched denoise requires guidance 1.0 and the resident path)");
            var done: usize = 0;
            errdefer for (results[0..done]) |r| allocator.free(r.pixels);
            for (seeds, 0..) |seed, i| {
                var req = request;
                req.seed = seed;
                results[i] = try self.generate(io, allocator, req);
                done += 1;
            }
            return results;
        }

        // Concurrent renders corrupt each other (genlock.zig); the serial
        // fallback locks inside generate, so only the batch arm locks here.
        genlock.acquire();
        defer genlock.release();
        const g = try geomFor(request);
        const nb = seeds.len;

        var lap = metrics.now();
        const tok_stage = try progress.begin(io, allocator, "encoding prompt");
        const embeds = try self.embedsFor(io, request.prompt);
        try progress.done(io, allocator, tok_stage);
        lap = metrics.lap("klein-encode", lap);

        const x = try allocator.alloc(f32, nb * g.latent_len);
        defer allocator.free(x);
        for (seeds, 0..) |seed, i| {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            for (x[i * g.latent_len ..][0..g.latent_len]) |*v| v.* = rng.floatNorm(f32);
        }

        var rope = try zflux2_dit.buildRope(allocator, g.gh, g.gw, rope_theta);
        defer rope.deinit(allocator);
        const sched = try zflux2_schedule.make(allocator, g.img_tokens, request.steps);
        defer sched.deinit(allocator);

        const denoise = try progress.begin(io, allocator, "sampling latents");
        const v = try allocator.alloc(f32, x.len);
        defer allocator.free(v);
        // The runtime's activation mode is left exactly as configured (a mode
        // override here would leak into later single-image generates on this
        // Runtime).
        const rc = &self.res.?;
        for (0..request.steps) |step| {
            try zflux2_res.forwardBatch(rc, allocator, v, x, embeds, &self.loaded, rope, sched.timesteps[step], nb);
            const dt = try zflux2_schedule.delta(sched, step);
            for (x, v) |*xi, vi| xi.* += dt * vi;
            for (0..nb) |i| {
                const seg = x[i * g.latent_len ..][0..g.latent_len];
                const vseg = v[i * g.latent_len ..][0..g.latent_len];
                const sn = sched.sigmas[step + 1];
                const n: usize = request.steps;
                try preview.write(
                    io,
                    allocator,
                    preview.dir(),
                    seg,
                    vseg,
                    sn,
                    g.gh,
                    g.gw,
                    step + 1,
                    n,
                    i,
                );
            }
            try progress.step(io, allocator, step + 1, request.steps);
        }
        try progress.done(io, allocator, denoise);
        lap = metrics.lap("klein-denoise", lap);

        const vae_stage = try progress.begin(io, allocator, "decoding image");
        var done: usize = 0;
        errdefer for (results[0..done]) |r| allocator.free(r.pixels);
        for (0..nb) |i| {
            const sample = try self.decodeSample(allocator, x[i * g.latent_len ..][0..g.latent_len], g);
            defer allocator.free(sample);
            const pixels = try allocator.alloc(u8, sample.len);
            errdefer allocator.free(pixels);
            try vrgb.run(pixels, sample, request.width, request.height);
            results[i] = .{ .pixels = pixels, .width = request.width, .height = request.height };
            done += 1;
        }
        try progress.done(io, allocator, vae_stage);
        lap = metrics.lap("klein-vae", lap);
        return results;
    }
};

fn dumpStepF32(
    allocator: std.mem.Allocator,
    dir: []const u8,
    tag: []const u8,
    step: usize,
    values: []const f32,
) !void {
    if (dir.len == 0) return;
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}_step{d}.bin", .{ dir, tag, step });
    defer allocator.free(path);
    const zpath = try allocator.dupeZ(u8, path);
    defer allocator.free(zpath);
    const file = std.c.fopen(zpath.ptr, "wb") orelse return error.AccessDenied;
    defer _ = std.c.fclose(file);
    const wrote = std.c.fwrite(
        @as([*]const u8, @ptrCast(values.ptr)),
        @sizeOf(f32),
        values.len,
        file,
    );
    if (wrote != values.len) return error.WriteFailed;
}
