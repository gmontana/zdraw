//! GPU-resident Qwen3 stacked text encode for FLUX.2 Klein.
//!
//! The per-op encoder path pays six synchronous command buffers per layer
//! (q/k/v/attention/o/FFN, each commit + wait + readback) with CPU norms,
//! qk-norms, rope and residuals interleaved: 162 command buffers and ~530 ms
//! of host sync per 27-layer encode. This module encodes the whole stacked
//! encode into ONE resident batch (mirroring zflux2_resident): the only
//! boundary crossings are the embedding upload (~5 MB) and one snapshot
//! readback (~16 MB). The per-op route (qwen_encoder.run) stays byte-
//! untouched as the oracle and the ZDRAW_KLEIN_TE_RES=0 fallback; Z-Image's
//! penultimate path never comes here.
//!
//! Numerics: GEMMs use the same ours16-bf16/gemm_half kernels as the per-op
//! GEMM route (same f16 A-staging class; acceptance is cos-no-worse, per the
//! pinned route-agreement bound). Norm reductions run sequentially per
//! thread in the new kernel, matching the CPU accumulation order; rope
//! cos/sin tables are CPU-built, so the rotation constants are bit-identical
//! to the per-op path.

const std = @import("std");

const env = @import("../runtime/env.zig");
const mattn = @import("../metal/mattn.zig");
const mblock_c = @import("../metal/mblock_c.zig");
const mblock_shader = @import("../metal/mblock_shader.zig");
const mbuffer = @import("../metal/mbuffer.zig");
const metal_c = @import("../metal/metal_c.zig");
const mgemm_shader = @import("../metal/mgemm_shader.zig");
const mpipe = @import("../metal/mpipe.zig");
const mres_util = @import("../metal/mres_util.zig");
const mswiglu_shader = @import("../metal/mswiglu_shader.zig");
const progress = @import("../cli/progress.zig");
const qattn = @import("qwen_attn.zig");
const qenc = @import("qwen_encoder.zig");
const qlayer = @import("qwen_layer.zig");
const qnames = @import("qwen_names.zig");
const qshader = @import("qwen_res_shader.zig");
const qtext = @import("qwen_text.zig");
const rope = @import("rope.zig");
const shards = @import("../pack/shards.zig");
const tensor = @import("../pack/tensor.zig");
const weight_index = @import("../pack/weight_index.zig");
const zconfig = @import("../zimage/zimage_config.zig");
const packed_w = @import("../klein/zflux2_packed.zig");
const glue_src = @import("../klein/zflux2_glue_shader.zig");

/// Uploaded via cbytes through the generic glue entry; the ObjC layer never
/// declares this struct, so there is no copy to drift (the AttnParams
/// incident class). The explicit pad puts w_offset at byte 24.
const QRopeParams = extern struct {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    w_dtype: u32,
    eps: f32,
    pad: u32 = 0,
    w_offset: u64,

    comptime {
        std.debug.assert(@sizeOf(QRopeParams) == 32);
        std.debug.assert(@offsetOf(QRopeParams, "w_offset") == 24);
    }
};

const AxpyParams = extern struct {
    count: u32,
    dt: f32,
};

const Pipelines = struct {
    qrms: *anyopaque,
    norm: *anyopaque,
    swiglu: *anyopaque,
    copy: *anyopaque,
    axpy: *anyopaque,
    gemm: *anyopaque,

    fn init(device: *anyopaque) !Pipelines {
        var err: [1024]u8 = undefined;
        const qrms = try mpipe.required(device, qshader.src, "qrms_rope", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(qrms);
        const norm = try mpipe.required(device, mblock_shader.block, "block_norm_scale", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(norm);
        const swiglu = try mpipe.required(device, mswiglu_shader.swiglu, "swiglu_f32", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(swiglu);
        const copy = try mpipe.required(device, glue_src.src, "kcopy", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(copy);
        const axpy = try mpipe.required(device, glue_src.src, "kaxpy", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(axpy);
        const gemm = try mpipe.required(device, mgemm_shader.gemm, "gemm_half", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(gemm);
        return .{
            .qrms = qrms,
            .norm = norm,
            .swiglu = swiglu,
            .copy = copy,
            .axpy = axpy,
            .gemm = gemm,
        };
    }

    fn deinit(self: *Pipelines) void {
        metal_c.zdraw_metal_release_pipeline(self.gemm);
        metal_c.zdraw_metal_release_pipeline(self.axpy);
        metal_c.zdraw_metal_release_pipeline(self.copy);
        metal_c.zdraw_metal_release_pipeline(self.swiglu);
        metal_c.zdraw_metal_release_pipeline(self.norm);
        metal_c.zdraw_metal_release_pipeline(self.qrms);
        self.* = undefined;
    }
};

/// Encoder pool identity: every buffer is sized from these dims (plus the
/// rope tables from tokens * head_dim / 2), so the pool is reusable iff all
/// match. theta is part of the identity because the tables bake it in.
const PoolShape = struct {
    tokens: usize,
    hidden: usize,
    inner: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    groups: usize,
    theta: f32,
};

const Pool = struct {
    shape: PoolShape,
    state: mbuffer.Buffer, // [tokens, hidden] residual stream
    normed: mbuffer.Buffer, // [tokens, hidden] norm output / GEMM A
    q: mbuffer.Buffer, // [tokens, heads * head_dim]
    k: mbuffer.Buffer, // [tokens, kv_heads * head_dim]
    v: mbuffer.Buffer,
    mix: mbuffer.Buffer, // attention output, heads * head_dim wide
    delta: mbuffer.Buffer, // attn_out AND mlp_out (serialized, hazard-tracked)
    gate: mbuffer.Buffer, // [tokens, inner]
    up: mbuffer.Buffer,
    rope_cos: mbuffer.Buffer, // [tokens, head_dim/2], CPU-built
    rope_sin: mbuffer.Buffer,
    snaps: mbuffer.Buffer, // [groups, tokens, hidden] snapshot staging

    fn deinit(self: *Pool) void {
        inline for (@typeInfo(Pool).@"struct".fields) |f| {
            if (f.type == mbuffer.Buffer) @field(self, f.name).deinit();
        }
        self.* = undefined;
    }
};

/// Temp binds (views without a backing mmap source) must outlive the open
/// batch, so they are parked here until batch end. Production encoder weights
/// are shard views and always bind no-copy: this list stays empty outside
/// the in-file tests.
const max_temps = 64;

pub const Ctx = struct {
    allocator: std.mem.Allocator,
    attn: mattn.Context,
    pipelines: Pipelines,
    wbind: mbuffer.Cache,
    pool: ?Pool = null,
    batch: ?*anyopaque = null,
    temps: [max_temps]?mbuffer.Buffer = @splat(null),
    temps_len: usize = 0,
    // ZDRAW_GEMM_BF16 (default on): try the fast staged GEMM first; a decline
    // falls back to gemm_half in the same batch, like the per-op route.
    fast_gemm: bool,

    pub fn init(allocator: std.mem.Allocator) !Ctx {
        // Own attention context pinned to the rows kernel: it is the only
        // kernel that honors AttnParams.valid (the padding mask).
        var attn = try mattn.Context.initKernel(.rows);
        errdefer attn.deinit();
        var pipelines = try Pipelines.init(attn.device);
        errdefer pipelines.deinit();
        const wbind = try mbuffer.Cache.init(attn.device);
        return .{
            .allocator = allocator,
            .attn = attn,
            .pipelines = pipelines,
            .wbind = wbind,
            .fast_gemm = env.flag("ZDRAW_GEMM_BF16", true),
        };
    }

    pub fn deinit(self: *Ctx) void {
        if (self.batch) |bt| {
            _ = metal_c.zdraw_metal_batch_end(bt);
            self.batch = null;
        }
        self.dropTemps();
        if (self.pool) |*p| p.deinit();
        self.wbind.deinit();
        self.pipelines.deinit();
        self.attn.deinit();
        self.* = undefined;
    }

    fn dropTemps(self: *Ctx) void {
        for (self.temps[0..self.temps_len]) |*slot| {
            if (slot.*) |*buf| buf.deinit();
            slot.* = null;
        }
        self.temps_len = 0;
    }

    /// Bind a weight view for use inside the open batch. No-copy for shard
    /// views; a temp copy is parked until batch end.
    fn bind(self: *Ctx, v: tensor.View) !mbuffer.Bind {
        var temp: ?mbuffer.Buffer = null;
        const b = try self.wbind.bindView(v, &temp);
        if (temp) |buf| {
            if (self.temps_len == max_temps) {
                var owned = buf;
                owned.deinit();
                return error.MetalCacheFull;
            }
            self.temps[self.temps_len] = buf;
            self.temps_len += 1;
        }
        return b;
    }

    fn ensurePool(self: *Ctx, shape: PoolShape) !*Pool {
        if (self.pool != null and std.meta.eql(self.pool.?.shape, shape)) return &self.pool.?;
        if (self.pool) |*p| p.deinit();
        self.pool = null;
        const dev = self.attn.device;
        const t = shape.tokens;
        const q_dim = shape.heads * shape.head_dim;
        const kv_dim = shape.kv_heads * shape.head_dim;
        const hd = shape.head_dim / 2;
        // CPU-built rope tables: bit-identical transcendentals to the per-op
        // route (rope.table is the shared truth).
        const cos_host = try self.allocator.alloc(f32, t * hd);
        defer self.allocator.free(cos_host);
        const sin_host = try self.allocator.alloc(f32, t * hd);
        defer self.allocator.free(sin_host);
        for (0..t) |pos| {
            try rope.table(
                cos_host[pos * hd ..][0..hd],
                sin_host[pos * hd ..][0..hd],
                pos,
                shape.theta,
            );
        }
        const pool = Pool{
            .shape = shape,
            .state = try empty(dev, t * shape.hidden),
            .normed = try empty(dev, t * shape.hidden),
            .q = try empty(dev, t * q_dim),
            .k = try empty(dev, t * kv_dim),
            .v = try empty(dev, t * kv_dim),
            .mix = try empty(dev, t * q_dim),
            .delta = try empty(dev, t * shape.hidden),
            .gate = try empty(dev, t * shape.inner),
            .up = try empty(dev, t * shape.inner),
            .rope_cos = try mbuffer.Buffer.fromBytes(dev, std.mem.sliceAsBytes(cos_host)),
            .rope_sin = try mbuffer.Buffer.fromBytes(dev, std.mem.sliceAsBytes(sin_host)),
            .snaps = try empty(dev, shape.groups * t * shape.hidden),
        };
        self.pool = pool;
        return &self.pool.?;
    }

    fn glue(
        self: *Ctx,
        pipe: *anyopaque,
        bufs: [5]?*anyopaque,
        offsets: ?[*]const usize,
        cbytes: ?*const anyopaque,
        cbytes_len: usize,
        cbytes_index: u32,
        grid: usize,
        threads: usize,
        as_groups: u32,
    ) !void {
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (metal_c.zdraw_metal_run_glue_enc(
            bt,
            pipe,
            bufs[0],
            bufs[1],
            bufs[2],
            bufs[3],
            bufs[4],
            offsets,
            cbytes,
            cbytes_len,
            cbytes_index,
            grid,
            threads,
            as_groups,
        ) != 0) return error.MetalDispatchFailed;
    }

    /// One encoder GEMM into the open batch: fast staged kernel when the
    /// shape and dtype admit it, gemm_half otherwise (both decode bf16).
    fn teGemm(
        self: *Ctx,
        a: *anyopaque,
        w: tensor.View,
        c_buf: *anyopaque,
        m: usize,
        k: usize,
        n: usize,
    ) !void {
        if (w.dtype == .u8 and w.source != null) {
            return self.teGemmPacked(a, w, c_buf, m, k, n);
        }
        const b = try self.bind(w);
        const params = metal_c.GemmParams{
            .m = try mres_util.toU32(m),
            .k = try mres_util.toU32(k),
            .n = try mres_util.toU32(n),
            .dtype = try mres_util.dtype(w.dtype),
            .mode = 2,
            .weight_offset = try mres_util.toU64(b.offset),
        };
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (self.fast_gemm and ours16Fits(&params)) {
            const rc = metal_c.zdraw_metal_run_gemm_ours16_enc(bt, a, b.handle, c_buf, &params, 0, 0);
            if (rc == 0) return;
        }
        if (metal_c.zdraw_metal_run_gemm_enc(
            bt,
            self.pipelines.gemm,
            a,
            b.handle,
            c_buf,
            &params,
        ) != 0) {
            return error.MetalDispatchFailed;
        }
    }

    /// A text-pack linear (the 4-bit `.u8` marker view): no-copy bind of
    /// the codes and scales, then the f32-A split-scales W4 kernel the Klein
    /// resident route uses, in the same batch.
    fn teGemmPacked(
        self: *Ctx,
        a: *anyopaque,
        w: tensor.View,
        c_buf: *anyopaque,
        m: usize,
        k: usize,
        n: usize,
    ) !void {
        if (w.shape.len != 2 or w.packed_bits != 4) return error.InvalidDType;
        if (w.shape[0] != n or w.shape[1] != k) return error.InvalidShape;
        const b = try self.bind(w);
        const params = metal_c.GemmParams{
            .m = try mres_util.toU32(m),
            .k = try mres_util.toU32(k),
            .n = try mres_util.toU32(n),
            .dtype = packed_w.dtype_w4,
            .mode = 2,
            .weight_offset = try mres_util.toU64(b.offset),
        };
        const scales_off = b.offset + try packed_w.scalesBase(4, n, k);
        const bt = self.batch orelse return error.MetalDispatchFailed;
        const rc = packed_w.runEnc(
            packed_w.dtype_w4,
            bt,
            a,
            b.handle,
            c_buf,
            &params,
            0,
            0,
            scales_off,
        );
        if (rc != 0) return error.MetalDispatchFailed;
    }

    /// Weighted RMSNorm over hidden (block_norm_scale, no extra scale vector).
    fn normScale(
        self: *Ctx,
        w: tensor.View,
        in_h: *anyopaque,
        out_h: *anyopaque,
        tokens: usize,
        hidden: usize,
        eps: f32,
    ) !void {
        const b = try self.bind(w);
        const params = mblock_c.Params{
            .tokens = try mres_util.toU32(tokens),
            .hidden = try mres_util.toU32(hidden),
            .dtype = try mres_util.dtype(w.dtype),
            .has_scale = 0,
            .eps = eps,
            .weight_offset = try mres_util.toU64(b.offset),
        };
        try self.glue(
            self.pipelines.norm,
            .{ in_h, b.handle, self.wbind.zero, out_h, null },
            null,
            &params,
            @sizeOf(mblock_c.Params),
            4,
            tokens,
            256,
            1,
        );
    }

    /// Fused per-head weighted RMSNorm + Qwen split-half rope, in place.
    fn qrmsRope(
        self: *Ctx,
        p: *Pool,
        w: tensor.View,
        buf: *anyopaque,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) !void {
        const b = try self.bind(w);
        const params = QRopeParams{
            .tokens = try mres_util.toU32(tokens),
            .heads = try mres_util.toU32(heads),
            .head_dim = try mres_util.toU32(head_dim),
            .w_dtype = try mres_util.dtype(w.dtype),
            .eps = eps,
            .w_offset = try mres_util.toU64(b.offset),
        };
        try self.glue(
            self.pipelines.qrms,
            .{ buf, b.handle, p.rope_cos.handle, p.rope_sin.handle, null },
            null,
            &params,
            @sizeOf(QRopeParams),
            4,
            tokens * heads,
            256,
            0,
        );
    }

    /// Causal masked attention on the pooled q/k/v. Only attention_rows reads
    /// AttnParams.valid, so any other pick is refused rather than silently
    /// dropping the padding mask.
    fn attention(self: *Ctx, p: *Pool, cfg: qattn.Config) !void {
        const picked = self.attn.pick(cfg.tokens, cfg.head_dim);
        if (picked.kernel != .rows) return error.UnsupportedShape;
        const params = metal_c.AttnParams{
            .tokens = try mres_util.toU32(cfg.tokens),
            .heads = try mres_util.toU32(cfg.heads),
            .kv_heads = try mres_util.toU32(cfg.kv_heads),
            .head_dim = try mres_util.toU32(cfg.head_dim),
            .causal = if (cfg.causal) 1 else 0,
            .valid = try mres_util.toU32(cfg.valid),
        };
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (metal_c.zdraw_metal_run_attention_enc(
            bt,
            picked.pipeline,
            p.q.handle,
            p.k.handle,
            p.v.handle,
            p.mix.handle,
            &params,
            @intFromEnum(picked.kernel),
            picked.threads,
        ) != 0) {
            return error.MetalDispatchFailed;
        }
    }

    fn axpy(self: *Ctx, state: *anyopaque, delta: *anyopaque, count: usize) !void {
        const params = AxpyParams{ .count = try mres_util.toU32(count), .dt = 1.0 };
        try self.glue(
            self.pipelines.axpy,
            .{ state, delta, null, null, null },
            null,
            &params,
            @sizeOf(AxpyParams),
            2,
            count,
            256,
            0,
        );
    }

    fn swiglu(self: *Ctx, gate: *anyopaque, up: *anyopaque, count: usize) !void {
        const c: u32 = try mres_util.toU32(count);
        try self.glue(
            self.pipelines.swiglu,
            .{ gate, up, null, null, null },
            null,
            &c,
            @sizeOf(u32),
            2,
            count,
            256,
            0,
        );
    }

    /// kcopy the live state into snapshot slot `slot` (byte-offset write).
    fn snapshot(self: *Ctx, p: *Pool, slot: usize, count: usize) !void {
        const c: u32 = try mres_util.toU32(count);
        const offs = [5]usize{ 0, slot * count * 4, 0, 0, 0 };
        try self.glue(
            self.pipelines.copy,
            .{ p.state.handle, p.snaps.handle, null, null, null },
            &offs,
            &c,
            @sizeOf(u32),
            2,
            count,
            256,
            0,
        );
    }

    fn encodeLayer(self: *Ctx, p: *Pool, w: qlayer.Weights, cfg: qattn.Config) !void {
        const tokens = cfg.tokens;
        const hidden = cfg.hidden;
        const inner = p.shape.inner;
        const q_dim = cfg.heads * cfg.head_dim;
        const kv_dim = cfg.kv_heads * cfg.head_dim;
        try self.normScale(
            w.attn.norm,
            p.state.handle,
            p.normed.handle,
            tokens,
            hidden,
            cfg.norm_eps,
        );
        try self.teGemm(p.normed.handle, w.attn.q, p.q.handle, tokens, hidden, q_dim);
        try self.teGemm(p.normed.handle, w.attn.k, p.k.handle, tokens, hidden, kv_dim);
        try self.teGemm(p.normed.handle, w.attn.v, p.v.handle, tokens, hidden, kv_dim);
        try self.qrmsRope(
            p,
            w.attn.q_norm,
            p.q.handle,
            tokens,
            cfg.heads,
            cfg.head_dim,
            cfg.norm_eps,
        );
        try self.qrmsRope(
            p,
            w.attn.k_norm,
            p.k.handle,
            tokens,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.norm_eps,
        );
        try self.attention(p, cfg);
        try self.teGemm(p.mix.handle, w.attn.o, p.delta.handle, tokens, q_dim, hidden);
        try self.axpy(p.state.handle, p.delta.handle, tokens * hidden);
        try self.normScale(
            w.post_norm,
            p.state.handle,
            p.normed.handle,
            tokens,
            hidden,
            cfg.norm_eps,
        );
        try self.teGemm(p.normed.handle, w.gate, p.gate.handle, tokens, hidden, inner);
        try self.teGemm(p.normed.handle, w.up, p.up.handle, tokens, hidden, inner);
        try self.swiglu(p.gate.handle, p.up.handle, tokens * inner);
        try self.teGemm(p.gate.handle, w.down, p.delta.handle, tokens, inner, hidden);
        try self.axpy(p.state.handle, p.delta.handle, tokens * hidden);
    }

    /// The resident stacked encode: embed on CPU, one batched command buffer
    /// for all layers, one snapshot readback, per-token concat into `out`.
    /// Mirrors qwen_encoder.runStacked semantics exactly (layers 0..max(groups),
    /// snapshot after each listed block count, pad rows computed unmasked
    /// except inside attention).
    pub fn encode(
        self: *Ctx,
        io: std.Io,
        allocator: std.mem.Allocator,
        out: []f32,
        ids: []const u32,
        store: *const shards.Store,
        index: weight_index.Index,
        text: zconfig.Text,
        groups: []const usize,
        valid: usize,
    ) !void {
        const cfg = qenc.maskedAttnCfg(text, ids.len, valid);
        const tokens = cfg.tokens;
        const hidden = cfg.hidden;
        if (out.len != tokens * groups.len * hidden) return error.InvalidShape;
        if (cfg.head_dim % 2 != 0) return error.InvalidShape;
        const state_len = tokens * hidden;
        const p = try self.stageInput(allocator, ids, store, index, text, cfg, groups.len);

        var last: usize = 0;
        for (groups) |g| last = @max(last, g);
        if (last == 0) return error.InvalidShape;

        self.batch = metal_c.zdraw_metal_batch_begin(
            self.attn.queue,
        ) orelse return error.MetalDispatchFailed;
        errdefer if (self.batch) |bt| {
            _ = metal_c.zdraw_metal_batch_end(bt);
            self.batch = null;
            self.dropTemps();
        };
        for (0..last) |layer| {
            try progress.layer(io, allocator, "text", layer + 1, last);
            const w = try qenc.layerWeights(allocator, store, index, layer);
            try self.encodeLayer(p, w, cfg);
            for (groups, 0..) |g, slot| {
                if (g != layer + 1) continue;
                try self.snapshot(p, slot, state_len);
            }
        }
        const bt = self.batch.?;
        self.batch = null;
        const rc = metal_c.zdraw_metal_batch_end(bt);
        self.dropTemps();
        if (rc != 0) return error.MetalDispatchFailed;

        try readSnapshots(allocator, p, out, tokens, hidden, groups.len);
        // Unwire the no-copy weight binds between encodes, preserving the
        // per-op route's memory envelope. Safe here: batch_end waited.
        self.wbind.clearSources();
    }

    /// CPU embed, pool (re)build, and the state upload.
    fn stageInput(
        self: *Ctx,
        allocator: std.mem.Allocator,
        ids: []const u32,
        store: *const shards.Store,
        index: weight_index.Index,
        text: zconfig.Text,
        cfg: qattn.Config,
        groups: usize,
    ) !*Pool {
        const state_len = cfg.tokens * cfg.hidden;
        const host = try allocator.alloc(f32, state_len);
        defer allocator.free(host);
        const embed_view = try store.view(index, qnames.embed);
        try qtext.embed(host, ids, embed_view);
        const p = try self.ensurePool(.{
            .tokens = cfg.tokens,
            .hidden = cfg.hidden,
            .inner = @intCast(text.intermediate_size),
            .heads = cfg.heads,
            .kv_heads = cfg.kv_heads,
            .head_dim = cfg.head_dim,
            .groups = groups,
            .theta = cfg.rope_theta,
        });
        metal_c.zdraw_metal_write_buffer(
            p.state.handle,
            std.mem.sliceAsBytes(host).ptr,
            state_len * 4,
        );
        return p;
    }
};

/// Whole-buffer snapshot readback (the read entry has no offset parameter),
/// then the shared per-token scatter into the stacked layout.
fn readSnapshots(
    allocator: std.mem.Allocator,
    p: *Pool,
    out: []f32,
    tokens: usize,
    hidden: usize,
    groups: usize,
) !void {
    const state_len = tokens * hidden;
    const snaps_host = try allocator.alloc(f32, groups * state_len);
    defer allocator.free(snaps_host);
    metal_c.zdraw_metal_read_buffer(
        p.snaps.handle,
        std.mem.sliceAsBytes(snaps_host).ptr,
        snaps_host.len * 4,
    );
    for (0..groups) |slot| {
        const src = snaps_host[slot * state_len ..][0..state_len];
        qenc.scatterSnapshot(out, src, tokens, hidden, slot, groups);
    }
}

/// Mirrors ours16_ok_w16 in metal_api.m: half mode, f16 or bf16 weights,
/// 32-aligned dims (same admission the per-op mgemm route uses; shared shape
/// math in mres_util).
fn ours16Fits(p: *const metal_c.GemmParams) bool {
    return p.mode == 2 and (p.dtype == 1 or p.dtype == 2) and
        mres_util.ours16Dims(p.m, p.k, p.n);
}

fn empty(dev: *anyopaque, n: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(dev, n * 4);
}

test "qrms_rope matches the CPU per-head norm and rope" {
    var ctx = Ctx.init(std.testing.allocator) catch return; // no Metal: skip
    defer ctx.deinit();
    const alloc = std.testing.allocator;
    const tokens = 5;
    const head_dim = 128;
    const eps: f32 = 1e-6;
    const theta: f32 = 1_000_000.0;
    var prng = std.Random.DefaultPrng.init(0x51ee);
    const rand = prng.random();

    inline for ([_]usize{ 32, 8 }) |heads| {
        const n = tokens * heads * head_dim;
        const data = try alloc.alloc(f32, n);
        defer alloc.free(data);
        for (data) |*v| v.* = rand.floatNorm(f32);
        var wbits: [head_dim]u16 = undefined;
        for (&wbits) |*w| w.* = wideBf16(rand);
        const wview = tensor.View{
            .dtype = .bf16,
            .shape = &.{head_dim},
            .bytes = std.mem.sliceAsBytes(&wbits),
        };

        const want = try alloc.dupe(f32, data);
        defer alloc.free(want);
        try qrmsCpuRef(want, wview, tokens, heads, head_dim, eps, theta);

        const got = try alloc.alloc(f32, n);
        defer alloc.free(got);
        try qrmsGpuOnce(&ctx, wview, data, got, tokens, heads, head_dim, eps, theta);
        for (want, got) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-4);
    }
}

/// Exactly qwen_attn's normalizeHeads + rotate, in place.
fn qrmsCpuRef(
    data: []f32,
    wview: tensor.View,
    tokens: usize,
    heads: usize,
    comptime head_dim: usize,
    eps: f32,
    theta: f32,
) !void {
    var tmp: [head_dim]f32 = undefined;
    var cos_tab: [head_dim / 2]f32 = undefined;
    var sin_tab: [head_dim / 2]f32 = undefined;
    for (0..tokens) |tok| {
        for (0..heads) |head| {
            const vec = data[(tok * heads + head) * head_dim ..][0..head_dim];
            try opsRms(&tmp, vec, wview, eps);
            @memcpy(vec, &tmp);
        }
    }
    for (0..tokens) |tok| {
        try rope.table(&cos_tab, &sin_tab, tok, theta);
        for (0..heads) |head| {
            const vec = data[(tok * heads + head) * head_dim ..][0..head_dim];
            rope.applyTable(vec, &cos_tab, &sin_tab);
        }
    }
}

/// One batched qrms dispatch on a fresh buffer, result read into `got`.
fn qrmsGpuOnce(
    ctx: *Ctx,
    wview: tensor.View,
    data: []const f32,
    got: []f32,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f32,
) !void {
    const p = try ctx.ensurePool(.{
        .tokens = tokens,
        .hidden = head_dim,
        .inner = head_dim,
        .heads = heads,
        .kv_heads = heads,
        .head_dim = head_dim,
        .groups = 1,
        .theta = theta,
    });
    var buf = try mbuffer.Buffer.fromBytes(ctx.attn.device, std.mem.sliceAsBytes(data));
    defer buf.deinit();
    ctx.batch = metal_c.zdraw_metal_batch_begin(
        ctx.attn.queue,
    ) orelse return error.MetalDispatchFailed;
    try ctx.qrmsRope(p, wview, buf.handle, tokens, heads, head_dim, eps);
    const bt = ctx.batch.?;
    ctx.batch = null;
    try std.testing.expectEqual(@as(c_int, 0), metal_c.zdraw_metal_batch_end(bt));
    ctx.dropTemps();
    metal_c.zdraw_metal_read_buffer(buf.handle, std.mem.sliceAsBytes(got).ptr, got.len * 4);
}

/// Synthetic bf16 layer weights spanning the checkpoint exponent range.
const Synth = struct {
    q: []u16,
    k: []u16,
    v: []u16,
    o: []u16,
    gate: []u16,
    up: []u16,
    down: []u16,
    norm: []u16,
    post: []u16,
    qn: []u16,
    kn: []u16,

    const hidden = 64;
    const q_dim = 64;
    const kv_dim = 32;
    const inner = 96;
    const head_dim = 32;

    fn mat(a: std.mem.Allocator, r: std.Random, n: usize) ![]u16 {
        const bits = try a.alloc(u16, n);
        for (bits) |*b| b.* = wideBf16(r);
        return bits;
    }

    fn init(a: std.mem.Allocator, r: std.Random) !Synth {
        return .{
            .q = try mat(a, r, q_dim * hidden),
            .k = try mat(a, r, kv_dim * hidden),
            .v = try mat(a, r, kv_dim * hidden),
            .o = try mat(a, r, hidden * q_dim),
            .gate = try mat(a, r, inner * hidden),
            .up = try mat(a, r, inner * hidden),
            .down = try mat(a, r, hidden * inner),
            .norm = try mat(a, r, hidden),
            .post = try mat(a, r, hidden),
            .qn = try mat(a, r, head_dim),
            .kn = try mat(a, r, head_dim),
        };
    }

    fn deinit(self: *Synth, a: std.mem.Allocator) void {
        inline for (@typeInfo(Synth).@"struct".fields) |f| a.free(@field(self, f.name));
    }

    fn view(bits: []const u16, comptime rows: usize, comptime cols: usize) tensor.View {
        return .{
            .dtype = .bf16,
            .shape = &[_]usize{ rows, cols },
            .bytes = std.mem.sliceAsBytes(bits),
        };
    }

    fn vec(bits: []const u16, comptime n: usize) tensor.View {
        return .{ .dtype = .bf16, .shape = &[_]usize{n}, .bytes = std.mem.sliceAsBytes(bits) };
    }

    fn weights(self: *const Synth) qlayer.Weights {
        return .{
            .attn = .{
                .norm = vec(self.norm, hidden),
                .q = view(self.q, q_dim, hidden),
                .k = view(self.k, kv_dim, hidden),
                .v = view(self.v, kv_dim, hidden),
                .o = view(self.o, hidden, q_dim),
                .q_norm = vec(self.qn, head_dim),
                .k_norm = vec(self.kn, head_dim),
            },
            .post_norm = vec(self.post, hidden),
            .gate = view(self.gate, inner, hidden),
            .up = view(self.up, inner, hidden),
            .down = view(self.down, hidden, inner),
        };
    }
};

/// The pure-CPU layer (metal = null) with correctly sized scratch.
fn cpuLayerRef(
    alloc: std.mem.Allocator,
    state: []f32,
    weights: qlayer.Weights,
    cfg: qattn.Config,
    inner: usize,
) !void {
    const q_dim = cfg.heads * cfg.head_dim;
    const kv_dim = cfg.kv_heads * cfg.head_dim;
    const norm_buf = try alloc.alloc(f32, cfg.hidden);
    defer alloc.free(norm_buf);
    const qb = try alloc.alloc(f32, cfg.tokens * q_dim);
    defer alloc.free(qb);
    const kb = try alloc.alloc(f32, cfg.tokens * kv_dim);
    defer alloc.free(kb);
    const vb = try alloc.alloc(f32, cfg.tokens * kv_dim);
    defer alloc.free(vb);
    const mixb = try alloc.alloc(f32, cfg.tokens * q_dim);
    defer alloc.free(mixb);
    const scores = try alloc.alloc(f32, cfg.tokens);
    defer alloc.free(scores);
    const attn_out = try alloc.alloc(f32, cfg.tokens * cfg.hidden);
    defer alloc.free(attn_out);
    const gate_s = try alloc.alloc(f32, cfg.tokens * inner);
    defer alloc.free(gate_s);
    const up_s = try alloc.alloc(f32, cfg.tokens * inner);
    defer alloc.free(up_s);
    const mlp_out = try alloc.alloc(f32, cfg.tokens * cfg.hidden);
    defer alloc.free(mlp_out);
    try qlayer.run(null, null, state, weights, .{
        .attn = .{ .norm = norm_buf, .q = qb, .k = kb, .v = vb, .mix = mixb, .scores = scores },
        .attn_out = attn_out,
        .mlp = .{ .gate = gate_s, .up = up_s },
        .mlp_out = mlp_out,
    }, cfg);
}

test "resident layer matches the CPU reference on a synthetic shape" {
    var ctx = Ctx.init(std.testing.allocator) catch return; // no Metal: skip
    defer ctx.deinit();
    errdefer |err| std.debug.print("layer parity failed: {s}\n", .{@errorName(err)});
    const alloc = std.testing.allocator;
    // %32 dims throughout so both GEMM kernels are exercised; GQA 2/1 heads;
    // right-padding mask via valid < tokens.
    const cfg = qattn.Config{
        .tokens = 64,
        .hidden = Synth.hidden,
        .heads = 2,
        .kv_heads = 1,
        .head_dim = Synth.head_dim,
        .norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .causal = true,
        .use_gemm = false,
        .valid = 48,
    };
    var prng = std.Random.DefaultPrng.init(0x9e37);
    const rand = prng.random();
    var synth = try Synth.init(alloc, rand);
    defer synth.deinit(alloc);
    const weights = synth.weights();

    const state0 = try alloc.alloc(f32, cfg.tokens * cfg.hidden);
    defer alloc.free(state0);
    for (state0) |*v| v.* = rand.floatNorm(f32) * 0.5;

    const want = try alloc.dupe(f32, state0);
    defer alloc.free(want);
    try cpuLayerRef(alloc, want, weights, cfg, Synth.inner);

    const got = try alloc.alloc(f32, state0.len);
    defer alloc.free(got);
    try gpuLayerOnce(&ctx, weights, cfg, state0, got);
    var worst: f32 = 0.0;
    for (want, got) |w, g| worst = @max(worst, @abs(w - g));
    if (worst >= 0.05) std.debug.print("layer parity worst |d| = {d}\n", .{worst});
    try std.testing.expect(worst < 0.05);
    ctx.wbind.clearSources();
}

/// One batched resident layer on `state0`, result read into `got`.
fn gpuLayerOnce(
    ctx: *Ctx,
    weights: qlayer.Weights,
    cfg: qattn.Config,
    state0: []const f32,
    got: []f32,
) !void {
    const p = try ctx.ensurePool(.{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .inner = Synth.inner,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .groups = 1,
        .theta = cfg.rope_theta,
    });
    metal_c.zdraw_metal_write_buffer(
        p.state.handle,
        std.mem.sliceAsBytes(state0).ptr,
        state0.len * 4,
    );
    ctx.batch = metal_c.zdraw_metal_batch_begin(
        ctx.attn.queue,
    ) orelse return error.MetalDispatchFailed;
    try ctx.encodeLayer(p, weights, cfg);
    const bt = ctx.batch.?;
    ctx.batch = null;
    try std.testing.expectEqual(@as(c_int, 0), metal_c.zdraw_metal_batch_end(bt));
    ctx.dropTemps();
    metal_c.zdraw_metal_read_buffer(p.state.handle, std.mem.sliceAsBytes(got).ptr, got.len * 4);
}

fn opsRms(out: []f32, input: []const f32, weight: tensor.View, eps: f32) !void {
    var mean: f32 = 0.0;
    for (input) |value| mean += value * value;
    const count: f32 = @floatFromInt(input.len);
    const scale = 1.0 / @sqrt(mean / count + eps);
    for (out, input, 0..) |*dst, value, i| dst.* = value * scale * weight.atF32Unchecked(i);
}

/// Random bf16 bit pattern spanning the exponent range real checkpoints
/// occupy (same idiom as the linear_fast route-agreement test).
fn wideBf16(rand: std.Random) u16 {
    if (rand.uintLessThan(u8, 64) == 0) return 0;
    const sign: u16 = if (rand.boolean()) 0x8000 else 0;
    const exp: u16 = 112 + rand.uintLessThan(u16, 16);
    return sign | (exp << 7) | (rand.int(u16) & 0x7F);
}
