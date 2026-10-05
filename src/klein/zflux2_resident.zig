//! Resident FLUX.2 Klein forward: every activation stays in GPU buffers;
//! GEMMs run on weight buffers via row-offsets (the fused projections need
//! no de-interleave at all); glue ops are the zflux2_glue kernels. Math is
//! identical to zflux2_dit's verified CPU path — gated against the same
//! block oracle.
//!
//! Backend policy — do NOT reopen "backend roulette" here. Both verdicts are
//! retained from the original engine measurements:
//!   - GEMM = gemm_half (zdraw's own simdgroup-MMA) plus the staged ours16
//!     fast kernel where shapes admit it. f16 steel is faster but numerically
//!     BROKEN in-chain (drift=inf, #26/#27) → killed; MPS is correct but
//!     vendor → REFERENCE-ONLY (engine-ownership thesis). Closed by the
//!     gemmbench kernel gate (cb241e3; the ZDRAW_KLEIN_GEMMBENCH flag itself
//!     is retired).
//!   - Attention = block16 (mattn .block). Attribution closed (fda01e2):
//!     block16 is 95.8% of the attention bucket, the q/k/v head-major converts
//!     4.2% — the cheap layout fix is neutral. The pass-3 instrument that
//!     produced this number is deleted (question closed); rebuild it from
//!     fda01e2 before reconsidering.
//!   - Attention above mfa_min_tokens (2048) = vendored MFA, default ON since
//!     2026-08-06. It was quarantined 06-26 for lower-half corruption; that
//!     corruption was incident-mfa-corruption (32-row tiles dispatched against
//!     Apple9's 16-row kernel body), fixed in b0b4a92, and the quarantine
//!     simply outlived its cause. Re-gated at 1024 across 8 content classes:
//!     worst PSNR 44.7 dB vs block16, corruption checker equal to block16 on
//!     every prompt, -4.1 s end-to-end. Opt out with ZDRAW_KLEIN_ATTN_MFA=0,
//!     which falls back to block16.
//!
//! Diagnostic trace: the ZDRAW_KLEIN_TRACE* fields and report*/traceMark*
//! functions on Ctx are a default-OFF attribution trace. They add GPU sync
//! boundaries only to bucket time (never product speed) and never run on the
//! product path — skip them when reading the normal forward.

const std = @import("std");

const mattn = @import("../metal/mattn.zig");
const mbuffer = @import("../metal/mbuffer.zig");
const mlend = @import("../metal/mlend.zig");
const zpool = @import("zflux2_pool.zig");
const metal_c = @import("../metal/metal_c.zig");
const mres_util = @import("../metal/mres_util.zig");
const mpipe = @import("../metal/mpipe.zig");
const tensor = @import("../pack/tensor.zig");
const zflux2 = @import("zflux2.zig");
const zflux2_dit = @import("zflux2_dit.zig");
const zflux2_pack = @import("zflux2_pack.zig");
const packed_w = @import("zflux2_packed.zig");
const glue_src = @import("zflux2_glue_shader.zig");
const mgemm_shader = @import("../metal/mgemm_shader.zig");
const mgemm_mpp = @import("../metal/mgemm_mpp_shader.zig");
const modvec_src = @import("zflux2_modvec_shader.zig");
const metrics = @import("../metal/metrics.zig");
const env = @import("../runtime/env.zig");

const head_dim = zflux2.head_dim;
const txt_len = zflux2.txt_len;
const Bufs = zpool.Bufs;
const PoolShape = zpool.PoolShape;
const Dims = zflux2_dit.Dims;
const ModOff = zflux2.ModOff;

/// Byte offset of modulation component `comp` of set `set` in an f32 mod
/// buffer (the glue kernels take byte offsets; the element layout is shared
/// truth in zflux2.modOffset). Keeps the resident offsets from drifting from
/// the CPU/oracle path.
fn modByteOff(set: usize, comp: usize, hidden: usize) usize {
    return zflux2.modOffset(set, comp, hidden) * 4;
}

/// Percent of `total` (trace attribution tables). Caller guards total > 0.
fn pctOf(x: u64, total: u64) f64 {
    return 100.0 * @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(total));
}

const GlueParams = extern struct {
    tokens: u32,
    hidden: u32,
    heads: u32,
    head_dim: u32,
};

/// krms_rope_hm constants: local token count for the dispatch, the global
/// token offset and total for the head-major destination index.
const RopeHmParams = extern struct {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    tok_off: u32,
    total: u32,
};

/// The step's modulation buffers; fmods = [scale, shift] (2*hidden).
const ModHandles = struct {
    img: *anyopaque,
    txt: *anyopaque,
    single: *anyopaque,
    fmods: *anyopaque,
};

const ModVecParams = extern struct {
    n: u32,
    k: u32,
    dtype: u32,
    pad: u32 = 0,
};

const AxpyParams = extern struct {
    count: u32,
    dt: f32,
};

/// Per-forward options for the resident path. Defaults reproduce the classic
/// contract (upload latents, read v back). ZDRAW_KLEIN_XRES sets dt_in on
/// steps > 0 (apply the previous Euler update on-GPU first) and read_v=false
/// so latents stay device-resident across the whole denoise.
pub const ForwardOpts = struct {
    dt_in: ?f32 = null,
    read_v: bool = true,
};

/// Weight-cache key: a bare pointer can alias across reloads/views, so the
/// length and dtype are part of the identity.
const WKey = struct { ptr: usize, len: usize, dtype: u8 };

const dtype_f16: u32 = 1;

const WeightBind = packed_w.WeightBind;

const TextProj = struct {
    key: u64,
    len: usize,
    hidden: usize,
    joint_dim: usize,
    buf: mbuffer.Buffer,

    fn matches(self: *const TextProj, key: u64, len: usize, hidden: usize, joint_dim: usize) bool {
        return self.key == key and
            self.len == len and
            self.hidden == hidden and
            self.joint_dim == joint_dim;
    }

    fn deinit(self: *TextProj) void {
        self.buf.deinit();
        self.* = undefined;
    }
};

fn wkey(v: tensor.View) WKey {
    return .{ .ptr = @intFromPtr(v.bytes.ptr), .len = v.bytes.len, .dtype = @intFromEnum(v.dtype) };
}

/// Per-timestep CPU modulation set. Depends only on (t, globals), so one
/// generation's 4 steps compute these once each and every later same-t forward
/// (rerolls, candidate sets) reuses them bit-identically.
const StepMods = struct {
    t_bits: u32,
    temb: []f32,
    mods_img: []f32,
    mods_txt: []f32,
    mods_single: []f32,
    fmods: []f32,

    fn deinit(self: *StepMods, allocator: std.mem.Allocator) void {
        allocator.free(self.temb);
        allocator.free(self.mods_img);
        allocator.free(self.mods_txt);
        allocator.free(self.mods_single);
        allocator.free(self.fmods);
        self.* = undefined;
    }
};

/// Pooled small upload buffer: reuse one MTLBuffer per role instead of a
/// fresh fromBytes/deinit every step (no-math change; contents re-written).
const Upload = struct {
    len: usize = 0,
    buf: ?mbuffer.Buffer = null,

    fn deinit(self: *Upload) void {
        if (self.buf) |*b| b.deinit();
        self.* = .{};
    }
};

const Pipelines = struct {
    ln_mod: *anyopaque,
    rms_rope: *anyopaque,
    rms_rope_hm: *anyopaque,
    unperm_hm_h: *anyopaque,
    unperm_hm_hh: *anyopaque,
    swiglu_hs: *anyopaque,
    swiglu: *anyopaque,
    gate_add: *anyopaque,
    cat_rows: *anyopaque,
    copy: *anyopaque,
    axpy: *anyopaque,
    ln_mod_h: *anyopaque,
    swiglu_h: *anyopaque,
    cat_rows_h: *anyopaque,
    gemm: *anyopaque,
    /// Metal 4 tensor-path GEMM for the f16-A route (default; ZDRAW_KLEIN_GEMM_MPP=0
    /// is the direct-kernel A/B arm). Byte-identical to the direct kernel and
    /// -0.9 to -1.2 s per 1024 render (klein-gemm-mpp-chain-20260827). Compiled
    /// at runtime at MSL 4.0; below macOS 26 it is null: one WARNING, the
    /// direct kernel runs and every substituted GEMM counts as an MPP fallback.
    gemm_mpp: ?*anyopaque,
    /// The flag was on but the pipeline is unavailable (counted per GEMM).
    mpp_fallback: bool,
    /// Per-step modulation matvecs on the GPU (ZDRAW_KLEIN_MODS_GPU).
    modvec: *anyopaque,

    fn init(device: *anyopaque) !Pipelines {
        var err: [1024]u8 = undefined;
        const ln_mod = try mpipe.required(device, glue_src.src, "kln_mod", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(ln_mod);
        const rms_rope = try mpipe.required(device, glue_src.src, "krms_rope", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(rms_rope);
        const rms_rope_hm = try mpipe.required(device, glue_src.src, "krms_rope_hm", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(rms_rope_hm);
        const unperm_hm_h = try mpipe.required(device, glue_src.src, "kunperm_hm_h", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(unperm_hm_h);
        const unperm_hm_hh = try mpipe.required(device, glue_src.src, "kunperm_hm_hh", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(unperm_hm_hh);
        const swiglu_hs = try mpipe.required(device, glue_src.src, "kswiglu_hs", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(swiglu_hs);
        const swiglu = try mpipe.required(device, glue_src.src, "kswiglu", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(swiglu);
        const gate_add = try mpipe.required(device, glue_src.src, "kgate_add", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(gate_add);
        const cat_rows = try mpipe.required(device, glue_src.src, "kcat_rows", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(cat_rows);
        const copy = try mpipe.required(device, glue_src.src, "kcopy", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(copy);
        const axpy = try mpipe.required(device, glue_src.src, "kaxpy", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(axpy);
        const ln_mod_h = try mpipe.required(device, glue_src.src, "kln_mod_h", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(ln_mod_h);
        const swiglu_h = try mpipe.required(device, glue_src.src, "kswiglu_h", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(swiglu_h);
        const cat_rows_h = try mpipe.required(device, glue_src.src, "kcat_rows_h", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(cat_rows_h);
        const gemm = try mpipe.required(device, mgemm_shader.gemm, "gemm_half", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(gemm);
        const modvec = try mpipe.required(device, modvec_src.src, "kmodvec", &err);
        errdefer metal_c.zdraw_metal_release_pipeline(modvec);
        const mpp_wanted = env.flag("ZDRAW_KLEIN_GEMM_MPP", true);
        const gemm_mpp = if (mpp_wanted) mppPipeline(device) else null;
        errdefer if (gemm_mpp) |p| metal_c.zdraw_metal_release_pipeline(p);
        return .{
            .ln_mod = ln_mod,
            .rms_rope = rms_rope,
            .rms_rope_hm = rms_rope_hm,
            .unperm_hm_h = unperm_hm_h,
            .unperm_hm_hh = unperm_hm_hh,
            .swiglu_hs = swiglu_hs,
            .swiglu = swiglu,
            .gate_add = gate_add,
            .cat_rows = cat_rows,
            .copy = copy,
            .axpy = axpy,
            .ln_mod_h = ln_mod_h,
            .swiglu_h = swiglu_h,
            .cat_rows_h = cat_rows_h,
            .gemm = gemm,
            .gemm_mpp = gemm_mpp,
            .mpp_fallback = mpp_wanted and gemm_mpp == null,
            .modvec = modvec,
        };
    }

    /// The Metal 4 tensor-path pipeline; null (one WARNING) on systems
    /// without MSL 4.0, where the direct kernel runs as a counted fallback.
    fn mppPipeline(device: *anyopaque) ?*anyopaque {
        const p = metal_c.zdraw_metal_compile_mpp(device, mgemm_mpp.src.ptr, "gemm_mpp64");
        if (p == null) metal_c.zdraw_metal_warn_mpp_unavailable();
        return p;
    }

    fn deinit(self: *Pipelines) void {
        metal_c.zdraw_metal_release_pipeline(self.modvec);
        if (self.gemm_mpp) |p| metal_c.zdraw_metal_release_pipeline(p);
        metal_c.zdraw_metal_release_pipeline(self.gemm);
        metal_c.zdraw_metal_release_pipeline(self.cat_rows_h);
        metal_c.zdraw_metal_release_pipeline(self.swiglu_h);
        metal_c.zdraw_metal_release_pipeline(self.ln_mod_h);
        metal_c.zdraw_metal_release_pipeline(self.axpy);
        metal_c.zdraw_metal_release_pipeline(self.copy);
        metal_c.zdraw_metal_release_pipeline(self.cat_rows);
        metal_c.zdraw_metal_release_pipeline(self.gate_add);
        metal_c.zdraw_metal_release_pipeline(self.swiglu);
        metal_c.zdraw_metal_release_pipeline(self.rms_rope);
        metal_c.zdraw_metal_release_pipeline(self.rms_rope_hm);
        metal_c.zdraw_metal_release_pipeline(self.unperm_hm_h);
        metal_c.zdraw_metal_release_pipeline(self.unperm_hm_hh);
        metal_c.zdraw_metal_release_pipeline(self.swiglu_hs);
        metal_c.zdraw_metal_release_pipeline(self.ln_mod);
        self.* = undefined;
    }
};

fn gemm64Ok(p: *const metal_c.GemmParams) bool {
    // dtype 1 only: this admits the half-A entry too, which has no bf16
    // kernel (the deliberate ours16_ok / ours16_ok_w16 split).
    return p.mode == 2 and p.dtype == 1 and mres_util.ours16Dims(p.m, p.k, p.n);
}

pub const Ctx = struct {
    allocator: std.mem.Allocator,
    attn: mattn.Context,
    pipelines: Pipelines,
    batch: ?*anyopaque = null,
    // Copied fallback buffers for non-f16/non-source weights.
    wcache: std.AutoHashMap(WKey, mbuffer.Buffer),
    wcache_f32: std.AutoHashMap(WKey, mbuffer.Buffer),
    // No-copy f16 source bindings for the W16 zpack. This is the hot path:
    // the Metal buffer wraps mmap-backed bytes and carries row offsets instead
    // of materializing duplicate f16 GPU weights during denoise step 1.
    wbind: mbuffer.Cache,
    use_gemm64: bool = false,
    // Half-activation mode (ZDRAW_KLEIN_ACT, DEFAULT ON since 71f6bca;
    // f32 opts out): the ln/swiglu/cat glue stores half and the big GEMMs run
    // the f16-A kernel. Requires the W16 pack (W6 weights fail loudly) and
    // %32-token shapes.
    act_f16: bool = false,
    // MFA above mfa_min_tokens, DEFAULT ON since 66b31fb: the 2026-06-26
    // quarantine's root cause (32-row tiles on the 16-row Apple9 kernel) was
    // fixed in b0b4a92 and the Klein re-test passed 8 content classes at
    // worst PSNR 44.7 dB. block16 remains the sub-threshold and fallback
    // kernel.
    use_mfa_attn: bool = false,
    mfa_min_tokens: usize = 2048,
    /// Emit q/k head-major half from the norm+rope kernel straight into the
    /// MFA scratch, skipping two of the four layout converts per attention.
    /// Only on the MFA route (block16 reads token-major). ZDRAW_KLEIN_QK_HM=0
    /// restores the convert path for A/B.
    hm_qk: bool = true,
    /// The vendored MLX steel attention on the head-major route (default;
    /// ZDRAW_KLEIN_ATTN_STEEL=0 restores MFA). 1.3x MFA in the bench, -1.4 to
    /// -2.2 s per 1024 render in-chain; gated 2026-08-26 by census n=10 one
    /// hash, klein_gate 8/8 incl. 1024, content, viewed (ledger
    /// klein-attn-steel-route-20260826). Falls back to MFA when the metallib
    /// lacks the kernel.
    attn_steel: bool = true,
    /// Set per attention when the steel route actually ran (falls back to
    /// MFA when the metallib lacks the kernel), so the tail reads the right O.
    steel_ran: bool = false,
    // Shape-keyed resident pool: the per-step Bufs set is allocated once per
    // PoolShape and reused across denoise steps and generations. rope/embeds
    // are re-uploaded into their pooled buffers every forward (a few MB, cheap
    // against ~6 s/step) — correctness over a host-pointer cache, which could
    // alias a freed allocation and feed a later generation stale conditioning.
    pool: ?Bufs = null,
    // A pool buffer is on loan to the VAE decode (mlend): ensurePool must
    // not free the set until the lease ends.
    lent_out: bool = false,
    // Two slots: CFG alternates positive/negative conditioning every step,
    // and a single slot would re-project both on every forward.
    txt_proj: [2]?TextProj = .{ null, null },
    txt_proj_next: usize = 0,
    // Per-timestep mods cache + pooled small upload buffers (A2: the per-step
    // CPU scalar matmuls and fresh MTLBuffers were serialized ahead of every
    // batch encode while the GPU sat idle).
    step_mods: [8]?StepMods = [_]?StepMods{null} ** 8,
    step_mods_next: usize = 0,
    up_mods_i: Upload = .{},
    up_mods_t: Upload = .{},
    up_mods_s: Upload = .{},
    up_fshift: Upload = .{},
    up_fscale: Upload = .{},
    up_lat: Upload = .{},
    // GPU modulation (default; ZDRAW_KLEIN_MODS_GPU=0 = the CPU cache A/B arm;
    // klein-mods-gpu-20260827: byte-identical, -0.85 s per 1024 render).
    mods_gpu: bool = false,
    up_act: Upload = .{},
    up_fmods: Upload = .{},
    mod_binds: [4]?WeightBind = .{ null, null, null, null },
    mod_temps: [4]?mbuffer.Buffer = .{ null, null, null, null },
    // In-denoise attribution trace (ZDRAW_KLEIN_TRACE=1, default off). When on,
    // forward splits its single batch into per-phase sub-batches (extra GPU
    // syncs) to bucket time by embedders/doubles/singles/final. The sync
    // boundaries inflate absolute time, so these are ATTRIBUTION numbers only,
    // never product speed. Off = single batch, behaviour/timing unchanged.
    trace: bool = false,
    tr_embed_ns: u64 = 0,
    tr_double_ns: u64 = 0,
    tr_single_ns: u64 = 0,
    tr_final_ns: u64 = 0,
    tr_steps: u64 = 0,
    // Pass 2: ONE selectable single block (ZDRAW_KLEIN_TRACE_SINGLE, default 0)
    // is sub-divided into qkv-gemm / attention / ffn-gemm / glue. Only that one
    // block syncs internally, so the extra sync overhead stays bounded.
    trace_single_idx: usize = 0,
    tr_sb_qkv_ns: u64 = 0,
    tr_sb_attn_ns: u64 = 0,
    tr_sb_ffn_ns: u64 = 0,
    tr_sb_glue_ns: u64 = 0,
    tr_sb_steps: u64 = 0,
    // GPU-active (gpu_ns) version of the pass-2 buckets: charged at the SAME
    // flush points, but from zdraw_metal_gpu_ns (GPUEnd-GPUStart) instead of
    // wall. Flush sync overhead is CPU wait, NOT GPU time, so these are the
    // UNBIASED per-category split (the wall buckets above over-charge whichever
    // category has more flush points — glue has 4). Read THIS split.
    tr_sb_qkv_gpu: u64 = 0,
    tr_sb_attn_gpu: u64 = 0,
    tr_sb_ffn_gpu: u64 = 0,
    tr_sb_glue_gpu: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) !Ctx {
        var attn = try mattn.Context.initKernel(.block);
        errdefer attn.deinit();
        const dev = attn.device;
        var pipelines = try Pipelines.init(dev);
        errdefer pipelines.deinit();
        return .{
            .allocator = allocator,
            .attn = attn,
            .pipelines = pipelines,
            .wcache = std.AutoHashMap(WKey, mbuffer.Buffer).init(allocator),
            .wcache_f32 = std.AutoHashMap(WKey, mbuffer.Buffer).init(allocator),
            .wbind = try mbuffer.Cache.init(dev),
            .use_gemm64 = env.flag("ZDRAW_KLEIN_GEMM64", true),
            // Default ON: gemm_f16_direct already stages A as half, so keeping
            // the activations half feeds the MMA the SAME values while halving
            // the read traffic - byte-identical output, -11% GPU busy. Costs
            // ~190 MB of peak RSS. ZDRAW_KLEIN_ACT=f32 opts out.
            .act_f16 = !env.equals("ZDRAW_KLEIN_ACT", "f32"),
            .use_mfa_attn = env.flag("ZDRAW_KLEIN_ATTN_MFA", true),
            .mfa_min_tokens = env.usizeVar("ZDRAW_KLEIN_MFA_MIN_TOKENS", 2048),
            .hm_qk = env.flag("ZDRAW_KLEIN_QK_HM", true),
            .attn_steel = env.flag("ZDRAW_KLEIN_ATTN_STEEL", true),
            .mods_gpu = env.flag("ZDRAW_KLEIN_MODS_GPU", true),
            .trace = std.c.getenv("ZDRAW_KLEIN_TRACE") != null,
            .trace_single_idx = blk: {
                const v = std.c.getenv("ZDRAW_KLEIN_TRACE_SINGLE") orelse break :blk 0;
                break :blk std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
            },
        };
    }

    /// Trace sub-batch boundary: end the current batch (syncs the GPU), charge
    /// the elapsed span to `accum`, and open the next. No-op when trace is off,
    /// so the production path keeps its single batch and unchanged timing.
    fn traceMark(self: *Ctx, prev: *u64, accum: *u64) !void {
        if (!self.trace) return;
        try self.flush(); // batch_end (GPU sync) + batch_begin
        const at = metrics.now();
        accum.* += at -% prev.*;
        prev.* = at;
    }

    /// Pass-2 mark that charges BOTH wall (wprev/wacc) and GPU-active
    /// (gprev/gacc) at the one flush. The flush drains the GPU so the gpu_ns
    /// delta is exactly this category's GPU execution time — unbiased by the
    /// per-flush CPU wait that distorts the wall buckets.
    fn traceMark2(self: *Ctx, wprev: *u64, wacc: *u64, gprev: *u64, gacc: *u64) !void {
        if (!self.trace) return;
        try self.flush();
        const w = metrics.now();
        wacc.* += w -% wprev.*;
        wprev.* = w;
        const g = metal_c.zdraw_metal_gpu_ns();
        gacc.* += g -% gprev.*;
        gprev.* = g;
    }

    /// Print the per-step attribution table (pass 1: coarse buckets). Diagnostic
    /// only — the sub-batch syncs inflate absolute ms; read the RELATIVE split.
    pub fn reportTrace(self: *const Ctx) void {
        if (!self.trace or self.tr_steps == 0) return;
        const n = self.tr_steps;
        const ms = struct {
            fn f(ns: u64, steps: u64) f64 {
                return @as(f64, @floatFromInt(ns / steps)) / 1.0e6;
            }
        }.f;
        const eb = ms(self.tr_embed_ns, n);
        const db = ms(self.tr_double_ns, n);
        const sb = ms(self.tr_single_ns, n);
        const fb = ms(self.tr_final_ns, n);
        std.debug.print("\nklein step avg (ZDRAW_KLEIN_TRACE; steps={d}):\n", .{n});
        std.debug.print(
            "  embedders:     {d:.1} ms\n" ++
                "  double blocks: {d:.1} ms\n" ++
                "  single blocks: {d:.1} ms\n" ++
                "  final:         {d:.1} ms\n",
            .{ eb, db, sb, fb },
        );
        self.reportSplit();
    }

    /// Pass 2: the selected single block's category split (% of that block).
    fn reportSplit(self: *const Ctx) void {
        const tot = self.tr_sb_qkv_ns + self.tr_sb_attn_ns + self.tr_sb_ffn_ns + self.tr_sb_glue_ns;
        if (self.tr_sb_steps == 0 or tot == 0) return;
        std.debug.print(
            "single block {d} WALL split (flush-biased, do not trust):\n" ++
                "  qkv/gate GEMM:   {d:.1}%\n" ++
                "  attention:       {d:.1}%\n" ++
                "  ffn/down GEMM:   {d:.1}%\n" ++
                "  glue/mod/norm:   {d:.1}%\n",
            .{
                self.trace_single_idx,
                pctOf(self.tr_sb_qkv_ns, tot),
                pctOf(self.tr_sb_attn_ns, tot),
                pctOf(self.tr_sb_ffn_ns, tot),
                pctOf(self.tr_sb_glue_ns, tot),
            },
        );
        const gtot = self.tr_sb_qkv_gpu + self.tr_sb_attn_gpu + self.tr_sb_ffn_gpu + self.tr_sb_glue_gpu;
        if (gtot == 0) return;
        const gms = struct {
            fn f(ns: u64, steps: u64) f64 {
                return @as(f64, @floatFromInt(ns / steps)) / 1.0e6;
            }
        }.f;
        std.debug.print(
            "single block {d} GPU-ACTIVE split (gpu_ns, UNBIASED) — per step:\n" ++
                "  qkv/gate GEMM:   {d:.1}%  ({d:.2} ms)\n" ++
                "  attention:       {d:.1}%  ({d:.2} ms)\n" ++
                "  ffn/down GEMM:   {d:.1}%  ({d:.2} ms)\n" ++
                "  glue/mod/norm:   {d:.1}%  ({d:.2} ms)\n",
            .{
                self.trace_single_idx,
                pctOf(self.tr_sb_qkv_gpu, gtot),
                gms(self.tr_sb_qkv_gpu, self.tr_sb_steps),
                pctOf(self.tr_sb_attn_gpu, gtot),
                gms(self.tr_sb_attn_gpu, self.tr_sb_steps),
                pctOf(self.tr_sb_ffn_gpu, gtot),
                gms(self.tr_sb_ffn_gpu, self.tr_sb_steps),
                pctOf(self.tr_sb_glue_gpu, gtot),
                gms(self.tr_sb_glue_gpu, self.tr_sb_steps),
            },
        );
    }

    pub fn deinit(self: *Ctx) void {
        var it = self.wcache.valueIterator();
        while (it.next()) |b| {
            var buf = b.*;
            buf.deinit();
        }
        self.wcache.deinit();
        var it_f32 = self.wcache_f32.valueIterator();
        while (it_f32.next()) |b| {
            var buf = b.*;
            buf.deinit();
        }
        self.wcache_f32.deinit();
        self.wbind.deinit();
        if (self.pool) |*p| p.deinit();
        for (&self.txt_proj) |*slot| {
            if (slot.*) |*t| t.deinit();
        }
        for (&self.step_mods) |*sm| {
            if (sm.*) |*m| m.deinit(self.allocator);
            sm.* = null;
        }
        self.up_mods_i.deinit();
        self.up_mods_t.deinit();
        self.up_mods_s.deinit();
        self.up_fshift.deinit();
        self.up_fscale.deinit();
        self.up_lat.deinit();
        self.up_act.deinit();
        self.up_fmods.deinit();
        for (&self.mod_temps) |*tb| {
            if (tb.*) |*buf| buf.deinit();
            tb.* = null;
        }
        self.pipelines.deinit();
        self.attn.deinit();
        self.* = undefined;
    }

    /// Cached per-timestep CPU mods (see StepMods). Bitwise-keyed on t.
    fn stepMods(self: *Ctx, loaded: *const zflux2.Loaded, t: f32, hidden: usize) !*const StepMods {
        const bits: u32 = @bitCast(t);
        for (&self.step_mods) |*slot| {
            if (slot.*) |*m| {
                if (m.t_bits == bits) return m;
            }
        }
        const allocator = self.allocator;
        const temb = try allocator.alloc(f32, hidden);
        errdefer allocator.free(temb);
        try zflux2_dit.timeEmbed(allocator, null, temb, loaded.globals, t);
        const mods_img = try zflux2_dit.modVectors(
            allocator,
            null,
            loaded.globals.mod_img,
            temb,
            2,
        );
        errdefer allocator.free(mods_img);
        const mods_txt = try zflux2_dit.modVectors(
            allocator,
            null,
            loaded.globals.mod_txt,
            temb,
            2,
        );
        errdefer allocator.free(mods_txt);
        const mods_single = try zflux2_dit.modVectors(
            allocator,
            null,
            loaded.globals.mod_single,
            temb,
            1,
        );
        errdefer allocator.free(mods_single);
        const fmods = try normOutMods(
            allocator,
            loaded.globals.norm_out,
            temb,
            hidden,
        );
        errdefer allocator.free(fmods);
        const idx = self.step_mods_next % self.step_mods.len;
        self.step_mods_next += 1;
        if (self.step_mods[idx]) |*old| old.deinit(self.allocator);
        self.step_mods[idx] = .{
            .t_bits = bits,
            .temb = temb,
            .mods_img = mods_img,
            .mods_txt = mods_txt,
            .mods_single = mods_single,
            .fmods = fmods,
        };
        return &self.step_mods[idx].?;
    }

    /// Write into the pooled upload buffer for this role (recreate on resize).
    fn upload(self: *Ctx, u: *Upload, bytes: []const u8) !*anyopaque {
        if (u.buf == null or u.len != bytes.len) {
            u.deinit();
            u.buf = try mbuffer.Buffer.fromBytes(self.attn.device, bytes);
            u.len = bytes.len;
            return u.buf.?.handle;
        }
        metal_c.zdraw_metal_write_buffer(u.buf.?.handle, bytes.ptr, bytes.len);
        return u.buf.?.handle;
    }

    /// A pooled device buffer of `len` bytes that a kernel writes (no upload).
    fn ensureUp(self: *Ctx, u: *Upload, len: usize) !*anyopaque {
        if (u.buf == null or u.len != len) {
            u.deinit();
            u.buf = try mbuffer.Buffer.empty(self.attn.device, len);
            u.len = len;
        }
        return u.buf.?.handle;
    }

    /// Raw no-copy bind of a mapped weight (its own dtype, no f16 promotion).
    fn rawBind(self: *Ctx, idx: usize, v: tensor.View) !WeightBind {
        if (self.mod_binds[idx]) |b| return b;
        const code: u32 = switch (v.dtype) {
            .f32 => 0,
            .f16 => 1,
            .bf16 => 2,
            .u8 => return error.InvalidDType,
        };
        var temp: ?mbuffer.Buffer = null;
        const bind = try self.wbind.bindView(v, &temp);
        if (temp) |tb| self.mod_temps[idx] = tb;
        const out = WeightBind{
            .handle = bind.handle,
            .offset = bind.offset,
            .dtype = code,
        };
        self.mod_binds[idx] = out;
        return out;
    }

    /// out[n] = W[n][:] . act on the GPU (kmodvec), inside the open batch.
    fn modVec(
        self: *Ctx,
        idx: usize,
        v: tensor.View,
        act_h: *anyopaque,
        out_h: *anyopaque,
    ) !void {
        if (v.shape.len != 2) return error.InvalidShape;
        const bind = try self.rawBind(idx, v);
        const p = ModVecParams{
            .n = @intCast(v.shape[0]),
            .k = @intCast(v.shape[1]),
            .dtype = bind.dtype,
        };
        const offs = [5]usize{ bind.offset, 0, 0, 0, 0 };
        const bufs = [5]?*anyopaque{ bind.handle, act_h, out_h, null, null };
        const n = v.shape[0];
        try self.glue(self.pipelines.modvec, bufs, &offs, &p, @sizeOf(ModVecParams), 3, n, 256, 0);
    }

    /// The step's modulation handles from the CPU cache (uploaded).
    fn stepModsCpu(self: *Ctx, loaded: *const zflux2.Loaded, t: f32, hidden: usize) !ModHandles {
        const mods = try self.stepMods(loaded, t, hidden);
        return .{
            .img = try self.upload(&self.up_mods_i, std.mem.sliceAsBytes(mods.mods_img)),
            .txt = try self.upload(&self.up_mods_t, std.mem.sliceAsBytes(mods.mods_txt)),
            .single = try self.upload(&self.up_mods_s, std.mem.sliceAsBytes(mods.mods_single)),
            .fmods = try self.upload(&self.up_fmods, std.mem.sliceAsBytes(mods.fmods)),
        };
    }

    /// GPU modulation: temb + SiLU on the CPU (tiny), four matvecs in-batch.
    fn stepModsGpu(
        self: *Ctx,
        loaded: *const zflux2.Loaded,
        t: f32,
        hidden: usize,
    ) !ModHandles {
        const a = self.allocator;
        const temb = try a.alloc(f32, hidden);
        defer a.free(temb);
        try zflux2_dit.timeEmbed(a, null, temb, loaded.globals, t);
        for (temb) |*v| v.* = v.* / (1.0 + @exp(-v.*));
        const act_h = try self.upload(&self.up_act, std.mem.sliceAsBytes(temb));
        const g = loaded.globals;
        const out = ModHandles{
            .img = try self.ensureUp(&self.up_mods_i, 6 * hidden * 4),
            .txt = try self.ensureUp(&self.up_mods_t, 6 * hidden * 4),
            .single = try self.ensureUp(&self.up_mods_s, 3 * hidden * 4),
            .fmods = try self.ensureUp(&self.up_fmods, 2 * hidden * 4),
        };
        try self.modVec(0, g.mod_img, act_h, out.img);
        try self.modVec(1, g.mod_txt, act_h, out.txt);
        try self.modVec(2, g.mod_single, act_h, out.single);
        try self.modVec(3, g.norm_out, act_h, out.fmods);
        return out;
    }

    /// Allocate (or reuse) the resident scratch set for this image-token count.
    fn ensurePool(self: *Ctx, dev: *anyopaque, img_len: usize, hidden: usize, inner: usize, joint_dim: usize, nb: usize) !*Bufs {
        const shape = PoolShape{ .img_len = img_len, .hidden = hidden, .inner = inner, .joint_dim = joint_dim, .batch = nb };
        if (self.pool != null and std.meta.eql(self.pool.?.shape, shape)) return &self.pool.?;
        if (self.lent_out) return error.PoolLeased;
        if (self.pool) |*p| p.deinit();
        self.pool = Bufs{
            .img = try zpool.sized(dev, shape, .img),
            .txt = try zpool.sized(dev, shape, .txt),
            .cat = try zpool.sized(dev, shape, .cat),
            .nscratch = try zpool.sized(dev, shape, .nscratch),
            .norm_t = try zpool.sized(dev, shape, .norm_t),
            .norm_c = try zpool.sized(dev, shape, .norm_c),
            .q = try zpool.sized(dev, shape, .q),
            .k = try zpool.sized(dev, shape, .k),
            .v = try zpool.sized(dev, shape, .v),
            .o = try zpool.sized(dev, shape, .o),
            .proj_t = try zpool.sized(dev, shape, .proj_t),
            .proj_i = try zpool.sized(dev, shape, .proj_i),
            .wide = try zpool.sized(dev, shape, .wide),
            .rope_cos = try zpool.sized(dev, shape, .rope),
            .rope_sin = try zpool.sized(dev, shape, .rope),
            .emb = try zpool.sized(dev, shape, .emb),
            .final_in = try zpool.sized(dev, shape, .final_in),
            .out128 = try zpool.sized(dev, shape, .out128),
        };
        if (self.act_f16) {
            self.pool.?.ah = try zpool.sized(dev, shape, .ah);
            self.pool.?.bh = try zpool.sized(dev, shape, .bh);
        }
        self.pool.?.shape = shape;
        return &self.pool.?;
    }

    /// Offer the idle resident pool to another phase (the VAE decode: the
    /// mid-attention's scratch and the conv1_out lease). Offering never
    /// allocates: a request is served only by an existing slot whose
    /// capacity covers it, else the borrower falls back to its own pool (no
    /// pool yet, a batch still open, f32 opt-out without ah/bh, or an
    /// unmapped slot).
    pub fn offer(self: *Ctx) ?mlend.Offer {
        const b = if (self.pool) |*p| p else return null;
        // A batch in flight means a forward is still encoding into these
        // buffers; the schedule never lends then, so fail closed if it does.
        if (self.batch != null) return null;
        // The offer outlives this call (one decode): pin the set until endOffer.
        self.lent_out = true;
        return zpool.offerFrom(b);
    }

    /// The borrow has ended (the decode returned): the pool may reshape again.
    pub fn endOffer(self: *Ctx) void {
        self.lent_out = false;
    }

    /// f32-branch scratch, allocated on first dispatch within the current
    /// PoolShape (the optional field is the only path to the handle, so an
    /// unallocated buffer cannot reach a dispatch). Freed with the pool.
    fn actHandle(self: *Ctx) !*anyopaque {
        const b = if (self.pool) |*p| p else return error.MetalDispatchFailed;
        if (b.act == null) {
            const s = b.shape;
            b.act = try empty(self.attn.device, s.batch * (txt_len + s.img_len) * s.inner);
        }
        return b.act.?.handle;
    }

    fn cat12Handle(self: *Ctx) !*anyopaque {
        const b = if (self.pool) |*p| p else return error.MetalDispatchFailed;
        if (b.cat12 == null) {
            const s = b.shape;
            b.cat12 = try empty(self.attn.device, s.batch * (txt_len + s.img_len) * (s.hidden + s.inner));
        }
        return b.cat12.?.handle;
    }

    fn lnPipe(self: *const Ctx) *anyopaque {
        return if (self.act_f16) self.pipelines.ln_mod_h else self.pipelines.ln_mod;
    }

    fn assertFloatView(v: tensor.View) !void {
        switch (v.dtype) {
            .bf16, .f16, .f32 => {},
            else => return error.InvalidDType,
        }
    }

    fn weight(self: *Ctx, v: tensor.View) !WeightBind {
        // Packed sidecar view (.u8 marker from zflux2_pack.swap, W6 or W4):
        // no-copy bind of the codes+scales; never the f16 materializer.
        if (v.dtype == .u8 and v.source != null) {
            if (v.shape.len != 2) return error.InvalidShape;
            var temp: ?mbuffer.Buffer = null;
            const bind = try self.wbind.bindView(v, &temp);
            // Source-backed views only: a temp bind here would hand back a freed
            // buffer. The guard above makes this unreachable; fail loudly if not.
            if (temp != null) return error.MetalDispatchFailed;
            return .{
                .handle = bind.handle,
                .offset = bind.offset,
                .dtype = try packed_w.dtypeFor(v.packed_bits),
                .scales_off = bind.offset + try packed_w.scalesBase(v.packed_bits, v.shape[0], v.shape[1]),
            };
        }
        // bf16 -> f16 promotion, once per weight: gemm_half's dtype-1 direct
        // path runs ~10 TF/s vs gemm_exact's ~1.5. Promotion is a value cast
        // (bf16 8-bit mantissa embeds in f16 within range; out-of-range
        // saturates) — Klein is candidate-tier, gated by the oracle cosines.
        try assertFloatView(v);
        if (v.dtype == .f16 and v.source != null) {
            var temp: ?mbuffer.Buffer = null;
            const bind = try self.wbind.bindView(v, &temp);
            // Source-backed views only: a temp bind here would hand back a freed
            // buffer. The guard above makes this unreachable; fail loudly if not.
            if (temp != null) return error.MetalDispatchFailed;
            return .{ .handle = bind.handle, .offset = bind.offset };
        }
        const key = wkey(v);
        if (self.wcache.get(key)) |b| return .{ .handle = b.handle };
        // Diagnostic (default off): print every materialization so cache
        // misses are attributable; a healthy run prints each weight ONCE.
        if (std.c.getenv("ZDRAW_WCACHE_DEBUG") != null) {
            std.debug.print("wcache miss ptr=0x{x} len={d} dtype={d} cached={d}\n", .{ key.ptr, key.len, key.dtype, self.wcache.count() });
        }
        const count = try v.elems();
        const tmp = try self.allocator.alloc(f16, count);
        defer self.allocator.free(tmp);
        for (tmp, 0..) |*d, i| {
            const f = v.atF32Unchecked(i);
            d.* = @floatCast(std.math.clamp(f, -65504.0, 65504.0));
        }
        var buf = try mbuffer.Buffer.fromBytes(self.attn.device, std.mem.sliceAsBytes(tmp));
        errdefer buf.deinit();
        try self.wcache.put(key, buf);
        return .{ .handle = buf.handle };
    }

    fn weightF32(self: *Ctx, v: tensor.View) !*anyopaque {
        try assertFloatView(v);
        const key = wkey(v);
        if (self.wcache_f32.get(key)) |b| return b.handle;
        const count = try v.elems();
        const tmp = try self.allocator.alloc(f32, count);
        defer self.allocator.free(tmp);
        for (tmp, 0..) |*d, i| d.* = v.atF32Unchecked(i);
        var buf = try mbuffer.Buffer.fromBytes(self.attn.device, std.mem.sliceAsBytes(tmp));
        errdefer buf.deinit();
        try self.wcache_f32.put(key, buf);
        return buf.handle;
    }

    /// W6 sidecar GEMM: batch-only dispatch through the split-scales kernel.
    /// Call sites pass w_off in f16 row bytes (row0 * k * 2); the W6 byte
    /// offsets for the same row window are derived from the shared zw6 layout.
    fn gemmRunW6(
        self: *Ctx,
        a: *anyopaque,
        w: WeightBind,
        c: *anyopaque,
        m: usize,
        k: usize,
        n: usize,
        w_off: usize,
        a_off: u64,
        c_off: u64,
    ) !void {
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (w_off % (k * 2) != 0) return error.InvalidShape;
        const row0 = w_off / (k * 2);
        const groups_per_row = (k + zflux2_pack.w6_group - 1) / zflux2_pack.w6_group;
        const p = metal_c.GemmParams{
            .m = @intCast(m),
            .k = @intCast(k),
            .n = @intCast(n),
            .dtype = w.dtype,
            .mode = 2,
            .weight_offset = w.offset + row0 * packed_w.codesPerRow(w.dtype, k),
        };
        const scales_off = w.scales_off + row0 * groups_per_row * 2;
        if (packed_w.runEnc(w.dtype, bt, a, w.handle, c, &p, a_off, c_off, scales_off) != 0) {
            return error.MetalDispatchFailed;
        }
    }

    /// f16-A GEMM (ZDRAW_KLEIN_ACT=f16): A is a half buffer, W the f16 sidecar,
    /// C stays f32 (gemm_f16a_direct's ABI). W6 packs are unsupported in this
    /// mode; non-%32 shapes are rejected by the kernel entry and fail loudly.
    fn gemmRunF16A(
        self: *Ctx,
        a: *anyopaque,
        w: WeightBind,
        c: *anyopaque,
        m: usize,
        k: usize,
        n: usize,
        w_off: usize,
        a_off: u64,
        c_off: u64,
    ) !void {
        if (w.dtype != dtype_f16)
            return packed_w.steel(self.batch, a, w, c, m, k, n, w_off, a_off, c_off);
        const p = metal_c.GemmParams{
            .m = @intCast(m),
            .k = @intCast(k),
            .n = @intCast(n),
            .dtype = w.dtype,
            .mode = 2,
            .weight_offset = w.offset + w_off,
        };
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (self.pipelines.gemm_mpp) |pipe| {
            if (m % 64 == 0 and n % 64 == 0 and k % 32 == 0) {
                const rc = metal_c.zdraw_metal_run_gemm_mpp_enc(
                    bt,
                    pipe,
                    a,
                    w.handle,
                    c,
                    &p,
                    a_off,
                    c_off,
                );
                if (rc != 0) return error.MetalDispatchFailed;
                return;
            }
        }
        if (self.pipelines.mpp_fallback) metal_c.zdraw_metal_note_mpp_fallback();
        if (metal_c.zdraw_metal_run_gemm_f16a_enc(bt, a, w.handle, c, &p, a_off, c_off) != 0) {
            return error.MetalDispatchFailed;
        }
    }

    fn gemmRun(self: *Ctx, a: *anyopaque, w: WeightBind, c: *anyopaque, m: usize, k: usize, n: usize, w_off: usize) !void {
        if (packed_w.isPacked(w.dtype)) return self.gemmRunW6(a, w, c, m, k, n, w_off, 0, 0);
        const p = metal_c.GemmParams{
            .m = @intCast(m),
            .k = @intCast(k),
            .n = @intCast(n),
            .dtype = w.dtype,
            .mode = 2,
            .weight_offset = w.offset + w_off,
        };
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (try self.gemm64Run(bt, a, w, c, &p, 0, 0)) return;
        if (metal_c.zdraw_metal_run_gemm_enc(
            bt,
            self.pipelines.gemm,
            a,
            w.handle,
            c,
            &p,
        ) != 0) {
            return error.MetalDispatchFailed;
        }
    }

    /// gemmRun with A (input) and C (output) byte offsets — the resident
    /// double-block writes a stream's rows straight into shared q/k/v (c_off)
    /// and reads o at the stream offset (a_off), so it needs no qkv/o staging.
    /// Batch path only (the double block always runs batched). ABI unchanged.
    fn gemmRunOff(
        self: *Ctx,
        a: *anyopaque,
        w: WeightBind,
        c: *anyopaque,
        m: usize,
        k: usize,
        n: usize,
        w_off: usize,
        a_off: u64,
        c_off: u64,
    ) !void {
        if (packed_w.isPacked(w.dtype)) return self.gemmRunW6(a, w, c, m, k, n, w_off, a_off, c_off);
        const p = metal_c.GemmParams{
            .m = @intCast(m),
            .k = @intCast(k),
            .n = @intCast(n),
            .dtype = w.dtype,
            .mode = 2,
            .weight_offset = w.offset + w_off,
        };
        const bt = self.batch orelse return error.MetalDispatchFailed;
        if (try self.gemm64Run(bt, a, w, c, &p, a_off, c_off)) return;
        if (metal_c.zdraw_metal_run_gemm_off_enc(bt, self.pipelines.gemm, a, w.handle, c, &p, a_off, c_off) != 0) {
            return error.MetalDispatchFailed;
        }
    }

    fn gemm64Run(
        self: *Ctx,
        batch: *anyopaque,
        a: *anyopaque,
        w: WeightBind,
        c: *anyopaque,
        p: *const metal_c.GemmParams,
        a_off: u64,
        c_off: u64,
    ) !bool {
        if (!self.use_gemm64 or !gemm64Ok(p)) return false;
        const rc = metal_c.zdraw_metal_run_gemm_ours16_enc(batch, a, w.handle, c, p, a_off, c_off);
        return rc == 0;
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

    fn flush(self: *Ctx) !void {
        if (self.batch) |bt| {
            if (metal_c.zdraw_metal_batch_end(bt) != 0) return error.MetalDispatchFailed;
            self.batch = metal_c.zdraw_metal_batch_begin(self.attn.queue) orelse return error.MetalDispatchFailed;
        }
    }
};

fn textProjKey(embeds: []const f32, loaded: *const zflux2.Loaded, hidden: usize) u64 {
    var h = std.hash.Wyhash.init(0x4b6c_6569_6e);
    const weight_ptr = @intFromPtr(loaded.globals.context_embed.bytes.ptr);
    const weight_len = loaded.globals.context_embed.bytes.len;
    const weight_dtype = @intFromEnum(loaded.globals.context_embed.dtype);
    h.update(std.mem.sliceAsBytes(embeds));
    h.update(std.mem.asBytes(&hidden));
    h.update(std.mem.asBytes(&loaded.cfg.joint_dim));
    h.update(std.mem.asBytes(&weight_ptr));
    h.update(std.mem.asBytes(&weight_len));
    h.update(std.mem.asBytes(&weight_dtype));
    return h.final();
}

fn empty(dev: *anyopaque, n: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(dev, n * 4);
}

fn copyF32(ctx: *Ctx, src: *anyopaque, dst: *anyopaque, count: usize) !void {
    const n: u32 = @intCast(count);
    try ctx.glue(
        ctx.pipelines.copy,
        .{ src, dst, null, null, null },
        null,
        &n,
        4,
        2,
        count,
        256,
        0,
    );
}

/// Final adaLN modulation: SiLU(temb) @ norm_out.linear^T -> [2*hidden]
/// (scale-first; see zflux2.final_scale_first). CPU 1-row matmul; caller frees.
fn normOutMods(a: std.mem.Allocator, w: tensor.View, temb: []const f32, hidden: usize) ![]f32 {
    const fmods = try a.alloc(f32, 2 * hidden);
    errdefer a.free(fmods);
    const act = try a.alloc(f32, hidden);
    defer a.free(act);
    for (act, temb) |*d2, v2| d2.* = v2 / (1.0 + @exp(-v2));
    for (0..2 * hidden) |o2| {
        var acc: f32 = 0;
        for (0..hidden) |ic| acc += w.atF32Unchecked(o2 * hidden + ic) * act[ic];
        fmods[o2] = acc;
    }
    return fmods;
}

fn projectText(
    ctx: *Ctx,
    dev: *anyopaque,
    b: *Bufs,
    embeds: []const f32,
    loaded: *const zflux2.Loaded,
    hidden: usize,
) !void {
    const joint_dim = loaded.cfg.joint_dim;
    const key = textProjKey(embeds, loaded, hidden);
    for (&ctx.txt_proj) |*slot| {
        if (slot.*) |*cache| {
            if (cache.matches(key, embeds.len, hidden, joint_dim)) {
                try copyF32(ctx, cache.buf.handle, b.txt.handle, txt_len * hidden);
                return;
            }
        }
    }
    const slot_i = ctx.txt_proj_next;
    if (ctx.txt_proj[slot_i]) |*stale| {
        stale.deinit();
        ctx.txt_proj[slot_i] = null;
    }
    ctx.txt_proj_next = (slot_i + 1) % ctx.txt_proj.len;

    var out = try empty(dev, txt_len * hidden);
    errdefer out.deinit();
    metal_c.zdraw_metal_write_buffer(b.emb.handle, std.mem.sliceAsBytes(embeds).ptr, embeds.len * 4);
    try ctx.gemmRun(
        b.emb.handle,
        try ctx.weight(loaded.globals.context_embed),
        out.handle,
        txt_len,
        joint_dim,
        hidden,
        0,
    );
    try copyF32(ctx, out.handle, b.txt.handle, txt_len * hidden);
    ctx.txt_proj[slot_i] = .{
        .key = key,
        .len = embeds.len,
        .hidden = hidden,
        .joint_dim = joint_dim,
        .buf = out,
    };
}

/// XRES epilogue: apply the last step's Euler update on-GPU and read the
/// final latents back for the VAE.
pub fn finishEuler(ctx: *Ctx, x_out: []f32, dt: f32) !void {
    const b = if (ctx.pool) |*p| p else return error.MetalDispatchFailed;
    const u = ctx.up_lat.buf orelse return error.MetalDispatchFailed;
    if (ctx.up_lat.len != x_out.len * 4) return error.InvalidShape;
    ctx.batch = metal_c.zdraw_metal_batch_begin(ctx.attn.queue) orelse
        return error.MetalDispatchFailed;
    errdefer if (ctx.batch) |bt| {
        _ = metal_c.zdraw_metal_batch_end(bt);
        ctx.batch = null;
    };
    const ap = AxpyParams{ .count = @intCast(x_out.len), .dt = dt };
    try ctx.glue(ctx.pipelines.axpy, .{ u.handle, b.out128.handle, null, null, null }, null, &ap, @sizeOf(AxpyParams), 2, x_out.len, 256, 0);
    if (ctx.batch) |bt| {
        if (metal_c.zdraw_metal_batch_end(bt) != 0) return error.MetalDispatchFailed;
        ctx.batch = null;
    }
    metal_c.zdraw_metal_read_buffer(u.handle, std.mem.sliceAsBytes(x_out).ptr, x_out.len * 4);
}

pub fn forward(
    ctx: *Ctx,
    out: []f32, // [img_tokens, 128]
    latents: []const f32,
    embeds: []const f32, // [txt_len, joint_dim]
    loaded: *const zflux2.Loaded,
    rope: zflux2_dit.Rope,
    t: f32,
    opts: ForwardOpts,
) !void {
    const dev = ctx.attn.device;
    const d = Dims.fromConfig(loaded.cfg);
    const hidden = d.hidden;
    const inner = d.inner;
    const img_len = latents.len / 128;
    const tokens = txt_len + img_len;
    // Attribution trace clock (zero cost when trace off). Buckets sum to the
    // full forward, so the embedder bucket includes the per-step CPU prep below.
    var t_prev: u64 = if (ctx.trace) metrics.now() else 0;

    const b = try ctx.ensurePool(dev, img_len, hidden, inner, loaded.cfg.joint_dim, 1);
    // Write the per-call inputs into the pooled buffers. Always upload (no
    // source-pointer cache): the allocator can reuse an address for a new
    // prompt/rope, so pointer identity is not a valid content key. The cost
    // (~20MB) is negligible against the ~6s/step denoise; a verified
    // generation-counter skip can come later if profiling asks for it.
    metal_c.zdraw_metal_write_buffer(b.rope_cos.handle, std.mem.sliceAsBytes(rope.cos).ptr, rope.cos.len * 4);
    metal_c.zdraw_metal_write_buffer(b.rope_sin.handle, std.mem.sliceAsBytes(rope.sin).ptr, rope.sin.len * 4);

    ctx.batch = metal_c.zdraw_metal_batch_begin(ctx.attn.queue) orelse
        return error.MetalDispatchFailed;
    // On any error before the explicit close, end the batch and clear it so
    // the retained cmd/encoder are released and the next forward starts clean.
    errdefer if (ctx.batch) |bt| {
        _ = metal_c.zdraw_metal_batch_end(bt);
        ctx.batch = null;
    };
    // Modulation vectors: the CPU cache (A2) uploaded, or GPU matvecs in-batch.
    const mh = if (ctx.mods_gpu)
        try ctx.stepModsGpu(loaded, t, hidden)
    else
        try ctx.stepModsCpu(loaded, t, hidden);
    const mods_i_h = mh.img;
    const mods_t_h = mh.txt;
    const mods_s_h = mh.single;
    // embedders (resident from the first GEMM). With dt_in the latents are
    // already device-resident from the previous step: apply the pending Euler
    // update (x += dt * v) as this batch's first op instead of re-uploading.
    const lat_h = if (opts.dt_in == null)
        try ctx.upload(&ctx.up_lat, std.mem.sliceAsBytes(latents))
    else blk: {
        const u = ctx.up_lat.buf orelse return error.MetalDispatchFailed;
        if (ctx.up_lat.len != latents.len * 4) return error.InvalidShape;
        break :blk u.handle;
    };
    if (opts.dt_in) |dt| {
        const ap = AxpyParams{ .count = @intCast(latents.len), .dt = dt };
        try ctx.glue(ctx.pipelines.axpy, .{ lat_h, b.out128.handle, null, null, null }, null, &ap, @sizeOf(AxpyParams), 2, latents.len, 256, 0);
    }
    try ctx.gemmRun(lat_h, try ctx.weight(loaded.globals.x_embed), b.img.handle, img_len, 128, hidden, 0);
    try projectText(ctx, dev, b, embeds, loaded, hidden);
    try ctx.traceMark(&t_prev, &ctx.tr_embed_ns);

    for (loaded.doubles) |blk| {
        try doubleBlock(ctx, b, blk, mods_i_h, mods_t_h, img_len, d);
    }

    // concat text-first (shared layout) -> cat (sequential rows: two copies)
    comptime std.debug.assert(zflux2.cat_txt_first);
    {
        const copy_t: u32 = @intCast(txt_len * hidden);
        const copy_i: u32 = @intCast(img_len * hidden);
        const off_seq = [5]usize{ 0, txt_len * hidden * 4, 0, 0, 0 };
        try ctx.glue(ctx.pipelines.copy, .{ b.txt.handle, b.cat.handle, null, null, null }, null, &copy_t, 4, 2, txt_len * hidden, 256, 0);
        try ctx.glue(ctx.pipelines.copy, .{ b.img.handle, b.cat.handle, null, null, null }, &off_seq, &copy_i, 4, 2, img_len * hidden, 256, 0);
    }
    try ctx.traceMark(&t_prev, &ctx.tr_double_ns);
    for (loaded.singles, 0..) |blk, si| {
        const inst = ctx.trace and si == ctx.trace_single_idx;
        try singleBlock(ctx, b, blk, mods_s_h, tokens, d, inst);
    }
    try ctx.traceMark(&t_prev, &ctx.tr_single_ns);

    // final: ln_mod on the img part + proj_out (fmods from the step cache)
    // norm_out emits its pair SCALE-FIRST (shared truth); scale is the first
    // hidden elements, shift the second. Pinned so a flip of the shared
    // constant is a compile error here, not a silent final-layer mismatch.
    comptime std.debug.assert(zflux2.final_scale_first);
    // img part of cat -> final_in
    {
        const offs = [5]usize{ @as(usize, txt_len) * hidden * 4, 0, 0, 0, 0 };
        const count: u32 = @intCast(img_len * hidden);
        try ctx.glue(ctx.pipelines.copy, .{ b.cat.handle, b.final_in.handle, null, null, null }, &offs, &count, 4, 2, img_len * hidden, 256, 0);
    }
    const gp = GlueParams{ .tokens = @intCast(img_len), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    // shift = fmods[hidden..], scale = fmods[0..hidden]: one buffer, two offsets.
    const f_offs = [5]usize{ 0, 0, hidden * 4, 0, 0 };
    const fb = [5]?*anyopaque{ b.final_in.handle, b.nscratch.handle, mh.fmods, mh.fmods, null };
    try ctx.glue(ctx.pipelines.ln_mod, fb, &f_offs, &gp, @sizeOf(GlueParams), 4, img_len, 256, 1);
    try ctx.gemmRun(b.nscratch.handle, try ctx.weight(loaded.globals.proj_out), b.out128.handle, img_len, hidden, 128, 0);
    if (ctx.batch) |bt| {
        if (metal_c.zdraw_metal_batch_end(bt) != 0) return error.MetalDispatchFailed;
        ctx.batch = null;
    }
    if (opts.read_v) {
        metal_c.zdraw_metal_read_buffer(b.out128.handle, std.mem.sliceAsBytes(out).ptr, out.len * 4);
    }
    if (ctx.trace) {
        ctx.tr_final_ns += metrics.now() -% t_prev;
        ctx.tr_steps += 1;
    }
}

/// Batched multi-seed forward: nb images share the prompt embeds, weights,
/// and timestep mods; latents/out are nb concatenated [img_tokens, 128]
/// segments. LN/FF/fused GEMMs run once over all rows (weights fetched once
/// per layer for the whole batch); attention runs per image. Plain product
/// path only: no XRES, no trace.
pub fn forwardBatch(
    ctx: *Ctx,
    allocator: std.mem.Allocator,
    out: []f32, // [nb * img_tokens, 128]
    latents: []const f32,
    embeds: []const f32, // [txt_len, joint_dim] (shared)
    loaded: *const zflux2.Loaded,
    rope: zflux2_dit.Rope,
    t: f32,
    nb: usize,
) !void {
    const dev = ctx.attn.device;
    const d = Dims.fromConfig(loaded.cfg);
    const hidden = d.hidden;
    const img_len = latents.len / 128 / nb;
    const tokens = txt_len + img_len;

    const mods = try ctx.stepMods(loaded, t, hidden);
    const mods_i_h = try ctx.upload(&ctx.up_mods_i, std.mem.sliceAsBytes(mods.mods_img));
    const mods_t_h = try ctx.upload(&ctx.up_mods_t, std.mem.sliceAsBytes(mods.mods_txt));
    const mods_s_h = try ctx.upload(&ctx.up_mods_s, std.mem.sliceAsBytes(mods.mods_single));

    const b = try ctx.ensurePool(dev, img_len, hidden, d.inner, loaded.cfg.joint_dim, nb);
    // Duplicated rope table: one CPU concat, one upload, so per-row rope
    // indexing works across every image segment.
    const rl = rope.cos.len;
    const rope2 = try allocator.alloc(f32, nb * rl * 2);
    defer allocator.free(rope2);
    for (0..nb) |i| {
        @memcpy(rope2[i * rl ..][0..rl], rope.cos);
        @memcpy(rope2[nb * rl + i * rl ..][0..rl], rope.sin);
    }
    metal_c.zdraw_metal_write_buffer(b.rope_cos.handle, std.mem.sliceAsBytes(rope2[0 .. nb * rl]).ptr, nb * rl * 4);
    metal_c.zdraw_metal_write_buffer(b.rope_sin.handle, std.mem.sliceAsBytes(rope2[nb * rl ..]).ptr, nb * rl * 4);

    ctx.batch = metal_c.zdraw_metal_batch_begin(ctx.attn.queue) orelse
        return error.MetalDispatchFailed;
    errdefer if (ctx.batch) |bt| {
        _ = metal_c.zdraw_metal_batch_end(bt);
        ctx.batch = null;
    };
    const lat_h = try ctx.upload(&ctx.up_lat, std.mem.sliceAsBytes(latents));
    try ctx.gemmRun(lat_h, try ctx.weight(loaded.globals.x_embed), b.img.handle, nb * img_len, 128, hidden, 0);
    // Shared prompt: project once into segment 0, then replicate on-GPU.
    try projectText(ctx, dev, b, embeds, loaded, hidden);
    const tseg: u32 = @intCast(txt_len * hidden);
    for (1..nb) |i| {
        const off = [5]usize{ 0, i * txt_len * hidden * 4, 0, 0, 0 };
        try ctx.glue(ctx.pipelines.copy, .{ b.txt.handle, b.txt.handle, null, null, null }, &off, &tseg, 4, 2, txt_len * hidden, 256, 0);
    }

    for (loaded.doubles) |blk| {
        try doubleBlockB(ctx, b, blk, mods_i_h, mods_t_h, img_len, d, nb);
    }

    comptime std.debug.assert(zflux2.cat_txt_first);
    const copy_t: u32 = @intCast(txt_len * hidden);
    const copy_i: u32 = @intCast(img_len * hidden);
    for (0..nb) |i| {
        const t_off = [5]usize{ i * txt_len * hidden * 4, i * tokens * hidden * 4, 0, 0, 0 };
        const i_off = [5]usize{ i * img_len * hidden * 4, (i * tokens + txt_len) * hidden * 4, 0, 0, 0 };
        try ctx.glue(ctx.pipelines.copy, .{ b.txt.handle, b.cat.handle, null, null, null }, &t_off, &copy_t, 4, 2, txt_len * hidden, 256, 0);
        try ctx.glue(ctx.pipelines.copy, .{ b.img.handle, b.cat.handle, null, null, null }, &i_off, &copy_i, 4, 2, img_len * hidden, 256, 0);
    }
    for (loaded.singles) |blk| {
        try singleBlockB(ctx, b, blk, mods_s_h, tokens, d, nb);
    }

    comptime std.debug.assert(zflux2.final_scale_first);
    const fshift_h = try ctx.upload(&ctx.up_fshift, std.mem.sliceAsBytes(mods.fmods[hidden..]));
    const fscale_h = try ctx.upload(&ctx.up_fscale, std.mem.sliceAsBytes(mods.fmods[0..hidden]));
    for (0..nb) |i| {
        const offs = [5]usize{ (i * tokens + txt_len) * hidden * 4, i * img_len * hidden * 4, 0, 0, 0 };
        try ctx.glue(ctx.pipelines.copy, .{ b.cat.handle, b.final_in.handle, null, null, null }, &offs, &copy_i, 4, 2, img_len * hidden, 256, 0);
    }
    const gp = GlueParams{ .tokens = @intCast(nb * img_len), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    try ctx.glue(ctx.pipelines.ln_mod, .{ b.final_in.handle, b.nscratch.handle, fshift_h, fscale_h, null }, null, &gp, @sizeOf(GlueParams), 4, nb * img_len, 256, 1);
    try ctx.gemmRun(b.nscratch.handle, try ctx.weight(loaded.globals.proj_out), b.out128.handle, nb * img_len, hidden, 128, 0);
    if (ctx.batch) |bt| {
        if (metal_c.zdraw_metal_batch_end(bt) != 0) return error.MetalDispatchFailed;
        ctx.batch = null;
    }
    metal_c.zdraw_metal_read_buffer(b.out128.handle, std.mem.sliceAsBytes(out).ptr, out.len * 4);
}

/// Batched double block: LN/gates/FF run once over all images' rows (full
/// weight amortization); qkv and out projections run per image so each
/// image's rows land at its contiguous [txt_i|img_i] q/k/v segment; attention
/// runs per image. Layout contract: img=[img_0|..], txt=[txt_0|..],
/// q/k/v/o=[txt_0|img_0|txt_1|img_1|..], rope table duplicated per image.
fn doubleBlockB(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Double,
    mods_i: *anyopaque,
    mods_t: *anyopaque,
    img_len: usize,
    d: Dims,
    nb: usize,
) !void {
    const hidden = d.hidden;
    const tokens = txt_len + img_len;
    const gp_i = GlueParams{ .tokens = @intCast(nb * img_len), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    const gp_t = GlueParams{ .tokens = @intCast(nb * txt_len), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    const gp1_i = GlueParams{ .tokens = @intCast(img_len), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    const gp1_t = GlueParams{ .tokens = txt_len, .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };

    const offs_msa = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.shift, hidden), modByteOff(zflux2.mod_msa, ModOff.scale, hidden), 0 };
    // Mirrors doubleBlock's af branch: half LN outputs feed the f16-A GEMMs.
    const af = ctx.act_f16;
    const img_a: *anyopaque = if (af) b.ah.?.handle else b.nscratch.handle;
    const txt_a: *anyopaque = if (af) b.bh.?.handle else b.norm_t.handle;
    try ctx.glue(ctx.lnPipe(), .{ b.img.handle, img_a, mods_i, mods_i, null }, &offs_msa, &gp_i, @sizeOf(GlueParams), 4, nb * img_len, 256, 1);
    try ctx.glue(ctx.lnPipe(), .{ b.txt.handle, txt_a, mods_t, mods_t, null }, &offs_msa, &gp_t, @sizeOf(GlueParams), 4, nb * txt_len, 256, 1);

    try qkvDualB(ctx, b, blk, txt_a, img_a, img_len, hidden, nb);

    // Duplicated rope table: rope row index == q/k row index per segment.
    // Steel arm (the single image's route) first; the MFA/owned path when
    // the steel route is unavailable.
    const steel_done = try steelDoubleB(ctx, b, blk, tokens, img_len, d.heads, nb);
    if (!steel_done) {
        for (0..nb) |i| {
            try rmsRope(ctx, b, blk.norm_added_q, &b.q, i * tokens, txt_len, gp1_t);
            try rmsRope(ctx, b, blk.norm_q, &b.q, i * tokens + txt_len, img_len, gp1_i);
            try rmsRope(ctx, b, blk.norm_added_k, &b.k, i * tokens, txt_len, gp1_t);
            try rmsRope(ctx, b, blk.norm_k, &b.k, i * tokens + txt_len, img_len, gp1_i);
        }
        try runAttnBatch(ctx, b, tokens, d.heads, nb, tokens * hidden * 4);
    }

    for (0..nb) |i| {
        const o_t: u64 = @intCast(i * tokens * hidden * 4);
        const o_i: u64 = @intCast((i * tokens + txt_len) * hidden * 4);
        try ctx.gemmRunOff(b.o.handle, try ctx.weight(blk.to_add_out), b.proj_t.handle, txt_len, hidden, hidden, 0, o_t, @intCast(i * txt_len * hidden * 4));
        try ctx.gemmRunOff(b.o.handle, try ctx.weight(blk.to_out), b.proj_i.handle, img_len, hidden, hidden, 0, o_i, @intCast(i * img_len * hidden * 4));
    }
    const gate_off = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.gate, hidden), 0, 0 };
    try ctx.glue(ctx.pipelines.gate_add, .{ b.txt.handle, b.proj_t.handle, mods_t, null, null }, &gate_off, &gp_t, @sizeOf(GlueParams), 3, nb * txt_len * hidden, 256, 0);
    try ctx.glue(ctx.pipelines.gate_add, .{ b.img.handle, b.proj_i.handle, mods_i, null, null }, &gate_off, &gp_i, @sizeOf(GlueParams), 3, nb * img_len * hidden, 256, 0);

    try ffStream(ctx, b, &b.img, blk.ff_in, blk.ff_out, mods_i, nb * img_len, gp_i, d);
    try ffStream(ctx, b, &b.txt, blk.ffc_in, blk.ffc_out, mods_t, nb * txt_len, gp_t, d);
}

/// The six qkv GEMMs of a double block, one image segment at a time in the
/// serial dispatch order (txt q/k/v then img q/k/v), written straight into
/// shared q/k/v at each stream's row offset via the GEMM C-offset (no
/// staging buffers, no concat copies). Serves the serial block at nb=1.
/// A-offsets scale by the activation element size (2 for half); C offsets
/// stay f32 (*4).
fn qkvDualB(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Double,
    txt_a: *anyopaque,
    img_a: *anyopaque,
    img_len: usize,
    hidden: usize,
    nb: usize,
) !void {
    const tokens = txt_len + img_len;
    const af = ctx.act_f16;
    const a_el: u64 = if (af) 2 else 4;
    const aq = try ctx.weight(blk.add_q);
    const ak = try ctx.weight(blk.add_k);
    const av = try ctx.weight(blk.add_v);
    const tq = try ctx.weight(blk.to_q);
    const tk = try ctx.weight(blk.to_k);
    const tv = try ctx.weight(blk.to_v);
    for (0..nb) |i| {
        const t_a: u64 = @intCast(i * txt_len * hidden * a_el);
        const t_c: u64 = @intCast(i * tokens * hidden * 4);
        const i_a: u64 = @intCast(i * img_len * hidden * a_el);
        const i_c: u64 = @intCast((i * tokens + txt_len) * hidden * 4);
        if (af) {
            try ctx.gemmRunF16A(txt_a, aq, b.q.handle, txt_len, hidden, hidden, 0, t_a, t_c);
            try ctx.gemmRunF16A(txt_a, ak, b.k.handle, txt_len, hidden, hidden, 0, t_a, t_c);
            try ctx.gemmRunF16A(txt_a, av, b.v.handle, txt_len, hidden, hidden, 0, t_a, t_c);
            try ctx.gemmRunF16A(img_a, tq, b.q.handle, img_len, hidden, hidden, 0, i_a, i_c);
            try ctx.gemmRunF16A(img_a, tk, b.k.handle, img_len, hidden, hidden, 0, i_a, i_c);
            try ctx.gemmRunF16A(img_a, tv, b.v.handle, img_len, hidden, hidden, 0, i_a, i_c);
        } else {
            try ctx.gemmRunOff(txt_a, aq, b.q.handle, txt_len, hidden, hidden, 0, t_a, t_c);
            try ctx.gemmRunOff(txt_a, ak, b.k.handle, txt_len, hidden, hidden, 0, t_a, t_c);
            try ctx.gemmRunOff(txt_a, av, b.v.handle, txt_len, hidden, hidden, 0, t_a, t_c);
            try ctx.gemmRunOff(img_a, tq, b.q.handle, img_len, hidden, hidden, 0, i_a, i_c);
            try ctx.gemmRunOff(img_a, tk, b.k.handle, img_len, hidden, hidden, 0, i_a, i_c);
            try ctx.gemmRunOff(img_a, tv, b.v.handle, img_len, hidden, hidden, 0, i_a, i_c);
        }
    }
}

fn doubleBlock(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Double,
    mods_i: *anyopaque,
    mods_t: *anyopaque,
    img_len: usize,
    d: Dims,
) !void {
    const hidden = d.hidden;
    const tokens = txt_len + img_len;
    const gp_i = GlueParams{ .tokens = @intCast(img_len), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    const gp_t = GlueParams{ .tokens = txt_len, .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };

    // norm + msa mods (shared layout: msa set then mlp set, [shift|scale|gate]).
    // kln_mod reads shift at buffer[2], scale at buffer[3] (byte offsets).
    const offs_msa = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.shift, hidden), modByteOff(zflux2.mod_msa, ModOff.scale, hidden), 0 };
    const af = ctx.act_f16;
    const img_a: *anyopaque = if (af) b.ah.?.handle else b.nscratch.handle;
    const txt_a: *anyopaque = if (af) b.bh.?.handle else b.norm_t.handle;
    try ctx.glue(ctx.lnPipe(), .{ b.img.handle, img_a, mods_i, mods_i, null }, &offs_msa, &gp_i, @sizeOf(GlueParams), 4, img_len, 256, 1);
    try ctx.glue(ctx.lnPipe(), .{ b.txt.handle, txt_a, mods_t, mods_t, null }, &offs_msa, &gp_t, @sizeOf(GlueParams), 4, txt_len, 256, 1);

    // qkv per stream via the shared helper at nb=1 (i=0 offsets reduce to
    // txt rows at 0 and img rows at txt_len*hidden*4, exactly the serial map).
    try qkvDualB(ctx, b, blk, txt_a, img_a, img_len, hidden, 1);
    const img_off: u64 = txt_len * hidden * 4;

    // qk norms + rope: txt part with norm_added_*, img part with norm_*
    if (hmRoute(ctx, tokens)) {
        const qhm = try hmScratch(ctx, 0, tokens, d.heads);
        const khm = try hmScratch(ctx, 1, tokens, d.heads);
        try rmsRopeHm(ctx, b, blk.norm_added_q, &b.q, 0, txt_len, tokens, d.heads, qhm);
        try rmsRopeHm(ctx, b, blk.norm_q, &b.q, txt_len, img_len, tokens, d.heads, qhm);
        try rmsRopeHm(ctx, b, blk.norm_added_k, &b.k, 0, txt_len, tokens, d.heads, khm);
        try rmsRopeHm(ctx, b, blk.norm_k, &b.k, txt_len, img_len, tokens, d.heads, khm);
        try runAttnHm(ctx, b, tokens, d.heads, qhm, khm, false);
    } else {
        try rmsRope(ctx, b, blk.norm_added_q, &b.q, 0, txt_len, gp_t);
        try rmsRope(ctx, b, blk.norm_q, &b.q, txt_len, img_len, gp_i);
        try rmsRope(ctx, b, blk.norm_added_k, &b.k, 0, txt_len, gp_t);
        try rmsRope(ctx, b, blk.norm_k, &b.k, txt_len, img_len, gp_i);
        try runAttn(ctx, b, tokens, d.heads);
    }

    // out projections read o directly at the stream's row offset (A-offset) — no
    // ot/oi split copies. Gated residuals unchanged.
    try ctx.gemmRun(b.o.handle, try ctx.weight(blk.to_add_out), b.proj_t.handle, txt_len, hidden, hidden, 0);
    try ctx.gemmRunOff(b.o.handle, try ctx.weight(blk.to_out), b.proj_i.handle, img_len, hidden, hidden, 0, img_off, 0);
    const gate_off = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.gate, hidden), 0, 0 };
    try ctx.glue(ctx.pipelines.gate_add, .{ b.txt.handle, b.proj_t.handle, mods_t, null, null }, &gate_off, &gp_t, @sizeOf(GlueParams), 3, txt_len * hidden, 256, 0);
    try ctx.glue(ctx.pipelines.gate_add, .{ b.img.handle, b.proj_i.handle, mods_i, null, null }, &gate_off, &gp_i, @sizeOf(GlueParams), 3, img_len * hidden, 256, 0);

    // FF per stream with the mlp mod set (offset 3*hidden)
    try ffStream(ctx, b, &b.img, blk.ff_in, blk.ff_out, mods_i, img_len, gp_i, d);
    try ffStream(ctx, b, &b.txt, blk.ffc_in, blk.ffc_out, mods_t, txt_len, gp_t, d);
}

fn rmsRope(ctx: *Ctx, b: *Bufs, w: tensor.View, buf: *mbuffer.Buffer, tok_off: usize, ntok: usize, gp: GlueParams) !void {
    const hidden: usize = gp.hidden;
    if (try w.elems() < head_dim) return error.InvalidShape;
    const wbuf = try ctx.weightF32(w);
    const offs = [5]usize{ tok_off * hidden * 4, 0, tok_off * head_dim * 4, tok_off * head_dim * 4, 0 };
    try ctx.glue(ctx.pipelines.rms_rope, .{ buf.handle, wbuf, b.rope_cos.handle, b.rope_sin.handle, null }, &offs, &gp, @sizeOf(GlueParams), 4, ntok * gp.heads, 256, 0);
}

fn runAttn(ctx: *Ctx, b: *Bufs, tokens: usize, heads: usize) !void {
    const picked = ctx.attn.pick(tokens, head_dim);
    const params = metal_c.AttnParams{
        .tokens = @intCast(tokens),
        .heads = @intCast(heads),
        .kv_heads = @intCast(heads),
        .head_dim = head_dim,
        .causal = 0,
    };
    const bt = ctx.batch orelse return error.MetalDispatchFailed;
    if (ctx.use_mfa_attn and tokens >= ctx.mfa_min_tokens) {
        const rc = metal_c.zdraw_metal_run_attention_mfa_enc(
            bt,
            b.q.handle,
            b.k.handle,
            b.v.handle,
            b.o.handle,
            &params,
            0,
            0,
        );
        if (rc == 0) return;
    }
    if (metal_c.zdraw_metal_run_attention_enc(bt, picked.pipeline, b.q.handle, b.k.handle, b.v.handle, b.o.handle, &params, @intFromEnum(picked.kernel), picked.threads) != 0) {
        return error.MetalDispatchFailed;
    }
}

fn hmRoute(ctx: *Ctx, tokens: usize) bool {
    return ctx.hm_qk and ctx.use_mfa_attn and tokens >= ctx.mfa_min_tokens;
}

fn hmScratch(ctx: *Ctx, slot: c_int, tokens: usize, heads: usize) !*anyopaque {
    return hmScratchBytes(ctx, slot, tokens * heads * head_dim * 2);
}

fn hmScratchBytes(ctx: *Ctx, slot: c_int, bytes: usize) !*anyopaque {
    return metal_c.zdraw_metal_attn_hm_scratch(ctx.attn.device, slot, bytes) orelse
        error.MetalDispatchFailed;
}

/// rmsRope variant writing head-major half into `out` (the MFA q or k
/// scratch) instead of the f32 row in place; x/cos/sin are offset by tok_off
/// exactly as rmsRope does, the destination index uses tok_off + t.
fn rmsRopeHm(
    ctx: *Ctx,
    b: *Bufs,
    w: tensor.View,
    buf: *mbuffer.Buffer,
    tok_off: usize,
    ntok: usize,
    total: usize,
    heads: usize,
    out: *anyopaque,
) !void {
    try rmsRopeHmAt(ctx, b, w, buf, tok_off, tok_off, ntok, total, heads, out);
}

/// The same with the source rows (x, cos, sin) at `src_row` and the
/// head-major destination at local token `dst_off`: image i of a --seeds
/// batch reads rows i*tokens + local and lands in a one-image scratch.
fn rmsRopeHmAt(
    ctx: *Ctx,
    b: *Bufs,
    w: tensor.View,
    buf: *mbuffer.Buffer,
    src_row: usize,
    dst_off: usize,
    ntok: usize,
    total: usize,
    heads: usize,
    out: *anyopaque,
) !void {
    const hidden = heads * head_dim;
    if (try w.elems() < head_dim) return error.InvalidShape;
    const wbuf = try ctx.weightF32(w);
    const p = RopeHmParams{
        .tokens = @intCast(ntok),
        .heads = @intCast(heads),
        .head_dim = head_dim,
        .tok_off = @intCast(dst_off),
        .total = @intCast(total),
    };
    const t_off = src_row * head_dim * 4;
    const offs = [5]usize{ src_row * hidden * 4, 0, t_off, t_off, 0 };
    const bufs = [5]?*anyopaque{ buf.handle, wbuf, b.rope_cos.handle, b.rope_sin.handle, out };
    const pipe = ctx.pipelines.rms_rope_hm;
    try ctx.glue(pipe, bufs, &offs, &p, @sizeOf(RopeHmParams), 5, ntok * heads, 256, 0);
}

/// MFA attention with q/k already head-major half in the scratch slots
/// (rmsRopeHm); v is converted inside, O is un-permuted into b.o as before.
fn runAttnHm(
    ctx: *Ctx,
    b: *Bufs,
    tokens: usize,
    heads: usize,
    qhm: *anyopaque,
    khm: *anyopaque,
    keep_o_hm: bool,
) !void {
    const params = metal_c.AttnParams{
        .tokens = @intCast(tokens),
        .heads = @intCast(heads),
        .kv_heads = @intCast(heads),
        .head_dim = head_dim,
        .causal = 0,
    };
    const bt = ctx.batch orelse return error.MetalDispatchFailed;
    const v = b.v.handle;
    const o = b.o.handle;
    const keep: c_int = if (keep_o_hm) 1 else 0;
    ctx.steel_ran = false;
    if (ctx.attn_steel) {
        const src = metal_c.zdraw_metal_run_attention_steel_route_enc(
            bt,
            qhm,
            khm,
            v,
            o,
            &params,
            keep,
            0,
            0,
        );
        if (src == 0) {
            ctx.steel_ran = true;
            return;
        }
    }
    const rc = metal_c.zdraw_metal_run_attention_mfa_hm_enc(
        bt,
        qhm,
        khm,
        v,
        o,
        &params,
        0,
        0,
        keep,
    );
    if (rc != 0) return error.MetalDispatchFailed;
}

/// Steel attention for image `i` of a --seeds batch: q/k already head-major
/// in the one-image scratch slots, v and o addressed by the segment offset.
/// Returns false when the steel route is unavailable (caller falls back).
fn steelBatchSeg(ctx: *Ctx, b: *Bufs, tokens: usize, heads: usize, i: usize) !bool {
    const params = metal_c.AttnParams{
        .tokens = @intCast(tokens),
        .heads = @intCast(heads),
        .kv_heads = @intCast(heads),
        .head_dim = head_dim,
        .causal = 0,
    };
    const bt = ctx.batch orelse return error.MetalDispatchFailed;
    const qhm = try hmScratch(ctx, 0, tokens, heads);
    const khm = try hmScratch(ctx, 1, tokens, heads);
    const off: usize = i * tokens * heads * head_dim * 4;
    const v = b.v.handle;
    const o = b.o.handle;
    const run = metal_c.zdraw_metal_run_attention_steel_route_enc;
    return run(bt, qhm, khm, v, o, &params, 0, off, off) == 0;
}

/// The batch takes the steel route when the single image would.
fn steelBatch(ctx: *Ctx, tokens: usize) bool {
    return ctx.attn_steel and hmRoute(ctx, tokens);
}

/// The double block's steel arm for a --seeds batch: per image, the txt and
/// img q/k rows (norm_added_* / norm_*) head-major into the one-image
/// scratch, then steel on that image's segment. False = not taken.
fn steelDoubleB(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Double,
    tokens: usize,
    img_len: usize,
    heads: usize,
    nb: usize,
) !bool {
    if (!steelBatch(ctx, tokens)) return false;
    for (0..nb) |i| {
        const qhm = try hmScratch(ctx, 0, tokens, heads);
        const khm = try hmScratch(ctx, 1, tokens, heads);
        const r0 = i * tokens;
        const ri = r0 + txt_len;
        try rmsRopeHmAt(ctx, b, blk.norm_added_q, &b.q, r0, 0, txt_len, tokens, heads, qhm);
        try rmsRopeHmAt(ctx, b, blk.norm_q, &b.q, ri, txt_len, img_len, tokens, heads, qhm);
        try rmsRopeHmAt(ctx, b, blk.norm_added_k, &b.k, r0, 0, txt_len, tokens, heads, khm);
        try rmsRopeHmAt(ctx, b, blk.norm_k, &b.k, ri, txt_len, img_len, tokens, heads, khm);
        if (!try steelBatchSeg(ctx, b, tokens, heads, i)) return false;
    }
    return true;
}

/// The single block's steel arm: one norm/rope pair per image.
fn steelSingleB(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Single,
    tokens: usize,
    heads: usize,
    nb: usize,
) !bool {
    if (!steelBatch(ctx, tokens)) return false;
    for (0..nb) |i| {
        const qhm = try hmScratch(ctx, 0, tokens, heads);
        const khm = try hmScratch(ctx, 1, tokens, heads);
        try rmsRopeHmAt(ctx, b, blk.norm_q, &b.q, i * tokens, 0, tokens, tokens, heads, qhm);
        try rmsRopeHmAt(ctx, b, blk.norm_k, &b.k, i * tokens, 0, tokens, tokens, heads, khm);
        if (!try steelBatchSeg(ctx, b, tokens, heads, i)) return false;
    }
    return true;
}

/// Batched multi-seed attention: one call per image over that image's
/// contiguous [tokens, hidden] segment of the shared q/k/v/o buffers (byte
/// offset i*seg_bytes). Never crosses images. Per-segment MFA first when the
/// route is on (identical math to the serial single-image MFA call, so the
/// batched output matches serial above the threshold), block16 otherwise.
fn runAttnBatch(ctx: *Ctx, b: *Bufs, tokens: usize, heads: usize, nb: usize, seg_bytes: usize) !void {
    const picked = ctx.attn.pick(tokens, head_dim);
    const params = metal_c.AttnParams{
        .tokens = @intCast(tokens),
        .heads = @intCast(heads),
        .kv_heads = @intCast(heads),
        .head_dim = head_dim,
        .causal = 0,
    };
    const bt = ctx.batch orelse return error.MetalDispatchFailed;
    const try_mfa = ctx.use_mfa_attn and tokens >= ctx.mfa_min_tokens;
    for (0..nb) |i| {
        const off: u64 = @intCast(i * seg_bytes);
        if (try_mfa) {
            const rc = metal_c.zdraw_metal_run_attention_mfa_enc(
                bt,
                b.q.handle,
                b.k.handle,
                b.v.handle,
                b.o.handle,
                &params,
                off,
                off,
            );
            if (rc == 0) continue;
        }
        if (metal_c.zdraw_metal_run_attention_enc_off(bt, picked.pipeline, b.q.handle, b.k.handle, b.v.handle, b.o.handle, &params, @intFromEnum(picked.kernel), picked.threads, off, off) != 0) {
            return error.MetalDispatchFailed;
        }
    }
}

fn ffStream(
    ctx: *Ctx,
    b: *Bufs,
    state: *mbuffer.Buffer,
    w_in: tensor.View,
    w_out: tensor.View,
    mods: *anyopaque,
    ntok: usize,
    gp: GlueParams,
    d: Dims,
) !void {
    const hidden = d.hidden;
    const inner = d.inner;
    const af = ctx.act_f16;
    // The mlp modulation set (second set, [shift|scale|gate]) within the vector.
    const offs_mlp = [5]usize{ 0, 0, modByteOff(zflux2.mod_mlp, ModOff.shift, hidden), modByteOff(zflux2.mod_mlp, ModOff.scale, hidden), 0 };
    const ln_a: *anyopaque = if (af) b.ah.?.handle else b.nscratch.handle;
    try ctx.glue(ctx.lnPipe(), .{ state.handle, ln_a, mods, mods, null }, &offs_mlp, &gp, @sizeOf(GlueParams), 4, ntok, 256, 1);
    if (af) {
        try ctx.gemmRunF16A(ln_a, try ctx.weight(w_in), b.wide.handle, ntok, hidden, inner * 2, 0, 0, 0);
    } else {
        try ctx.gemmRun(ln_a, try ctx.weight(w_in), b.wide.handle, ntok, hidden, inner * 2, 0);
    }
    const tk: u32 = @intCast(ntok);
    const in32: u32 = @intCast(inner);
    const cb: [2]u32 = .{ tk, in32 };
    if (af) {
        try ctx.glue(ctx.pipelines.swiglu_h, .{ b.wide.handle, b.bh.?.handle, null, null, null }, null, &cb, 8, 2, ntok * inner, 256, 0);
        try ctx.gemmRunF16A(b.bh.?.handle, try ctx.weight(w_out), b.nscratch.handle, ntok, inner, hidden, 0, 0, 0);
    } else {
        const act_h = try ctx.actHandle();
        try ctx.glue(ctx.pipelines.swiglu, .{ b.wide.handle, act_h, null, null, null }, null, &cb, 8, 2, ntok * inner, 256, 0);
        try ctx.gemmRun(act_h, try ctx.weight(w_out), b.nscratch.handle, ntok, inner, hidden, 0);
    }
    const gate_off = [5]usize{ 0, 0, modByteOff(zflux2.mod_mlp, ModOff.gate, hidden), 0, 0 };
    try ctx.glue(ctx.pipelines.gate_add, .{ state.handle, b.nscratch.handle, mods, null, null }, &gate_off, &gp, @sizeOf(GlueParams), 3, ntok * hidden, 256, 0);
}

/// Batched single block: every op except attention runs once over all
/// nb*tokens rows (cat is per-image contiguous [txt_i|img_i], so the fused
/// qkv/mlp and out GEMMs get full weight amortization); attention runs per
/// image segment. rope table is duplicated per image, so one rope pass covers
/// all rows.
fn singleBlockB(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Single,
    mods: *anyopaque,
    tokens: usize,
    d: Dims,
    nb: usize,
) !void {
    const hidden = d.hidden;
    const inner = d.inner;
    const all = nb * tokens;
    const gp = GlueParams{ .tokens = @intCast(all), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    const offs_msa = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.shift, hidden), modByteOff(zflux2.mod_msa, ModOff.scale, hidden), 0 };
    // The batched single block is singleBlock's body over all = nb*tokens
    // contiguous rows; the af branch mirrors it with zero extra offsets.
    const af = ctx.act_f16;
    const norm_a: *anyopaque = if (af) b.ah.?.handle else b.norm_c.handle;
    try ctx.glue(ctx.lnPipe(), .{ b.cat.handle, norm_a, mods, mods, null }, &offs_msa, &gp, @sizeOf(GlueParams), 4, all, 256, 1);

    const wq = try ctx.weight(blk.qkv_mlp);
    const row_bytes = hidden * 2;
    const r_q = zflux2.SingleRows.q * hidden * row_bytes;
    const r_k = zflux2.SingleRows.k * hidden * row_bytes;
    const r_v = zflux2.SingleRows.v * hidden * row_bytes;
    const r_m = zflux2.SingleRows.qkv * hidden * row_bytes;
    if (af) {
        try ctx.gemmRunF16A(norm_a, wq, b.q.handle, all, hidden, hidden, r_q, 0, 0);
        try ctx.gemmRunF16A(norm_a, wq, b.k.handle, all, hidden, hidden, r_k, 0, 0);
        try ctx.gemmRunF16A(norm_a, wq, b.v.handle, all, hidden, hidden, r_v, 0, 0);
        try ctx.gemmRunF16A(norm_a, wq, b.wide.handle, all, hidden, inner * 2, r_m, 0, 0);
    } else {
        try ctx.gemmRun(norm_a, wq, b.q.handle, all, hidden, hidden, r_q);
        try ctx.gemmRun(norm_a, wq, b.k.handle, all, hidden, hidden, r_k);
        try ctx.gemmRun(norm_a, wq, b.v.handle, all, hidden, hidden, r_v);
        try ctx.gemmRun(norm_a, wq, b.wide.handle, all, hidden, inner * 2, r_m);
    }

    const steel_done = try steelSingleB(ctx, b, blk, tokens, d.heads, nb);
    if (!steel_done) {
        try rmsRope(ctx, b, blk.norm_q, &b.q, 0, all, gp);
        try rmsRope(ctx, b, blk.norm_k, &b.k, 0, all, gp);
        try runAttnBatch(ctx, b, tokens, d.heads, nb, tokens * hidden * 4);
    }

    try outFusedB(ctx, b, blk, all, hidden, inner);
    const gate_off = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.gate, hidden), 0, 0 };
    try ctx.glue(ctx.pipelines.gate_add, .{ b.cat.handle, b.norm_c.handle, mods, null, null }, &gate_off, &gp, @sizeOf(GlueParams), 3, all * hidden, 256, 0);
}

/// The batched single block's fused tail: swiglu, cat_rows, and the out
/// GEMM over all = nb*tokens contiguous rows (zero per-image offsets).
fn outFusedB(ctx: *Ctx, b: *Bufs, blk: zflux2.Single, all: usize, hidden: usize, inner: usize) !void {
    const cb: [2]u32 = .{ @intCast(all), @intCast(inner) };
    const cat_dims: [2]u32 = .{ @intCast(hidden), @intCast(inner) };
    const wide = b.wide.handle;
    const o = b.o.handle;
    const wout = try ctx.weight(blk.out);
    const cat_n = all * (hidden + inner);
    if (ctx.act_f16) {
        const bh = b.bh.?.handle;
        const ah = b.ah.?.handle;
        const sw = ctx.pipelines.swiglu_h;
        const cr = ctx.pipelines.cat_rows_h;
        try ctx.glue(sw, .{ wide, bh, null, null, null }, null, &cb, 8, 2, all * inner, 256, 0);
        try ctx.glue(cr, .{ o, bh, ah, null, null }, null, &cat_dims, 8, 3, cat_n, 256, 0);
        try ctx.gemmRunF16A(ah, wout, b.norm_c.handle, all, hidden + inner, hidden, 0, 0, 0);
    } else {
        const act_h = try ctx.actHandle();
        const cat12_h = try ctx.cat12Handle();
        const sw = ctx.pipelines.swiglu;
        const cr = ctx.pipelines.cat_rows;
        try ctx.glue(sw, .{ wide, act_h, null, null, null }, null, &cb, 8, 2, all * inner, 256, 0);
        try ctx.glue(cr, .{ o, act_h, cat12_h, null, null }, null, &cat_dims, 8, 3, cat_n, 256, 0);
        try ctx.gemmRun(cat12_h, wout, b.norm_c.handle, all, hidden + inner, hidden, 0);
    }
}

fn singleBlock(
    ctx: *Ctx,
    b: *Bufs,
    blk: zflux2.Single,
    mods: *anyopaque,
    tokens: usize,
    d: Dims,
    instrument: bool, // pass-2: sub-divide THIS block into category buckets
) !void {
    const hidden = d.hidden;
    const inner = d.inner;
    const gp = GlueParams{ .tokens = @intCast(tokens), .hidden = @intCast(hidden), .heads = @intCast(d.heads), .head_dim = head_dim };
    // Drain the preceding (un-instrumented) blocks so this block's first mark
    // measures only this block — otherwise the first glue bucket absorbs all of
    // blocks 0..N-1 (they run without a flush). Instrumented block only.
    if (instrument) try ctx.flush();
    // pass-2 local clocks: wall (t2, flush-biased) + GPU-active (g2, unbiased).
    var t2: u64 = if (instrument) metrics.now() else 0;
    var g2: u64 = if (instrument) metal_c.zdraw_metal_gpu_ns() else 0;
    // mod_single carries one [shift|scale|gate] set; index it msa-style.
    const offs_msa = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.shift, hidden), modByteOff(zflux2.mod_msa, ModOff.scale, hidden), 0 };
    // f16 activation mode: ln output and the qkv/mlp A operand live in the
    // half scratch (ah); math is unchanged (f32 accumulate, half store).
    const af = ctx.act_f16;
    const norm_a: *anyopaque = if (af) b.ah.?.handle else b.norm_c.handle;
    try ctx.glue(ctx.lnPipe(), .{ b.cat.handle, norm_a, mods, mods, null }, &offs_msa, &gp, @sizeOf(GlueParams), 4, tokens, 256, 1);
    if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);

    // fused projection split by weight row offsets (bf16 = 2 bytes/elem); the
    // qkv/mlp row layout is shared truth (zflux2.SingleRows).
    const wq = try ctx.weight(blk.qkv_mlp);
    const row_bytes = hidden * 2;
    if (af) {
        try ctx.gemmRunF16A(norm_a, wq, b.q.handle, tokens, hidden, hidden, zflux2.SingleRows.q * hidden * row_bytes, 0, 0);
        try ctx.gemmRunF16A(norm_a, wq, b.k.handle, tokens, hidden, hidden, zflux2.SingleRows.k * hidden * row_bytes, 0, 0);
        try ctx.gemmRunF16A(norm_a, wq, b.v.handle, tokens, hidden, hidden, zflux2.SingleRows.v * hidden * row_bytes, 0, 0);
        try ctx.gemmRunF16A(norm_a, wq, b.wide.handle, tokens, hidden, inner * 2, zflux2.SingleRows.qkv * hidden * row_bytes, 0, 0);
    } else {
        try ctx.gemmRun(norm_a, wq, b.q.handle, tokens, hidden, hidden, zflux2.SingleRows.q * hidden * row_bytes);
        try ctx.gemmRun(norm_a, wq, b.k.handle, tokens, hidden, hidden, zflux2.SingleRows.k * hidden * row_bytes);
        try ctx.gemmRun(norm_a, wq, b.v.handle, tokens, hidden, hidden, zflux2.SingleRows.v * hidden * row_bytes);
        try ctx.gemmRun(norm_a, wq, b.wide.handle, tokens, hidden, inner * 2, zflux2.SingleRows.qkv * hidden * row_bytes);
    }
    if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_qkv_ns, &g2, &ctx.tr_sb_qkv_gpu);

    // On the MFA route with half activations the attention output stays
    // head-major in the scratch and is cast straight into the concat operand
    // (kunperm_hm_h), so neither the f32 un-permute nor kcat_rows_h runs.
    const o_hm = hmRoute(ctx, tokens) and af;
    if (hmRoute(ctx, tokens)) {
        const qhm = try hmScratch(ctx, 0, tokens, d.heads);
        const khm = try hmScratch(ctx, 1, tokens, d.heads);
        try rmsRopeHm(ctx, b, blk.norm_q, &b.q, 0, tokens, tokens, d.heads, qhm);
        try rmsRopeHm(ctx, b, blk.norm_k, &b.k, 0, tokens, tokens, d.heads, khm);
        if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);
        try runAttnHm(ctx, b, tokens, d.heads, qhm, khm, o_hm);
    } else {
        try rmsRope(ctx, b, blk.norm_q, &b.q, 0, tokens, gp);
        try rmsRope(ctx, b, blk.norm_k, &b.k, 0, tokens, gp);
        if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);
        try runAttn(ctx, b, tokens, d.heads);
    }
    if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_attn_ns, &g2, &ctx.tr_sb_attn_gpu);

    const tk: u32 = @intCast(tokens);
    const cb: [2]u32 = .{ tk, @intCast(inner) };
    const cat_dims: [2]u32 = .{ @intCast(hidden), @intCast(inner) };
    if (o_hm) {
        // O head-major f32 (attention scratch slot 3) -> ah[:, 0..hidden] as
        // half, swiglu -> ah[:, hidden..] directly: no bh, no concat copy.
        const stride: u32 = @intCast(hidden + inner);
        const ah = b.ah.?.handle;
        const o_scratch = try hmScratchBytes(ctx, 3, tokens * hidden * 4);
        const uq: [4]u32 = .{ tk, @intCast(d.heads), head_dim, stride };
        const up = if (ctx.steel_ran) ctx.pipelines.unperm_hm_hh else ctx.pipelines.unperm_hm_h;
        const un = tokens * hidden;
        try ctx.glue(up, .{ o_scratch, ah, null, null, null }, null, &uq, 16, 2, un, 256, 0);
        const sq: [4]u32 = .{ tk, @intCast(inner), stride, @intCast(hidden) };
        const sp = ctx.pipelines.swiglu_hs;
        const wide = b.wide.handle;
        try ctx.glue(sp, .{ wide, ah, null, null, null }, null, &sq, 16, 2, tokens * inner, 256, 0);
        if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);
        const wout = try ctx.weight(blk.out);
        try ctx.gemmRunF16A(ah, wout, b.norm_c.handle, tokens, hidden + inner, hidden, 0, 0, 0);
    } else if (af) {
        // half swiglu out (bh), half concat (ah reused: the qkv/mlp reads are
        // encoded above, hazard tracking serializes the overwrite), f16-A out
        // GEMM; C stays the f32 norm_c the gated residual reads.
        try ctx.glue(ctx.pipelines.swiglu_h, .{ b.wide.handle, b.bh.?.handle, null, null, null }, null, &cb, 8, 2, tokens * inner, 256, 0);
        try ctx.glue(ctx.pipelines.cat_rows_h, .{ b.o.handle, b.bh.?.handle, b.ah.?.handle, null, null }, null, &cat_dims, 8, 3, tokens * (hidden + inner), 256, 0);
        if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);
        try ctx.gemmRunF16A(b.ah.?.handle, try ctx.weight(blk.out), b.norm_c.handle, tokens, hidden + inner, hidden, 0, 0, 0);
    } else {
        const act_h = try ctx.actHandle();
        const cat12_h = try ctx.cat12Handle();
        try ctx.glue(ctx.pipelines.swiglu, .{ b.wide.handle, act_h, null, null, null }, null, &cb, 8, 2, tokens * inner, 256, 0);
        try ctx.glue(ctx.pipelines.cat_rows, .{ b.o.handle, act_h, cat12_h, null, null }, null, &cat_dims, 8, 3, tokens * (hidden + inner), 256, 0);
        if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);
        try ctx.gemmRun(cat12_h, try ctx.weight(blk.out), b.norm_c.handle, tokens, hidden + inner, hidden, 0);
    }
    if (instrument) try ctx.traceMark2(&t2, &ctx.tr_sb_ffn_ns, &g2, &ctx.tr_sb_ffn_gpu);
    const gate_off = [5]usize{ 0, 0, modByteOff(zflux2.mod_msa, ModOff.gate, hidden), 0, 0 };
    try ctx.glue(ctx.pipelines.gate_add, .{ b.cat.handle, b.norm_c.handle, mods, null, null }, &gate_off, &gp, @sizeOf(GlueParams), 3, tokens * hidden, 256, 0);
    if (instrument) {
        try ctx.traceMark2(&t2, &ctx.tr_sb_glue_ns, &g2, &ctx.tr_sb_glue_gpu);
        ctx.tr_sb_steps += 1;
    }
}

test "pool shape reallocates when joint_dim differs" {
    // emb is the only joint_dim-sized pooled buffer; ensurePool reuses the
    // pool iff std.meta.eql(pool.shape, shape). Two shapes that differ ONLY in
    // joint_dim MUST compare unequal — otherwise a joint_dim change (e.g. a 4B
    // vs 9B context width) would silently reuse a wrong-sized emb buffer.
    const a = PoolShape{ .img_len = 1024, .hidden = 3072, .inner = 9216, .joint_dim = 2560 };
    const b = PoolShape{ .img_len = 1024, .hidden = 3072, .inner = 9216, .joint_dim = 4096 };
    try std.testing.expect(!std.meta.eql(a, b));
    try std.testing.expect(std.meta.eql(a, a));
}
