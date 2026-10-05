//! FLUX.2 Klein DiT forward — phase-1 compute path.
//!
//! Built stage by stage against the captured block oracle
//! (~/anchors/klein4b/blocks): embedders + time embed first, then the
//! double/single blocks. Correctness before speed: GEMMs go through the
//! shared mlinear path; glue runs on CPU.

const std = @import("std");

const mlinear = @import("../metal/mlinear.zig");
const ops = @import("../runtime/ops.zig");
const tensor = @import("../pack/tensor.zig");
const vproj = @import("../vae/vproj.zig");
const zflux2 = @import("zflux2.zig");

/// Sinusoidal timestep embedding, diffusers `get_timestep_embedding` with
/// flip_sin_to_cos=true, downscale_freq_shift=0 (the FLUX family layout:
/// [cos | sin]).
pub fn timeSinusoid(out: []f32, t: f32) void {
    const half = out.len / 2;
    const hf: f32 = @floatFromInt(half);
    for (0..half) |i| {
        const exponent = -@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / hf;
        const freq = @exp(exponent);
        const arg = t * freq;
        out[i] = @cos(arg);
        out[half + i] = @sin(arg);
    }
}

fn silu(x: []f32) void {
    for (x) |*v| v.* = ops.silu(v.*);
}

/// time_guidance_embed: sinusoid(256) -> linear_1(3072) -> SiLU -> linear_2.
pub fn timeEmbed(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    out: []f32,
    g: zflux2.Globals,
    t: f32,
) !void {
    var sin_buf: [256]f32 = undefined;
    timeSinusoid(&sin_buf, t);
    const mid = try allocator.alloc(f32, out.len);
    defer allocator.free(mid);
    try vproj.run(metal, mid, &sin_buf, g.time_in_1, null, 1);
    silu(mid);
    try vproj.run(metal, out, mid, g.time_in_2, null, 1);
}

/// x_embedder: packed latents [tokens,128] -> [tokens,hidden].
pub fn xEmbed(
    metal: ?*mlinear.Context,
    out: []f32,
    latents: []const f32,
    g: zflux2.Globals,
    tokens: usize,
) !void {
    try vproj.run(metal, out, latents, g.x_embed, null, tokens);
}

/// context_embedder: text features [tokens,joint_dim] -> [tokens,hidden].
pub fn contextEmbed(
    metal: ?*mlinear.Context,
    out: []f32,
    feats: []const f32,
    g: zflux2.Globals,
    tokens: usize,
) !void {
    try vproj.run(metal, out, feats, g.context_embed, null, tokens);
}

// ---------------------------------------------------------------------------
// Block-level forward (correctness path: mlinear GEMMs, CPU glue, standalone
// attention). Gated stage-by-stage against the captured block oracle.

const mattn = @import("../metal/mattn.zig");

pub const head_dim = zflux2.head_dim;
pub const txt_len = zflux2.txt_len;
const ModOff = zflux2.ModOff;

/// Size-dependent dims, derived from the checkpoint Config. head_dim and
/// txt_len are family constants (zflux2); these three vary 4B vs 9B.
pub const Dims = struct {
    hidden: usize,
    heads: usize,
    inner: usize,

    pub fn fromConfig(cfg: zflux2.Config) Dims {
        return .{
            .hidden = cfg.hidden,
            .heads = cfg.heads,
            .inner = cfg.ffn_inner,
        };
    }
};

/// 4-axis rope tables, repeat-interleaved reals (cos/sin each [tokens][128]).
/// Text tokens: ids (0,0,0,l); image tokens: (0,h,w,0) over the packed grid.
pub const Rope = struct {
    cos: []f32,
    sin: []f32,

    pub fn deinit(self: *Rope, allocator: std.mem.Allocator) void {
        allocator.free(self.cos);
        allocator.free(self.sin);
    }
};

pub fn buildRope(
    allocator: std.mem.Allocator,
    grid_h: usize,
    grid_w: usize,
    theta: f64,
) !Rope {
    return buildRopeRef(allocator, grid_h, grid_w, 0, 0, theta);
}

/// The rope table with a reference image's tokens after the image's: the
/// same H/W ids on their own grid and T = 10 (diffusers' `_prepare_image_ids`
/// with one reference), so the transformer tells the two apart. ref grid
/// 0x0 = no reference.
pub fn buildRopeRef(
    allocator: std.mem.Allocator,
    grid_h: usize,
    grid_w: usize,
    ref_h: usize,
    ref_w: usize,
    theta: f64,
) !Rope {
    const img = grid_h * grid_w;
    const tokens = txt_len + img + ref_h * ref_w;
    const cos = try allocator.alloc(f32, tokens * head_dim);
    errdefer allocator.free(cos);
    const sin = try allocator.alloc(f32, tokens * head_dim);
    errdefer allocator.free(sin);
    for (0..tokens) |t| {
        var ids: [4]f64 = .{ 0, 0, 0, 0 };
        if (t < txt_len) {
            ids[3] = @floatFromInt(t);
        } else if (t < txt_len + img) {
            const p = t - txt_len;
            ids[1] = @floatFromInt(p / grid_w);
            ids[2] = @floatFromInt(p % grid_w);
        } else {
            const p = t - txt_len - img;
            ids[0] = 10;
            ids[1] = @floatFromInt(p / ref_w);
            ids[2] = @floatFromInt(p % ref_w);
        }
        for (0..4) |ax| {
            // 32-dim axis: 16 freqs, each repeated 2x interleaved.
            for (0..16) |i| {
                const exponent = @as(f64, @floatFromInt(2 * i)) / 32.0;
                const freq = 1.0 / std.math.pow(f64, theta, exponent);
                const arg = ids[ax] * freq;
                const c: f32 = @floatCast(@cos(arg));
                const s: f32 = @floatCast(@sin(arg));
                const base = t * head_dim + ax * 32 + 2 * i;
                cos[base] = c;
                cos[base + 1] = c;
                sin[base] = s;
                sin[base + 1] = s;
            }
        }
    }
    return .{ .cos = cos, .sin = sin };
}

fn layerNorm(out: []f32, x: []const f32, tokens: usize, hidden: usize) void {
    for (0..tokens) |t| {
        const row = x[t * hidden ..][0..hidden];
        var mean: f64 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(hidden);
        var varr: f64 = 0;
        for (row) |v| varr += (v - mean) * (v - mean);
        varr /= @floatFromInt(hidden);
        const inv = 1.0 / @sqrt(varr + 1e-6);
        const dst = out[t * hidden ..][0..hidden];
        for (dst, row) |*d, v| d.* = @floatCast((v - mean) * inv);
    }
}

fn modApply(out: []f32, normed: []const f32, shift: []const f32, scale: []const f32, tokens: usize, hidden: usize) void {
    for (0..tokens) |t| {
        const src = normed[t * hidden ..][0..hidden];
        const dst = out[t * hidden ..][0..hidden];
        for (dst, src, 0..) |*d, v, j| d.* = (1.0 + scale[j]) * v + shift[j];
    }
}

fn gateAdd(state: []f32, delta: []const f32, gate: []const f32, tokens: usize, hidden: usize) void {
    for (0..tokens) |t| {
        const dst = state[t * hidden ..][0..hidden];
        const src = delta[t * hidden ..][0..hidden];
        for (dst, src, 0..) |*d, v, j| d.* += gate[j] * v;
    }
}

fn rmsNormHeads(x: []f32, w: tensor.View, tokens: usize, heads: usize) void {
    for (0..tokens * heads) |th| {
        const row = x[th * head_dim ..][0..head_dim];
        var ss: f64 = 0;
        for (row) |v| ss += @as(f64, v) * v;
        const inv = 1.0 / @sqrt(ss / head_dim + 1e-6);
        for (row, 0..) |*v, j| v.* = @floatCast(@as(f64, v.*) * inv * w.atF32Unchecked(j));
    }
}

fn applyRope(x: []f32, rope: Rope, tokens: usize, heads: usize) void {
    for (0..tokens) |t| {
        for (0..heads) |h| {
            const row = x[(t * heads + h) * head_dim ..][0..head_dim];
            const c = rope.cos[t * head_dim ..][0..head_dim];
            const s = rope.sin[t * head_dim ..][0..head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const x0 = row[i];
                const x1 = row[i + 1];
                row[i] = x0 * c[i] - x1 * s[i];
                row[i + 1] = x1 * c[i + 1] + x0 * s[i + 1];
            }
        }
    }
}

fn swiglu(out: []f32, x: []const f32, tokens: usize, inner: usize) void {
    for (0..tokens) |t| {
        const g = x[t * inner * 2 ..][0..inner];
        const u = x[t * inner * 2 + inner ..][0..inner];
        const dst = out[t * inner ..][0..inner];
        for (dst, g, u) |*dv, gv, uv| dv.* = gv / (1.0 + @exp(-gv)) * uv;
    }
}

/// Shared modulation vectors: linear(SiLU(temb)) -> sets of (shift,scale,gate).
pub fn modVectors(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    w: tensor.View,
    temb: []const f32,
    sets: usize,
) ![]f32 {
    const act = try allocator.alloc(f32, temb.len);
    defer allocator.free(act);
    for (act, temb) |*dv, v| dv.* = v / (1.0 + @exp(-v));
    const out = try allocator.alloc(f32, sets * 3 * temb.len);
    errdefer allocator.free(out);
    try vproj.run(metal, out, act, w, null, 1);
    return out;
}

pub const Streams = struct {
    img: []f32, // [img_len, hidden]
    txt: []f32, // [txt_len, hidden]
};

/// One double block, in place on the streams. mods are the SHARED global
/// img/txt modulation vectors (6*hidden each).
pub fn doubleBlock(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: *mattn.Context,
    s: Streams,
    blk: zflux2.Double,
    mods_img: []const f32,
    mods_txt: []const f32,
    rope: Rope,
    d: Dims,
) !void {
    const hidden = d.hidden;
    const heads = d.heads;
    const img_len = s.img.len / hidden;
    const tokens = txt_len + img_len;
    const a = allocator;

    // Modulation component offsets within the 6*hidden img/txt mod vectors
    // (shared layout: msa set then mlp set, each [shift, scale, gate]).
    const msa_shift = zflux2.modOffset(zflux2.mod_msa, ModOff.shift, hidden);
    const msa_scale = zflux2.modOffset(zflux2.mod_msa, ModOff.scale, hidden);
    const msa_gate = zflux2.modOffset(zflux2.mod_msa, ModOff.gate, hidden);
    const mlp_base = zflux2.modOffset(zflux2.mod_mlp, ModOff.shift, hidden);

    // norm + msa modulation
    const ni = try a.alloc(f32, s.img.len);
    defer a.free(ni);
    layerNorm(ni, s.img, img_len, hidden);
    modApply(ni, ni, mods_img[msa_shift..][0..hidden], mods_img[msa_scale..][0..hidden], img_len, hidden);
    const nt = try a.alloc(f32, s.txt.len);
    defer a.free(nt);
    layerNorm(nt, s.txt, txt_len, hidden);
    modApply(nt, nt, mods_txt[msa_shift..][0..hidden], mods_txt[msa_scale..][0..hidden], txt_len, hidden);

    // qkv on both streams, concat [txt, img]
    const q = try a.alloc(f32, tokens * hidden);
    defer a.free(q);
    const k = try a.alloc(f32, tokens * hidden);
    defer a.free(k);
    const v = try a.alloc(f32, tokens * hidden);
    defer a.free(v);
    try vproj.run(metal, q[0 .. txt_len * hidden], nt, blk.add_q, null, txt_len);
    try vproj.run(metal, k[0 .. txt_len * hidden], nt, blk.add_k, null, txt_len);
    try vproj.run(metal, v[0 .. txt_len * hidden], nt, blk.add_v, null, txt_len);
    try vproj.run(metal, q[txt_len * hidden ..], ni, blk.to_q, null, img_len);
    try vproj.run(metal, k[txt_len * hidden ..], ni, blk.to_k, null, img_len);
    try vproj.run(metal, v[txt_len * hidden ..], ni, blk.to_v, null, img_len);

    rmsNormHeads(q[0 .. txt_len * hidden], blk.norm_added_q, txt_len, heads);
    rmsNormHeads(k[0 .. txt_len * hidden], blk.norm_added_k, txt_len, heads);
    rmsNormHeads(q[txt_len * hidden ..], blk.norm_q, img_len, heads);
    rmsNormHeads(k[txt_len * hidden ..], blk.norm_k, img_len, heads);
    applyRope(q, rope, tokens, heads);
    applyRope(k, rope, tokens, heads);

    const ao = try a.alloc(f32, tokens * hidden);
    defer a.free(ao);
    try attn.run(ao, q, k, v, .{
        .tokens = tokens,
        .heads = heads,
        .kv_heads = heads,
        .head_dim = head_dim,
        .causal = false,
    });

    // out projections + gated residual (msa)
    const proj = try a.alloc(f32, tokens * hidden);
    defer a.free(proj);
    try vproj.run(metal, proj[0 .. txt_len * hidden], ao[0 .. txt_len * hidden], blk.to_add_out, null, txt_len);
    try vproj.run(metal, proj[txt_len * hidden ..], ao[txt_len * hidden ..], blk.to_out, null, img_len);
    gateAdd(s.txt, proj[0 .. txt_len * hidden], mods_txt[msa_gate..][0..hidden], txt_len, hidden);
    gateAdd(s.img, proj[txt_len * hidden ..], mods_img[msa_gate..][0..hidden], img_len, hidden);

    // FF on each stream (mlp modulation set, second half of the mod vector)
    try ffStream(a, metal, s.img, blk.ff_in, blk.ff_out, mods_img[mlp_base..], img_len, d);
    try ffStream(a, metal, s.txt, blk.ffc_in, blk.ffc_out, mods_txt[mlp_base..], txt_len, d);
}

fn ffStream(
    a: std.mem.Allocator,
    metal: ?*mlinear.Context,
    state: []f32,
    w_in: tensor.View,
    w_out: tensor.View,
    mods: []const f32, // [shift, scale, gate] x hidden
    tokens: usize,
    d: Dims,
) !void {
    const hidden = d.hidden;
    const inner = d.inner;
    // `mods` is one [shift, scale, gate] set (the mlp set sliced by the caller).
    const sh = ModOff.shift * hidden;
    const sc = ModOff.scale * hidden;
    const ga = ModOff.gate * hidden;
    const n = try a.alloc(f32, state.len);
    defer a.free(n);
    layerNorm(n, state, tokens, hidden);
    modApply(n, n, mods[sh..][0..hidden], mods[sc..][0..hidden], tokens, hidden);
    const wide = try a.alloc(f32, tokens * inner * 2);
    defer a.free(wide);
    try vproj.run(metal, wide, n, w_in, null, tokens);
    const act = try a.alloc(f32, tokens * inner);
    defer a.free(act);
    swiglu(act, wide, tokens, inner);
    const ff = try a.alloc(f32, state.len);
    defer a.free(ff);
    try vproj.run(metal, ff, act, w_out, null, tokens);
    gateAdd(state, ff, mods[ga..][0..hidden], tokens, hidden);
}

/// One single (parallel) block, in place on the concatenated [txt,img]
/// stream. One shared mod set (3*hidden).
pub fn singleBlock(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: *mattn.Context,
    state: []f32,
    blk: zflux2.Single,
    mods: []const f32,
    rope: Rope,
    d: Dims,
) !void {
    const hidden = d.hidden;
    const heads = d.heads;
    const tokens = state.len / hidden;
    const a = allocator;
    const qkv_w = zflux2.SingleRows.qkv * hidden;
    const mlp_w = 2 * d.inner;

    const n = try a.alloc(f32, state.len);
    defer a.free(n);
    layerNorm(n, state, tokens, hidden);
    modApply(n, n, mods[0..hidden], mods[hidden .. 2 * hidden], tokens, hidden);

    const wide = try a.alloc(f32, tokens * (qkv_w + mlp_w));
    defer a.free(wide);
    try vproj.run(metal, wide, n, blk.qkv_mlp, null, tokens);

    // de-interleave the fused output into q/k/v and mlp halves
    const q = try a.alloc(f32, tokens * hidden);
    defer a.free(q);
    const k = try a.alloc(f32, tokens * hidden);
    defer a.free(k);
    const v = try a.alloc(f32, tokens * hidden);
    defer a.free(v);
    const mlp = try a.alloc(f32, tokens * mlp_w);
    defer a.free(mlp);
    for (0..tokens) |t| {
        const row = wide[t * (qkv_w + mlp_w) ..];
        @memcpy(q[t * hidden ..][0..hidden], row[0..hidden]);
        @memcpy(k[t * hidden ..][0..hidden], row[hidden .. 2 * hidden]);
        @memcpy(v[t * hidden ..][0..hidden], row[2 * hidden .. 3 * hidden]);
        @memcpy(mlp[t * mlp_w ..][0..mlp_w], row[qkv_w .. qkv_w + mlp_w]);
    }
    rmsNormHeads(q, blk.norm_q, tokens, heads);
    rmsNormHeads(k, blk.norm_k, tokens, heads);
    applyRope(q, rope, tokens, heads);
    applyRope(k, rope, tokens, heads);

    const ao = try a.alloc(f32, tokens * hidden);
    defer a.free(ao);
    try attn.run(ao, q, k, v, .{
        .tokens = tokens,
        .heads = heads,
        .kv_heads = heads,
        .head_dim = head_dim,
        .causal = false,
    });

    const act = try a.alloc(f32, tokens * d.inner);
    defer a.free(act);
    swiglu(act, mlp, tokens, d.inner);

    // concat [attn | act] -> to_out
    const cat = try a.alloc(f32, tokens * (hidden + d.inner));
    defer a.free(cat);
    for (0..tokens) |t| {
        @memcpy(cat[t * (hidden + d.inner) ..][0..hidden], ao[t * hidden ..][0..hidden]);
        @memcpy(cat[t * (hidden + d.inner) + hidden ..][0..d.inner], act[t * d.inner ..][0..d.inner]);
    }
    const out = try a.alloc(f32, state.len);
    defer a.free(out);
    try vproj.run(metal, out, cat, blk.out, null, tokens);
    gateAdd(state, out, mods[2 * hidden .. 3 * hidden], tokens, hidden);
}

/// Final adaLN (norm_out: [scale, shift] from linear(SiLU(temb)), FLUX
/// order scale-first) + proj_out on the image stream.
pub fn finalProj(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    out: []f32,
    img: []const f32,
    g: zflux2.Globals,
    temb: []const f32,
    tokens: usize,
    scale_first: bool,
    d: Dims,
) !void {
    const hidden = d.hidden;
    const a = allocator;
    const act = try a.alloc(f32, temb.len);
    defer a.free(act);
    for (act, temb) |*dv, v| dv.* = v / (1.0 + @exp(-v));
    const mods = try a.alloc(f32, 2 * hidden);
    defer a.free(mods);
    try vproj.run(metal, mods, act, g.norm_out, null, 1);
    const shift = if (scale_first) mods[hidden..] else mods[0..hidden];
    const scale = if (scale_first) mods[0..hidden] else mods[hidden..];
    const n = try a.alloc(f32, img.len);
    defer a.free(n);
    layerNorm(n, img, tokens, hidden);
    modApply(n, n, shift, scale, tokens, hidden);
    try vproj.run(metal, out, n, g.proj_out, null, tokens);
}

/// Full DiT forward: packed latents + text features + t -> velocity.
/// Every stage of this path is oracle-verified (see flux2DitGate).
pub fn forward(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: *mattn.Context,
    out: []f32, // [img_tokens, 128]
    latents: []const f32, // [img_tokens, 128] packed
    text_feats: []const f32, // [txt_len, joint_dim]
    loaded: *const zflux2.Loaded,
    rope: Rope,
    t: f32,
) !void {
    const a = allocator;
    const d = Dims.fromConfig(loaded.cfg);
    const hidden = d.hidden;
    const img_len = latents.len / 128;

    const temb = try a.alloc(f32, hidden);
    defer a.free(temb);
    try timeEmbed(a, metal, temb, loaded.globals, t);
    const mods_img = try modVectors(a, metal, loaded.globals.mod_img, temb, 2);
    defer a.free(mods_img);
    const mods_txt = try modVectors(a, metal, loaded.globals.mod_txt, temb, 2);
    defer a.free(mods_txt);
    const mods_single = try modVectors(a, metal, loaded.globals.mod_single, temb, 1);
    defer a.free(mods_single);

    const img = try a.alloc(f32, img_len * hidden);
    defer a.free(img);
    try xEmbed(metal, img, latents, loaded.globals, img_len);
    const txt = try a.alloc(f32, txt_len * hidden);
    defer a.free(txt);
    try contextEmbed(metal, txt, text_feats, loaded.globals, txt_len);

    for (loaded.doubles) |blk| {
        try doubleBlock(a, metal, attn, .{ .img = img, .txt = txt }, blk, mods_img, mods_txt, rope, d);
    }
    // Single stream is the two double-stream streams concatenated text-first
    // (shared layout): text rows, then image rows.
    comptime std.debug.assert(zflux2.cat_txt_first);
    const cat = try a.alloc(f32, (txt_len + img_len) * hidden);
    defer a.free(cat);
    @memcpy(cat[0 .. txt_len * hidden], txt);
    @memcpy(cat[txt_len * hidden ..], img);
    for (loaded.singles) |blk| {
        try singleBlock(a, metal, attn, cat, blk, mods_single, rope, d);
    }
    try finalProj(a, metal, out, cat[txt_len * hidden ..], loaded.globals, temb, img_len, zflux2.final_scale_first, d);
}
