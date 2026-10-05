//! Metal shader source for scaled dot-product attention.
//!
//! Four kernels (shape gate; grid):
//! - attention_rows: tokens <= 6144; one threadgroup per token*head.
//! - attention_flash_rows: head_dim <= 128, no token cap; one threadgroup
//!   per token*head.
//! - attention_flash_wide: tokens > 6144, head_dim <= 512; 16 (token,head)
//!   pairs per group, 512 threads.
//! - attention_wide_mma: heads == 1, head_dim == 512, tokens % 64 == 0;
//!   16 query rows x 4 dim-sliced simdgroups per group, 128 threads.

pub const flash: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\#define BK 32
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint heads;
    \\    uint kv_heads;
    \\    uint head_dim;
    \\    uint causal;
    \\};
    \\
    \\static inline uint q_index(uint tok, uint head, uint dim, constant Params& p) {
    \\    return (tok * p.heads + head) * p.head_dim + dim;
    \\}
    \\
    \\static inline uint kv_index(uint tok, uint head, uint dim, constant Params& p) {
    \\    return (tok * p.kv_heads + head) * p.head_dim + dim;
    \\}
    \\
    \\kernel void attention_flash_rows(
    \\    const device float* q [[buffer(0)]],
    \\    const device float* k [[buffer(1)]],
    \\    const device float* v [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]]
    \\) {
    \\    uint tok = group / p.heads;
    \\    uint head = group - tok * p.heads;
    \\    uint kv_head = head * p.kv_heads / p.heads;
    \\    uint limit = p.causal == 0 ? p.tokens : tok + 1;
    \\    float scale = rsqrt(float(p.head_dim));
    \\    uint sg_count = (tg_size + 31) / 32;
    \\    threadgroup float q_vec[128];
    \\    threadgroup float v_tile[BK][128];
    \\    threadgroup float p_tile[BK];
    \\    threadgroup float carry[2];
    \\    for (uint dim = tid; dim < p.head_dim; dim += tg_size) {
    \\        q_vec[dim] = q[q_index(tok, head, dim, p)];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float acc = 0.0f;
    \\    float row_max = -INFINITY;
    \\    float row_sum = 0.0f;
    \\    for (uint k0 = 0; k0 < limit; k0 += BK) {
    \\        uint tile = min((uint)BK, limit - k0);
    \\        if (tid < p.head_dim) {
    \\            for (uint key = 0; key < tile; key++) {
    \\                v_tile[key][tid] = v[kv_index(k0 + key, kv_head, tid, p)];
    \\            }
    \\        }
    \\        for (uint key = sg; key < tile; key += sg_count) {
    \\            float part = 0.0f;
    \\            for (uint dim = lane; dim < p.head_dim; dim += 32) {
    \\                part += q_vec[dim] * k[kv_index(k0 + key, kv_head, dim, p)];
    \\            }
    \\            float dot = simd_sum(part);
    \\            if (lane == 0) p_tile[key] = dot * scale;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // The first simdgroup folds this tile into the online softmax state.
    \\        if (sg == 0) {
    \\            float score = lane < tile ? p_tile[lane] : -INFINITY;
    \\            float next = max(row_max, simd_max(score));
    \\            float prob = lane < tile ? exp(score - next) : 0.0f;
    \\            p_tile[lane] = prob;
    \\            float sum = simd_sum(prob);
    \\            if (lane == 0) {
    \\                carry[0] = next;
    \\                carry[1] = sum;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        float next_max = carry[0];
    \\        float alpha = exp(row_max - next_max);
    \\        acc *= alpha;
    \\        if (tid < p.head_dim) {
    \\            for (uint key = 0; key < tile; key++) {
    \\                acc += p_tile[key] * v_tile[key][tid];
    \\            }
    \\        }
    \\        row_sum = alpha * row_sum + carry[1];
    \\        row_max = next_max;
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid < p.head_dim) {
    \\        output[q_index(tok, head, tid, p)] = acc / row_sum;
    \\    }
    \\}
;

// Wide-head variant for the shapes no other kernel takes (the VAE mid block:
// tokens > 6144 AND head_dim in 129..512, e.g. 16384x512 at 1024px). One
// simdgroup per query, WQ queries per threadgroup sharing each streamed
// f32 K/V tile (a one-query-per-TG version re-read all of K/V per query and
// measured >10x slower than the SDPA graph). All per-query state lives in
// registers: q and the O accumulator are strided lane+32*j exactly like the
// flash kernel's per-thread loops, the softmax fold matches its simd-op
// shape, and every reduction has a fixed order - deterministic, unlike the
// MPSGraph route this replaces (ledger 2026-08-04). K+V tiles at BKW=6 use
// 24 KB of the 32 KB threadgroup budget.
pub const wide: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\#define BKW 6
    \\#define WQ 16
    \\#define MAX_WIDE_DIM 512
    \\#define DIM_REGS (MAX_WIDE_DIM / 32)
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint heads;
    \\    uint kv_heads;
    \\    uint head_dim;
    \\    uint causal;
    \\};
    \\
    \\static inline uint q_index(uint tok, uint head, uint dim, constant Params& p) {
    \\    return (tok * p.heads + head) * p.head_dim + dim;
    \\}
    \\
    \\static inline uint kv_index(uint tok, uint head, uint dim, constant Params& p) {
    \\    return (tok * p.kv_heads + head) * p.head_dim + dim;
    \\}
    \\
    \\kernel void attention_flash_wide(
    \\    const device float* q [[buffer(0)]],
    \\    const device float* k [[buffer(1)]],
    \\    const device float* v [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]]
    \\) {
    \\    uint pair = group * WQ + sg;
    \\    uint tok = pair / p.heads;
    \\    uint head = pair - tok * p.heads;
    \\    bool live = tok < p.tokens;
    \\    uint kv_head = head * p.kv_heads / p.heads;
    \\    uint dims = (p.head_dim + 31) / 32;
    \\    float scale = rsqrt(float(p.head_dim));
    \\    threadgroup float k_tile[BKW][MAX_WIDE_DIM];
    \\    threadgroup float v_tile[BKW][MAX_WIDE_DIM];
    \\    float q_reg[DIM_REGS];
    \\    float acc[DIM_REGS];
    \\    for (uint j = 0; j < dims; j++) {
    \\        uint dim = lane + 32 * j;
    \\        q_reg[j] = (live && dim < p.head_dim) ? q[q_index(tok, head, dim, p)] : 0.0f;
    \\        acc[j] = 0.0f;
    \\    }
    \\    float row_max = -INFINITY;
    \\    float row_sum = 0.0f;
    \\    // Every query in the group is non-causal over the same kv_head range
    \\    // (heads==kv_heads on this route), so one shared stream serves all WQ.
    \\    for (uint k0 = 0; k0 < p.tokens; k0 += BKW) {
    \\        uint tile = min((uint)BKW, p.tokens - k0);
    \\        for (uint idx = tid; idx < tile * p.head_dim; idx += tg_size) {
    \\            uint key = idx / p.head_dim;
    \\            uint dim = idx - key * p.head_dim;
    \\            k_tile[key][dim] = k[kv_index(k0 + key, kv_head, dim, p)];
    \\            v_tile[key][dim] = v[kv_index(k0 + key, kv_head, dim, p)];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // Per-lane prob for key==lane, folded with the flash kernel's
    \\        // simd-op shape; broadcast back per key for the O accumulate.
    \\        float score = -INFINITY;
    \\        for (uint key = 0; key < tile; key++) {
    \\            float part = 0.0f;
    \\            for (uint j = 0; j < dims; j++) {
    \\                uint dim = lane + 32 * j;
    \\                float kv = dim < p.head_dim ? k_tile[key][dim] : 0.0f;
    \\                part += q_reg[j] * kv;
    \\            }
    \\            float dot = simd_sum(part);
    \\            if (lane == key) score = dot * scale;
    \\        }
    \\        float next = max(row_max, simd_max(score));
    \\        float prob = lane < tile ? exp(score - next) : 0.0f;
    \\        float alpha = exp(row_max - next);
    \\        row_sum = alpha * row_sum + simd_sum(prob);
    \\        row_max = next;
    \\        for (uint j = 0; j < dims; j++) acc[j] *= alpha;
    \\        for (uint key = 0; key < tile; key++) {
    \\            float pk = simd_broadcast(prob, (ushort)key);
    \\            for (uint j = 0; j < dims; j++) {
    \\                uint dim = lane + 32 * j;
    \\                float vv = dim < p.head_dim ? v_tile[key][dim] : 0.0f;
    \\                acc[j] += pk * vv;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (!live) return;
    \\    for (uint j = 0; j < dims; j++) {
    \\        uint dim = lane + 32 * j;
    \\        if (dim < p.head_dim) {
    \\            output[q_index(tok, head, dim, p)] = acc[j] / row_sum;
    \\        }
    \\    }
    \\}
;

// Chunked-D MMA wide-head kernel (the rewrite of the killed 6ad8330 kernel,
// whose full-width of[64] accumulator = 128 f32/lane was architecturally
// guaranteed to spill the 128-GPR file). 128 threads = 4 simdgroups; every
// threadgroup owns 16 query rows and every simdgroup owns a DISJOINT
// 128-dim D-slice of head_dim 512, so each K/V byte is read once per
// threadgroup (the killed kernel read them 4x) and the per-lane accumulator
// is of[2][16] = 64 f32 with ~18% headroom under the register cap. The full
// logit needs the 4 partial 128-dim dots summed: partials stage through
// threadgroup memory with a WRITTEN-OUT fixed-order sum, then every
// simdgroup redundantly recomputes the identical online softmax (the killed
// kernel's certified idiom verbatim: base-2 exp, thread_elements, the
// simd_shuffle_xor 1/8 row reductions) because P must sit in each
// simdgroup's fragments for its own V-slice MMA anyway. PV accumulation is
// dimension-sliced, so it needs no cross-simdgroup work. Exactly 3 barriers
// per KV tile. The host routes here only when heads == 1, head_dim == 512
// and tokens % 64 == 0, so every tile is full and edge paths do not exist.
// Fixed reduction order end to end: run-to-run byte-stable; NOT bit-equal
// to the killed kernel (the 512-dim dot is reassociated into 4 x 128).
// All Params fields except tokens are host-asserted rather than read;
// causal shapes are refused by sdpaShape on the host, not by the kernel.
pub const wide_mma: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\#define WQ16 16
    \\#define WBKV 32
    \\#define WHD 512
    \\#define WSLICE 128
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint heads;
    \\    uint kv_heads;
    \\    uint head_dim;
    \\    uint causal;
    \\};
    \\
    \\kernel void attention_wide_mma(
    \\    const device float* q [[buffer(0)]],
    \\    const device float* k [[buffer(1)]],
    \\    const device float* v [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint sgid [[simdgroup_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]]
    \\) {
    \\    const uint q0 = group * WQ16;
    \\    const uint ds = sgid * WSLICE; // this simdgroup's dim-slice base
    \\    const float scale = 1.442695041f * rsqrt(float(WHD)); // log2(e)/sqrt(D)
    \\
    \\    // Partial S tiles per simdgroup, then the summed+scaled logits.
    \\    threadgroup float part[4 * WQ16 * WBKV];
    \\    threadgroup float full[WQ16 * WBKV];
    \\
    \\    simdgroup_float8x8 of[2][WSLICE / 8];
    \\    for (uint r = 0; r < 2; r++) {
    \\        for (uint d = 0; d < WSLICE / 8; d++) {
    \\            of[r][d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\        }
    \\    }
    \\    float row_m[2] = { -INFINITY, -INFINITY };
    \\    float row_l[2] = { 0.0f, 0.0f };
    \\
    \\    for (uint kv0 = 0; kv0 < p.tokens; kv0 += WBKV) {
    \\        simdgroup_float8x8 sf[2][WBKV / 8];
    \\        for (uint r = 0; r < 2; r++) {
    \\            for (uint j = 0; j < WBKV / 8; j++) {
    \\                sf[r][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\            }
    \\        }
    \\        for (uint d = 0; d < WSLICE / 8; d++) {
    \\            simdgroup_float8x8 kf[WBKV / 8];
    \\            for (uint j = 0; j < WBKV / 8; j++) {
    \\                simdgroup_load(kf[j], k + (kv0 + j * 8) * WHD + ds + d * 8,
    \\                               WHD, ulong2(0, 0), true);
    \\            }
    \\            for (uint r = 0; r < 2; r++) {
    \\                simdgroup_float8x8 qf;
    \\                simdgroup_load(qf, q + (q0 + r * 8) * WHD + ds + d * 8, WHD);
    \\                for (uint j = 0; j < WBKV / 8; j++) {
    \\                    simdgroup_multiply_accumulate(sf[r][j], qf, kf[j], sf[r][j]);
    \\                }
    \\            }
    \\        }
    \\        // Stage this simdgroup's partial 16x32 S tile.
    \\        for (uint r = 0; r < 2; r++) {
    \\            for (uint j = 0; j < WBKV / 8; j++) {
    \\                simdgroup_store(sf[r][j],
    \\                                part + sgid * WQ16 * WBKV + (r * 8) * WBKV + j * 8,
    \\                                WBKV);
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // Fixed-associativity cross-simdgroup sum; one thread per element.
    \\        for (uint i = tid; i < WQ16 * WBKV; i += 128) {
    \\            full[i] = (((part[i] + part[WQ16 * WBKV + i]) +
    \\                        part[2 * WQ16 * WBKV + i]) +
    \\                       part[3 * WQ16 * WBKV + i]) * scale;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // Every simdgroup reloads the identical summed logits and runs the
    \\        // certified online softmax redundantly (bit-identical across
    \\        // simdgroups; scale is already applied above).
    \\        for (uint r = 0; r < 2; r++) {
    \\            for (uint j = 0; j < WBKV / 8; j++) {
    \\                simdgroup_load(sf[r][j], full + (r * 8) * WBKV + j * 8, WBKV);
    \\            }
    \\            float pm = -INFINITY;
    \\            for (uint j = 0; j < WBKV / 8; j++) {
    \\                thread auto& te = sf[r][j].thread_elements();
    \\                pm = max(pm, max(float(te[0]), float(te[1])));
    \\            }
    \\            pm = max(pm, simd_shuffle_xor(pm, 1));
    \\            pm = max(pm, simd_shuffle_xor(pm, 8));
    \\            const float new_m = max(row_m[r], pm);
    \\            const float corr =
    \\                row_m[r] == -INFINITY ? 0.0f : fast::exp2(row_m[r] - new_m);
    \\            row_m[r] = new_m;
    \\
    \\            float ps = 0.0f;
    \\            for (uint j = 0; j < WBKV / 8; j++) {
    \\                thread auto& te = sf[r][j].thread_elements();
    \\                const float e0 = fast::exp2(float(te[0]) - new_m);
    \\                const float e1 = fast::exp2(float(te[1]) - new_m);
    \\                ps += e0 + e1;
    \\                te[0] = e0;
    \\                te[1] = e1;
    \\            }
    \\            ps += simd_shuffle_xor(ps, 1);
    \\            ps += simd_shuffle_xor(ps, 8);
    \\            row_l[r] = row_l[r] * corr + ps;
    \\
    \\            for (uint d = 0; d < WSLICE / 8; d++) {
    \\                thread auto& oe = of[r][d].thread_elements();
    \\                oe[0] *= corr;
    \\                oe[1] *= corr;
    \\            }
    \\        }
    \\        for (uint j = 0; j < WBKV / 8; j++) {
    \\            for (uint d = 0; d < WSLICE / 8; d++) {
    \\                simdgroup_float8x8 vf;
    \\                simdgroup_load(vf, v + (kv0 + j * 8) * WHD + ds + d * 8, WHD);
    \\                simdgroup_multiply_accumulate(of[0][d], sf[0][j], vf, of[0][d]);
    \\                simdgroup_multiply_accumulate(of[1][d], sf[1][j], vf, of[1][d]);
    \\            }
    \\        }
    \\        // part[] and full[] are rewritten next tile; the softmax reloads
    \\        // above must complete first.
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    for (uint r = 0; r < 2; r++) {
    \\        const float inv_l = row_l[r] > 0.0f ? 1.0f / row_l[r] : 0.0f;
    \\        for (uint d = 0; d < WSLICE / 8; d++) {
    \\            thread auto& oe = of[r][d].thread_elements();
    \\            oe[0] *= inv_l;
    \\            oe[1] *= inv_l;
    \\            simdgroup_store(of[r][d],
    \\                            output + (q0 + r * 8) * WHD + ds + d * 8, WHD);
    \\        }
    \\    }
    \\}
;

pub const attn: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\#define MAX_TOKENS 6144
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint heads;
    \\    uint kv_heads;
    \\    uint head_dim;
    \\    uint causal;
    \\    uint valid; // keys >= valid are padding; 0 = none
    \\};
    \\
    \\static inline uint q_index(uint tok, uint head, uint dim, constant Params& p) {
    \\    return (tok * p.heads + head) * p.head_dim + dim;
    \\}
    \\
    \\static inline uint kv_index(uint tok, uint head, uint dim, constant Params& p) {
    \\    return (tok * p.kv_heads + head) * p.head_dim + dim;
    \\}
    \\
    \\kernel void attention_rows(
    \\    const device float* q [[buffer(0)]],
    \\    const device float* k [[buffer(1)]],
    \\    const device float* v [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint tok = group / p.heads;
    \\    uint head = group - tok * p.heads;
    \\    uint kv_head = head * p.kv_heads / p.heads;
    \\    uint limit = p.causal == 0 ? p.tokens : tok + 1;
    \\    if (p.valid != 0 && limit > p.valid) limit = p.valid;
    \\    float scale = rsqrt(float(p.head_dim));
    \\    threadgroup float scores[MAX_TOKENS];
    \\    threadgroup float reduce[256];
    \\    float local_max = -INFINITY;
    \\    for (uint key = tid; key < limit; key += tg_size) {
    \\        float dot = 0.0f;
    \\        for (uint dim = 0; dim < p.head_dim; dim++) {
    \\            dot += q[q_index(tok, head, dim, p)] * k[kv_index(key, kv_head, dim, p)];
    \\        }
    \\        float score = dot * scale;
    \\        scores[key] = score;
    \\        local_max = max(local_max, score);
    \\    }
    \\    reduce[tid] = local_max;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] = max(reduce[tid], reduce[tid + stride]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float max_score = reduce[0];
    \\    float local_sum = 0.0f;
    \\    for (uint key = tid; key < limit; key += tg_size) {
    \\        float value = exp(scores[key] - max_score);
    \\        scores[key] = value;
    \\        local_sum += value;
    \\    }
    \\    reduce[tid] = local_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv_sum = 1.0f / reduce[0];
    \\    for (uint dim = tid; dim < p.head_dim; dim += tg_size) {
    \\        float acc = 0.0f;
    \\        for (uint key = 0; key < limit; key++) {
    \\            acc += scores[key] * v[kv_index(key, kv_head, dim, p)];
    \\        }
    \\        output[q_index(tok, head, dim, p)] = acc * inv_sum;
    \\    }
    \\}
;
