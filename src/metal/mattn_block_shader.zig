//! Variant-B v3 flash attention: register-resident softmax on simdgroup
//! fragments via the empirically-verified M-series thread->element mapping
//! (thread t owns row (t%8)/2 + (t/16)*4, cols (t%2)*2 + ((t/8)%2)*4 ..+1;
//! see zdraw_metal_frag_map_probe). Row max/sum = simd_shuffle_xor 1 and 8;
//! per-row m/l state is thread-local. No threadgroup traffic for S/P/O.

pub const block: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\#define BQ 16
    \\#define BKV 128
    \\#define HD 128
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint heads;
    \\    uint kv_heads;
    \\    uint head_dim;
    \\    uint causal;
    \\};
    \\
    \\kernel void attention_block16(
    \\    const device half* q [[buffer(0)]],
    \\    const device half* k [[buffer(1)]],
    \\    const device half* v [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint sgid [[simdgroup_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]]
    \\) {
    \\    const uint qblocks = (p.tokens + BQ - 1) / BQ;
    \\    const uint head = group / qblocks;
    \\    const uint qb = group - head * qblocks;
    \\    const uint q0 = qb * BQ;
    \\    const float scale = 1.442695041f * rsqrt(float(HD)); // log2(e)/sqrt(D)
    \\
    \\    // Q loaded directly from device (half); rows past the end are
    \\    // handled by clamping the base row so loads stay in-bounds, and
    \\    // masked at the final store.
    \\    const uint ldq = HD; // head-major: dense per-head rows
    \\    const uint qrow = min(q0 + sgid * 8, p.tokens - 1);
    \\    const device half* qbase = q + (head * p.tokens + qrow) * HD;
    \\
    \\    simdgroup_float8x8 of[HD / 8];
    \\    for (uint d = 0; d < HD / 8; d++) {
    \\        of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    }
    \\    // Q is loop-invariant: this simdgroup's 8 rows are reused for every
    \\    // KV block and every j. Loading it once costs 16 registers and drops
    \\    // a third of the kernel's device loads (48 -> 32 per 8-token step).
    \\    simdgroup_half8x8 qf[HD / 8];
    \\    for (uint d = 0; d < HD / 8; d++) {
    \\        simdgroup_load(qf[d], qbase + d * 8, ldq);
    \\    }
    \\    float row_m = -INFINITY; // softmax state for THIS thread's row
    \\    float row_l = 0.0f;
    \\
    \\    const uint kv_head = head * p.kv_heads / p.heads;
    \\    const uint ldkv = HD;
    \\    const device half* kbase = k + kv_head * p.tokens * HD;
    \\    const device half* vbase = v + kv_head * p.tokens * HD;
    \\    for (uint kv0 = 0; kv0 < p.tokens; kv0 += BKV) {
    \\        // S = Q . K^T: half K rows. Full 8-row tiles load direct from
    \\        // device into MMA; the ragged tail stages an 8xHD half tile
    \\        // through threadgroup with two barriers.
    \\        const uint limit = min(uint(BKV), p.tokens - kv0);
    \\        const uint jmax = (limit + 7) / 8;
    \\        simdgroup_half8x8 sf[BKV / 8];
    \\        for (uint j = 0; j < jmax; j++) {
    \\            sf[j] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
    \\            const bool full = kv0 + j * 8 + 8 <= p.tokens;
    \\            if (full) {
    \\                for (uint d = 0; d < HD / 8; d++) {
    \\                    simdgroup_half8x8 kf;
    \\                    simdgroup_load(kf, kbase + (kv0 + j * 8) * ldkv + d * 8,
    \\                                   ldkv, ulong2(0, 0), true);
    \\                    simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
    \\                }
    \\            } else {
    \\                threadgroup half kt[8][HD];
    \\                for (uint i = tid; i < 8 * HD; i += 64) {
    \\                    const uint r = i / HD;
    \\                    const uint tok = kv0 + j * 8 + r;
    \\                    kt[r][i - r * HD] = tok < p.tokens
    \\                        ? kbase[tok * ldkv + (i - r * HD)] : half(0.0h);
    \\                }
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\                for (uint d = 0; d < HD / 8; d++) {
    \\                    simdgroup_half8x8 kf;
    \\                    simdgroup_load(kf, &kt[0][d * 8], HD, ulong2(0, 0), true);
    \\                    simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
    \\                }
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            }
    \\        }
    \\        const uint c0 = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
    \\        const float mask_val = -57344.0f; // 0.875 * -HALF_MAX
    \\        const float mask_min = -49152.0f;
    \\        float pm = -INFINITY;
    \\        for (uint j = 0; j < jmax; j++) {
    \\            thread auto& te = sf[j].thread_elements();
    \\            float s0 = float(te[0]) * scale;
    \\            float s1 = float(te[1]) * scale;
    \\            if (j * 8 + c0 >= limit) s0 = mask_val;
    \\            if (j * 8 + c0 + 1 >= limit) s1 = mask_val;
    \\            te[0] = half(s0);
    \\            te[1] = half(s1);
    \\            pm = max(pm, max(s0, s1));
    \\        }
    \\        pm = max(pm, simd_shuffle_xor(pm, 1));
    \\        pm = max(pm, simd_shuffle_xor(pm, 8));
    \\        const float new_m = max(row_m, pm);
    \\        const float corr = row_m == -INFINITY ? 0.0f : fast::exp2(row_m - new_m);
    \\        row_m = new_m;
    \\
    \\        float ps = 0.0f;
    \\        for (uint j = 0; j < jmax; j++) {
    \\            thread auto& te = sf[j].thread_elements();
    \\            const float s0 = float(te[0]);
    \\            const float s1 = float(te[1]);
    \\            const float e0 = s0 <= mask_min ? 0.0f : fast::exp2(s0 - new_m);
    \\            const float e1 = s1 <= mask_min ? 0.0f : fast::exp2(s1 - new_m);
    \\            ps += e0 + e1;
    \\            te[0] = half(e0);
    \\            te[1] = half(e1);
    \\        }
    \\        ps += simd_shuffle_xor(ps, 1);
    \\        ps += simd_shuffle_xor(ps, 8);
    \\        row_l = row_l * corr + ps;
    \\
    \\        for (uint d = 0; d < HD / 8; d++) {
    \\            thread auto& oe = of[d].thread_elements();
    \\            oe[0] *= corr;
    \\            oe[1] *= corr;
    \\        }
    \\        // O += P . V: half V rows; full tiles load direct, the ragged
    \\        // tail stages an 8xHD half tile through threadgroup with two
    \\        // barriers.
    \\        for (uint j = 0; j < jmax; j++) {
    \\            const uint rows = min(uint(8), p.tokens - (kv0 + j * 8));
    \\            if (rows == 8) {
    \\                for (uint d = 0; d < HD / 8; d++) {
    \\                    simdgroup_half8x8 vf;
    \\                    simdgroup_load(vf, vbase + (kv0 + j * 8) * ldkv + d * 8, ldkv);
    \\                    simdgroup_multiply_accumulate(of[d], sf[j], vf, of[d]);
    \\                }
    \\            } else {
    \\                threadgroup half vtt[8][HD];
    \\                for (uint i = tid; i < 8 * HD; i += 64) {
    \\                    const uint r = i / HD;
    \\                    const uint tok = kv0 + j * 8 + r;
    \\                    vtt[r][i - r * HD] = tok < p.tokens
    \\                        ? vbase[tok * ldkv + (i - r * HD)] : half(0.0h);
    \\                }
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\                for (uint d = 0; d < HD / 8; d++) {
    \\                    simdgroup_half8x8 vf;
    \\                    simdgroup_load(vf, &vtt[0][d * 8], HD);
    \\                    simdgroup_multiply_accumulate(of[d], sf[j], vf, of[d]);
    \\                }
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            }
    \\        }
    \\    }
    \\
    \\    const float inv_l = row_l > 0.0f ? 1.0f / row_l : 0.0f;
    \\    threadgroup float ot[BQ][HD];
    \\    for (uint d = 0; d < HD / 8; d++) {
    \\        thread auto& oe = of[d].thread_elements();
    \\        oe[0] *= inv_l;
    \\        oe[1] *= inv_l;
    \\        simdgroup_store(of[d], &ot[sgid * 8][d * 8], HD);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = tid; i < BQ * HD; i += 64) {
    \\        const uint r = i / HD;
    \\        const uint c = i - r * HD;
    \\        const uint tok = q0 + r;
    \\        if (tok < p.tokens) {
    \\            output[(tok * p.heads + head) * HD + c] = ot[r][c]; // out stays token-major
    \\        }
    \\    }
    \\}
;
