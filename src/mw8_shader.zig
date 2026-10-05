//! Experimental W8 fused-dequant GEMM kernel.

pub const source: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct GemmParams {
    \\    uint m;
    \\    uint k;
    \\    uint n;
    \\    uint dtype;
    \\    uint mode;
    \\    ulong weight_offset;
    \\};
    \\
    \\constant uint group_size = 64;
    \\
    \\kernel void gemm_w8(
    \\    const device float* A [[buffer(0)]],
    \\    const device uchar* Wbytes [[buffer(1)]],
    \\    device float* C [[buffer(2)]],
    \\    constant GemmParams& p [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint tile_m = tg.y * 32;
    \\    const uint tile_n = tg.x * 32;
    \\    const device uchar* base = Wbytes + p.weight_offset;
    \\    const device char* q = (const device char*)base;
    \\    const device float* scales = (const device float*)(base + ulong(p.n) * ulong(p.k));
    \\    threadgroup half a_stage[32 * 8];
    \\    threadgroup half w_stage[32 * 8];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    const uint groups = (p.k + group_size - 1) / group_size;
    \\    for (uint kg = 0; kg < p.k; kg += group_size) {
    \\        float s = scales[(tile_n + tid) * groups + kg / group_size];
    \\        for (uint k0 = kg; k0 < min(kg + group_size, p.k); k0 += 8) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                a_stage[tid * 8 + kk] = half(A[(tile_m + tid) * p.k + k0 + kk]);
    \\                int qv = int(q[(tile_n + tid) * p.k + k0 + kk]);
    \\                w_stage[tid * 8 + kk] = half(float(qv) * s);
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], a_stage + i * 64, 8);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], w_stage + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\}
;
