//! Winograd F(4x4,3x3) conv for the product f16 decoder
//! (vae-winograd-20260827). A 3x3 stride-1 conv as 36 batched GEMMs on
//! transformed operands: 4x fewer multiplies than the direct kernels, which
//! already run at the chip's f16 MMA rate. Product tier only; the transforms
//! run in f32, the GEMM operands and the M planes are half (f32 accumulate).
//!
//! Layouts (per tile batch of `tile_count` tiles, row-major over the map):
//!   U  half [36][out_ch][in_ch]     transformed weights (G g G^T), static per conv
//!   V  half [36][in_ch][tile_count] transformed inputs  (B^T d B), prenorm fused
//!   M  half [36][out_ch][tile_count] GEMM output, then Y = A^T M A + bias
pub const src: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct WinoParams {
    \\    uint in_ch;
    \\    uint out_ch;
    \\    uint height;
    \\    uint width;
    \\    uint tiles_x;
    \\    uint tile0;
    \\    uint tile_count;
    \\    uint groups;
    \\    uint has_bias;
    \\    uint bias_dtype;
    \\    uint norm_dtype;
    \\    uint norm_bias_dtype;
    \\    ulong weight_offset;
    \\    ulong bias_offset;
    \\    ulong norm_weight_offset;
    \\    ulong norm_bias_offset;
    \\    // Strip-memory decode (memory-ladder wall 1): the input buffer holds
    \\    // only rows [in_row0, in_row0+in_rows) of each plane, the output
    \\    // buffer rows [out_row0, out_row0+out_rows) and, when out_local is
    \\    // set, only the channels [oc0, oc0+oc_count). Whole-map callers set
    \\    // in_row0 = out_row0 = oc0 = 0, in_rows = out_rows = height,
    \\    // oc_count = out_ch, out_local = 0 and compute exactly as before.
    \\    uint in_row0;
    \\    uint in_rows;
    \\    uint out_row0;
    \\    uint out_rows;
    \\    uint oc0;
    \\    uint oc_count;
    \\    uint out_local;
    \\    // In-place strips (tier 2): input rows [stash_row0, stash_row0+stash_rows)
    \\    // are read from the stash buffer (buffer 6), which holds the rows the
    \\    // previous strip overwrote; stash_rows = 0 reads the input only.
    \\    uint stash_row0;
    \\    uint stash_rows;
    \\    uint pad2_;
    \\    uint pad3_;
    \\};
    \\
    \\struct WinoGemmParams {
    \\    uint m;       // out_ch (the M plane stride)
    \\    uint k;       // in_ch
    \\    uint n;       // tile_count
    \\    uint m0;      // first output-channel row of this dispatch
    \\};
    \\
    \\static inline float wino_read(const device uchar* base, uint index,
    \\                              uint dtype) {
    \\    if (dtype == 1)
    \\        return float(reinterpret_cast<const device half*>(base)[index]);
    \\    if (dtype == 2) {
    \\        ushort b = reinterpret_cast<const device ushort*>(base)[index];
    \\        return as_type<float>(uint(b) << 16);
    \\    }
    \\    return reinterpret_cast<const device float*>(base)[index];
    \\}
    \\
    \\// Weight transform U = G g G^T, one thread per (oc, ic); g is the f16 3x3
    \\// kernel at weight_offset (layout [oc][ic][3][3]).
    \\kernel void wino_weight_h(
    \\    const device uchar* weight [[buffer(0)]],
    \\    device half* U [[buffer(1)]],
    \\    constant WinoParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint total = p.out_ch * p.in_ch;
    \\    if (gid >= total) return;
    \\    const device half* g =
    \\        reinterpret_cast<const device half*>(weight + p.weight_offset) + gid * 9;
    \\    float w[3][3];
    \\    for (uint r = 0; r < 3; r++)
    \\        for (uint c = 0; c < 3; c++) w[r][c] = float(g[r * 3 + c]);
    \\    // G (6x3): rows {1,0,0} {1,3/4,9/16} {1,-3/4,9/16} {1,5/4,25/16} {1,-5/4,25/16} {0,0,1}
    \\    float t[6][3];
    \\    for (uint c = 0; c < 3; c++) {
    \\        float a = w[0][c], b = w[1][c], d = w[2][c];
    \\        t[0][c] = a;
    \\        t[1][c] = a + 0.75f * b + 0.5625f * d;
    \\        t[2][c] = a - 0.75f * b + 0.5625f * d;
    \\        t[3][c] = a + 1.25f * b + 1.5625f * d;
    \\        t[4][c] = a - 1.25f * b + 1.5625f * d;
    \\        t[5][c] = d;
    \\    }
    \\    for (uint r = 0; r < 6; r++) {
    \\        float a = t[r][0], b = t[r][1], d = t[r][2];
    \\        float u[6];
    \\        u[0] = a;
    \\        u[1] = a + 0.75f * b + 0.5625f * d;
    \\        u[2] = a - 0.75f * b + 0.5625f * d;
    \\        u[3] = a + 1.25f * b + 1.5625f * d;
    \\        u[4] = a - 1.25f * b + 1.5625f * d;
    \\        u[5] = d;
    \\        for (uint c = 0; c < 6; c++) U[(r * 6 + c) * total + gid] = half(u[c]);
    \\    }
    \\}
    \\
    \\// B^T (6x6) for the points {0, +-3/4, +-5/4, inf} applied to a 6-vector.
    \\static inline void wino_bt(thread const float* d, thread float* o) {
    \\    o[0] = d[0] + (-2.41777778f) * d[2] + 1.13777778f * d[4];
    \\    o[1] = 1.04166667f * d[1] + 1.38888889f * d[2];
    \\    o[1] = o[1] - 0.666666667f * d[3] - 0.888888889f * d[4];
    \\    o[2] = (-1.04166667f) * d[1] + 1.38888889f * d[2];
    \\    o[2] = o[2] + 0.666666667f * d[3] - 0.888888889f * d[4];
    \\    o[3] = (-0.225f) * d[1] + (-0.18f) * d[2] + 0.4f * d[3] + 0.32f * d[4];
    \\    o[4] = 0.225f * d[1] + (-0.18f) * d[2] + (-0.4f) * d[3] + 0.32f * d[4];
    \\    o[5] = 0.87890625f * d[1] + (-2.125f) * d[3] + d[5];
    \\}
    \\
    \\// Input transform with GroupNorm+SiLU fused into the read (the prenorm
    \\// kernels' op order: (x-mean)*scale, *w+b, silu). One thread per
    \\// (tile, ic); lanes run over consecutive tiles so the V stores coalesce.
    \\// Threadgroup 128 = 32 tiles x 4 channels (tg.y steps in_ch by 4).
    \\kernel void wino_input_h(
    \\    const device half* input [[buffer(0)]],
    \\    device half* V [[buffer(1)]],
    \\    constant WinoParams& p [[buffer(2)]],
    \\    const device float* stats [[buffer(3)]],
    \\    const device uchar* norm_weight [[buffer(4)]],
    \\    const device uchar* norm_bias [[buffer(5)]],
    \\    const device half* stash [[buffer(6)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint lane = tid & 31u;
    \\    uint ic = tg.y * 4 + (tid >> 5);
    \\    uint t = tg.x * 32 + lane;
    \\    if (ic >= p.in_ch || t >= p.tile_count) return;
    \\    uint tile = p.tile0 + t;
    \\    uint ty = tile / p.tiles_x;
    \\    uint tx = tile - ty * p.tiles_x;
    \\    int y0 = int(ty) * 4 - 1;
    \\    int x0 = int(tx) * 4 - 1;
    \\    uint group_ch = p.in_ch / p.groups;
    \\    uint g = ic / group_ch;
    \\    float mean = stats[g * 2 + 0];
    \\    float scale = stats[g * 2 + 1];
    \\    float wn = wino_read(norm_weight + p.norm_weight_offset, ic, p.norm_dtype);
    \\    float bn = wino_read(norm_bias + p.norm_bias_offset, ic, p.norm_bias_dtype);
    \\    const device half* plane = input + ic * p.in_rows * p.width;
    \\    float d[6][6];
    \\    for (uint r = 0; r < 6; r++) {
    \\        int yy = y0 + int(r);
    \\        for (uint c = 0; c < 6; c++) {
    \\            int xx = x0 + int(c);
    \\            float v = 0.0f;
    \\            if (yy >= 0 && yy < int(p.height) && xx >= 0 && xx < int(p.width) &&
    \\                yy >= int(p.in_row0) && yy < int(p.in_row0 + p.in_rows)) {
    \\                bool st = p.stash_rows != 0u && yy >= int(p.stash_row0) &&
    \\                          yy < int(p.stash_row0 + p.stash_rows);
    \\                float raw = st
    \\                    ? float(stash[(ic * p.stash_rows + (uint(yy) - p.stash_row0)) * p.width
    \\                        + uint(xx)])
    \\                    : float(plane[(uint(yy) - p.in_row0) * p.width + uint(xx)]);
    \\                float x = (raw - mean) * scale;
    \\                x = x * wn + bn;
    \\                v = x / (1.0f + exp(-x));
    \\            }
    \\            d[r][c] = v;
    \\        }
    \\    }
    \\    // rows: T = B^T d ; then columns: V = T B (apply B^T to each row of T)
    \\    float tmp[6][6];
    \\    for (uint c = 0; c < 6; c++) {
    \\        float col[6], o[6];
    \\        for (uint r = 0; r < 6; r++) col[r] = d[r][c];
    \\        wino_bt(col, o);
    \\        for (uint r = 0; r < 6; r++) tmp[r][c] = o[r];
    \\    }
    \\    device half* out = V + ic * p.tile_count + t;
    \\    uint plane_stride = p.in_ch * p.tile_count;
    \\    for (uint r = 0; r < 6; r++) {
    \\        float o[6];
    \\        wino_bt(tmp[r], o);
    \\        for (uint c = 0; c < 6; c++) out[(r * 6 + c) * plane_stride] = half(o[c]);
    \\    }
    \\}
    \\
    \\// Input transform over the nearest-2x upsampled map (the fused upsample
    \\// conv): the tile's 6x6 window reads input[(y/2, x/2)] of the half-size
    \\// map (in_ch planes of (height/2 x width/2)); no prenorm.
    \\kernel void wino_input_up_h(
    \\    const device half* input [[buffer(0)]],
    \\    device half* V [[buffer(1)]],
    \\    constant WinoParams& p [[buffer(2)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint lane = tid & 31u;
    \\    uint ic = tg.y * 4 + (tid >> 5);
    \\    uint t = tg.x * 32 + lane;
    \\    if (ic >= p.in_ch || t >= p.tile_count) return;
    \\    uint tile = p.tile0 + t;
    \\    uint ty = tile / p.tiles_x;
    \\    uint tx = tile - ty * p.tiles_x;
    \\    int y0 = int(ty) * 4 - 1;
    \\    int x0 = int(tx) * 4 - 1;
    \\    uint in_h = p.height / 2u;
    \\    uint in_w = p.width / 2u;
    \\    const device half* plane = input + ic * in_h * in_w;
    \\    float d[6][6];
    \\    for (uint r = 0; r < 6; r++) {
    \\        int yy = y0 + int(r);
    \\        for (uint c = 0; c < 6; c++) {
    \\            int xx = x0 + int(c);
    \\            float v = 0.0f;
    \\            if (yy >= 0 && yy < int(p.height) && xx >= 0 && xx < int(p.width))
    \\                v = float(plane[(uint(yy) >> 1) * in_w + (uint(xx) >> 1)]);
    \\            d[r][c] = v;
    \\        }
    \\    }
    \\    float tmp[6][6];
    \\    for (uint c = 0; c < 6; c++) {
    \\        float col[6], o[6];
    \\        for (uint r = 0; r < 6; r++) col[r] = d[r][c];
    \\        wino_bt(col, o);
    \\        for (uint r = 0; r < 6; r++) tmp[r][c] = o[r];
    \\    }
    \\    device half* out = V + ic * p.tile_count + t;
    \\    uint plane_stride = p.in_ch * p.tile_count;
    \\    for (uint r = 0; r < 6; r++) {
    \\        float o[6];
    \\        wino_bt(tmp[r], o);
    \\        for (uint c = 0; c < 6; c++) out[(r * 6 + c) * plane_stride] = half(o[c]);
    \\    }
    \\}
    \\
    \\// Batched GEMM over the 36 planes (tg.z): M[z][m][n] = U[z][m][k] . V[z][k][n],
    \\// half operands, f32 accumulate, half store. 64x64 tile, 4 simdgroups,
    \\// K double-buffered (the gemm_f16a_direct pattern); V is K-major so its
    \\// stage is loaded along n and B needs no transpose. Contract: m % 64,
    \\// k % 32, n % 64 == 0.
    \\kernel void wino_gemm_h(
    \\    const device half* U [[buffer(0)]],
    \\    const device half* V [[buffer(1)]],
    \\    device half* M [[buffer(2)]],
    \\    constant WinoGemmParams& p [[buffer(3)]],
    \\    uint3 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint sgid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const device half* A = U + tg.z * p.m * p.k;
    \\    const device half* W = V + tg.z * p.k * p.n;
    \\    device half* C = M + tg.z * p.m * p.n;
    \\    const uint tile_m = p.m0 + tg.y * 64;
    \\    const uint tile_n = tg.x * 64;
    \\    const uint sm = (sgid >> 1) * 32;
    \\    const uint sn = (sgid & 1) * 32;
    \\    threadgroup half As[2][64 * 32];
    \\    threadgroup half Ws[2][32 * 64];
    \\    threadgroup float Cs[4][32 * 32];
    \\    const uint arow = tid >> 1;
    \\    const uint aseg = (tid & 1) * 16;
    \\    const uint wrow = tid >> 2;
    \\    const uint wseg = (tid & 3) * 16;
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    const device half4* Arow =
    \\        reinterpret_cast<const device half4*>(A + (tile_m + arow) * p.k);
    \\    uint cur = 0;
    \\    {
    \\        threadgroup half4* ad =
    \\            reinterpret_cast<threadgroup half4*>(&As[0][arow * 32 + aseg]);
    \\        for (uint q = 0; q < 4; q++) ad[q] = Arow[(aseg >> 2) + q];
    \\        const device half4* wsrc =
    \\            reinterpret_cast<const device half4*>(W + wrow * p.n + tile_n + wseg);
    \\        threadgroup half4* wd =
    \\            reinterpret_cast<threadgroup half4*>(&Ws[0][wrow * 64 + wseg]);
    \\        for (uint q = 0; q < 4; q++) wd[q] = wsrc[q];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint k0 = 0; k0 < p.k; k0 += 32) {
    \\        const uint nk = k0 + 32;
    \\        if (nk < p.k) {
    \\            threadgroup half4* ad =
    \\                reinterpret_cast<threadgroup half4*>(&As[1 - cur][arow * 32 + aseg]);
    \\            const uint base = (nk + aseg) >> 2;
    \\            for (uint q = 0; q < 4; q++) ad[q] = Arow[base + q];
    \\            const device half4* wsrc = reinterpret_cast<const device half4*>(
    \\                W + (nk + wrow) * p.n + tile_n + wseg);
    \\            threadgroup half4* wd =
    \\                reinterpret_cast<threadgroup half4*>(&Ws[1 - cur][wrow * 64 + wseg]);
    \\            for (uint q = 0; q < 4; q++) wd[q] = wsrc[q];
    \\        }
    \\        for (uint kk = 0; kk < 32; kk += 8) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], &Ws[cur][kk * 64 + sn + j * 8], 64);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
    \\        cur = 1 - cur;
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], Cs[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // each lane writes one row (32 n) of its simdgroup's 32x32 block as half4
    \\    uint lane = tid & 31u;
    \\    device half4* dst = reinterpret_cast<device half4*>(
    \\        C + (tile_m + sm + lane) * p.n + tile_n + sn);
    \\    const threadgroup float4* srcv =
    \\        reinterpret_cast<const threadgroup float4*>(Cs[sgid] + lane * 32);
    \\    for (uint q = 0; q < 8; q++) dst[q] = half4(srcv[q]);
    \\}
    \\
    \\// Output transform Y = A^T M A (+ bias) -> NCHW f16 output; one thread per
    \\// (tile, oc), lanes over consecutive tiles (M reads and the row writes
    \\// coalesce). Threadgroup 128 = 32 tiles x 4 output channels.
    \\kernel void wino_output_h(
    \\    const device half* M [[buffer(0)]],
    \\    device half* output [[buffer(1)]],
    \\    constant WinoParams& p [[buffer(2)]],
    \\    const device uchar* bias [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint lane = tid & 31u;
    \\    uint oc = p.oc0 + tg.y * 4 + (tid >> 5);
    \\    uint t = tg.x * 32 + lane;
    \\    if (oc >= p.out_ch || oc >= p.oc0 + p.oc_count || t >= p.tile_count) return;
    \\    uint tile = p.tile0 + t;
    \\    uint ty = tile / p.tiles_x;
    \\    uint tx = tile - ty * p.tiles_x;
    \\    uint plane_stride = p.out_ch * p.tile_count;
    \\    const device half* m = M + oc * p.tile_count + t;
    \\    float mm[6][6];
    \\    for (uint r = 0; r < 6; r++)
    \\        for (uint c = 0; c < 6; c++) mm[r][c] = float(m[(r * 6 + c) * plane_stride]);
    \\    // A^T (4x6): {1,1,1,1,1,0} {0,3/4,-3/4,5/4,-5/4,0}
    \\    //             {0,9/16,9/16,25/16,25/16,0} {0,27/64,-27/64,125/64,-125/64,1}
    \\    float tmp[4][6];
    \\    for (uint c = 0; c < 6; c++) {
    \\        float m0 = mm[0][c], m1 = mm[1][c], m2 = mm[2][c];
    \\        float m3 = mm[3][c], m4 = mm[4][c], m5 = mm[5][c];
    \\        tmp[0][c] = m0 + m1 + m2 + m3 + m4;
    \\        tmp[1][c] = 0.75f * (m1 - m2) + 1.25f * (m3 - m4);
    \\        tmp[2][c] = 0.5625f * (m1 + m2) + 1.5625f * (m3 + m4);
    \\        tmp[3][c] = 0.421875f * (m1 - m2) + 1.953125f * (m3 - m4) + m5;
    \\    }
    \\    float bval = 0.0f;
    \\    if (p.has_bias != 0) bval = wino_read(bias + p.bias_offset, oc, p.bias_dtype);
    \\    uint oc_out = (p.out_local != 0) ? (oc - p.oc0) : oc;
    \\    device half* out = output + oc_out * p.out_rows * p.width;
    \\    for (uint r = 0; r < 4; r++) {
    \\        float m0 = tmp[r][0], m1 = tmp[r][1], m2 = tmp[r][2];
    \\        float m3 = tmp[r][3], m4 = tmp[r][4], m5 = tmp[r][5];
    \\        float y[4];
    \\        y[0] = m0 + m1 + m2 + m3 + m4;
    \\        y[1] = 0.75f * (m1 - m2) + 1.25f * (m3 - m4);
    \\        y[2] = 0.5625f * (m1 + m2) + 1.5625f * (m3 + m4);
    \\        y[3] = 0.421875f * (m1 - m2) + 1.953125f * (m3 - m4) + m5;
    \\        uint yy = ty * 4 + r;
    \\        if (yy >= p.height || yy < p.out_row0 || yy >= p.out_row0 + p.out_rows) continue;
    \\        for (uint c = 0; c < 4; c++) {
    \\            uint xx = tx * 4 + c;
    \\            if (xx < p.width) out[(yy - p.out_row0) * p.width + xx] = half(y[c] + bval);
    \\        }
    \\    }
    \\}
;
