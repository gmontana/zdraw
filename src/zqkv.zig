//! Exact Q/K/V projection helper for Z-Image attention.

const linear = @import("linear_fast.zig");
const mlinear = @import("mlinear.zig");
const mfallback = @import("metal_fallback.zig");
const mtri = @import("mtri.zig");
const tensor = @import("tensor.zig");

pub fn batch(
    metal: ?*mlinear.Context,
    q: []f32,
    k: []f32,
    v: []f32,
    input: []const f32,
    q_weight: tensor.View,
    k_weight: tensor.View,
    v_weight: tensor.View,
    count: usize,
) !void {
    if (metal) |ctx| {
        if (ctx.gemm_mode != .off and allGemm(q_weight, k_weight, v_weight)) {
            mtri.batch(ctx, q, k, v, input, q_weight, k_weight, v_weight, count) catch |err| {
                if (mfallback.isFallback(err)) {
                    return fallback(metal, q, k, v, input, q_weight, k_weight, v_weight, count);
                }
                return err;
            };
            return;
        }
    }
    try fallback(metal, q, k, v, input, q_weight, k_weight, v_weight, count);
}

fn fallback(
    metal: ?*mlinear.Context,
    q: []f32,
    k: []f32,
    v: []f32,
    input: []const f32,
    q_weight: tensor.View,
    k_weight: tensor.View,
    v_weight: tensor.View,
    count: usize,
) !void {
    try linear.gemmBatch(metal, q, input, q_weight, null, count);
    try linear.gemmBatch(metal, k, input, k_weight, null, count);
    try linear.gemmBatch(metal, v, input, v_weight, null, count);
}

fn allGemm(q: tensor.View, k: tensor.View, v: tensor.View) bool {
    return gemmType(q.dtype) and gemmType(k.dtype) and gemmType(v.dtype);
}

fn gemmType(dtype: tensor.DType) bool {
    return dtype == .f16 or dtype == .bf16 or dtype == .f32;
}
