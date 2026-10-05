//! VAE projection helper with biased GEMM when the shape fits.

const linear = @import("linear_fast.zig");
const mgemm_bias = @import("mgemm_bias.zig");
const mlinear = @import("mlinear.zig");
const mfallback = @import("metal_fallback.zig");
const tensor = @import("tensor.zig");

pub fn run(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    batch: usize,
) !void {
    if (metal) |ctx| {
        if (bias) |b| {
            mgemm_bias.batch(ctx, out, input, weight, b, batch) catch |err| {
                if (mfallback.isGemmRefusal(err)) {
                    return linear.runBatch(metal, out, input, weight, bias, batch);
                }
                return err;
            };
            return;
        }
    }
    try linear.gemmBatch(metal, out, input, weight, bias, batch);
}
