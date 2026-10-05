//! Shared convolution helper with an optional Metal backend.

const conv = @import("conv.zig");
const mconv = @import("../metal/mconv.zig");
const tensor = @import("../pack/tensor.zig");

pub fn run(
    metal: ?*mconv.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: conv.Config,
) !void {
    if (metal) |ctx| {
        ctx.run(out, input, weight, bias, cfg) catch |err| switch (err) {
            error.UnsupportedDType => {},
            else => return err,
        };
        if (supported(weight, bias)) return;
    }
    try conv.run(out, input, weight, bias, cfg);
}

fn supported(weight: tensor.View, bias: ?tensor.View) bool {
    if (!supportedType(weight.dtype)) return false;
    if (bias) |b| return supportedType(b.dtype);
    return true;
}

fn supportedType(dtype: tensor.DType) bool {
    return dtype == .f32 or dtype == .f16 or dtype == .bf16;
}
