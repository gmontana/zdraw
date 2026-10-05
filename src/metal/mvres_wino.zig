//! Winograd F(4x4,3x3) host side for the product decoder
//! (vae-winograd-20260827): the pipes, the persistent scratch and the set the
//! C chain carries; the kernels live in mconv_wino_shader.zig.
const std = @import("std");
const c = @import("metal_c.zig");
const mpipe = @import("mpipe.zig");
const mbuffer = @import("mbuffer.zig");
const mconv_wino = @import("mconv_wino_shader.zig");
const chain = @import("mvres_stream_chain.zig");

/// The product profile sets ZDRAW_VAE_WINO=1 (gated 2026-08-27: PSNR vs
/// strict 48.7 dB, census, content); `0` restores the direct kernels for A/B.
/// Strict never sets it and its f32 route does not engage it.
pub fn enabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_WINO") orelse return false;
    return raw[0] != '0';
}

/// Mirrors the C ZdrawWinoSet.
pub const Set = extern struct {
    weight_pipe: *anyopaque,
    input_pipe: *anyopaque,
    input_up_pipe: *anyopaque,
    gemm_pipe: *anyopaque,
    output_pipe: *anyopaque,
    u: *anyopaque,
    v: *anyopaque,
    m: *anyopaque,
    v_bytes: u32,
};

extern fn zdraw_metal_run_wino_upsample(
    queue: *anyopaque,
    ws: *const Set,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    up: *const chain.ConvUpsampleWindowParams,
) c_int;

pub const Pipes = struct {
    weight: *anyopaque,
    input: *anyopaque,
    input_up: *anyopaque,
    gemm: *anyopaque,
    output: *anyopaque,

    pub fn init(dev: *anyopaque) !Pipes {
        var err: [1024]u8 = undefined;
        const src = mconv_wino.src.ptr;
        const weight = try mpipe.required(dev, src, "wino_weight_h", &err);
        errdefer c.zdraw_metal_release_pipeline(weight);
        const input = try mpipe.required(dev, src, "wino_input_h", &err);
        errdefer c.zdraw_metal_release_pipeline(input);
        const input_up = try mpipe.required(dev, src, "wino_input_up_h", &err);
        errdefer c.zdraw_metal_release_pipeline(input_up);
        const gemm = try mpipe.required(dev, src, "wino_gemm_h", &err);
        errdefer c.zdraw_metal_release_pipeline(gemm);
        const output = try mpipe.required(dev, src, "wino_output_h", &err);
        errdefer c.zdraw_metal_release_pipeline(output);
        return .{
            .weight = weight,
            .input = input,
            .input_up = input_up,
            .gemm = gemm,
            .output = output,
        };
    }

    pub fn deinit(self: Pipes) void {
        c.zdraw_metal_release_pipeline(self.output);
        c.zdraw_metal_release_pipeline(self.gemm);
        c.zdraw_metal_release_pipeline(self.input_up);
        c.zdraw_metal_release_pipeline(self.input);
        c.zdraw_metal_release_pipeline(self.weight);
    }
};

/// Persistent scratch: U planes sized for the largest conv, V and M at the
/// tile-batch cap (the C side sizes each conv's batch to fit); ~67 MB.
pub const Scratch = struct {
    u: Grow = .{},
    v: Grow = .{},
    m: Grow = .{},

    pub fn deinit(self: *Scratch) void {
        self.m.deinit();
        self.v.deinit();
        self.u.deinit();
    }
};

const Grow = struct {
    buf: ?mbuffer.Buffer = null,
    cap: usize = 0,

    fn handle(self: *Grow, device: *anyopaque, bytes: usize) !*anyopaque {
        if (self.buf == null or self.cap < bytes) {
            if (self.buf) |*b| b.deinit();
            self.buf = try mbuffer.Buffer.empty(device, bytes);
            self.cap = bytes;
        }
        return self.buf.?.handle;
    }

    fn deinit(self: *Grow) void {
        if (self.buf) |*b| b.deinit();
        self.buf = null;
        self.cap = 0;
    }
};

const batch_bytes: usize = 24 * 1024 * 1024;

pub fn set(scratch: *Scratch, pipes: Pipes, dev: *anyopaque, in_ch: usize, out_ch: usize) !Set {
    const big = @max(in_ch, out_ch);
    return .{
        .weight_pipe = pipes.weight,
        .input_pipe = pipes.input,
        .input_up_pipe = pipes.input_up,
        .gemm_pipe = pipes.gemm,
        .output_pipe = pipes.output,
        .u = try scratch.u.handle(dev, 36 * big * big * 2),
        .v = try scratch.v.handle(dev, batch_bytes),
        .m = try scratch.m.handle(dev, batch_bytes),
        .v_bytes = @intCast(batch_bytes),
    };
}

/// The fused upsample conv over the whole map; false when the shape is not
/// Winograd-eligible (the caller's strip loop then runs).
pub fn upsample(
    queue: *anyopaque,
    scratch: *Scratch,
    pipes: Pipes,
    dev: *anyopaque,
    cfg: chain.Config,
    conv: c.ConvParams,
    up_h: u32,
    up_w: u32,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
) !bool {
    const ws = try set(scratch, pipes, dev, cfg.out_ch, cfg.out_ch);
    const whole = chain.ConvUpsampleWindowParams{
        .channels = @intCast(cfg.out_ch),
        .out_height = up_h,
        .out_width = up_w,
        .in_height = @intCast(cfg.height),
        .in_width = @intCast(cfg.width),
        .ksize = 3,
        .pad = 1,
        .dtype = conv.dtype,
        .bias_dtype = conv.bias_dtype,
        .has_bias = conv.has_bias,
        .weight_offset = conv.weight_offset,
        .bias_offset = conv.bias_offset,
        .row0 = 0,
        .row1 = up_h,
    };
    const rc = zdraw_metal_run_wino_upsample(queue, &ws, input, weight, bias, output, &whole);
    if (rc < 0) return error.MetalDispatchFailed;
    return rc == 0;
}
