//! Resident Metal VAE residual block runner.

const std = @import("std");

const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const bufs = @import("mvres_buf.zig");
const params = @import("mvres_param.zig");
const vres = @import("../vae/vres.zig");

extern fn zdraw_metal_run_vae_res(
    queue: *anyopaque,
    norm_pipeline: *anyopaque,
    conv_pipeline: *anyopaque,
    add_pipeline: *anyopaque,
    buffers: *const bufs.ResBuffers,
    ps: *const params.ResParams,
    norm_threads: usize,
    conv_threads: usize,
    add_threads: usize,
) c_int;

pub fn run(
    ctx: *mconv.Context,
    out: []f32,
    input: []const f32,
    views: vres.Views,
    cfg: vres.Config,
    recycle: ?mbuffer.Recycle,
) !void {
    try params.check(out, input, views, cfg);
    var set = try bufs.make(ctx, input, out.len, views.skip_w != null, recycle);
    defer set.deinit();
    var temps = bufs.Temps{};
    defer temps.deinit();
    const binds = try bufs.bindAll(ctx, views, &temps);
    const ps = try params.make(views, binds, cfg);
    const cbufs = bufs.resBuffers(set, binds);
    const code = zdraw_metal_run_vae_res(
        ctx.queue,
        ctx.norm_pipeline,
        ctx.pipeline,
        ctx.add_pipeline,
        &cbufs,
        &ps,
        ctx.norm_threads,
        ctx.threads,
        ctx.add_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
    readBack(set.output, out);
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(
        buf.handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}
