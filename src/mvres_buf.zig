//! Buffers and weight bindings for resident VAE residual blocks.

const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const vres = @import("vres.zig");

pub const ResBuffers = extern struct {
    input: *anyopaque,
    norm: *anyopaque,
    work: *anyopaque,
    output: *anyopaque,
    skip: *anyopaque,
    norm1_w: *anyopaque,
    norm1_b: *anyopaque,
    conv1_w: *anyopaque,
    conv1_b: *anyopaque,
    norm2_w: *anyopaque,
    norm2_b: *anyopaque,
    conv2_w: *anyopaque,
    conv2_b: *anyopaque,
    skip_w: *anyopaque,
    skip_b: *anyopaque,
};

pub const Binds = struct {
    norm1_w: mbuffer.Bind,
    norm1_b: mbuffer.Bind,
    conv1_w: mbuffer.Bind,
    conv1_b: mbuffer.Bind,
    norm2_w: mbuffer.Bind,
    norm2_b: mbuffer.Bind,
    conv2_w: mbuffer.Bind,
    conv2_b: mbuffer.Bind,
    skip_w: mbuffer.Bind,
    skip_b: mbuffer.Bind,
};

pub const Set = struct {
    input: mbuffer.Buffer,
    norm: mbuffer.Buffer,
    work: mbuffer.Buffer,
    output: mbuffer.Buffer,
    skip: ?mbuffer.Buffer,

    pub fn deinit(self: *Set) void {
        if (self.skip) |*buf| buf.deinit();
        self.output.deinit();
        self.work.deinit();
        self.norm.deinit();
        self.input.deinit();
    }
};

pub const Temps = struct {
    norm1_w: ?mbuffer.Buffer = null,
    norm1_b: ?mbuffer.Buffer = null,
    conv1_w: ?mbuffer.Buffer = null,
    conv1_b: ?mbuffer.Buffer = null,
    norm2_w: ?mbuffer.Buffer = null,
    norm2_b: ?mbuffer.Buffer = null,
    conv2_w: ?mbuffer.Buffer = null,
    conv2_b: ?mbuffer.Buffer = null,
    skip_w: ?mbuffer.Buffer = null,
    skip_b: ?mbuffer.Buffer = null,

    pub fn deinit(self: *Temps) void {
        if (self.skip_b) |*buf| buf.deinit();
        if (self.skip_w) |*buf| buf.deinit();
        if (self.conv2_b) |*buf| buf.deinit();
        if (self.conv2_w) |*buf| buf.deinit();
        if (self.norm2_b) |*buf| buf.deinit();
        if (self.norm2_w) |*buf| buf.deinit();
        if (self.conv1_b) |*buf| buf.deinit();
        if (self.conv1_w) |*buf| buf.deinit();
        if (self.norm1_b) |*buf| buf.deinit();
        if (self.norm1_w) |*buf| buf.deinit();
    }
};

pub fn make(
    ctx: *mconv.Context,
    input: []const f32,
    out_len: usize,
    has_skip: bool,
    recycle: ?mbuffer.Recycle,
) !Set {
    var input_buf = try mbuffer.Buffer.fromInput(ctx.device, input, recycle);
    errdefer input_buf.deinit();
    var norm_buf = try empty(ctx, input.len);
    errdefer norm_buf.deinit();
    var work_buf = try empty(ctx, out_len);
    errdefer work_buf.deinit();
    var out_buf = try empty(ctx, out_len);
    errdefer out_buf.deinit();
    const skip_buf = if (has_skip) try empty(ctx, out_len) else null;
    return .{
        .input = input_buf,
        .norm = norm_buf,
        .work = work_buf,
        .output = out_buf,
        .skip = skip_buf,
    };
}

pub fn bindAll(ctx: *mconv.Context, views: vres.Views, temps: *Temps) !Binds {
    return .{
        .norm1_w = try ctx.buffers.bindView(views.norm1_w, &temps.norm1_w),
        .norm1_b = try ctx.buffers.bindView(views.norm1_b, &temps.norm1_b),
        .conv1_w = try ctx.buffers.bindView(views.conv1_w, &temps.conv1_w),
        .conv1_b = try ctx.buffers.bindView(views.conv1_b, &temps.conv1_b),
        .norm2_w = try ctx.buffers.bindView(views.norm2_w, &temps.norm2_w),
        .norm2_b = try ctx.buffers.bindView(views.norm2_b, &temps.norm2_b),
        .conv2_w = try ctx.buffers.bindView(views.conv2_w, &temps.conv2_w),
        .conv2_b = try ctx.buffers.bindView(views.conv2_b, &temps.conv2_b),
        .skip_w = try ctx.buffers.bindBias(views.skip_w, &temps.skip_w),
        .skip_b = try ctx.buffers.bindBias(views.skip_b, &temps.skip_b),
    };
}

pub fn resBuffers(set: Set, binds: Binds) ResBuffers {
    const skip = if (set.skip) |buf| buf.handle else set.output.handle;
    return .{
        .input = set.input.handle,
        .norm = set.norm.handle,
        .work = set.work.handle,
        .output = set.output.handle,
        .skip = skip,
        .norm1_w = binds.norm1_w.handle,
        .norm1_b = binds.norm1_b.handle,
        .conv1_w = binds.conv1_w.handle,
        .conv1_b = binds.conv1_b.handle,
        .norm2_w = binds.norm2_w.handle,
        .norm2_b = binds.norm2_b.handle,
        .conv2_w = binds.conv2_w.handle,
        .conv2_b = binds.conv2_b.handle,
        .skip_w = binds.skip_w.handle,
        .skip_b = binds.skip_b.handle,
    };
}

fn empty(ctx: *mconv.Context, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(ctx.device, count * @sizeOf(f32));
}
