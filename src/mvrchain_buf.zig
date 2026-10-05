//! Buffer ownership for chained resident VAE residual blocks.
//!
//! The three blocks in an up group share one spatial size and out_ch, and the
//! command encoder runs them serially, so their scratch never overlaps in time:
//! one norm/work buffer is reused by every block, and the per-block result
//! ping-pongs between just two output buffers (block i's input is block i-1's
//! output, which is dead once block i has read it). The skip projection (only
//! block 0 of a group ever has one) lands in the work buffer, which is already
//! dead by then within that block. That holds the resident GPU working set to
//! input + norm + work + 2 outputs instead of a fresh norm/work/output/skip per
//! block, without changing a single computed value.

const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const bufs = @import("mvres_buf.zig");
const vres = @import("vres.zig");

pub const block_count = 3;
const output_slots = 2;

pub const Config = struct {
    in_ch: usize,
    out_ch: usize,
    height: usize,
    width: usize,
};

pub const Set = struct {
    input: mbuffer.Buffer,
    norm: ?mbuffer.Buffer = null,
    work: ?mbuffer.Buffer = null,
    output: [output_slots]?mbuffer.Buffer = [_]?mbuffer.Buffer{null} ** output_slots,

    pub fn make(
        ctx: *mconv.Context,
        input: []const f32,
        views: [block_count]vres.Views,
        cfg: Config,
        recycle: ?mbuffer.Recycle,
    ) !Set {
        var set = Set{ .input = try mbuffer.Buffer.fromInput(ctx.device, input, recycle) };
        errdefer set.deinit();
        try set.allocShared(ctx, views, cfg);
        return set;
    }

    pub fn resBuffers(self: *const Set, idx: usize, binds: bufs.Binds) bufs.ResBuffers {
        return .{
            .input = self.inputHandle(idx),
            .norm = self.norm.?.handle,
            .work = self.work.?.handle,
            .output = self.outputHandle(idx),
            .skip = self.skipHandle(idx),
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

    // The buffer holding the final block's result, for readback.
    pub fn finalOutput(self: *const Set) mbuffer.Buffer {
        return self.output[(block_count - 1) % output_slots].?;
    }

    pub fn deinit(self: *Set) void {
        releaseAll(&self.output);
        if (self.work) |*b| b.deinit();
        if (self.norm) |*b| b.deinit();
        self.input.deinit();
    }

    fn allocShared(
        self: *Set,
        ctx: *mconv.Context,
        views: [block_count]vres.Views,
        cfg: Config,
    ) !void {
        _ = views;
        // norm is read at each block's in_ch; block 0 has the widest in_ch, so
        // size to it. work/output all run at out_ch.
        const norm_elems = count(cfg.in_ch, blockCfg(cfg, 0));
        const out_elems = count(cfg.out_ch, blockCfg(cfg, 0));
        self.norm = try empty(ctx, @max(norm_elems, out_elems));
        self.work = try empty(ctx, out_elems);
        for (&self.output) |*slot| slot.* = try empty(ctx, out_elems);
    }

    fn inputHandle(self: *const Set, idx: usize) *anyopaque {
        return if (idx == 0) self.input.handle else self.outputHandle(idx - 1);
    }

    fn outputHandle(self: *const Set, idx: usize) *anyopaque {
        return self.output[idx % output_slots].?.handle;
    }

    // Block 0 of a group is the only one that ever takes a skip projection, and
    // its work buffer is already dead by the time the skip conv runs, so the
    // skip lands there. Blocks without a skip ignore this handle (the encoder
    // only reads b->skip when has_skip is set).
    fn skipHandle(self: *const Set, idx: usize) *anyopaque {
        _ = idx;
        return self.work.?.handle;
    }
};

pub fn blockCfg(cfg: Config, idx: usize) vres.Config {
    return .{
        .in_ch = if (idx == 0) cfg.in_ch else cfg.out_ch,
        .out_ch = cfg.out_ch,
        .height = cfg.height,
        .width = cfg.width,
    };
}

pub fn elemCount(channels: usize, cfg: Config) usize {
    return channels * cfg.height * cfg.width;
}

fn count(channels: usize, cfg: vres.Config) usize {
    return channels * cfg.height * cfg.width;
}

fn empty(ctx: *mconv.Context, len: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(ctx.device, len * @sizeOf(f32));
}

fn releaseAll(buffers: *[output_slots]?mbuffer.Buffer) void {
    for (buffers) |*buf| if (buf.*) |*b| b.deinit();
}
