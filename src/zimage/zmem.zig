//! Scratch memory for one Z-Image transformer block.
//!
//! A single owned float buffer is split into the slices expected by `zlayer`.

const std = @import("std");

const zlayer = @import("zlayer.zig");

pub const Owned = struct {
    buf: []f32,
    layer: zlayer.Scratch,

    pub fn deinit(self: *Owned, allocator: std.mem.Allocator) void {
        allocator.free(self.buf);
        self.* = undefined;
    }
};

pub fn init(
    allocator: std.mem.Allocator,
    cfg: zlayer.Config,
    ffn_dim: usize,
) !Owned {
    const total = count(cfg, ffn_dim);
    const buf = try allocator.alloc(f32, total);
    var pos: usize = 0;
    const state_len = cfg.tokens * cfg.hidden;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;

    const layer = zlayer.Scratch{
        .norm = take(buf, &pos, cfg.hidden),
        .mod = take(buf, &pos, 4 * cfg.hidden),
        .attn = .{
            .norm = take(buf, &pos, cfg.hidden),
            .q = take(buf, &pos, q_len),
            .k = take(buf, &pos, kv_len),
            .v = take(buf, &pos, kv_len),
            .mix = take(buf, &pos, q_len),
            .scores = take(buf, &pos, cfg.tokens),
        },
        .attn_out = take(buf, &pos, state_len),
        .ffn = .{
            .norm = take(buf, &pos, cfg.hidden),
            .gate = take(buf, &pos, cfg.tokens * ffn_dim),
            .up = take(buf, &pos, cfg.tokens * ffn_dim),
        },
        .ffn_out = take(buf, &pos, state_len),
    };
    return .{ .buf = buf, .layer = layer };
}

pub fn count(cfg: zlayer.Config, ffn_dim: usize) usize {
    const state_len = cfg.tokens * cfg.hidden;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    const ffn_len = cfg.tokens * ffn_dim;
    return cfg.hidden + 4 * cfg.hidden + cfg.hidden + q_len + kv_len +
        kv_len + q_len + cfg.tokens + state_len + cfg.hidden + ffn_len +
        ffn_len + state_len;
}

fn take(buf: []f32, pos: *usize, len: usize) []f32 {
    const start = pos.*;
    pos.* += len;
    return buf[start..pos.*];
}

test "allocate layer scratch slices" {
    const cfg = zlayer.Config{
        .tokens = 1,
        .hidden = 2,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .norm_eps = 0.0,
    };
    var owned = try init(std.testing.allocator, cfg, 3);
    defer owned.deinit(std.testing.allocator);

    try std.testing.expectEqual(count(cfg, 3), owned.buf.len);
    try std.testing.expectEqual(@as(usize, 8), owned.layer.mod.len);
    try std.testing.expectEqual(@as(usize, 3), owned.layer.ffn.gate.len);
}
