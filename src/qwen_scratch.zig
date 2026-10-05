//! Scratch buffers reused while running the Qwen text encoder.

const std = @import("std");

const qattn = @import("qwen_attn.zig");
const qlayer = @import("qwen_layer.zig");

pub const Scratch = struct {
    state: []f32,
    layer: qlayer.Scratch,

    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        allocator.free(self.state);
        allocator.free(self.layer.attn.norm);
        allocator.free(self.layer.attn.q);
        allocator.free(self.layer.attn.k);
        allocator.free(self.layer.attn.v);
        allocator.free(self.layer.attn.mix);
        allocator.free(self.layer.attn.scores);
        allocator.free(self.layer.attn_out);
        allocator.free(self.layer.mlp.gate);
        allocator.free(self.layer.mlp.up);
        allocator.free(self.layer.mlp_out);
        self.* = undefined;
    }
};

pub fn init(
    allocator: std.mem.Allocator,
    cfg: qattn.Config,
    intermediate: usize,
) !Scratch {
    var out = empty();
    errdefer out.deinit(allocator);
    try allocBuffers(&out, allocator, cfg, intermediate);
    return out;
}

fn empty() Scratch {
    return .{
        .state = &.{},
        .layer = .{
            .attn = .{
                .norm = &.{},
                .q = &.{},
                .k = &.{},
                .v = &.{},
                .mix = &.{},
                .scores = &.{},
            },
            .attn_out = &.{},
            .mlp = .{ .gate = &.{}, .up = &.{} },
            .mlp_out = &.{},
        },
    };
}

fn allocBuffers(
    out: *Scratch,
    allocator: std.mem.Allocator,
    cfg: qattn.Config,
    intermediate: usize,
) !void {
    const state_len = cfg.tokens * cfg.hidden;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    out.state = try allocator.alloc(f32, state_len);
    out.layer.attn.norm = try allocator.alloc(f32, cfg.hidden);
    out.layer.attn.q = try allocator.alloc(f32, q_len);
    out.layer.attn.k = try allocator.alloc(f32, kv_len);
    out.layer.attn.v = try allocator.alloc(f32, kv_len);
    out.layer.attn.mix = try allocator.alloc(f32, q_len);
    out.layer.attn.scores = try allocator.alloc(f32, cfg.tokens);
    out.layer.attn_out = try allocator.alloc(f32, state_len);
    out.layer.mlp.gate = try allocator.alloc(f32, cfg.tokens * intermediate);
    out.layer.mlp.up = try allocator.alloc(f32, cfg.tokens * intermediate);
    out.layer.mlp_out = try allocator.alloc(f32, state_len);
}

test "allocate and free scratch" {
    const cfg = qattn.Config{
        .tokens = 2,
        .hidden = 4,
        .heads = 2,
        .kv_heads = 1,
        .head_dim = 2,
        .norm_eps = 0.000001,
        .rope_theta = 10000.0,
        .causal = true,
    };
    var scratch = try init(std.testing.allocator, cfg, 8);
    defer scratch.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 8), scratch.state.len);
    try std.testing.expectEqual(@as(usize, 8), scratch.layer.attn.q.len);
    try std.testing.expectEqual(@as(usize, 4), scratch.layer.attn.k.len);
    try std.testing.expectEqual(@as(usize, 16), scratch.layer.mlp.gate.len);
    try std.testing.expectEqual(@as(usize, 8), scratch.layer.mlp_out.len);
}
