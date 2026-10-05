//! Shared types for chained resident block execution.

const zmod = @import("../zimage/zmod.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
};

pub fn attnScale(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.attn_scale;
    return null;
}

pub fn attnGate(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.attn_gate;
    return null;
}

pub fn mlpScale(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.mlp_scale;
    return null;
}

pub fn mlpGate(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.mlp_gate;
    return null;
}
