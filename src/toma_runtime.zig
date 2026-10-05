//! Persistent owner for ToMA pipelines, construction scratch, and patterns.
//!
//! The runtime borrows the engine's Metal device and queue. It owns only its
//! compiled pipelines and reusable ToMA buffers, so teardown order is explicit:
//! destroy Runtime before the parent Metal context.

const std = @import("std");

const toma_config = @import("toma_config.zig");
const toma_metal = @import("toma_metal.zig");

pub const Runtime = struct {
    context: toma_metal.Context,
    pattern: ?toma_metal.Pattern = null,

    pub fn initBorrowed(device: *anyopaque, queue: *anyopaque) !Runtime {
        return .{ .context = try toma_metal.Context.initBorrowed(device, queue) };
    }

    pub fn deinit(self: *Runtime) void {
        if (self.pattern) |*pattern| pattern.deinit();
        self.context.deinit();
        self.* = undefined;
    }

    /// Rebuild a shape-compatible pattern in place; replace its allocation
    /// only when the semantic shape or algorithm changes.
    pub fn prepare(
        self: *Runtime,
        features: *anyopaque,
        source_tokens: usize,
        feature_width: usize,
        config: toma_config.Config,
    ) !*toma_metal.Pattern {
        const shape = try toma_config.shape(source_tokens, feature_width, config);
        if (self.pattern) |*pattern| {
            if (std.meta.eql(pattern.shape, shape) and
                std.meta.eql(pattern.config, config))
            {
                try self.context.rebuildBuffer(features, pattern);
                return pattern;
            }
            pattern.deinit();
            self.pattern = null;
        }
        self.pattern = try self.context.buildBuffer(
            features,
            source_tokens,
            feature_width,
            config,
        );
        return &self.pattern.?;
    }
};
