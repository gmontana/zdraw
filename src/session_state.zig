//! Session state for the interactive zdraw shell.

const std = @import("std");

const args = @import("args.zig");
const runtime = @import("session_runtime.zig");

pub const Frame = struct {
    pixels: []u8,
    width: u32,
    height: u32,
};

pub const Stats = struct {
    verbose: bool = false,
};

pub const State = struct {
    request: args.Session,
    env: *const std.process.Environ.Map,
    rt: ?runtime.Runtime = null,
    /// Owns the directory selected by a model switch; startup borrows the CLI path.
    weights_owned: ?[]u8 = null,
    count: u32 = 0,
    prompt: []u8 = &.{},
    last_path: []u8 = &.{},
    last_frame: ?Frame = null,
    history: std.ArrayList([]u8),
    stats: Stats = .{},

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        env: *const std.process.Environ.Map,
        request: args.Session,
    ) !State {
        _ = io;
        return .{
            .request = request,
            .env = env,
            .history = try std.ArrayList([]u8).initCapacity(allocator, 8),
        };
    }

    pub fn deinit(self: *State, io: std.Io, allocator: std.mem.Allocator) void {
        if (self.rt) |*rt| rt.deinit(io, allocator);
        if (self.weights_owned) |path| allocator.free(path);
        allocator.free(self.prompt);
        allocator.free(self.last_path);
        self.clearLastFrame(allocator);
        for (self.history.items) |item| allocator.free(item);
        self.history.deinit(allocator);
    }

    pub fn clearLastFrame(self: *State, allocator: std.mem.Allocator) void {
        if (self.last_frame) |frame| allocator.free(frame.pixels);
        self.last_frame = null;
    }
};
