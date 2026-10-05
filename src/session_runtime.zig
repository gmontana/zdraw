//! Runtime adapter for the interactive session.
//!
//! The session workflow is model-agnostic: it needs a warm runtime that can
//! generate RGB pixels. Z-Image and Klein have intentionally different runtime
//! types, so this small tagged union keeps that difference at the boundary.

const std = @import("std");

const kind_mod = @import("model_kind.zig");
const model = @import("model.zig");
const seed_field = @import("seed_field.zig");
const zimage_runtime = @import("model_runtime.zig");

pub const Request = struct {
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    // Guided sampling reaches only the Klein base variants; Z-Image Turbo
    // and distilled Klein run at their fixed 1.0 (main.sampling enforces it).
    guidance: f32 = 1.0,
    /// img2img (Klein only; Z-Image ignores it).
    init_image: []const u8 = "",
    strength: f32 = 1.0,
    mask: []const u8 = "",
    /// Instruction editing (Klein only): the photo to change.
    ref_image: []const u8 = "",
    /// Wander's corner-seed blend (Klein only).
    field: seed_field.Field = .{},
};

pub const Result = struct {
    pixels: []u8,
    width: u32,
    height: u32,
};

pub const Runtime = union(enum) {
    z_image: zimage_runtime.Runtime,
    klein: model.KleinRuntime,

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        kind: kind_mod.ModelKind,
        weights_dir: []const u8,
    ) !Runtime {
        return switch (kind) {
            .z_image_turbo => .{
                .z_image = try zimage_runtime.Runtime.init(io, allocator, kind, weights_dir),
            },
            else => .{
                .klein = try model.openKlein(io, allocator, kind, weights_dir),
            },
        };
    }

    pub fn deinit(self: *Runtime, io: std.Io, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .z_image => |*rt| rt.deinit(io, allocator),
            .klein => |*rt| rt.deinit(io, allocator),
        }
        self.* = undefined;
    }

    /// One image per seed. Klein runs them as a single batched denoise when
    /// the resident path allows it (guidance 1.0), otherwise it falls back to
    /// a serial loop internally. Z-Image has no batch entry, so it loops here.
    /// Caller owns the slice and every `pixels` in it.
    pub fn generateMulti(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
        seeds: []const u64,
    ) ![]Result {
        const out = try allocator.alloc(Result, seeds.len);
        errdefer allocator.free(out);
        switch (self.*) {
            .klein => |*rt| {
                const batch = try rt.generateMulti(io, allocator, .{
                    .prompt = request.prompt,
                    .width = request.width,
                    .height = request.height,
                    .steps = request.steps,
                    .seed = request.seed,
                    .guidance = request.guidance,
                    .init_image = request.init_image,
                    .strength = request.strength,
                    .mask = request.mask,
                    .ref_image = request.ref_image,
                    .field = request.field,
                }, seeds);
                defer allocator.free(batch);
                for (batch, 0..) |one, i| {
                    out[i] = .{ .pixels = one.pixels, .width = one.width, .height = one.height };
                }
            },
            .z_image => |*rt| {
                var done: usize = 0;
                errdefer for (out[0..done]) |r| allocator.free(r.pixels);
                for (seeds, 0..) |seed, i| {
                    const one = try rt.generate(io, allocator, .{
                        .prompt = request.prompt,
                        .width = request.width,
                        .height = request.height,
                        .steps = request.steps,
                        .seed = seed,
                    });
                    out[i] = .{ .pixels = one.pixels, .width = one.width, .height = one.height };
                    done += 1;
                }
            },
        }
        return out;
    }

    pub fn generate(
        self: *Runtime,
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
    ) !Result {
        switch (self.*) {
            .z_image => |*rt| {
                const out = try rt.generate(io, allocator, .{
                    .prompt = request.prompt,
                    .width = request.width,
                    .height = request.height,
                    .steps = request.steps,
                    .seed = request.seed,
                });
                return .{ .pixels = out.pixels, .width = out.width, .height = out.height };
            },
            .klein => |*rt| {
                const out = try rt.generate(io, allocator, .{
                    .prompt = request.prompt,
                    .width = request.width,
                    .height = request.height,
                    .steps = request.steps,
                    .seed = request.seed,
                    .guidance = request.guidance,
                });
                return .{ .pixels = out.pixels, .width = out.width, .height = out.height };
            },
        }
    }
};
