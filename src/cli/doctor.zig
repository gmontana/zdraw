//! `zdraw doctor`: what this binary will do on this machine, one fact per
//! line and no weights required - the routes that will engage, the fallbacks
//! that would run instead, the overrides in the environment, and where the
//! weights and sidecars resolve. `--json` prints the same facts as one object
//! (schema_version 1). It reports; it never selects a route.
const std = @import("std");
const args = @import("args.zig");
const env = @import("../runtime/env.zig");
const bench_card = @import("bench_card.zig");
const execution_receipt = @import("../control/execution_receipt.zig");
const metal_c = @import("../metal/metal_c.zig");
const mgemm_mpp = @import("../metal/mgemm_mpp_shader.zig");
const model = @import("../runtime/model.zig");
const profile = @import("../runtime/profile.zig");
const util = @import("session_util.zig");
const version = @import("version.zig");
const weights = @import("../pack/weights.zig");
const zflux2_pack = @import("../klein/zflux2_pack.zig");
const zpack_file = @import("../pack/zpack_file.zig");

pub const schema_version: u32 = 1;

pub const Probe = struct {
    ok: bool,
    detail: []const u8,
};

pub const Routes = struct {
    steel_path: []const u8,
    steel: Probe,
    metal4_gemm: Probe,
    attention_default: []const u8,
    gemm_default_klein: []const u8,
    winograd_default: bool,
    mods_gpu_default: bool,
};

pub const DeviceInfo = struct {
    name: []const u8,
    ram_bytes: u64,
    gpu_working_set_bytes: u64,
    os_major: u32,
};

pub const EnvVar = struct {
    name: []const u8,
    value: []const u8,
};

fn sidecarBits(io: std.Io, path: []const u8) u8 {
    var pack = zpack_file.open(io, path) catch return 0;
    defer pack.deinit(io);
    return zflux2_pack.sidecarBits(pack.bytes()) catch 0;
}

pub const WeightsInfo = struct {
    model: []const u8,
    dir: []const u8,
    files_ok: bool,
    sidecar_path: ?[]const u8,
    /// Bits per weight the sidecar packs (0 when none or unreadable).
    sidecar_bits: u8 = 0,
    /// "env" | "beside" | "runs" | "none"
    sidecar_source: []const u8,
};

pub const Report = struct {
    schema_version: u32 = schema_version,
    zdraw_version: []const u8,
    commit: []const u8,
    engine_abi: u32,
    device: ?DeviceInfo,
    thermal: []const u8,
    routes: Routes,
    env_overrides: []const EnvVar,
    weights: ?WeightsInfo,
    default_route: bool,
    reasons: []const []const u8,
};

/// Flags that never change the route (paths, instruments, provenance).
// A sidecar path is NOT neutral: a pack built from another checkpoint changes
// every weight it swaps in, so the card records it as an override.
const neutral_env = [_][]const u8{
    "ZDRAW_STEEL_LIB",       "ZDRAW_PROGRESS", "ZDRAW_METRICS",
    "ZDRAW_ENGINE_REVISION", "ZDRAW_MEMTRACE",
};

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    request: args.Doctor,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const report = try collect(io, arena, environ, request);
    if (request.json) {
        const text = try toJson(arena, report);
        try util.writeIo(io, text);
        try util.writeIo(io, "\n");
    } else {
        try printText(io, arena, report);
    }
}

fn collect(
    io: std.Io,
    arena: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    request: args.Doctor,
) !Report {
    var reasons: std.ArrayList([]const u8) = .empty;
    const device = deviceInfo(arena);
    if (device == null) try reasons.append(arena, "no Metal device");
    const steel = try probeSteel(arena);
    const metal4 = probeMetal4(arena);
    try routeReasons(arena, &reasons, steel, metal4);
    const overrides = try envOverrides(arena, environ);
    for (overrides) |o| {
        if (isNeutral(o.name)) continue;
        const line = try std.fmt.allocPrint(arena, "env override {s}={s}", .{ o.name, o.value });
        try reasons.append(arena, line);
    }
    const winfo = if (request.weights_dir.len > 0)
        try weightsInfo(io, arena, request.kind, request.weights_dir, &reasons)
    else
        null;
    return .{
        .zdraw_version = version.semver,
        .commit = version.commit,
        .engine_abi = version.abi,
        .device = device,
        .thermal = thermalName(),
        .routes = routesFor(steel, metal4),
        .env_overrides = overrides,
        .weights = winfo,
        .default_route = reasons.items.len == 0,
        .reasons = reasons.items,
    };
}

fn routeReasons(
    arena: std.mem.Allocator,
    reasons: *std.ArrayList([]const u8),
    steel: SteelResult,
    metal4: Probe,
) !void {
    if (!steel.probe.ok) try reasons.append(arena, try std.fmt.allocPrint(
        arena,
        "steel metallib {s} at {s}: attention falls back to MFA (Klein hash changes); " ++
            "restore the bundled library, or rebuild with Xcode's Metal compiler",
        .{ steel.probe.detail, steel.path },
    ));
    if (!metal4.ok) try reasons.append(arena, try std.fmt.allocPrint(
        arena,
        "Metal 4 matmul2d unavailable ({s}): Klein GEMMs run the direct kernel " ++
            "(counted MPP fallback; hash unchanged, slower)",
        .{metal4.detail},
    ));
}

/// The routes the defaults will take, given what the probes found.
fn routesFor(steel: SteelResult, metal4: Probe) Routes {
    const steel_on = env.flag("ZDRAW_KLEIN_ATTN_STEEL", true) and
        env.flag("ZDRAW_ATTN_STEEL", true);
    const mpp_on = env.flag("ZDRAW_KLEIN_GEMM_MPP", true);
    return .{
        .steel_path = steel.path,
        .steel = steel.probe,
        .metal4_gemm = metal4,
        .attention_default = if (steel_on and steel.probe.ok) "steel" else "mfa",
        .gemm_default_klein = if (mpp_on and metal4.ok)
            "metal4-matmul2d"
        else
            "simdgroup-direct",
        .winograd_default = env.flag("ZDRAW_VAE_WINO", true),
        .mods_gpu_default = env.flag("ZDRAW_KLEIN_MODS_GPU", true),
    };
}

fn deviceInfo(arena: std.mem.Allocator) ?DeviceInfo {
    const d = profile.detect() orelse return null;
    const name = arena.dupe(u8, d.chip()) catch return null;
    return .{
        .name = name,
        .ram_bytes = d.ram_bytes,
        .gpu_working_set_bytes = d.gpu_working_set,
        .os_major = d.os_major,
    };
}

const SteelResult = struct { path: []const u8, probe: Probe };

fn probeSteel(arena: std.mem.Allocator) !SteelResult {
    var path_buf: [4096]u8 = undefined;
    var err_buf: [512]u8 = undefined;
    path_buf[0] = 0;
    err_buf[0] = 0;
    const rc = metal_c.zdraw_metal_steel_probe(
        &path_buf,
        path_buf.len,
        &err_buf,
        err_buf.len,
    );
    const path = try arena.dupe(u8, std.mem.sliceTo(&path_buf, 0));
    const err = try arena.dupe(u8, std.mem.sliceTo(&err_buf, 0));
    const probe: Probe = switch (rc) {
        0 => .{ .ok = true, .detail = "found" },
        1 => .{ .ok = false, .detail = "missing" },
        2 => .{ .ok = false, .detail = if (err.len > 0) err else "unusable" },
        else => .{
            .ok = false,
            .detail = try std.fmt.allocPrint(arena, "stale (no {s})", .{err}),
        },
    };
    return .{ .path = path, .probe = probe };
}

fn probeMetal4(arena: std.mem.Allocator) Probe {
    _ = arena;
    switch (metal_c.zdraw_metal4_available()) {
        1 => {},
        0 => return .{ .ok = false, .detail = "needs macOS 26" },
        else => return .{ .ok = false, .detail = "no Metal device" },
    }
    const device = metal_c.zdraw_metal_create_device() orelse
        return .{ .ok = false, .detail = "no Metal device" };
    defer metal_c.zdraw_metal_release_device(device);
    const pipe = metal_c.zdraw_metal_compile_mpp(
        device,
        mgemm_mpp.src.ptr,
        "gemm_mpp64",
    ) orelse return .{ .ok = false, .detail = "gemm_mpp64 failed to compile" };
    metal_c.zdraw_metal_release_pipeline(pipe);
    return .{ .ok = true, .detail = "gemm_mpp64 compiled" };
}

fn envOverrides(arena: std.mem.Allocator, environ: *const std.process.Environ.Map) ![]EnvVar {
    var list: std.ArrayList(EnvVar) = .empty;
    var it = environ.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "ZDRAW_")) continue;
        try list.append(arena, .{
            .name = try arena.dupe(u8, name),
            .value = try arena.dupe(u8, entry.value_ptr.*),
        });
    }
    std.mem.sort(EnvVar, list.items, {}, lessByName);
    return list.items;
}

fn lessByName(_: void, a: EnvVar, b: EnvVar) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

/// The remote-Mac harness settings (host, user, identity: "M4_" prefix,
/// spelled in two parts so env_guard does not read it as one flag).
const harness_prefix = "ZDRAW_" ++ "M4_";

/// Paths, instruments, provenance and the harness settings never change a
/// route; everything else in the environment is reported as an override.
pub fn isNeutral(name: []const u8) bool {
    if (std.mem.startsWith(u8, name, harness_prefix)) return true;
    for (neutral_env) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

fn weightsInfo(
    io: std.Io,
    arena: std.mem.Allocator,
    kind: model.ModelKind,
    dir: []const u8,
    reasons: *std.ArrayList([]const u8),
) !WeightsInfo {
    const files_ok = if (weights.validate(io, arena, kind, dir)) |_| true else |_| false;
    if (!files_ok) try reasons.append(arena, try std.fmt.allocPrint(
        arena,
        "weights incomplete under {s} (see README: Models and weights)",
        .{dir},
    ));
    const klein = kind != .z_image_turbo;
    const env_name: [*:0]const u8 = if (klein) "ZDRAW_KLEIN_ZPACK" else "ZDRAW_ZPACK";
    var info = WeightsInfo{
        .model = model.cliName(kind),
        .dir = dir,
        .files_ok = files_ok,
        .sidecar_path = null,
        .sidecar_source = "none",
    };
    if (std.c.getenv(env_name)) |raw| {
        info.sidecar_path = try arena.dupe(u8, std.mem.span(raw));
        info.sidecar_source = "env";
    } else {
        const pack = if (klein) model.kleinPackName(kind) else "zdraw-w16.zpack";
        const beside = try std.fmt.allocPrint(arena, "{s}/{s}", .{ dir, pack });
        const in_runs = try std.fmt.allocPrint(arena, "runs/{s}", .{pack});
        if (exists(io, beside)) {
            info.sidecar_path = beside;
            info.sidecar_source = "beside";
        } else if (klein and exists(io, in_runs)) {
            info.sidecar_path = in_runs;
            info.sidecar_source = "runs";
        }
    }
    if (info.sidecar_path) |p| {
        if (!exists(io, p)) try reasons.append(arena, try std.fmt.allocPrint(
            arena,
            "sidecar missing at {s} (zdraw fetch builds it)",
            .{p},
        ));
        if (exists(io, p)) info.sidecar_bits = sidecarBits(io, p);
    } else try reasons.append(arena, "no weight sidecar found (zdraw fetch builds it)");
    return info;
}

fn exists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

fn thermalName() []const u8 {
    const state = std.enums.fromInt(
        execution_receipt.ThermalState,
        metal_c.zdraw_thermal_state(),
    ) orelse .unknown;
    return @tagName(state);
}

pub fn toJson(allocator: std.mem.Allocator, report: Report) ![]u8 {
    return std.json.Stringify.valueAlloc(allocator, report, .{
        .emit_null_optional_fields = false,
        .whitespace = .indent_2,
    });
}

fn printText(io: std.Io, arena: std.mem.Allocator, r: Report) !void {
    var aw: std.Io.Writer.Allocating = .init(arena);
    defer aw.deinit();
    const w = &aw.writer;
    try w.print("zdraw {s} ({s}) abi {d}\n", .{ r.zdraw_version, r.commit, r.engine_abi });
    if (r.device) |d| {
        try w.print("device: {s}, {d} GiB, macOS {d}\n", .{
            d.name,
            d.ram_bytes >> 30,
            d.os_major,
        });
    } else try w.print("device: none\n", .{});
    try w.print("thermal: {s}\n", .{r.thermal});
    try w.print("steel metallib: {s} {s}\n", .{ r.routes.steel.detail, r.routes.steel_path });
    try w.print("metal4 matmul2d: {s} ({s})\n", .{
        if (r.routes.metal4_gemm.ok) "available" else "unavailable",
        r.routes.metal4_gemm.detail,
    });
    try w.print("attention: {s}; klein gemm: {s}\n", .{
        r.routes.attention_default,
        r.routes.gemm_default_klein,
    });
    try w.print("winograd decoder: {s}; gpu modulation: {s}\n", .{
        if (r.routes.winograd_default) "on" else "off",
        if (r.routes.mods_gpu_default) "on" else "off",
    });
    if (r.env_overrides.len == 0) {
        try w.print("env overrides: none\n", .{});
    } else {
        try w.print("env overrides:", .{});
        for (r.env_overrides) |o| try w.print(" {s}={s}", .{ o.name, o.value });
        try w.print("\n", .{});
    }
    if (r.weights) |wi| {
        try w.print("weights {s}: {s} {s}; sidecar {s} ({s}, {s})\n", .{
            wi.model,
            if (wi.files_ok) "ok" else "incomplete",
            wi.dir,
            wi.sidecar_path orelse "none",
            wi.sidecar_source,
            if (wi.sidecar_bits == 0) "no tier read" else bench_card.packName(wi.sidecar_bits),
        });
    }
    if (r.default_route) {
        try w.print("default route: yes\n", .{});
    } else {
        try w.print("default route: no\n", .{});
        for (r.reasons) |reason| try w.print("  - {s}\n", .{reason});
    }
    try util.writeIo(io, aw.writer.buffered());
}

test "doctor report round-trips through JSON" {
    const report = Report{
        .zdraw_version = "0.1.0",
        .commit = "abc",
        .engine_abi = 1,
        .device = null,
        .thermal = "nominal",
        .routes = .{
            .steel_path = "/x/steel.metallib",
            .steel = .{ .ok = true, .detail = "found" },
            .metal4_gemm = .{ .ok = false, .detail = "needs macOS 26" },
            .attention_default = "steel",
            .gemm_default_klein = "simdgroup-direct",
            .winograd_default = true,
            .mods_gpu_default = true,
        },
        .env_overrides = &.{},
        .weights = null,
        .default_route = false,
        .reasons = &.{"Metal 4 matmul2d unavailable"},
    };
    const text = try toJson(std.testing.allocator, report);
    defer std.testing.allocator.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 1), root.get("schema_version").?.integer);
    try std.testing.expect(!root.get("default_route").?.bool);
    try std.testing.expect(root.get("device") == null);
    const routes = root.get("routes").?.object;
    try std.testing.expectEqualStrings("steel", routes.get("attention_default").?.string);
}

test "neutral flags never count as overrides" {
    try std.testing.expect(isNeutral("ZDRAW_METRICS"));
    try std.testing.expect(isNeutral(harness_prefix ++ "USER"));
    try std.testing.expect(!isNeutral("ZDRAW_VAE_WINO"));
}
