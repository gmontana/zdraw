//! `zdraw bench --card`: the census case (the certified prompt at 1024, 4
//! steps, seed 46) rendered once cold in this process, reported as a human
//! block and one JSON line (schema_version 1) that the community table
//! accepts. The card only reports: hash, phases, routes, fallbacks, memory,
//! thermal state, and whether the output matches `certified/hashes.json`.
const std = @import("std");
const doctor = @import("doctor.zig");
const env = @import("../runtime/env.zig");
const execution_receipt = @import("../control/execution_receipt.zig");
const metrics = @import("../metal/metrics.zig");
const model = @import("../runtime/model.zig");
const profile = @import("../runtime/profile.zig");
const util = @import("session_util.zig");
const version = @import("version.zig");

pub const prompt = "a red fox sitting in deep snow, golden hour light";
pub const size: u32 = 1024;
pub const steps: u32 = 4;
pub const seed: u64 = 46;

/// The census step count per model: the distilled models' fixed 4, the
/// base model's 50-step guided regime (its certified entry is keyed on it).
pub fn stepsFor(kind: model.ModelKind) u32 {
    return if (model.policy(kind).cfg_allowed) model.policy(kind).default_steps else steps;
}
pub const schema_version: u32 = 1;

const certified_json = @embedFile("certified_hashes");

pub const Entry = struct {
    model: []const u8,
    profile: []const u8,
    /// The weight tier the hash was certified with: "w16" (default), "w6", "w4", "mixed".
    pack: []const u8 = "w16",
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    sha256: []const u8,
    zdraw_version: []const u8,
    commit: []const u8,
    date: []const u8,
    ledger: []const u8,
    routes: []const u8,
};

pub const Certified = struct {
    schema_version: u32,
    prompt: []const u8,
    prompt_sha256: []const u8,
    entries: []const Entry,
};

pub const Phases = struct {
    encode_s: f64,
    denoise_s: f64,
    decode_s: f64,
};

pub const Routes = struct {
    attention: []const u8,
    gemm: []const u8,
    decoder: []const u8,
};

pub const Fallbacks = struct {
    steel: u64,
    mpp: u64,
    mps: u64,
};

pub const Card = struct {
    schema_version: u32 = schema_version,
    chip: []const u8,
    ram_gb: f64,
    macos: u32,
    zdraw_version: []const u8,
    commit: []const u8,
    model: []const u8,
    profile: []const u8,
    /// The weight tier rendered with ("w16", "w6", "w4").
    pack: []const u8,
    size: u32,
    steps: u32,
    seed: u64,
    prompt_sha256: []const u8,
    wall_s: f64,
    runs_s: []const f64,
    phases: Phases,
    gpu_active_s: f64,
    rss_gb: f64,
    footprint_gb: f64,
    /// Peak resident pages of the mapped weight files over the run's phase
    /// boundaries (mincore); RSS and footprint exclude them.
    mapped_resident_gb: f64,
    /// Peak of footprint + mapped_resident at a phase boundary: what the
    /// host had to hold.
    total_gb: f64,
    /// SHA-256 of the raw RGB bytes (the certified hash since 2026-08-28).
    output_sha256: []const u8,
    /// SHA-256 of the PNG file (pixels + the recipe chunk), for the record.
    png_sha256: []const u8,
    expected_sha256: ?[]const u8,
    hash_match: ?bool,
    routes: Routes,
    fallbacks: Fallbacks,
    thermal_before: []const u8,
    thermal_after: []const u8,
    env_overrides: []const []const u8,
    /// "on" | "off": whether the safety filter was armed for this card.
    safety: []const u8,
};

pub const Inputs = struct {
    kind: model.ModelKind,
    profile: []const u8,
    /// Bits per weight of the loaded sidecar (16 without one).
    pack_bits: u8,
    /// Bits per weight of the text encoder's pack (16 from the bf16 shards).
    text_bits: u8 = 16,
    safety: bool,
    wall_ns: u64,
    output_sha256: []const u8,
    png_sha256: []const u8,
    thermal_before: execution_receipt.ThermalState,
    thermal_after: execution_receipt.ThermalState,
};

pub fn parseCertified(allocator: std.mem.Allocator) !std.json.Parsed(Certified) {
    const opts = std.json.ParseOptions{ .ignore_unknown_fields = true };
    return std.json.parseFromSlice(Certified, allocator, certified_json, opts);
}

/// The certified entry for a model/profile/pack at the census case, if any.
pub fn lookup(cert: Certified, model_name: []const u8, prof: []const u8, pack: []const u8) ?Entry {
    for (cert.entries) |e| {
        if (std.mem.eql(u8, e.model, model_name) and std.mem.eql(u8, e.profile, prof) and
            std.mem.eql(u8, e.pack, pack)) return e;
    }
    return null;
}

/// The tier name for a sidecar bit width ("w16", "w6", "w4", ...); 9 is
/// zflux2_pack.mixed_bits, the mixed-allocation pack.
pub fn packName(bits: u8) []const u8 {
    return switch (bits) {
        16 => "w16",
        8 => "w8",
        6 => "w6",
        4 => "w4",
        3 => "w3",
        2 => "w2",
        9 => "mixed",
        else => "w?",
    };
}

/// The tier label: the transformer's pack name, with "/t4" when the text
/// encoder renders from the 4-bit text pack (a different picture, so its
/// own certificate entry).
pub fn packLabel(pack_bits: u8, text_bits: u8) []const u8 {
    if (text_bits == 16) return packName(pack_bits);
    return switch (pack_bits) {
        16 => "w16/t4",
        6 => "w6/t4",
        4 => "w4/t4",
        2 => "w2/t4",
        9 => "mixed/t4",
        else => "w?/t4",
    };
}

pub fn promptSha(allocator: std.mem.Allocator) ![]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(prompt, &digest, .{});
    return std.fmt.allocPrint(allocator, "{x}", .{digest});
}

pub fn build(
    arena: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    in: Inputs,
) !Card {
    const cert = try parseCertified(arena);
    const model_name = model.cliName(in.kind);
    const pack = packLabel(in.pack_bits, in.text_bits);
    const entry = lookup(cert.value, model_name, in.profile, pack);
    const dev = profile.detect();
    const counters = metrics.snapshot();
    const mem = metrics.memory();
    const fallbacks = Fallbacks{
        .steel = counters.steel_fallback,
        .mpp = counters.mpp_fallback,
        .mps = counters.mps_fallback,
    };
    const expected: ?[]const u8 = if (entry) |e| e.sha256 else null;
    return .{
        .chip = if (dev) |d| try arena.dupe(u8, d.chip()) else "unknown",
        .ram_gb = if (dev) |d| gb(d.ram_bytes) else 0,
        .macos = if (dev) |d| d.os_major else 0,
        .zdraw_version = version.semver,
        .commit = version.commit,
        .model = model_name,
        .profile = in.profile,
        .pack = pack,
        .size = size,
        .steps = stepsFor(in.kind),
        .seed = seed,
        .prompt_sha256 = try promptSha(arena),
        .wall_s = coldWall(in.wall_ns),
        .runs_s = try runSeconds(arena),
        .phases = phases(in.kind),
        .gpu_active_s = secs(counters.gpu_ns),
        .rss_gb = gb(mem.peak_rss_bytes),
        .footprint_gb = gb(mem.phys_footprint_bytes),
        .mapped_resident_gb = gb(mem.mapped_resident_peak_bytes),
        .total_gb = gb(mem.total_peak_bytes),
        .output_sha256 = in.output_sha256,
        .png_sha256 = in.png_sha256,
        .expected_sha256 = expected,
        .hash_match = if (expected) |e| std.mem.eql(u8, e, in.output_sha256) else null,
        .routes = routesFor(in.kind, in.profile, fallbacks),
        .fallbacks = fallbacks,
        .thermal_before = @tagName(in.thermal_before),
        .thermal_after = @tagName(in.thermal_after),
        .env_overrides = try overrides(arena, environ),
        .safety = if (in.safety) "on" else "off",
    };
}

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

fn firstSample(name: []const u8) f64 {
    const samples = metrics.samplesOf(name);
    return if (samples.len > 0) secs(samples[0]) else 0;
}

/// Run-1 laps: Klein records its own three; Z-Image's are the progress stages.
fn phases(kind: model.ModelKind) Phases {
    if (kind != .z_image_turbo) return .{
        .encode_s = firstSample("klein-encode"),
        .denoise_s = firstSample("klein-denoise"),
        .decode_s = firstSample("klein-vae"),
    };
    return .{
        .encode_s = firstSample("encoding prompt"),
        .denoise_s = firstSample("sampling latents"),
        .decode_s = firstSample("decoding image"),
    };
}

/// The cold one-shot equivalent: the process wall minus the warm runs that
/// --repeat added, so cards with different --repeat settings compare.
fn coldWall(wall_ns: u64) f64 {
    const samples = metrics.samplesOf("generate");
    var warm: f64 = 0;
    if (samples.len > 1) for (samples[1..]) |s| {
        warm += secs(s);
    };
    return secs(wall_ns) - warm;
}

fn runSeconds(arena: std.mem.Allocator) ![]const f64 {
    const samples = metrics.samplesOf("generate");
    const out = try arena.alloc(f64, samples.len);
    for (out, samples) |*o, s| o.* = secs(s);
    return out;
}

fn routesFor(kind: model.ModelKind, prof: []const u8, fb: Fallbacks) Routes {
    const strict = std.mem.eql(u8, prof, "strict");
    const klein = kind != .z_image_turbo;
    const steel_flag = if (klein)
        env.flag("ZDRAW_KLEIN_ATTN_STEEL", true)
    else
        env.flag("ZDRAW_ATTN_STEEL", true);
    const attention: []const u8 = if (steel_flag and fb.steel == 0) "steel" else "mfa";
    const gemm: []const u8 = if (!klein)
        (if (strict) "exact" else "ours-f16")
    else if (env.flag("ZDRAW_KLEIN_GEMM_MPP", true) and fb.mpp == 0)
        "metal4-matmul2d"
    else
        "simdgroup-direct";
    const decoder: []const u8 = if (strict)
        "reference-f32"
    else if (env.flag("ZDRAW_VAE_WINO", true))
        "winograd"
    else
        "direct";
    return .{ .attention = attention, .gemm = gemm, .decoder = decoder };
}

fn overrides(
    arena: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var it = environ.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "ZDRAW_") or doctor.isNeutral(name)) continue;
        const line = try std.fmt.allocPrint(arena, "{s}={s}", .{ name, entry.value_ptr.* });
        try list.append(arena, line);
    }
    std.mem.sort([]const u8, list.items, {}, lessThan);
    return list.items;
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

pub fn toJson(allocator: std.mem.Allocator, card: Card) ![]u8 {
    return std.json.Stringify.valueAlloc(allocator, card, .{ .emit_null_optional_fields = true });
}

/// The human block, then the JSON line as the last line of stdout.
pub fn emit(io: std.Io, arena: std.mem.Allocator, card: Card) !void {
    var aw: std.Io.Writer.Allocating = .init(arena);
    defer aw.deinit();
    const w = &aw.writer;
    try w.print("zdraw bench card\n", .{});
    try w.print("  machine   {s}, {d:.0} GB, macOS {d}\n", .{ card.chip, card.ram_gb, card.macos });
    try w.print("  zdraw     {s} ({s})\n", .{ card.zdraw_version, card.commit });
    try w.print("  workload  {s} {s} {s} {d}x{d}, {d} steps, seed {d}\n", .{
        card.model, card.profile, card.pack, card.size,
        card.size,  card.steps,   card.seed,
    });
    const wall_fmt = "  wall      {d:.1} s cold one-shot (encode {d:.2}, denoise {d:.2}, decode {d:.2})\n";
    try w.print(wall_fmt, .{
        card.wall_s,
        card.phases.encode_s,
        card.phases.denoise_s,
        card.phases.decode_s,
    });
    try w.print("  gpu       {d:.1} s active\n", .{card.gpu_active_s});
    if (card.runs_s.len > 1) {
        try w.print("  runs     ", .{});
        for (card.runs_s) |r| try w.print(" {d:.1}", .{r});
        try w.print(" s\n", .{});
    }
    try w.print("  memory    RSS {d:.2} GB, footprint {d:.2} GB\n", .{
        card.rss_gb,
        card.footprint_gb,
    });
    try w.print("  mapped    {d:.2} GB resident weight pages at peak, total {d:.2} GB\n", .{
        card.mapped_resident_gb,
        card.total_gb,
    });
    try w.print("  routes    attention {s}, gemm {s}, decoder {s}\n", .{
        card.routes.attention,
        card.routes.gemm,
        card.routes.decoder,
    });
    try w.print("  fallbacks steel {d}, mpp {d}, mps {d}\n", .{
        card.fallbacks.steel,
        card.fallbacks.mpp,
        card.fallbacks.mps,
    });
    try w.print("  thermal   {s} -> {s}\n", .{ card.thermal_before, card.thermal_after });
    try printHash(w, card);
    try w.print("{s}\n", .{try toJson(arena, card)});
    try util.writeIo(io, aw.writer.buffered());
}

fn printHash(w: *std.Io.Writer, card: Card) !void {
    const short = card.output_sha256[0..@min(12, card.output_sha256.len)];
    if (card.hash_match) |match| {
        if (match) {
            try w.print("  hash      {s} MATCH (certified)\n", .{short});
        } else {
            const want = card.expected_sha256.?[0..12];
            try w.print("  hash      {s} MISMATCH (certified {s})\n", .{ short, want });
            try w.print("  why       ", .{});
            if (card.fallbacks.steel > 0) {
                try w.print("steel attention fell back to MFA (changes the hash); ", .{});
            }
            if (card.env_overrides.len > 0) try w.print("env overrides set; ", .{});
            try w.print("a different zdraw version or profile also changes it\n", .{});
        }
    } else {
        try w.print("  hash      {s} (no certified entry for this model/profile/pack)\n", .{short});
    }
}

test "the embedded certified file is well formed" {
    var parsed = try parseCertified(std.testing.allocator);
    defer parsed.deinit();
    const cert = parsed.value;
    try std.testing.expectEqual(@as(u32, 1), cert.schema_version);
    try std.testing.expectEqualStrings(prompt, cert.prompt);
    const sha = try promptSha(std.testing.allocator);
    defer std.testing.allocator.free(sha);
    try std.testing.expectEqualStrings(sha, cert.prompt_sha256);
    for (cert.entries, 0..) |e, i| {
        try std.testing.expectEqual(@as(usize, 64), e.sha256.len);
        for (cert.entries[i + 1 ..]) |other| {
            const same = std.mem.eql(u8, e.model, other.model) and
                std.mem.eql(u8, e.profile, other.profile) and
                std.mem.eql(u8, e.pack, other.pack);
            try std.testing.expect(!same);
        }
    }
    try std.testing.expect(lookup(cert, "flux2-klein-4b", "-", "w16") != null);
    try std.testing.expect(lookup(cert, "flux2-klein-4b", "-", "w4") != null);
    try std.testing.expect(lookup(cert, "flux2-klein-4b", "-", "w2") == null);
    try std.testing.expectEqualStrings("w4", packName(4));
    try std.testing.expectEqualStrings("w4", packLabel(4, 16));
    try std.testing.expectEqualStrings("mixed/t4", packLabel(9, 4));
}

test "card JSON keeps a null hash_match and the run list" {
    const card = Card{
        .chip = "Apple M4 Max",
        .ram_gb = 128,
        .macos = 26,
        .zdraw_version = "0.1.0",
        .commit = "abc",
        .model = "flux2-klein-4b",
        .profile = "-",
        .pack = "w16",
        .size = size,
        .steps = steps,
        .seed = seed,
        .prompt_sha256 = "00",
        .wall_s = 12.3,
        .runs_s = &.{ 12.3, 11.9 },
        .phases = .{ .encode_s = 0.4, .denoise_s = 10.8, .decode_s = 0.8 },
        .gpu_active_s = 11.2,
        .rss_gb = 4.2,
        .footprint_gb = 4.9,
        .mapped_resident_gb = 2.0,
        .total_gb = 6.9,
        .output_sha256 = "ff",
        .png_sha256 = "ee",
        .expected_sha256 = null,
        .hash_match = null,
        .routes = .{ .attention = "steel", .gemm = "metal4-matmul2d", .decoder = "winograd" },
        .fallbacks = .{ .steel = 0, .mpp = 0, .mps = 0 },
        .thermal_before = "nominal",
        .thermal_after = "nominal",
        .env_overrides = &.{},
        .safety = "on",
    };
    const text = try toJson(std.testing.allocator, card);
    defer std.testing.allocator.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expect(root.get("hash_match").? == .null);
    try std.testing.expectEqual(@as(usize, 2), root.get("runs_s").?.array.items.len);
}
