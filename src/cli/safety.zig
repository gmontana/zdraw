//! The prompt stage of the safety filter: the categories the model card names
//! (sexual content involving minors, non-consensual sexual imagery of real
//! people, gore/torture), as whole-word phrase groups in
//! `safety/prompt_rules.json`. A category fires when every group has a phrase
//! present. Regex-free and partial by design; the image stage is the second
//! line. Fixtures in `safety/tests.json` run under `zig build test`.
const std = @import("std");
const mlinear = @import("../metal/mlinear.zig");
const nsfw_vit = @import("nsfw_vit.zig");
const progress = @import("progress.zig");

const rules_json = @embedFile("safety_rules");
const tests_json = @embedFile("safety_tests");

pub const Category = struct {
    name: []const u8,
    message: []const u8,
    groups: []const []const []const u8,
};

pub const Rules = struct {
    schema_version: u32,
    note: []const u8 = "",
    categories: []const Category,
};

pub const Verdict = struct {
    blocked: bool,
    category: []const u8 = "",
    message: []const u8 = "",
};

pub fn parseRules(allocator: std.mem.Allocator) !std.json.Parsed(Rules) {
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    return std.json.parseFromSlice(Rules, allocator, rules_json, opts);
}

/// The verdict for one prompt (allocation-free apart from the rule parse).
pub fn checkPrompt(allocator: std.mem.Allocator, prompt: []const u8) !Verdict {
    var parsed = try parseRules(allocator);
    defer parsed.deinit();
    var buf: [4096]u8 = undefined;
    const folded = fold(prompt, &buf);
    for (parsed.value.categories) |cat| {
        var all = true;
        for (cat.groups) |group| {
            var any = false;
            for (group) |phrase| {
                var pbuf: [128]u8 = undefined;
                if (containsPhrase(folded, fold(phrase, &pbuf))) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                all = false;
                break;
            }
        }
        if (all) return .{
            .blocked = true,
            // The rules are embedded for the program's lifetime.
            .category = cat.name.ptr[0..cat.name.len],
            .message = cat.message.ptr[0..cat.message.len],
        };
    }
    return .{ .blocked = false };
}

/// Lowercase ASCII with every non-alphanumeric run collapsed to one space
/// and a space at both ends, so whole-word matching is a substring test.
fn fold(text: []const u8, buf: []u8) []const u8 {
    var n: usize = 0;
    buf[n] = ' ';
    n += 1;
    var last_space = true;
    for (text) |c| {
        if (n + 2 >= buf.len) break;
        const lower = std.ascii.toLower(c);
        if (std.ascii.isAlphanumeric(lower)) {
            buf[n] = lower;
            n += 1;
            last_space = false;
        } else if (!last_space) {
            buf[n] = ' ';
            n += 1;
            last_space = true;
        }
    }
    if (!last_space) {
        buf[n] = ' ';
        n += 1;
    }
    return buf[0..n];
}

fn containsPhrase(folded_text: []const u8, folded_phrase: []const u8) bool {
    return std.mem.indexOf(u8, folded_text, folded_phrase) != null;
}

/// Print the refusal and return an error so callers can release resources.
pub fn gate(io: std.Io, allocator: std.mem.Allocator, prompt: []const u8) !void {
    const verdict = try checkPrompt(allocator, prompt);
    if (!verdict.blocked) return;
    const text = try std.fmt.allocPrint(
        allocator,
        "zdraw: prompt blocked by the safety filter ({s}: {s}). " ++
            "--safety off disables it in the CLI.\n",
        .{ verdict.category, verdict.message },
    );
    defer allocator.free(text);
    try std.Io.File.stderr().writeStreamingAll(io, text);
    return error.SafetyBlocked;
}

const Case = struct { prompt: []const u8, expect: ?[]const u8 };
const Fixtures = struct { schema_version: u32, cases: []const Case };

test "the rules parse and every category has a group" {
    var parsed = try parseRules(std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u32, 1), parsed.value.schema_version);
    for (parsed.value.categories) |cat| try std.testing.expect(cat.groups.len > 0);
}

test "fixtures: the QC prompts pass, the blocked prompts name their category" {
    var parsed = try std.json.parseFromSlice(Fixtures, std.testing.allocator, tests_json, .{});
    defer parsed.deinit();
    for (parsed.value.cases) |case| {
        const v = try checkPrompt(std.testing.allocator, case.prompt);
        if (case.expect) |want| {
            try std.testing.expect(v.blocked);
            try std.testing.expectEqualStrings(want, v.category);
        } else {
            try std.testing.expect(!v.blocked);
        }
    }
}

test "whole words only" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings(" a teen s toy ", fold("A teen's toy!", &b));
    try std.testing.expect(!containsPhrase(" a teenager reads ", " teen "));
}

/// Block when p(nsfw) reaches this.
pub const image_threshold: f32 = 0.8;
/// The last image-stage score of this process (the receipt reads it).
pub var last_image_score: ?f64 = null;

/// Where the classifier lives: ZDRAW_SAFETY_MODEL, else the fetch default.
fn classifierDir(allocator: std.mem.Allocator, environ: *const std.process.Environ.Map) !?[]u8 {
    if (environ.get("ZDRAW_SAFETY_MODEL")) |p| return try allocator.dupe(u8, p);
    const home = environ.get("HOME") orelse return null;
    const fmt = "{s}/.zdraw/models/nsfw_image_detection";
    return try std.fmt.allocPrint(allocator, fmt, .{home});
}

/// The image stage: score the decoded pixels and return SafetyBlocked when
/// blocked. Without the classifier (not fetched) it says so once and passes.
pub fn imageGate(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    pixels: []const u8,
    width: usize,
    height: usize,
) !void {
    const dir = (try classifierDir(allocator, environ)) orelse return;
    defer allocator.free(dir);
    var model = nsfw_vit.open(io, allocator, dir) catch {
        const hint = "safety: image stage skipped (run `zdraw fetch nsfw-classifier`)";
        try progress.event(io, allocator, hint);
        return;
    };
    defer model.deinit(io, allocator);
    var metal: ?mlinear.Context = mlinear.Context.init() catch null;
    defer if (metal) |*m| m.deinit();
    const ctx: ?*mlinear.Context = if (metal) |*m| m else null;
    const p = try nsfw_vit.score(allocator, &model, ctx, pixels, width, height);
    last_image_score = p;
    var msg: [96]u8 = undefined;
    const line = try std.fmt.bufPrint(&msg, "safety: image score {d:.3}", .{p});
    try progress.event(io, allocator, line);
    if (p < image_threshold) return;
    const text = try std.fmt.allocPrint(
        allocator,
        "zdraw: image withheld by the safety filter (nsfw score {d:.2}). " ++
            "--safety off disables it in the CLI.\n",
        .{p},
    );
    defer allocator.free(text);
    try std.Io.File.stderr().writeStreamingAll(io, text);
    return error.SafetyBlocked;
}
