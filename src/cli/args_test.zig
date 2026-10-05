const std = @import("std");

const args = @import("args.zig");
const model = @import("model_kind.zig");

test "parse generate command" {
    const test_args = std.process.Args{ .vector = &.{
        "zdraw",     "generate",
        "--model",   "z-image-turbo",
        "--weights", "weights",
        "--prompt",  "cat",
        "--out",     "cat.png",
        "--show",
    } };
    var iter = std.process.Args.Iterator.init(test_args);
    const command = try args.parse(&iter);

    try std.testing.expectEqual(model.ModelKind.z_image_turbo, command.generate.kind);
    try std.testing.expectEqualStrings("weights", command.generate.weights_dir);
    try std.testing.expectEqualStrings("cat", command.generate.prompt);
    try std.testing.expectEqualStrings("cat.png", command.generate.output_path);
    // 0 = unset: main resolves the per-model default (4 distilled, 50 base).
    try std.testing.expectEqual(@as(u32, 0), command.generate.steps);
    try std.testing.expectEqual(@as(f32, 0), command.generate.guidance);
    try std.testing.expect(command.generate.show);
}

test "parse bare command as help" {
    const test_args = std.process.Args{ .vector = &.{"zdraw"} };
    var iter = std.process.Args.Iterator.init(test_args);
    const command = try args.parse(&iter);

    try std.testing.expectEqual(args.Help.overview, command.help);
}

test "parse session command" {
    const test_args = std.process.Args{ .vector = &.{
        "zdraw",          "session",
        "--preview",      "--out-dir",
        "runs",           "--width",
        "64",             "--no-show",
        "--no-auto-save",
    } };
    var iter = std.process.Args.Iterator.init(test_args);
    const command = try args.parse(&iter);

    try std.testing.expectEqualStrings("runs", command.session.output_dir);
    try std.testing.expectEqual(@as(u32, 64), command.session.width);
    try std.testing.expect(!command.session.show);
    try std.testing.expect(command.session.preview);
    try std.testing.expect(!command.session.auto_save);
}

test "parse session without output dir when auto-save is off" {
    const test_args = std.process.Args{ .vector = &.{
        "zdraw", "session", "--preview", "--no-auto-save",
    } };
    var iter = std.process.Args.Iterator.init(test_args);
    const command = try args.parse(&iter);

    try std.testing.expect(command.session.preview);
    try std.testing.expect(!command.session.auto_save);
}

test "parse version command, both spellings" {
    for ([_][*:0]const u8{ "version", "--version" }) |word| {
        const test_args = std.process.Args{ .vector = &.{ "zdraw", word } };
        var iter = std.process.Args.Iterator.init(test_args);
        const command = try args.parse(&iter);
        try std.testing.expectEqual(args.Command.version, command);
    }
}

test "parse doctor without weights and with --json" {
    const bare = std.process.Args{ .vector = &.{ "zdraw", "doctor" } };
    var iter = std.process.Args.Iterator.init(bare);
    const command = try args.parse(&iter);
    try std.testing.expectEqualStrings("", command.doctor.weights_dir);
    try std.testing.expect(!command.doctor.json);

    const json = std.process.Args{
        .vector = &.{ "zdraw", "doctor", "--json", "--weights", "w" },
    };
    var iter2 = std.process.Args.Iterator.init(json);
    const command2 = try args.parse(&iter2);
    try std.testing.expect(command2.doctor.json);
    try std.testing.expectEqualStrings("w", command2.doctor.weights_dir);
}

test "parse bench --card and reject a size override" {
    const ok = std.process.Args{
        .vector = &.{
            "zdraw",     "bench", "--card",   "--model", "flux2-klein-4b",
            "--weights", "w",     "--repeat", "3",
        },
    };
    var iter = std.process.Args.Iterator.init(ok);
    const command = try args.parse(&iter);
    try std.testing.expect(command.bench.card);
    try std.testing.expectEqual(@as(u32, 3), command.bench.repeat);
    try std.testing.expectEqualStrings("w", command.bench.weights_dir);

    const bad = std.process.Args{
        .vector = &.{ "zdraw", "bench", "--weights", "w", "--width", "512" },
    };
    var iter2 = std.process.Args.Iterator.init(bad);
    try std.testing.expectError(error.UnknownOption, args.parse(&iter2));
}

test "parse fetch with a model and a directory" {
    const test_args = std.process.Args{
        .vector = &.{ "zdraw", "fetch", "z-image-turbo", "--dir", "/m" },
    };
    var iter = std.process.Args.Iterator.init(test_args);
    const command = try args.parse(&iter);
    try std.testing.expectEqual(args.Command.fetch, std.meta.activeTag(command));
    try std.testing.expectEqualStrings("/m", command.fetch.dir);
}

test "command help works before required arguments" {
    const commands = [_][*:0]const u8{
        "generate", "session", "fetch", "bench", "doctor", "preview", "inspect", "version",
    };
    for (commands) |name| {
        const direct = try parseTest(&.{ "zdraw", name, "--help" });
        const named = try parseTest(&.{ "zdraw", "help", name });
        try std.testing.expectEqual(named.help, direct.help);
        const short = try parseTest(&.{ "zdraw", name, "-h" });
        try std.testing.expectEqual(named.help, short.help);
    }
    const after = try parseTest(&.{ "zdraw", "generate", "--model", "flux2-klein-4b", "--help" });
    try std.testing.expectEqual(args.Help.generate, after.help);
    try std.testing.expectError(error.InvalidCommand, parseTest(&.{ "zdraw", "help", "missing" }));
    try std.testing.expectError(error.UnknownOption, parseTest(&.{ "zdraw", "version", "extra" }));
}

test "model commands can resolve downloaded weights after parsing" {
    const gen = try parseTest(&.{ "zdraw", "generate", "--prompt", "cat", "--out", "cat.png" });
    try std.testing.expectEqualStrings("", gen.generate.weights_dir);
    const bench = try parseTest(&.{ "zdraw", "bench" });
    try std.testing.expectEqualStrings("", bench.bench.weights_dir);
    const session = try parseTest(&.{ "zdraw", "session", "--out-dir", "images" });
    try std.testing.expectEqualStrings("", session.session.weights_dir);
}

test "invalid edit combinations fail before model loading" {
    const prefix = .{ "zdraw", "generate", "--prompt", "cat", "--out", "cat.png" };
    try std.testing.expectError(
        error.EditUnsupported,
        parseTest(&(prefix ++ .{ "--edit", "photo.png" })),
    );
    try std.testing.expectError(
        error.EditUnsupported,
        parseTest(&(prefix ++ .{ "--init-image", "photo.png" })),
    );
    try std.testing.expectError(
        error.MissingInitImage,
        parseTest(&(prefix ++ .{ "--mask", "mask.png" })),
    );
    const klein = prefix ++ .{ "--model", "flux2-klein-4b" };
    try std.testing.expectError(
        error.MissingInitImage,
        parseTest(&(klein ++ .{ "--strength", "0.3" })),
    );
    try std.testing.expectError(error.ConflictingEdits, parseTest(&(klein ++ .{
        "--edit", "a.png", "--init-image", "b.png",
    })));
    const init = klein ++ .{ "--init-image", "photo.png" };
    try std.testing.expectError(error.EditSeedsUnsupported, parseTest(&(init ++ .{ "--seeds", "7,11" })));
    try std.testing.expectError(
        error.InvalidStrength,
        parseTest(&(init ++ .{ "--strength", "1.1" })),
    );
    try std.testing.expectError(
        error.InvalidStrength,
        parseTest(&(init ++ .{ "--strength", "-0.1" })),
    );
    const masked = try parseTest(&(init ++ .{ "--strength", "0", "--mask", "mask.png" }));
    try std.testing.expectEqual(@as(f32, 0), masked.generate.strength);
}

test "nonfinite numbers missing values and empty seeds are rejected" {
    const prefix = .{ "zdraw", "generate", "--prompt", "cat", "--out", "cat.png" };
    inline for (.{ "nan", "inf", "-inf" }) |value| {
        try std.testing.expectError(
            error.InvalidNumber,
            parseTest(&(prefix ++ .{ "--guidance", value })),
        );
    }
    try std.testing.expectError(
        error.InvalidGuidance,
        parseTest(&(prefix ++ .{ "--guidance", "0" })),
    );
    try std.testing.expectError(error.InvalidNumber, parseTest(&(prefix ++ .{ "--steps", "0" })));
    try std.testing.expectError(error.MissingOptionValue, parseTest(&.{
        "zdraw", "generate", "--prompt", "--out", "cat.png",
    }));
    const klein = prefix ++ .{ "--model", "flux2-klein-4b" };
    try std.testing.expectError(error.ConflictingSeeds, parseTest(&(klein ++ .{
        "--seed", "7", "--seeds", "7,11",
    })));
    inline for (.{ "7,", ",7", "7,,11" }) |value| {
        try std.testing.expectError(
            error.InvalidNumber,
            parseTest(&(klein ++ .{ "--seeds", value })),
        );
    }
    try std.testing.expectError(error.ProfileUnsupported, parseTest(&.{
        "zdraw", "bench", "--model", "flux2-klein-4b", "--profile", "strict",
    }));
}

fn parseTest(argv: []const [*:0]const u8) !args.Command {
    var iter = std.process.Args.Iterator.init(.{ .vector = argv });
    return args.parse(&iter);
}

test "public CLI rejects unfinished commands and research variants" {
    try std.testing.expectError(error.InvalidCommand, parseTest(&.{ "zdraw", "serve" }));
    try std.testing.expectError(error.InvalidModel, parseTest(&.{
        "zdraw", "fetch", "flux2-klein-9b",
    }));
    inline for (.{ "--project", "--vae-reference" }) |flag| {
        try std.testing.expectError(error.UnknownOption, parseTest(&.{ "zdraw", "session", flag }));
    }
    inline for (.{ "--execution-plan", "--receipt" }) |flag| {
        try std.testing.expectError(error.UnknownOption, parseTest(&.{ "zdraw", "generate", flag }));
    }
}

test "model dimensions and distilled guidance fail before loading weights" {
    const prefix = .{ "zdraw", "generate", "--prompt", "cat", "--out", "cat.png" };
    try std.testing.expectError(error.InvalidGuidance, parseTest(&(prefix ++ .{ "--guidance", "4" })));
    const distilled = prefix ++ .{ "--model", "flux2-klein-4b" };
    const distilled_guided = parseTest(&(distilled ++ .{ "--guidance", "4" }));
    try std.testing.expectError(error.InvalidGuidance, distilled_guided);
    // The base model accepts guidance, in either option order, and an explicit 1.
    const base = prefix ++ .{ "--model", "flux2-klein-base-4b" };
    const guided = try parseTest(&(base ++ .{ "--guidance", "4" }));
    try std.testing.expectEqual(@as(f32, 4.0), guided.generate.guidance);
    const base_first = try parseTest(&(prefix ++ .{
        "--guidance", "3.5", "--model", "flux2-klein-base-4b",
    }));
    try std.testing.expectEqual(@as(f32, 3.5), base_first.generate.guidance);
    _ = try parseTest(&(base ++ .{ "--guidance", "1" }));
    const session = try parseTest(&.{
        "zdraw", "session", "--no-auto-save", "--model", "flux2-klein-base-4b", "--guidance", "3",
    });
    try std.testing.expectEqual(@as(f32, 3.0), session.session.guidance);
    try std.testing.expectError(error.InvalidGuidance, parseTest(&.{
        "zdraw", "session", "--no-auto-save", "--guidance", "3",
    }));
    try std.testing.expectError(error.InvalidDimension, parseTest(&(prefix ++ .{ "--width", "513" })));
    _ = try parseTest(&(prefix ++ .{ "--width", "528" }));
    try std.testing.expectError(error.InvalidDimension, parseTest(&(prefix ++ .{ "--model", "flux2-klein-4b", "--width", "528" })));
    _ = try parseTest(&.{ "zdraw", "session", "--preview", "--no-auto-save", "--width", "17" });
    try std.testing.expectError(error.InvalidDimension, parseTest(&.{
        "zdraw", "session", "--no-auto-save", "--width", "17",
    }));
}

test "duplicate seed values cannot silently overwrite a candidate" {
    const prefix = .{
        "zdraw",    "generate", "--model", "flux2-klein-4b",
        "--prompt", "cat",      "--out",   "cat.png",
        "--seeds",
    };
    const repeated = .{
        "7,7", "7,11,07", "0, 00", "18446744073709551615,18446744073709551615",
    };
    inline for (repeated) |value| {
        try std.testing.expectError(error.DuplicateSeed, parseTest(&(prefix ++ .{value})));
    }
    _ = try parseTest(&(prefix ++ .{"0,7,18446744073709551615"}));
}

test "unsupported Klein resolution fails before weights and previews remain free-sized" {
    const prefix = .{
        "zdraw", "generate", "--model", "flux2-klein-4b", "--prompt", "cat", "--out", "cat.png",
    };
    inline for (.{ .{ "64", "64" }, .{ "544", "800" }, .{ "1536", "1536" } }) |size| {
        const command = prefix ++ .{ "--width", size[0], "--height", size[1] };
        try std.testing.expectError(error.UnsupportedKleinResolution, parseTest(&command));
    }
    _ = try parseTest(&(prefix ++ .{ "--width", "1024", "--height", "768" }));
    try std.testing.expectError(error.UnsupportedKleinResolution, parseTest(&.{
        "zdraw",   "session", "--model",  "flux2-klein-4b", "--no-auto-save",
        "--width", "544",     "--height", "800",
    }));
    _ = try parseTest(&.{
        "zdraw",   "session", "--model", "flux2-klein-4b", "--preview", "--no-auto-save",
        "--width", "17",
    });
}

test "progressive display defaults on and can be disabled independently" {
    const prefix = .{ "zdraw", "generate", "--prompt", "fox", "--out", "fox.png", "--show" };
    const gen = try parseTest(&prefix);
    try std.testing.expect(gen.generate.progressive);
    const final = try parseTest(&(prefix ++ .{"--no-progressive"}));
    try std.testing.expect(final.generate.show);
    try std.testing.expect(!final.generate.progressive);
    const session = try parseTest(&.{ "zdraw", "session", "--no-auto-save", "--no-progressive" });
    try std.testing.expect(session.session.show);
    try std.testing.expect(!session.session.progressive);
}
