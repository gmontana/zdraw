//! Build the CLI, developer tools, unit tests and QA checks.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Keep -Doptimize explicit: standardOptimizeOption would default to Debug.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size",
    ) orelse .ReleaseFast;
    const is_macos = target.result.os.tag == .macos;
    const stanza_dep = b.dependency("stanza", .{
        .target = target,
        .optimize = optimize,
    });
    const stanza = stanza_dep.module("stanza");
    const stanza_licence = b.addInstallFile(stanza_dep.path("LICENSE"), "share/licenses/stanza/LICENSE");
    b.getInstallStep().dependOn(&stanza_licence.step);
    // The vendored MFA kernels, embedded so the binary never reads them from
    // the working directory at runtime (vendor/mfa/embed.zig).
    mfa_module = b.createModule(.{ .root_source_file = b.path("vendor/mfa/embed.zig") });

    const abi_gen = b.addExecutable(.{
        .name = "zdraw-abi-gen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/abi_gen.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    abi_gen.root_module.addImport("mfa", mfa_module.?);
    const abi_run = b.addRunArtifact(abi_gen);
    const abi_header = abi_run.addOutputFileArg("zdraw_abi.h");
    abi_include_dir = abi_header.dirname();

    // Binary identity for `zdraw version`, receipts and bench cards.
    const manifest = @import("build.zig.zon");
    const version_str = b.option([]const u8, "version", "Version string (default: build.zig.zon)") orelse
        manifest.version;
    const commit_str = b.option([]const u8, "commit", "Commit id (default: git describe)") orelse
        gitDescribe(b);
    const opts = b.addOptions();
    opts.addOption([]const u8, "version", version_str);
    opts.addOption([]const u8, "commit", commit_str);
    build_opts = opts;

    const exe = b.addExecutable(.{
        .name = "zdraw",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("stanza", stanza);
    exe.root_module.addImport("mfa", mfa_module.?);
    exe.root_module.addOptions("build_options", opts);
    addCertified(exe, b);
    addMetal(exe, b, is_macos);
    b.installArtifact(exe);
    // The vendored steel kernels (GEMM W6 route, steel attention) ship as
    // lib/steel.metallib next to the binary; the engine looks there after
    // ZDRAW_STEEL_LIB and before the legacy /tmp path. macOS only (xcrun).
    if (is_macos) {
        // The sources and the include tree are file inputs so the cache
        // rebuilds the metallib when a kernel header changes.
        // The metallib carries the same minimum macOS as the binary (the
        // target's version, the host's for a native build).
        const min_os = target.result.os.version_range.semver.min;
        const min_flag = b.fmt("-mmacosx-version-min={d}.{d}", .{ min_os.major, min_os.minor });
        const gemm_air = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", min_flag, "-I" });
        gemm_air.addDirectoryArg(b.path("vendor/steel/include"));
        gemm_air.addArg("-c");
        gemm_air.addFileArg(b.path("vendor/steel/gemm_entry.metal"));
        gemm_air.addArg("-o");
        const gemm_out = gemm_air.addOutputFileArg("gemm.air");
        const attn_air = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", min_flag, "-I" });
        attn_air.addDirectoryArg(b.path("vendor/steel/include"));
        attn_air.addArg("-c");
        attn_air.addFileArg(b.path("vendor/steel/attn_entry.metal"));
        attn_air.addArg("-o");
        const attn_out = attn_air.addOutputFileArg("attn.air");
        const lib_cmd = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib" });
        lib_cmd.addFileArg(gemm_out);
        lib_cmd.addFileArg(attn_out);
        lib_cmd.addArg("-o");
        const lib_out = lib_cmd.addOutputFileArg("steel.metallib");
        const lib_install = b.addInstallFile(lib_out, "lib/steel.metallib");
        b.getInstallStep().dependOn(&lib_install.step);
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the zdraw CLI");
    run_step.dependOn(&run_cmd.step);

    // One row per tool executable; addTool wires the exe, its run step,
    // and (when bin_step is set) an install-only build step. The check step
    // compiles EVERY row, which is what keeps a src/ rename from silently
    // breaking a tool that test/qa never compile (the mvres_stream lesson).
    const tools = [_]Tool{
        .{ .step = "refcheck", .root = "cmd/refcheck.zig", .desc = "Compare zdraw against reference tensors" },
        .{ .step = "bench", .root = "cmd/bench.zig", .desc = "Benchmark a fixed generate case" },
        .{ .step = "trace", .root = "cmd/trace.zig", .desc = "Write a runtime engine trace report" },
        .{
            .step = "gemmbench",
            .root = "cmd/gemmbench.zig",
            .desc = "Benchmark our GEMM kernel vs MPS",
            .mps_oracle = true,
        },
        .{ .step = "packbench", .root = "cmd/packbench.zig", .desc = "Benchmark W8 fused-dequant GEMM" },
        .{
            .step = "vencodegate",
            .root = "cmd/vencodegate.zig",
            .desc = "Run the CPU VAE encoder on the oracle's input tensor",
        },
        .{
            .step = "quality",
            .root = "cmd/quality.zig",
            .desc = "Gate a precision mode on image quality (PSNR/SSIM)",
        },
        .{
            .step = "schedulegate",
            .root = "cmd/schedulegate.zig",
            .desc = "Compare lower-step output against an exact schedule reference",
        },
        .{
            .step = "cacheprobe",
            .root = "cmd/cacheprobe.zig",
            .desc = "Measure cross-step transformer layer drift",
        },
        .{ .step = "sensitivity", .root = "cmd/sensitivity.zig", .desc = "Scan layer-band precision quality" },
        .{
            .step = "quantreport",
            .root = "cmd/quantreport.zig",
            .desc = "Screen transformer weights for W8/W4 packing",
            .metal = false,
        },
        .{
            .step = "zpackbuild",
            .root = "cmd/zpackbuild.zig",
            .desc = "Build packed W8 sidecars",
            .metal = false,
        },
        .{
            .step = "kleinpack",
            .root = "cmd/kleinpack.zig",
            .desc = "Build FLUX.2 Klein W16 sidecar",
            .metal = false,
        },
    };
    const check_step = b.step("check", "Compile every tool executable without running it");
    for (tools) |t| {
        const tool_exe = addTool(b, target, optimize, is_macos, t);
        check_step.dependOn(&tool_exe.step);
    }

    const test_step = b.step("test", "Run unit tests");
    // One test root (src/tests.zig) references every module: files in the
    // src/ subfolders import each other with relative paths, which only a
    // module rooted at src/ permits.
    const test_exe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_exe.root_module.addImport("stanza", stanza);
    test_exe.root_module.addImport("mfa", mfa_module.?);
    test_exe.root_module.addOptions("build_options", opts);
    addCertified(test_exe, b);
    addMetal(test_exe, b, is_macos);
    test_step.dependOn(&b.addRunArtifact(test_exe).step);

    const fmt = b.addSystemCommand(&.{ "zig", "fmt", "--check", "." });
    const guard = b.addSystemCommand(&.{ "python3", "tools/style/zig_guard.py", "--check" });
    const names = b.addSystemCommand(&.{ "python3", "tools/style/name_guard.py", "--all" });
    const zlint = b.addSystemCommand(&.{ "python3", "tools/style/zlint_check.py", "--check" });
    const architecture_cmd =
        b.addSystemCommand(&.{ "python3", "tools/style/architecture_guard.py" });
    const architecture_test_cmd =
        b.addSystemCommand(&.{ "python3", "tools/style/architecture_guard.py", "--self-test" });
    const env_guard_cmd =
        b.addSystemCommand(&.{ "python3", "tools/style/env_guard.py" });
    const env_guard_test_cmd =
        b.addSystemCommand(&.{ "python3", "tools/style/env_guard.py", "--self-test" });
    const licence_guard_cmd =
        b.addSystemCommand(&.{ "python3", "tools/style/licence_guard.py" });
    const licence_guard_test_cmd =
        b.addSystemCommand(&.{ "python3", "tools/style/licence_guard.py", "--self-test" });
    const manifest_test_cmd =
        b.addSystemCommand(&.{ "python3", "tools/pack_manifest.py", "--self-test" });
    const architecture_step =
        b.step("architecture", "Check ownership and dependency invariants");
    architecture_step.dependOn(&architecture_cmd.step);
    architecture_step.dependOn(&architecture_test_cmd.step);
    architecture_step.dependOn(&env_guard_cmd.step);
    architecture_step.dependOn(&env_guard_test_cmd.step);
    architecture_step.dependOn(&licence_guard_cmd.step);
    architecture_step.dependOn(&licence_guard_test_cmd.step);
    architecture_step.dependOn(&manifest_test_cmd.step);

    const qa_step = b.step("qa", "Run fmt, architecture, style guards, zlint, and tool compiles");
    const census_test_cmd = b.addSystemCommand(&.{ "python3", "tools/quality/test_repro_census.py" });
    qa_step.dependOn(&census_test_cmd.step);
    qa_step.dependOn(check_step);
    qa_step.dependOn(&fmt.step);
    qa_step.dependOn(architecture_step);
    qa_step.dependOn(&guard.step);
    qa_step.dependOn(&names.step);
    qa_step.dependOn(&zlint.step);
}

const Tool = struct {
    step: []const u8,
    root: []const u8,
    desc: []const u8,
    metal: bool = true,
    mps_oracle: bool = false,
    bin_step: ?[]const u8 = null,
    bin_desc: []const u8 = "",
};

fn addTool(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    is_macos: bool,
    t: Tool,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = b.fmt("zdraw-{s}", .{t.step}),
        .root_module = b.createModule(.{
            .root_source_file = b.path(t.root),
            .target = target,
            .optimize = optimize,
        }),
    });
    // The tool's own root is `cmd/`; the engine comes in as one package whose
    // root is `src/lib.zig`, carrying the vendored MFA sources, the build
    // options and the embedded JSON the engine modules import.
    const lib = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.addImport("mfa", mfa_module.?);
    if (build_opts) |o| lib.addOptions("build_options", o);
    addCertifiedTo(lib, b);
    exe.root_module.addImport("zdraw", lib);
    exe.root_module.addImport("mfa", mfa_module.?);
    if (build_opts) |o| exe.root_module.addOptions("build_options", o);
    if (t.metal) addMetal(exe, b, is_macos);
    if (t.mps_oracle) addMpsOracle(exe, b, is_macos);
    const cmd = b.addRunArtifact(exe);
    if (b.args) |args| cmd.addArgs(args);
    b.step(t.step, t.desc).dependOn(&cmd.step);
    if (t.bin_step) |bin| {
        const install = b.addInstallArtifact(exe, .{});
        b.step(bin, t.bin_desc).dependOn(&install.step);
    }
    return exe;
}

// Set once in build() BEFORE the tool loop: the generated zdraw_abi.h
// directory every Metal compile includes (the single ABI size-literal
// source, src/abi_gen.zig). addMetal silently skips the include when this
// is still null, so ordering matters.
var abi_include_dir: ?std.Build.LazyPath = null;
var mfa_module: ?*std.Build.Module = null;
var build_opts: ?*std.Build.Step.Options = null;

/// Embedded data: the certified census hashes and the safety rules/fixtures.
fn addCertified(compile: *std.Build.Step.Compile, b: *std.Build) void {
    addCertifiedTo(compile.root_module, b);
}

fn addCertifiedTo(module: *std.Build.Module, b: *std.Build) void {
    module.addAnonymousImport("certified_hashes", .{
        .root_source_file = b.path("certified/hashes.json"),
    });
    module.addAnonymousImport("safety_rules", .{
        .root_source_file = b.path("safety/prompt_rules.json"),
    });
    module.addAnonymousImport("safety_tests", .{
        .root_source_file = b.path("safety/tests.json"),
    });
}

/// `git describe --always --dirty --tags` of the build root, or "unknown"
/// outside a checkout (release tarballs pass -Dcommit instead).
fn gitDescribe(b: *std.Build) []const u8 {
    var code: u8 = 0;
    const root = b.build_root.path orelse ".";
    const raw = b.runAllowFail(
        &.{ "git", "-C", root, "describe", "--always", "--dirty", "--tags" },
        &code,
        .ignore,
    ) catch return "unknown";
    const trimmed = std.mem.trim(u8, raw, " \n\r\t");
    return if (trimmed.len == 0 or code != 0) "unknown" else trimmed;
}

fn addMetal(compile: *std.Build.Step.Compile, b: *std.Build, enabled: bool) void {
    if (!enabled) return;
    if (abi_include_dir) |dir| compile.root_module.addIncludePath(dir);
    // An explicit -Dtarget (the release builds: aarch64-macos.14.0) switches
    // zig's native SDK detection off; the SDK then comes from --sysroot.
    if (b.sysroot) |sdk| {
        const m = compile.root_module;
        m.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "System/Library/Frameworks" }) });
        m.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/include" }) });
    }
    compile.root_module.addCSourceFile(.{
        .file = b.path("src/metal/metal_api.m"),
        .flags = &[_][]const u8{ "-O3", "-fobjc-arc" },
    });
    compile.root_module.addCSourceFile(.{
        .file = b.path("src/cli/image_io.m"),
        .flags = &[_][]const u8{ "-O2", "-fobjc-arc" },
    });
    compile.root_module.linkFramework("CoreGraphics", .{});
    compile.root_module.linkFramework("ImageIO", .{});
    compile.root_module.linkFramework("Metal", .{});
    compile.root_module.linkFramework("MetalPerformanceShaders", .{});
    compile.root_module.linkFramework("MetalPerformanceShadersGraph", .{});
    compile.root_module.linkFramework("Foundation", .{});
}

fn addMpsOracle(compile: *std.Build.Step.Compile, b: *std.Build, enabled: bool) void {
    if (!enabled) return;
    compile.root_module.addCSourceFile(.{
        .file = b.path("src/metal/mps_api.m"),
        .flags = &[_][]const u8{ "-O3", "-fobjc-arc" },
    });
    compile.root_module.linkFramework("MetalPerformanceShaders", .{});
}

fn fatal(b: *std.Build, comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("build.zig: " ++ fmt ++ "\n", args);
    b.invalid_user_input = true;
    std.process.exit(1);
}
