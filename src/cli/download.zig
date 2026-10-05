//! Resumable downloads from Hugging Face for `zdraw fetch`, in Zig: no
//! Python, no `hf` CLI. One file at a time: the first hop
//! (huggingface.co/<repo>/resolve/main/<path>) answers 302 for LFS files with
//! `x-linked-size` and `x-linked-etag` (the file's SHA-256), then the CDN URL
//! is read with a `Range` header into `<file>.part`; the hash is checked and
//! the file renamed. Small non-LFS files come back 200 on the first hop.
//! Progress goes to stdout as `fetch: file <k>/<n> <path>` then
//! `fetch: <path> <done>/<total> MB` lines (the app's progress bar reads them).
const std = @import("std");
const util = @import("session_util.zig");
const version = @import("version.zig");

const chunk = 1 << 20;
const Sha256 = std.crypto.hash.sha2.Sha256;
/// "Bearer <HF_TOKEN>" for the session, or null (set once per repoFiles).
var auth: ?[]const u8 = null;

fn headers() std.http.Client.Request.Headers {
    return .{
        .user_agent = .{ .override = "zdraw/" ++ version.semver },
        .authorization = if (auth) |a| .{ .override = a } else .default,
    };
}

pub const Error = error{ HttpStatus, HashMismatch, ShortBody };

/// Every path under `dest`; existing complete files are skipped.
pub fn repoFiles(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    repo: []const u8,
    paths: []const []const u8,
    dest: []const u8,
) !void {
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    try client.initDefaultProxies(arena_state.allocator(), environ);
    // Gated repositories: `HF_TOKEN` as on the Hugging Face CLI.
    auth = if (environ.get("HF_TOKEN")) |t|
        try std.fmt.allocPrint(arena_state.allocator(), "Bearer {s}", .{t})
    else
        null;
    for (paths, 1..) |path, k| {
        const out = try std.fs.path.join(allocator, &.{ dest, path });
        defer allocator.free(out);
        if (std.Io.Dir.cwd().access(io, out, .{})) |_| continue else |_| {}
        var line: [512]u8 = undefined;
        const fmt = "fetch: file {d}/{d} {s}\n";
        try util.writeIo(io, try std.fmt.bufPrint(&line, fmt, .{ k, paths.len, path }));
        if (std.fs.path.dirname(out)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
        try oneFile(io, allocator, &client, repo, path, out);
    }
}

const Head = struct {
    /// The CDN URL (302) or null when the first hop carried the body (200).
    location: ?[]const u8,
    size: ?u64,
    sha256: ?[64]u8,
};

fn oneFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: []const u8,
    path: []const u8,
    out: []const u8,
) !void {
    const url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/{s}/resolve/main/{s}",
        .{ repo, path },
    );
    defer allocator.free(url);
    const part = try std.fmt.allocPrint(allocator, "{s}.part", .{out});
    defer allocator.free(part);
    // A connection that dies mid-file surfaces as a read error or a 30 s
    // receive timeout (armTimeout); each attempt resumes from the part size
    // with a fresh CDN location.
    var attempt: u32 = 0;
    const sha = while (true) {
        var have: u64 = 0;
        if (std.Io.Dir.cwd().statFile(io, part, .{})) |st| have = st.size else |_| {}
        if (transfer(io, allocator, client, url, part, have, path)) |sha| {
            break sha;
        } else |err| {
            attempt += 1;
            if (attempt >= 6) return err;
            var buf: [512]u8 = undefined;
            const fmt = "fetch: {s} interrupted ({s}), resuming\n";
            try util.writeIo(io, try std.fmt.bufPrint(&buf, fmt, .{ path, @errorName(err) }));
        }
    };
    if (sha) |want| try verify(io, allocator, part, &want);
    try std.Io.Dir.cwd().rename(part, std.Io.Dir.cwd(), out, io);
}

/// One attempt: first hop, then the body from `have`. Returns the SHA-256
/// to verify when the server gave one.
fn transfer(
    io: std.Io,
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    part: []const u8,
    have: u64,
    path: []const u8,
) !?[64]u8 {
    const head = try firstHop(io, allocator, client, url, part);
    defer if (head.location) |loc| allocator.free(loc);
    if (head.location) |loc| {
        try body(io, allocator, client, loc, part, have, head.size, path);
    }
    return head.sha256;
}

/// 30 s receive timeout on the request's socket: a stalled CDN connection
/// then fails the read instead of hanging the download forever.
fn armTimeout(req: *std.http.Client.Request) void {
    const conn = req.connection orelse return;
    const fd = conn.stream_reader.stream.socket.handle;
    const tv = std.posix.timeval{ .sec = 30, .usec = 0 };
    const opt = std.mem.asBytes(&tv);
    std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, opt) catch |err| {
        std.log.warn("fetch: receive timeout not set: {s}", .{@errorName(err)});
    };
}

/// GET the resolve URL without following redirects: LFS files answer 302
/// with size and hash headers; small files answer 200 and are written here.
fn firstHop(
    io: std.Io,
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    part: []const u8,
) !Head {
    var req = try client.request(.GET, try std.Uri.parse(url), .{
        .redirect_behavior = .unhandled,
        .headers = headers(),
    });
    defer req.deinit();
    armTimeout(&req);
    try req.sendBodiless();
    var response = try req.receiveHead(&.{});
    var head = Head{ .location = null, .size = null, .sha256 = null };
    var it = response.head.iterateHeaders();
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "x-linked-size")) {
            head.size = std.fmt.parseInt(u64, h.value, 10) catch null;
        } else if (std.ascii.eqlIgnoreCase(h.name, "x-linked-etag")) {
            const v = std.mem.trim(u8, h.value, "\"");
            if (v.len == 64) head.sha256 = v[0..64].*;
        }
    }
    switch (response.head.status) {
        .ok => {
            var tbuf: [chunk]u8 = undefined;
            const reader = response.reader(&tbuf);
            try streamTo(io, allocator, reader, part, 0, response.head.content_length, url);
            return head;
        },
        .found, .moved_permanently, .temporary_redirect, .permanent_redirect => {
            // LFS files: an absolute CDN URL. Small files: a relative
            // /api/resolve-cache/... path on huggingface.co.
            const loc = response.head.location orelse return Error.HttpStatus;
            head.location = if (std.mem.startsWith(u8, loc, "/"))
                try std.fmt.allocPrint(allocator, "https://huggingface.co{s}", .{loc})
            else
                try allocator.dupe(u8, loc);
            return head;
        },
        else => return Error.HttpStatus,
    }
}

/// GET the CDN URL from byte `have` onwards, appended to the part file.
fn body(
    io: std.Io,
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    part: []const u8,
    have: u64,
    total: ?u64,
    label: []const u8,
) !void {
    if (total) |t| if (have >= t) return;
    var range_buf: [64]u8 = undefined;
    const range = try std.fmt.bufPrint(&range_buf, "bytes={d}-", .{have});
    const extra = [_]std.http.Header{.{ .name = "Range", .value = range }};
    var req = try client.request(.GET, try std.Uri.parse(url), .{
        .extra_headers = &extra,
        .headers = headers(),
    });
    defer req.deinit();
    armTimeout(&req);
    try req.sendBodiless();
    var redirect_buf: [8192]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    const len = response.head.content_length;
    // The expected final size comes from the hop that carries the body:
    // 206 = the rest after `have`, 200 = the whole file, 416 = nothing
    // left (the part file is already complete).
    const start: u64 = switch (response.head.status) {
        .partial_content => have,
        .ok => 0,
        .range_not_satisfiable => return,
        else => return Error.HttpStatus,
    };
    const expect: ?u64 = if (len) |n| start + n else total;
    var tbuf: [chunk]u8 = undefined;
    const reader = response.reader(&tbuf);
    try streamTo(io, allocator, reader, part, start, expect, label);
}

/// Copy the response body into the part file from `start`, printing
/// progress every ~2%.
fn streamTo(
    io: std.Io,
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    part: []const u8,
    start: u64,
    total: ?u64,
    label: []const u8,
) !void {
    _ = allocator;
    const file = if (start == 0)
        try std.Io.Dir.cwd().createFile(io, part, .{})
    else
        try std.Io.Dir.cwd().openFile(io, part, .{ .mode = .read_write });
    defer file.close(io);
    var wbuf: [chunk]u8 = undefined;
    var fw = std.Io.File.Writer.init(file, io, &wbuf);
    if (start != 0) try fw.seekTo(start);
    var done: u64 = start;
    var last_pct: u64 = 0;
    var line: [512]u8 = undefined;
    while (true) {
        const n = reader.stream(&fw.interface, .limited(chunk)) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        done += n;
        if (total) |t| {
            const pct = if (t == 0) 100 else done * 100 / t;
            if (pct >= last_pct + 2 or done == t) {
                last_pct = pct;
                const fmt = "fetch: {s} {d}/{d} MB\n";
                const text = try std.fmt.bufPrint(&line, fmt, .{ label, done >> 20, t >> 20 });
                try util.writeIo(io, text);
            }
        }
    }
    try fw.interface.flush();
    if (total) |t| if (done != t) return Error.ShortBody;
}

/// SHA-256 of the part file against the hex from `x-linked-etag`.
fn verify(io: std.Io, allocator: std.mem.Allocator, part: []const u8, want: *const [64]u8) !void {
    _ = allocator;
    const file = try std.Io.Dir.cwd().openFile(io, part, .{});
    defer file.close(io);
    var rbuf: [chunk]u8 = undefined;
    var fr = file.reader(io, &rbuf);
    var hasher = Sha256.init(.{});
    var buf: [chunk]u8 = undefined;
    while (true) {
        const n = try fr.interface.readSliceShort(&buf);
        if (n == 0) break;
        hasher.update(buf[0..n]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    var hex: [64]u8 = undefined;
    const got = try std.fmt.bufPrint(&hex, "{x}", .{digest});
    if (!std.mem.eql(u8, got, want)) return Error.HashMismatch;
}

test "the hash check compares the lowercase hex digest" {
    var digest: [32]u8 = undefined;
    Sha256.hash("abc", &digest, .{});
    var hex: [64]u8 = undefined;
    const got = try std.fmt.bufPrint(&hex, "{x}", .{digest});
    try std.testing.expectEqualStrings(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        got,
    );
}
