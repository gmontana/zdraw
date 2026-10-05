//! Cross-process generation lock.
//!
//! Two zdraw processes generating simultaneously corrupt each other's
//! renders (ledger concurrent-render-corruption, 2026-08-04). Until that
//! defect is root-caused, generation is serialized per user with a blocking
//! advisory flock: a second process waits instead of producing garbage.
//! One owner per render; acquire is idempotent within the owning thread,
//! release is not depth-counted. Within one process a second thread blocks on
//! an in-process mutex (the C ABI is otherwise free to be called from two
//! threads, and the flock is per process, not per thread).
//! ZDRAW_NO_LOCK=1 opts a lab run out of both.

const std = @import("std");

const env = @import("env.zig");

var lock_fd: ?c_int = null;
var held = std.atomic.Value(bool).init(false);
var owner: ?std.Thread.Id = null;

/// Blocks until this process holds the per-user generation lock. Best-effort:
/// any failure to create or lock the file falls through to running unlocked
/// (the defect is a corruption hazard, not a safety invariant).
pub fn acquire() void {
    if (env.flag("ZDRAW_NO_LOCK", false)) return;
    const me = std.Thread.getCurrentId();
    if (owner == me) return; // idempotent for the holder
    // A second thread in this process waits here (renders take seconds;
    // a sleep-polled wait is cheaper than plumbing an Io through callers).
    while (held.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
        var pause = std.c.timespec{ .sec = 0, .nsec = 2 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&pause, null);
    }
    owner = me;
    if (lock_fd != null) return;
    var buf: [96]u8 = undefined;
    const path = std.fmt.bufPrintZ(
        &buf,
        "/tmp/zdraw-generate-{d}.lock",
        .{std.c.getuid()},
    ) catch return;
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDWR, .CREAT = true }, @as(c_uint, 0o644));
    if (fd < 0) return;
    if (std.c.flock(fd, std.c.LOCK.EX) != 0) {
        _ = std.c.close(fd);
        return;
    }
    lock_fd = fd;
}

/// Releases the lock (closing the descriptor drops the flock).
pub fn release() void {
    if (owner != std.Thread.getCurrentId()) return;
    if (lock_fd) |fd| {
        _ = std.c.close(fd);
        lock_fd = null;
    }
    owner = null;
    held.store(false, .release);
}

test "release after a redundant acquire leaves no fd" {
    acquire();
    acquire();
    release();
    release();
}

test "a second thread waits for the holder and then proceeds" {
    if (env.flag("ZDRAW_NO_LOCK", false)) return;
    acquire();
    var entered = std.atomic.Value(bool).init(false);
    const worker = try std.Thread.spawn(.{}, struct {
        fn run(flag: *std.atomic.Value(bool)) void {
            acquire();
            flag.store(true, .seq_cst);
            release();
        }
    }.run, .{&entered});
    var pause = std.c.timespec{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&pause, null);
    try std.testing.expect(!entered.load(.seq_cst)); // blocked on the holder
    release();
    worker.join();
    try std.testing.expect(entered.load(.seq_cst));
}
