//! Binary identity from build options: version and source commit ("unknown"
//! outside a checkout). An installed binary needs no repository to report them.
const std = @import("std");
const build_options = @import("build_options");

pub const semver: []const u8 = build_options.version;
pub const commit: []const u8 = build_options.commit;
pub const abi: u32 = 1;
/// "0.1.0+5d4ea75": what receipts and cards stamp as the engine revision.
pub const revision: []const u8 = semver ++ "+" ++ commit;

test "version strings are populated" {
    try std.testing.expect(std.mem.indexOfScalar(u8, semver, '.') != null);
    try std.testing.expect(commit.len > 0);
    try std.testing.expect(std.mem.indexOfScalar(u8, revision, '+') != null);
}
