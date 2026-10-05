//! Shared classifiers for Metal route fallback decisions.

pub fn isGemmRefusal(err: anyerror) bool {
    return switch (err) {
        error.GemmUnavailable,
        error.GemmShape,
        error.UnsupportedDType,
        => true,
        else => false,
    };
}

pub fn isFallback(err: anyerror) bool {
    return isGemmRefusal(err) or err == error.InvalidShape;
}

test "classifies GEMM fallback errors" {
    try @import("std").testing.expect(isGemmRefusal(error.GemmUnavailable));
    try @import("std").testing.expect(isGemmRefusal(error.GemmShape));
    try @import("std").testing.expect(isGemmRefusal(error.UnsupportedDType));
    try @import("std").testing.expect(!isGemmRefusal(error.InvalidShape));
    try @import("std").testing.expect(isFallback(error.InvalidShape));
}
