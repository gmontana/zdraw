//! Bench-only MPS declarations.
//!
//! Production zdraw inference uses the raw Metal bridge. These symbols are
//! linked only into `zig build gemmbench` so MPS stays an oracle, not a runtime
//! dependency.

pub extern fn zdraw_mps_gemm_make(
    device: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    m: u32,
    n: u32,
    k: u32,
) ?*anyopaque;
pub extern fn zdraw_mps_gemm_run(queue: *anyopaque, ctx: *anyopaque) c_int;
pub extern fn zdraw_mps_gemm_free(ctx: *anyopaque) void;
