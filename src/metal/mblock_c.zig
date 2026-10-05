//! C declarations for block-local Metal kernels.

pub extern fn zdraw_metal_run_block_kernel(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    scale: *anyopaque,
    output: *anyopaque,
    params: *const Params,
    thread_count: usize,
) c_int;

pub const Params = extern struct {
    tokens: u32,
    hidden: u32,
    dtype: u32,
    has_scale: u32,
    eps: f32,
    weight_offset: u64,
};
