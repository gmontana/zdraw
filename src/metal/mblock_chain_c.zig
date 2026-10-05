//! C declaration for one-command-buffer block execution.

const c = @import("metal_c.zig");
const mblock = @import("mblock_c.zig");

comptime {
    _ = @import("../runtime/abi_assert.zig"); // shared-struct layout guards
}

pub extern fn zdraw_metal_run_block_chain(
    queue: *anyopaque,
    norm_pipeline: *anyopaque,
    resid_pipeline: *anyopaque,
    gemm_exact_pipeline: *anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    qk_pipeline: *anyopaque,
    attn_pipeline: *anyopaque,
    swiglu_pipeline: *anyopaque,
    swiglu_fused_pipeline: ?*anyopaque,
    resid_norm_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_encode_chain_layer(
    batch: *anyopaque,
    norm_pipeline: *anyopaque,
    resid_pipeline: *anyopaque,
    gemm_exact_pipeline: *anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    qk_pipeline: *anyopaque,
    attn_pipeline: *anyopaque,
    swiglu_pipeline: *anyopaque,
    swiglu_fused_pipeline: ?*anyopaque,
    resid_norm_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_encode_toma_attn_norm(
    batch: *anyopaque,
    norm_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_encode_toma_attn_core(
    batch: *anyopaque,
    gemm_exact_pipeline: *anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    qk_pipeline: *anyopaque,
    attn_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_encode_toma_attn_finish(
    batch: *anyopaque,
    gemm_exact_pipeline: *anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    resid_norm_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_encode_toma_ffn_core(
    batch: *anyopaque,
    gemm_exact_pipeline: *anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    swiglu_pipeline: *anyopaque,
    swiglu_fused_pipeline: ?*anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_encode_toma_ffn_finish(
    batch: *anyopaque,
    resid_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: *const Weights,
    params: *const Params,
    threads: *const Threads,
) c_int;

pub extern fn zdraw_metal_run_stack_chain(
    queue: *anyopaque,
    norm_pipeline: *anyopaque,
    resid_pipeline: *anyopaque,
    gemm_exact_pipeline: *anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    qk_pipeline: *anyopaque,
    attn_pipeline: *anyopaque,
    swiglu_pipeline: *anyopaque,
    swiglu_fused_pipeline: ?*anyopaque,
    resid_norm_pipeline: *anyopaque,
    buffers: *const Buffers,
    weights: [*]const Weights,
    params: [*]const Params,
    layer_count: usize,
    threads: *const Threads,
) c_int;

pub const Buffers = extern struct {
    state: *anyopaque,
    norm: *anyopaque,
    attn: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    mix: *anyopaque,
    ffn: *anyopaque,
    gate: *anyopaque,
    up: *anyopaque,
    gateup: *anyopaque, // fused [M, 2*inner] gate+up output
};

pub const Weights = extern struct {
    attn_in: *anyopaque,
    attn_scale: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    q_norm: *anyopaque,
    k_norm: *anyopaque,
    pos: *anyopaque,
    rope: *anyopaque,
    proj: *anyopaque,
    attn_out: *anyopaque,
    attn_gate: *anyopaque,
    ffn_in: *anyopaque,
    mlp_scale: *anyopaque,
    ffn_gate: *anyopaque,
    ffn_up: *anyopaque,
    ffn_down: *anyopaque,
    ffn_out: *anyopaque,
    mlp_gate: *anyopaque,
    ffn_fused: *anyopaque, // fused [2*inner, hidden] gate+up weight (or zero)
};

pub const Params = extern struct {
    attn_norm: mblock.Params,
    q: c.GemmParams,
    k: c.GemmParams,
    v: c.GemmParams,
    qk: c.QkNormParams,
    attn: c.AttnParams,
    proj: c.GemmParams,
    attn_resid: mblock.Params,
    ffn_norm: mblock.Params,
    ffn_gate: c.GemmParams,
    ffn_up: c.GemmParams,
    ffn_down: c.GemmParams,
    ffn_resid: mblock.Params,
    ffn_fused: c.GemmParams, // mode 0 disables the fused gate+up path
};

pub const Threads = extern struct {
    block: usize,
    qk: usize,
    attn: usize,
    swiglu: usize,
    /// mattn.Kernel ordinal for `attn`; the C side keys the grid on it.
    attn_kernel: usize,
};
