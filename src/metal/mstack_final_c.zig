//! C declaration for fused stack-to-final execution.

const c = @import("metal_c.zig");
const chain_c = @import("mblock_chain_c.zig");
const mfinal = @import("mfinal.zig");

pub const FinalBufs = extern struct {
    scale: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    batch: *anyopaque,
    output: *anyopaque,
};

pub extern fn zdraw_metal_run_stack_final(
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
    final_pipeline: *anyopaque,
    bias_pipeline: *anyopaque,
    buffers: *const chain_c.Buffers,
    weights: [*]const chain_c.Weights,
    params: [*]const chain_c.Params,
    layer_count: usize,
    final_buffers: *const FinalBufs,
    final_norm: *const mfinal.NormParams,
    final_gemm: *const c.GemmParams,
    final_bias: *const mfinal.BiasParams,
    threads: *const chain_c.Threads,
    final_threads: usize,
) c_int;

pub extern fn zdraw_metal_encode_final(
    batch: *anyopaque,
    final_pipeline: *anyopaque,
    gemm_pipeline: *anyopaque,
    bias_pipeline: *anyopaque,
    state: *anyopaque,
    final_buffers: *const FinalBufs,
    final_norm: *const mfinal.NormParams,
    final_gemm: *const c.GemmParams,
    final_bias: *const mfinal.BiasParams,
    norm_threads: usize,
) c_int;
