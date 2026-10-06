//! Raw declarations for the tiny Metal bridge.

// The embedded MFA kernel sources ride along with every bridge link.
comptime {
    _ = @import("mfa_embed.zig");
}

pub extern fn zdraw_metal_create_device() ?*anyopaque;
pub extern fn zdraw_metal_release_device(device: ?*anyopaque) void;
pub extern fn zdraw_metal_create_queue(device: *anyopaque) ?*anyopaque;
pub extern fn zdraw_metal_release_queue(queue: ?*anyopaque) void;
pub extern fn zdraw_metal_create_buffer(device: *anyopaque, size: usize) ?*anyopaque;
pub extern fn zdraw_metal_create_buffer_with_data(
    device: *anyopaque,
    data: [*]const u8,
    size: usize,
) ?*anyopaque;
pub extern fn zdraw_metal_create_buffer_no_copy(
    device: *anyopaque,
    data: [*]const u8,
    size: usize,
) ?*anyopaque;
pub extern fn zdraw_metal_read_buffer(buffer: *anyopaque, dst: [*]u8, size: usize) void;
pub extern fn zdraw_metal_buffer_length(buffer: *anyopaque) usize;
pub extern fn zdraw_metal_read_buffer_any(
    queue: *anyopaque,
    buffer: *anyopaque,
    dst: [*]u8,
    size: usize,
) c_int;
pub extern fn zdraw_metal_write_buffer(buffer: *anyopaque, src: [*]const u8, size: usize) void;
pub extern fn zdraw_metal_release_buffer(buffer: ?*anyopaque) void;
pub extern fn zdraw_metal_release_weight_buffer(buffer: ?*anyopaque) void;
pub extern fn zdraw_metal_clear_transient_caches() void;
pub extern fn zdraw_metal_pool_push() ?*anyopaque;
pub extern fn zdraw_metal_pool_pop(pool: ?*anyopaque) void;

pub extern fn zdraw_metal_metrics_reset() void;
pub extern fn zdraw_metal_peak_bytes() u64;
pub extern fn zdraw_metal_live_bytes() u64;
pub extern fn zdraw_metal_weight_bytes() u64;
pub extern fn zdraw_metal_dispatch_count() u64;
pub extern fn zdraw_metal_command_count() u64;
pub extern fn zdraw_metal_wait_count() u64;
pub extern fn zdraw_metal_readback_count() u64;
pub extern fn zdraw_metal_gemm_count() u64;
pub extern fn zdraw_metal_gemm_exact_count() u64;
pub extern fn zdraw_metal_gemm_half_count() u64;
pub extern fn zdraw_metal_gemm_w8_count() u64;
pub extern fn zdraw_metal_gemm_w6_count() u64;
pub extern fn zdraw_metal_gemm_w4_count() u64;
pub extern fn zdraw_metal_gemm_w2_count() u64;
pub extern fn zdraw_metal_gemm_mps_count() u64;
pub extern fn zdraw_metal_gemm_mps_fallback_count() u64;
pub extern fn zdraw_metal_attn_steel_fallback_count() u64;
pub extern fn zdraw_metal_gemm_mpp_fallback_count() u64;
pub extern fn zdraw_metal_note_mpp_fallback() void;
pub extern fn zdraw_metal_warn_mpp_unavailable() void;
/// Doctor probe of the steel metallib: 0 ok, 1 missing, 2 unusable, 3 stale.
pub extern fn zdraw_metal_steel_probe(
    path_out: [*]u8,
    path_len: usize,
    err_out: [*]u8,
    err_len: usize,
) c_int;
/// 1 when the Metal 4 shading language (macOS 26) and a device are present.
pub extern fn zdraw_metal4_available() c_int;
pub extern fn zdraw_metal_compile_mpp(
    device: *anyopaque,
    source: [*:0]const u8,
    entry: [*:0]const u8,
) ?*anyopaque;
pub extern fn zdraw_metal_run_gemm_mpp_enc(
    batch: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c_out: *anyopaque,
    params: *const GemmParams,
    a_off: u64,
    c_off: u64,
) c_int;
pub extern fn zdraw_metal_gpu_ns() u64;
pub extern fn zdraw_proc_peak_rss_bytes() u64;
pub extern fn zdraw_proc_phys_footprint_bytes() u64;
pub extern fn zdraw_proc_mapped_resident_bytes() u64;
pub extern fn zdraw_proc_sample_memory() void;
pub extern fn zdraw_proc_mapped_resident_peak_bytes() u64;
pub extern fn zdraw_proc_total_peak_bytes() u64;
pub extern fn zdraw_device_identity(
    name_out: [*]u8,
    name_len: usize,
    ram_bytes: *u64,
    gpu_working_set: *u64,
    os_major: *u32,
) c_int;
pub extern fn zdraw_thermal_state() c_int;
pub extern fn zdraw_now_ns() u64;
pub extern fn zdraw_metal_compile(
    device: *anyopaque,
    source: [*:0]const u8,
    entry: [*:0]const u8,
    error_buf: [*]u8,
    error_len: usize,
) ?*anyopaque;
pub extern fn zdraw_metal_release_pipeline(pipeline: ?*anyopaque) void;
pub extern fn zdraw_metal_pipeline_threads(pipeline: *anyopaque) usize;
pub extern fn zdraw_metal_run_generated_1d(
    queue: *anyopaque,
    pipeline: *anyopaque,
    buffers: [*]const *anyopaque,
    buffer_count: usize,
    element_count: u32,
    thread_count: usize,
) c_int;

pub extern fn zdraw_metal_run_linear(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const LinearParams,
    thread_count: usize,
) c_int;
pub extern fn zdraw_metal_run_conv(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
    params: *const ConvParams,
    thread_count: usize,
) c_int;
pub extern fn zdraw_metal_run_attention(
    queue: *anyopaque,
    pipeline: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    output: *anyopaque,
    params: *const AttnParams,
    kernel: u32,
    thread_count: usize,
) c_int;
pub extern fn zdraw_metal_run_qk_norm_rope(
    queue: *anyopaque,
    pipeline: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    q_weight: *anyopaque,
    k_weight: *anyopaque,
    pos: *anyopaque,
    rope: *anyopaque,
    params: *const QkNormParams,
    thread_count: usize,
) c_int;
pub extern fn zdraw_metal_run_gemm(
    queue: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
) c_int;
pub extern fn zdraw_metal_run_gemm_mps(
    queue: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
) c_int;
pub extern fn zdraw_metal_run_gemm_pair(
    queue: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w0: *anyopaque,
    c0: *anyopaque,
    params0: *const GemmParams,
    w1: *anyopaque,
    c1: *anyopaque,
    params1: *const GemmParams,
) c_int;
pub extern fn zdraw_metal_run_gemm_triple(
    queue: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w0: *anyopaque,
    c0: *anyopaque,
    params0: *const GemmParams,
    w1: *anyopaque,
    c1: *anyopaque,
    params1: *const GemmParams,
    w2: *anyopaque,
    c2: *anyopaque,
    params2: *const GemmParams,
) c_int;
pub extern fn zdraw_metal_run_ffn(
    queue: *anyopaque,
    gemm_pipeline: *anyopaque,
    swiglu_pipeline: *anyopaque,
    input: *anyopaque,
    gate_weight: *anyopaque,
    up_weight: *anyopaque,
    down_weight: *anyopaque,
    gate: *anyopaque,
    up: *anyopaque,
    out: *anyopaque,
    gate_params: *const GemmParams,
    up_params: *const GemmParams,
    down_params: *const GemmParams,
    swiglu_count: u32,
    swiglu_threads: usize,
) c_int;

pub const LinearParams = extern struct {
    rows: u32,
    cols: u32,
    batch: u32,
    dtype: u32,
    bias_dtype: u32,
    has_bias: u32,
    pad: u32 = 0,
    weight_offset: u64,
    bias_offset: u64,
};

pub const ConvParams = extern struct {
    in_ch: u32,
    out_ch: u32,
    height: u32,
    width: u32,
    ksize: u32,
    pad: u32,
    dtype: u32,
    bias_dtype: u32,
    has_bias: u32,
    pad1: u32 = 0,
    weight_offset: u64,
    bias_offset: u64,
};

pub const AttnParams = extern struct {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    causal: u32,
    /// Right-padding limit: keys at index >= valid are masked for every query.
    /// 0 = no padding. Only `attention_rows` declares this field on the MSL
    /// side; every other attention kernel keeps its 5-field struct and reads
    /// the same prefix, so their behaviour is untouched.
    valid: u32 = 0,
};

pub const QkNormParams = extern struct {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    q_dtype: u32,
    k_dtype: u32,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    eps: f32,
    q_offset: u64,
    k_offset: u64,
    base0: u64,
    base1: u64,
    base2: u64,
};

pub const GemmParams = extern struct {
    m: u32,
    k: u32,
    n: u32,
    dtype: u32,
    mode: u32 = 0,
    weight_offset: u64 = 0,
};

// Resident-batch ABI (one command buffer across many encodes).
pub extern fn zdraw_metal_batch_begin(queue: *anyopaque) ?*anyopaque;
pub extern fn zdraw_metal_run_attention_enc(
    batch: *anyopaque,
    pipeline: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    o: *anyopaque,
    params: *const AttnParams,
    kernel: u32,
    thread_count: usize,
) c_int;
pub extern fn zdraw_metal_run_attention_enc_off(
    batch: *anyopaque,
    pipeline: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    o: *anyopaque,
    params: *const AttnParams,
    kernel: u32,
    thread_count: usize,
    in_off: u64,
    out_off: u64,
) c_int;
pub extern fn zdraw_metal_run_attention_mfa_enc(
    batch: *anyopaque,
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    output: *anyopaque,
    params: *const AttnParams,
    in_off: usize,
    out_off: usize,
) c_int;
pub extern fn zdraw_metal_attn_hm_scratch(
    device: *anyopaque,
    slot: c_int,
    bytes: usize,
) ?*anyopaque;
pub extern fn zdraw_metal_run_attention_mfa_hm_enc(
    batch: *anyopaque,
    qhm: *anyopaque,
    khm: *anyopaque,
    v: *anyopaque,
    output: *anyopaque,
    params: *const AttnParams,
    in_off: usize,
    out_off: usize,
    keep_o_hm: c_int,
) c_int;
pub extern fn zdraw_metal_run_attention_steel_hm_enc(
    batch: *anyopaque,
    qhm: *anyopaque,
    khm: *anyopaque,
    vhm: *anyopaque,
    o_hm: *anyopaque,
    params: *const AttnParams,
) c_int;
pub extern fn zdraw_metal_run_attention_steel_route_enc(
    batch: *anyopaque,
    qhm: *anyopaque,
    khm: *anyopaque,
    v: *anyopaque,
    output: *anyopaque,
    params: *const AttnParams,
    keep_o_hm: c_int,
    in_off: usize,
    out_off: usize,
) c_int;
pub extern fn zdraw_metal_batch_end(batch: *anyopaque) c_int;
pub extern fn zdraw_metal_run_gemm_enc(
    batch: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
) c_int;
// P6: Klein double-block GEMM with A/C byte offsets (setBuffer:offset:, not
// struct fields — GemmParams ABI unchanged). Writes/reads a stream's rows at an
// offset so the double block needs no qkv/o staging buffers.
pub extern fn zdraw_metal_run_gemm_off_enc(
    batch: *anyopaque,
    pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    a_offset: u64,
    c_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_ours16_enc(
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    a_offset: u64,
    c_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_w6_enc(
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    a_offset: u64,
    c_offset: u64,
    scales_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_w4_enc(
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    a_offset: u64,
    c_offset: u64,
    scales_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_steel_enc(
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    a_offset: u64,
    c_offset: u64,
    scales_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_w6_split(
    queue: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    scales_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_f16a_enc(
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
    a_offset: u64,
    c_offset: u64,
) c_int;
pub extern fn zdraw_metal_run_gemm_mps_enc(
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    params: *const GemmParams,
) c_int;
pub extern fn zdraw_metal_run_glue_enc(
    batch: *anyopaque,
    pipeline: *anyopaque,
    b0: ?*anyopaque,
    b1: ?*anyopaque,
    b2: ?*anyopaque,
    b3: ?*anyopaque,
    b4: ?*anyopaque,
    offsets: ?[*]const usize,
    cbytes: ?*const anyopaque,
    cbytes_len: usize,
    cbytes_index: u32,
    grid: usize,
    tg_threads: usize,
    grid_is_threadgroups: u32,
) c_int;
pub extern fn zdraw_metal_run_glue(
    queue: *anyopaque,
    pipeline: *anyopaque,
    b0: ?*anyopaque,
    b1: ?*anyopaque,
    b2: ?*anyopaque,
    b3: ?*anyopaque,
    b4: ?*anyopaque,
    offsets: ?[*]const usize,
    cbytes: ?*const anyopaque,
    cbytes_len: usize,
    cbytes_index: u32,
    grid: usize,
    tg_threads: usize,
    grid_is_threadgroups: u32,
) c_int;
