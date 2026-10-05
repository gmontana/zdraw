// SPDX-License-Identifier: Apache-2.0
// Metal bridge for zdraw.
//
// Zig owns the model code. This file turns Objective-C Metal calls into C
// functions with plain opaque pointers. It is deliberately one file (split
// deferred by ruling); sections appear in this order:
//
//   1. device/queue/buffer lifecycle, metrics counters, process memory
//   2. generic runners: linear, conv, SDPA/attention, qk-norm-rope, GEMM,
//      swiglu/ffn (queue-per-call ABI)
//   3. resident batch ABI (batch_begin/end) and _enc encoders that ride an
//      open command buffer, including attention offset variants
//   4. vendored steel GEMM loaders (metallib at ZDRAW_STEEL_LIB) and the
//      quarantined steel-W6 route
//   5. probes
//   6. native GEMM pipeline family: gemm_half source, gemm_f16_direct
//      (ours16), f16a, f32_staged, W6 staged/split, and their _enc entries
//   7. conv/VAE window kernels: conv2d_window family, prenorm/upsample
//      variants (v1/v3/v7), norm stats/apply, add, f16 tail kernels
//   8. streamed VAE res-chain encoder (encode_vae_res_stream_block and the
//      chain entry) and stream buffer structs
//
// Extern declarations live once in src/metal_c.zig (or the single consumer
// module); never redeclare them locally elsewhere (ABI-drift segfault class).

#import <Foundation/Foundation.h>
#include "zdraw_abi.h"
#include <sys/sysctl.h>
#include <mach-o/dyld.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <mach/mach.h>
#import <mach/mach_time.h>
#import <objc/runtime.h>
extern void* objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void*);
#import <mach/task_info.h>
#import <os/lock.h>
#import <stddef.h>
#import <stdint.h>
#import <stdlib.h>
#import <string.h>
#import <sys/mman.h>
#import <sys/resource.h>
#import <unistd.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t batch;
    uint32_t dtype;
    uint32_t bias_dtype;
    uint32_t has_bias;
    uint32_t pad;
    uint64_t weight_offset;
    uint64_t bias_offset;
} ZdrawLinearParams;

typedef struct {
    uint32_t in_ch;
    uint32_t out_ch;
    uint32_t height;
    uint32_t width;
    uint32_t ksize;
    uint32_t pad;
    uint32_t dtype;
    uint32_t bias_dtype;
    uint32_t has_bias;
    uint32_t pad1;
    uint64_t weight_offset;
    uint64_t bias_offset;
} ZdrawConvParams;

// Exact-streaming windowed conv params (mirrors MSL ConvWindowParams):
// ZdrawConvParams fields + the contiguous output row-strip [row0, row1).
typedef struct {
    uint32_t in_ch;
    uint32_t out_ch;
    uint32_t height;
    uint32_t width;
    uint32_t ksize;
    uint32_t pad;
    uint32_t dtype;
    uint32_t bias_dtype;
    uint32_t has_bias;
    uint32_t pad1;
    uint64_t weight_offset;
    uint64_t bias_offset;
    uint32_t row0;
    uint32_t row1;
} ZdrawConvWindowParams;

// conv2d_strip_in params (mirrors MSL ConvStripParams): the windowed conv
// fields plus the strip-local input rows [in_row0, in_row0+in_rows).
typedef struct {
    uint32_t in_ch;
    uint32_t out_ch;
    uint32_t height;
    uint32_t width;
    uint32_t ksize;
    uint32_t pad;
    uint32_t dtype;
    uint32_t bias_dtype;
    uint32_t has_bias;
    uint32_t pad1;
    uint64_t weight_offset;
    uint64_t bias_offset;
    uint32_t row0;
    uint32_t row1;
    uint32_t in_row0;
    uint32_t in_rows;
} ZdrawConvStripParams;

// Exact-streaming windowed conv with GroupNorm+SiLU fused into the input read
// (mirrors MSL ConvPrenormWindowParams). The conv fields match ZdrawConvWindow
// (minus pad1), plus the norm group count and the norm weight/bias dtype+offset.
typedef struct {
    uint32_t in_ch;
    uint32_t out_ch;
    uint32_t height;
    uint32_t width;
    uint32_t ksize;
    uint32_t pad;
    uint32_t dtype;
    uint32_t bias_dtype;
    uint32_t has_bias;
    uint32_t groups;
    uint64_t weight_offset;
    uint64_t bias_offset;
    uint32_t row0;
    uint32_t row1;
    uint32_t norm_dtype;
    uint32_t norm_bias_dtype;
    uint64_t norm_weight_offset;
    uint64_t norm_bias_offset;
} ZdrawConvPrenormWindowParams;

// Exact-streaming windowed conv with nearest 2x upsample fused into the input
// read (mirrors MSL ConvUpsampleWindowParams). out_height/width are 2x the
// in_height/width; the conv loads the low-res input via the upsample index map,
// so no full 2x intermediate is materialized.
typedef struct {
    uint32_t channels;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t in_height;
    uint32_t in_width;
    uint32_t ksize;
    uint32_t pad;
    uint32_t dtype;
    uint32_t bias_dtype;
    uint32_t has_bias;
    uint32_t pad1;
    uint64_t weight_offset;
    uint64_t bias_offset;
    uint32_t row0;
    uint32_t row1;
} ZdrawConvUpsampleWindowParams;

typedef struct {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
} ZdrawUpParams;

typedef struct {
    void* input; void* high; void* weight; void* bias; void* output;
} ZdrawVaeUpBuffers;

typedef struct {
    ZdrawUpParams up;
    ZdrawConvParams conv;
} ZdrawVaeUpParams;

typedef struct {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t groups;
    uint32_t dtype;
    uint32_t bias_dtype;
    float eps;
    uint64_t weight_offset;
    uint64_t bias_offset;
} ZdrawVaeNormParams;

// Exact-streaming windowed apply params (mirrors MSL NormWindowParams).
typedef struct {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t groups;
    uint32_t dtype;
    uint32_t bias_dtype;
    float eps;
    uint64_t weight_offset;
    uint64_t bias_offset;
    uint32_t row0;
    uint32_t row1;
    uint32_t col0;
    uint32_t col1;
} ZdrawVaeNormWindowParams;

// Exact-streaming windowed residual add params (mirrors MSL AddWindowParams):
// add residual to output over the contiguous output row-strip [row0, row1).
typedef struct {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t row0;
    uint32_t row1;
} ZdrawVaeAddWindowParams;

typedef struct {
    void* input; void* norm; void* norm_w; void* norm_b;
    void* conv_w; void* conv_b; void* output;
} ZdrawVaeFinalBuffers;

typedef struct {
    ZdrawVaeNormParams norm;
    ZdrawConvParams conv;
} ZdrawVaeFinalParams;

typedef struct {
    void* input; void* norm; void* work; void* output; void* skip;
    void* norm1_w; void* norm1_b; void* conv1_w; void* conv1_b;
    void* norm2_w; void* norm2_b; void* conv2_w; void* conv2_b;
    void* skip_w; void* skip_b;
} ZdrawVaeResBuffers;

typedef struct {
    ZdrawVaeNormParams norm1;
    ZdrawConvParams conv1;
    ZdrawVaeNormParams norm2;
    ZdrawConvParams conv2;
    ZdrawConvParams skip;
    uint32_t has_skip;
    uint32_t out_count;
} ZdrawVaeResParams;

// Resident exact-streaming resblock chain: per-block GPU buffers (input/output
// ping-pong owned by the caller) + shared scratch (conv1_out full, two stats
// buffers, optional skip). The convs fuse GroupNorm+SiLU, so no full normed
// buffer is materialized. Weight handles are the bound shard slices (dtype +
// offset live in ZdrawVaeResParams).
typedef struct {
    void* input;       // residual feature in (in_ch * hw)
    void* output;      // resblock result out (out_ch * hw)
    void* conv1_out;   // conv1 result, read globally by stats2 + conv2 halo
    void* stats1;      // [mean, scale] per group, norm1
    void* stats2;      // [mean, scale] per group, norm2
    void* skip;        // 1x1 skip projection (out_ch * hw); ignored if !has_skip
    void* norm1_w; void* norm1_b; void* conv1_w; void* conv1_b;
    void* norm2_w; void* norm2_b; void* conv2_w; void* conv2_b;
    void* skip_w; void* skip_b;
    void* norm;        // normed+SiLU scratch (max(in,out)_ch * hw); unfuse path
} ZdrawVaeResStreamBuffers;

typedef struct {
    uint32_t tokens;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t causal;
    // Right-padding limit: keys at index >= valid are masked for every query
    // (0 = none). Only attention_rows declares it MSL-side; the other kernels
    // keep the 5-field struct and read the same prefix. This field MUST stay
    // in step with metal_c.AttnParams - setBytes uploads sizeof(this), so a
    // missing field here silently feeds the kernel garbage.
    uint32_t valid;
} ZdrawAttnParams;

typedef struct {
    uint32_t tokens;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    float eps;
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t base0;
    uint64_t base1;
    uint64_t base2;
} ZdrawQkNormParams;

typedef struct {
    uint32_t tokens;
    uint32_t hidden;
    uint32_t dtype;
    uint32_t has_scale;
    float eps;
    uint64_t weight_offset;
} ZdrawBlockParams;

typedef struct {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t dtype;
    uint32_t mode;
    uint64_t weight_offset;
} GemmParams;

typedef struct {
    uint32_t n;
    uint32_t dtype;
    uint64_t bias_offset;
} BiasParams;

static void count_gemm(const GemmParams* params);

typedef struct {
    uint32_t tokens;
    uint32_t hidden;
    float eps;
    uint32_t pad;
} FinalNormParams;

typedef struct {
    void* scale;
    void* weight;
    void* bias;
    void* batch;
    void* output;
} FinalBuffers;

// Lightweight metrics for the benchmark harness (single-threaded, no atomics).
static uint64_t g_live = 0;     // live MTLBuffer bytes
static uint64_t g_peak = 0;     // peak live bytes
static uint64_t g_weights = 0;  // no-copy (mmap'd weight) bytes wrapped
static uint64_t g_dispatch = 0; // Metal kernel dispatches
static uint64_t g_command = 0;  // command buffers committed
static uint64_t g_wait = 0;     // command-buffer completion waits
static uint64_t g_readback = 0; // GPU->host buffer reads
static uint64_t g_gemm = 0;     // simdgroup GEMM dispatches (subset of g_dispatch)
static uint64_t g_gemm_exact = 0;
static uint64_t g_gemm_half = 0;
static uint64_t g_gemm_w8 = 0;
static uint64_t g_gemm_w6 = 0;
static uint64_t g_gemm_w4 = 0;
static uint64_t g_gemm_w2 = 0;
static uint64_t g_gemm_mps = 0;
static uint64_t g_gemm_mps_fallback = 0;
static uint64_t g_attn_steel_fallback = 0; // steel attention asked for, kernel unavailable -> MFA ran
static uint64_t g_gemm_mpp_fallback = 0;   // Metal 4 tensor GEMM asked for, unavailable -> direct kernel ran
static double g_gpu_seconds = 0.0; // summed GPUEndTime-GPUStartTime over waits

enum {
    SP_ATTN_NORM,
    SP_QKV,
    SP_QK_ROPE,
    SP_ATTENTION,
    SP_PROJ,
    SP_ATTN_RESID_NORM,
    SP_FFN_GATEUP,
    SP_SWIGLU,
    SP_FFN_DOWN,
    SP_FFN_RESID,
    SP_FINAL_NORM,
    SP_FINAL_PROJ,
    SP_FINAL_BIAS,
    SP_COUNT,
};

static const char* g_stack_profile_names[SP_COUNT] = {
    "attn-norm",
    "qkv-gemm",
    "qk-rope",
    "attention",
    "proj-gemm",
    "attn-resid-norm",
    "ffn-gateup",
    "swiglu",
    "ffn-down",
    "ffn-resid",
    "final-norm",
    "final-proj",
    "final-bias",
};
static uint64_t g_stack_profile_ns[SP_COUNT] = {0};
static uint64_t g_stack_profile_samples[SP_COUNT] = {0};
static double g_stack_profile_gpu[SP_COUNT] = {0};

static void metrics_add(uint64_t size) {
    g_live += size;
    if (g_live > g_peak) g_peak = g_live;
}

static NSMutableDictionary<NSString*, id<MTLBuffer>>* g_conv_slice_cache = nil;
static uint64_t g_conv_slice_bytes = 0;

static NSMutableDictionary<NSString*, id<MTLBuffer>>* conv_slice_cache(void) {
    if (!g_conv_slice_cache) g_conv_slice_cache = [[NSMutableDictionary alloc] init];
    return g_conv_slice_cache;
}

void zdraw_metal_clear_transient_caches(void) {
    if (!g_conv_slice_cache) return;
    if (g_conv_slice_bytes > 0) {
        g_live = (g_conv_slice_bytes <= g_live) ? (g_live - g_conv_slice_bytes) : 0;
        g_conv_slice_bytes = 0;
    }
    [g_conv_slice_cache removeAllObjects];
}

static void stack_profile_reset(void) {
    memset(g_stack_profile_ns, 0, sizeof(g_stack_profile_ns));
    memset(g_stack_profile_samples, 0, sizeof(g_stack_profile_samples));
    memset(g_stack_profile_gpu, 0, sizeof(g_stack_profile_gpu));
}

// Wait for a command buffer and bank its true GPU execution time.
static double cmd_gpu_seconds(id<MTLCommandBuffer> cmd) {
    double span = cmd.GPUEndTime - cmd.GPUStartTime;
    return span > 0.0 ? span : 0.0;
}

static void wait_cmd(id<MTLCommandBuffer> cmd) {
    g_wait += 1;
    // Sole direct wait on product paths. Bench probes wait bare deliberately and
    // read GPU timing themselves: zdraw_metal_ab_probe, zdraw_metal_mix_probe,
    // zdraw_metal_frag_map_probe, zdraw_metal_chain_probe (also the bench-only
    // zdraw_metal_steel_w6_run_cfg and run_stack_final's ZDRAW_MIX_PROBE block).
    [cmd waitUntilCompleted];
    g_gpu_seconds += cmd_gpu_seconds(cmd);
}

void zdraw_metal_metrics_reset(void) {
    g_peak = g_live; g_dispatch = 0;
    g_command = 0; g_wait = 0; g_readback = 0;
    g_gemm = 0; g_gemm_exact = 0; g_gemm_half = 0; g_gemm_w8 = 0; g_gemm_w6 = 0; g_gemm_w4 = 0; g_gemm_w2 = 0;
    g_gemm_mps = 0; g_gemm_mps_fallback = 0; g_attn_steel_fallback = 0; g_gemm_mpp_fallback = 0;
    g_gpu_seconds = 0.0;
    stack_profile_reset();
}
uint64_t zdraw_metal_peak_bytes(void) { return g_peak; }
uint64_t zdraw_metal_live_bytes(void) { return g_live; }
uint64_t zdraw_metal_weight_bytes(void) { return g_weights; }
uint64_t zdraw_metal_dispatch_count(void) { return g_dispatch; }
uint64_t zdraw_metal_command_count(void) { return g_command; }
uint64_t zdraw_metal_wait_count(void) { return g_wait; }
uint64_t zdraw_metal_readback_count(void) { return g_readback; }
uint64_t zdraw_metal_gemm_count(void) { return g_gemm; }
uint64_t zdraw_metal_gemm_exact_count(void) { return g_gemm_exact; }
uint64_t zdraw_metal_gemm_half_count(void) { return g_gemm_half; }
uint64_t zdraw_metal_gemm_w8_count(void) { return g_gemm_w8; }
uint64_t zdraw_metal_gemm_w6_count(void) { return g_gemm_w6; }
uint64_t zdraw_metal_gemm_w4_count(void) { return g_gemm_w4; }
uint64_t zdraw_metal_gemm_w2_count(void) { return g_gemm_w2; }
uint64_t zdraw_metal_gemm_mps_count(void) { return g_gemm_mps; }
uint64_t zdraw_metal_gemm_mps_fallback_count(void) { return g_gemm_mps_fallback; }
uint64_t zdraw_metal_attn_steel_fallback_count(void) { return g_attn_steel_fallback; }
uint64_t zdraw_metal_gemm_mpp_fallback_count(void) { return g_gemm_mpp_fallback; }
void zdraw_metal_note_mpp_fallback(void) { g_gemm_mpp_fallback++; }
void zdraw_metal_warn_mpp_unavailable(void) {
    fprintf(stderr, "zdraw: WARNING Metal 4 tensor GEMM unavailable (needs macOS 26); Klein GEMMs "
                    "run on the direct kernel: NOT the default route\n");
}
uint64_t zdraw_metal_stack_profile_count(void) { return SP_COUNT; }
const char* zdraw_metal_stack_profile_name(uint64_t i) {
    return i < SP_COUNT ? g_stack_profile_names[i] : "";
}
uint64_t zdraw_metal_stack_profile_ns(uint64_t i) {
    return i < SP_COUNT ? g_stack_profile_ns[i] : 0;
}
uint64_t zdraw_metal_stack_profile_samples(uint64_t i) {
    return i < SP_COUNT ? g_stack_profile_samples[i] : 0;
}
uint64_t zdraw_metal_gpu_ns(void) { return (uint64_t)(g_gpu_seconds * 1e9); }
uint64_t zdraw_metal_stack_profile_gpu_ns(uint64_t i) {
    return i < SP_COUNT ? (uint64_t)(g_stack_profile_gpu[i] * 1e9) : 0;
}

uint64_t zdraw_proc_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
    return (uint64_t)usage.ru_maxrss; // bytes on macOS
}

// Activity Monitor's "Memory" number. Like RSS it excludes clean file-backed
// pages, so memory-mapped weights appear in neither figure (measured: a 2.8 GB
// W6 pack and the 7.4 GB W16 pack give identical RSS and footprint, ledger
// klein-memory-panel-20260826); what it does charge is every zero-filled pool
// buffer, which is why pool capacity is the engine's memory authority.
uint64_t zdraw_proc_phys_footprint_bytes(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t)info.phys_footprint;
}

// Device identity for profile selection: chip name, GPU core count proxy
// (Metal working-set hints), RAM, and OS — the axes every measured policy
// is conditional on.
int zdraw_device_identity(
    char* name_out,
    size_t name_len,
    uint64_t* ram_bytes,
    uint64_t* gpu_working_set,
    uint32_t* os_major
) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return -1;
    strlcpy(name_out, device.name.UTF8String, name_len);
    *gpu_working_set = device.recommendedMaxWorkingSetSize;
    size_t len = sizeof(uint64_t);
    if (sysctlbyname("hw.memsize", ram_bytes, &len, NULL, 0) != 0) *ram_bytes = 0;
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    *os_major = (uint32_t)v.majorVersion;
    return 0;
}

int zdraw_thermal_state(void) {
    if (@available(macOS 10.10.3, *)) {
        return (int)[[NSProcessInfo processInfo] thermalState];
    }
    return -1;
}

uint64_t zdraw_now_ns(void) {
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
}

void* zdraw_metal_create_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)device;
}

void zdraw_metal_release_device(void* device) {
    if (!device) return;
    id<MTLDevice> value = (__bridge_transfer id<MTLDevice>)device;
    (void)value;
}

void* zdraw_metal_create_queue(void* device) {
    id<MTLDevice> value = (__bridge id<MTLDevice>)device;
    return (__bridge_retained void*)[value newCommandQueue];
}

void zdraw_metal_release_queue(void* queue) {
    if (!queue) return;
    id<MTLCommandQueue> value = (__bridge_transfer id<MTLCommandQueue>)queue;
    (void)value;
}

void* zdraw_metal_create_buffer(void* device, size_t size) {
    id<MTLDevice> value = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buffer = [value newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (buffer) {
        memset([buffer contents], 0, size);
        metrics_add(size);
    }
    return (__bridge_retained void*)buffer;
}

void* zdraw_metal_create_buffer_with_data(void* device, const void* data, size_t size) {
    id<MTLDevice> value = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buffer = [value newBufferWithBytes:data
                                              length:size
                                             options:MTLResourceStorageModeShared];
    if (buffer) metrics_add(size);
    return (__bridge_retained void*)buffer;
}

// Registry of the no-copy weight mappings for zdraw_proc_mapped_resident_bytes.
#define MAPPING_MAX 4096
struct mapping { const void* ptr; size_t len; };
static struct mapping g_mappings[MAPPING_MAX];
static int g_mapping_n = 0;

static void mapping_register(const void* ptr, size_t len) {
    if (!ptr || g_mapping_n >= MAPPING_MAX) return;
    g_mappings[g_mapping_n].ptr = ptr;
    g_mappings[g_mapping_n].len = len;
    g_mapping_n++;
}

static void mapping_unregister(const void* ptr) {
    for (int i = 0; i < g_mapping_n; i++) {
        if (g_mappings[i].ptr == ptr) {
            g_mappings[i] = g_mappings[g_mapping_n - 1];
            g_mapping_n--;
            return;
        }
    }
}

void* zdraw_metal_create_buffer_no_copy(void* device, const void* data, size_t size) {
    id<MTLDevice> value = (__bridge id<MTLDevice>)device;
    // A whole W16 sidecar binds as ONE buffer (the 9B pack is ~18 GiB); fail
    // loudly rather than let Metal reject it downstream with a nil buffer.
    if (size > value.maxBufferLength) {
        fprintf(stderr, "zdraw: buffer %zu bytes exceeds device maxBufferLength %lu\n",
                size, (unsigned long)value.maxBufferLength);
        return NULL;
    }
    // Residency probe: ZDRAW_COPY_WEIGHTS=1 trades RAM for copied weight
    // buffers to measure whether mmap-backed reads throttle the GPU.
    if (getenv("ZDRAW_COPY_WEIGHTS")) {
        id<MTLBuffer> copied = [value newBufferWithBytes:(void*)data
                                                  length:size
                                                 options:MTLResourceStorageModeShared];
        if (copied) g_weights += size;
        return (__bridge_retained void*)copied;
    }
    id<MTLBuffer> buffer = [value newBufferWithBytesNoCopy:(void*)data
                                                    length:size
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    // mmap-shared: counts toward the weight footprint, not a real device alloc.
    if (buffer) {
        g_weights += size;
        mapping_register(data, size);
    }
    return (__bridge_retained void*)buffer;
}

void zdraw_metal_read_buffer(void* buffer, void* dst, size_t size) {
    id<MTLBuffer> value = (__bridge id<MTLBuffer>)buffer;
    memcpy(dst, [value contents], size);
    g_readback++;
}

// Host -> shared-buffer upload into a caller-owned (e.g. pooled) MTLBuffer,
// so a persistent input buffer can be refilled without allocating a fresh one.
void zdraw_metal_write_buffer(void* buffer, const void* src, size_t size) {
    id<MTLBuffer> value = (__bridge id<MTLBuffer>)buffer;
    memcpy([value contents], src, size);
}

void zdraw_metal_release_buffer(void* buffer) {
    if (!buffer) return;
    id<MTLBuffer> value = (__bridge_transfer id<MTLBuffer>)buffer;
    uint64_t len = (uint64_t)[value length];
    g_live = (len <= g_live) ? (g_live - len) : 0;
}

void zdraw_metal_release_weight_buffer(void* buffer) {
    if (!buffer) return;
    id<MTLBuffer> value = (__bridge_transfer id<MTLBuffer>)buffer;
    uint64_t len = (uint64_t)[value length];
    g_weights = (len <= g_weights) ? (g_weights - len) : 0;
    mapping_unregister([value contents]);
}

// Resident pages of the no-copy weight mappings (ledger
// memory-ladder-w0-instrument-20261006). RSS and phys_footprint exclude clean
// file-backed pages, so a 2.8 GB and a 7.4 GB pack give the same footprint
// while the GPU reads every page of them. Every no-copy wrap registers its
// range; this walks the ranges with mincore (merged, so a slice wrap inside
// a whole-sidecar wrap is not counted twice) and sums the resident pages.
static uint64_t g_mapped_peak = 0;  // highest mapped-resident sample
static uint64_t g_total_peak = 0;   // highest phys_footprint + mapped-resident sample

// `stride` = 1 walks every page (exact, about 0.3 s over a 7 GiB mapping);
// a larger stride samples every stride-th page and scales, which the
// always-on phase sampler uses so a render pays milliseconds, not seconds.
static uint64_t mapped_resident_walk(size_t stride) {
    static unsigned char vec[65536];  // 64 Ki pages per window: 1 GiB at 16 KiB pages
    size_t page = (size_t)getpagesize();
    if (page == 0 || stride == 0) return 0;
    // Sort a copy by address and merge overlaps (a few entries; insertion sort).
    struct mapping sorted[MAPPING_MAX];
    int n = g_mapping_n;
    memcpy(sorted, g_mappings, sizeof(struct mapping) * (size_t)n);
    for (int i = 1; i < n; i++) {
        struct mapping key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j].ptr > key.ptr) { sorted[j + 1] = sorted[j]; j--; }
        sorted[j + 1] = key;
    }
    uint64_t resident = 0;
    const unsigned char* cursor = NULL;  // end of the last counted range
    for (int i = 0; i < n; i++) {
        const unsigned char* base = (const unsigned char*)sorted[i].ptr;
        const unsigned char* end = base + sorted[i].len;
        if (cursor && base < cursor) base = cursor;
        if (base >= end) continue;
        // Align the start down to a page (mincore needs page-aligned addresses).
        uintptr_t aligned = (uintptr_t)base & ~(uintptr_t)(page - 1);
        base = (const unsigned char*)aligned;
        while (base < end) {
            size_t win = (size_t)(end - base);
            if (win > sizeof(vec) * page) win = sizeof(vec) * page;
            if (stride == 1) {
                if (mincore((void*)base, win, (char*)vec) != 0) break;
                size_t pages = (win + page - 1) / page;
                for (size_t p = 0; p < pages; p++) if (vec[p] & MINCORE_INCORE) resident += page;
            } else {
                // One page per stride: query single pages and scale.
                size_t pages = (win + page - 1) / page;
                uint64_t hit = 0, asked = 0;
                for (size_t p = 0; p < pages; p += stride) {
                    unsigned char one = 0;
                    if (mincore((void*)(base + p * page), page, (char*)&one) != 0) break;
                    asked++;
                    if (one & MINCORE_INCORE) hit++;
                }
                if (asked) resident += (uint64_t)((double)hit / (double)asked * (double)pages * (double)page);
            }
            base += win;
        }
        cursor = end;
    }
    if (resident > g_mapped_peak) g_mapped_peak = resident;
    uint64_t total = resident + zdraw_proc_phys_footprint_bytes();
    if (total > g_total_peak) g_total_peak = total;
    return resident;
}

uint64_t zdraw_proc_mapped_resident_bytes(void) { return mapped_resident_walk(1); }

// Sample the mapping residency (updates the peaks) without printing; the
// pipeline calls this at every phase boundary so the card can report the
// peak even when ZDRAW_MEMTRACE is off. Every 256th page is queried, so a
// 7 GiB mapping costs about a millisecond rather than a third of a second.
void zdraw_proc_sample_memory(void) { (void)mapped_resident_walk(256); }
uint64_t zdraw_proc_mapped_resident_peak_bytes(void) { return g_mapped_peak; }
uint64_t zdraw_proc_total_peak_bytes(void) { return g_total_peak; }

// Math precision: default RELAXED honors INF/NaN
// (our attention/softmax/VAE produce large magnitudes; .fast makes those
// UB - relevant to the f16-overflow history) at ~fast speed. ZDRAW_MATH=
// fast (peak ALU), relaxed (default), safe (correctly-rounded div/sqrt +
// precise transcendentals, closest to the diffusers reference).
static MTLCompileOptions* zdraw_compile_options(void) {
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    const char* env = getenv("ZDRAW_MATH");
    const int fast = env && strcmp(env, "fast") == 0;
    const int safe = env && strcmp(env, "safe") == 0;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        options.mathMode = fast ? MTLMathModeFast
                                : (safe ? MTLMathModeSafe : MTLMathModeRelaxed);
        options.mathFloatingPointFunctions =
            safe ? MTLMathFloatingPointFunctionsPrecise : MTLMathFloatingPointFunctionsFast;
        return options;
    }
#endif
    // Pre-15.0 fallback; dead on macOS >= 15 (the @available branch returns
    // above). fastMathEnabled is deprecated on the 15+ SDK, so silence it here.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = safe ? NO : YES;
#pragma clang diagnostic pop
    return options;
}

void* zdraw_metal_compile(
    void* device,
    const char* source,
    const char* entry,
    char* error_buf,
    size_t error_len
) {
    const uint64_t t0 = zdraw_now_ns();
    id<MTLDevice> value = (__bridge id<MTLDevice>)device;
    NSString* text = [NSString stringWithUTF8String:source];
    NSString* name = [NSString stringWithUTF8String:entry];

    NSError* error = nil;
    id<MTLLibrary> library =
        [value newLibraryWithSource:text options:zdraw_compile_options() error:&error];
    if (!library) {
        if (error_buf && error_len > 0) {
            const char* msg = [[error localizedDescription] UTF8String];
            strncpy(error_buf, msg ? msg : "unknown Metal error", error_len - 1);
            error_buf[error_len - 1] = '\0';
        }
        return nil;
    }

    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) return nil;
    id<MTLComputePipelineState> pipe = [value newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (getenv("ZDRAW_COMPILE_TIMES")) {
        fprintf(stderr, "compile %s: %.0f ms\n", entry,
                (double)(zdraw_now_ns() - t0) / 1e6);
    }
    if (!pipe && error_buf && error_len > 0) {
        const char* msg = [[error localizedDescription] UTF8String];
        strncpy(error_buf, msg ? msg : "unknown Metal error", error_len - 1);
        error_buf[error_len - 1] = '\0';
    }
    return (__bridge_retained void*)pipe;
}

void zdraw_metal_release_pipeline(void* pipeline) {
    if (!pipeline) return;
    id<MTLComputePipelineState> value = (__bridge_transfer id<MTLComputePipelineState>)pipeline;
    (void)value;
}

size_t zdraw_metal_pipeline_threads(void* pipeline) {
    id<MTLComputePipelineState> value = (__bridge id<MTLComputePipelineState>)pipeline;
    return [value maxTotalThreadsPerThreadgroup];
}

int zdraw_metal_run_generated_1d(
    void* queue,
    void* pipeline,
    void* const* buffers,
    size_t buffer_count,
    uint32_t element_count,
    size_t thread_count
) {
    if (!queue || !pipeline || !buffers || buffer_count == 0 ||
        buffer_count > 30 || element_count == 0 || thread_count == 0) {
        return -1;
    }
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pipe];
    for (size_t index = 0; index < buffer_count; index++) {
        if (!buffers[index]) {
            [enc endEncoding];
            return -1;
        }
        [enc setBuffer:(__bridge id<MTLBuffer>)buffers[index] offset:0 atIndex:index];
    }
    [enc setBytes:&element_count length:sizeof(element_count) atIndex:buffer_count];
    const size_t maximum = [pipe maxTotalThreadsPerThreadgroup];
    const size_t width = thread_count < maximum ? thread_count : maximum;
    const size_t groups = ((size_t)element_count + width - 1) / width;
    g_dispatch++;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_linear(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawLinearParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawLinearParams) atIndex:4];
    g_dispatch++;

    size_t batch_tiles = ((size_t)params->batch + 3u) / 4u;
    MTLSize groups = MTLSizeMake((size_t)params->rows * batch_tiles, 1, 1);
    MTLSize threads = MTLSizeMake(thread_count, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_mods(
    void* queue,
    void* pipeline,
    void* input,
    void* const* weights,
    void* const* biases,
    void* output,
    const ZdrawLinearParams* params,
    size_t count,
    size_t output_stride,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)input;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)output;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    [enc setComputePipelineState:pipe];
    [enc setBuffer:in_buf offset:0 atIndex:0];
    for (size_t i = 0; i < count; i++) {
        [enc setBuffer:(__bridge id<MTLBuffer>)weights[i] offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)biases[i] offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:i * output_stride atIndex:3];
        [enc setBytes:&params[i] length:sizeof(ZdrawLinearParams) atIndex:4];
        size_t batch_tiles = ((size_t)params[i].batch + 3u) / 4u;
        MTLSize groups = MTLSizeMake((size_t)params[i].rows * batch_tiles, 1, 1);
        MTLSize threads = MTLSizeMake(thread_count, 1, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    }
    [enc endEncoding];

    g_dispatch += count;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static void encode_conv(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvParams* params
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawConvParams) atIndex:4];
    size_t hw = (size_t)params->height * (size_t)params->width;
    MTLSize groups = MTLSizeMake((hw + 31u) / 32u, ((size_t)params->out_ch + 31u) / 32u, 1);
    MTLSize threads = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

int zdraw_metal_run_conv(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvParams* params,
    size_t thread_count
) {
    // Per-call pool: the command buffer is autoreleased and retains the
    // caller's transient in/out/weight buffers; without a pool in a CLI
    // process it never drains, and a 1024x1024 reference encode leaked
    // ~5.6 GB per photo (memorystatus killed serve after 84 edits).
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) return -1;

        encode_conv(enc, pipe, input, weight, bias, output, params);
        [enc endEncoding];
        g_dispatch++;
        g_command++;
        [cmd commit];
        wait_cmd(cmd);
        return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
    }
}

// Request-level autorelease pools for the Zig callers (serve loop, generate):
// everything Metal autoreleases during a request drains when the request ends.
void* zdraw_metal_pool_push(void) { return objc_autoreleasePoolPush(); }
void zdraw_metal_pool_pop(void* pool) { objc_autoreleasePoolPop(pool); }

// Exact-streaming windowed conv: conv2d restricted to the output row-strip
// [row0,row1). tg.x covers the strip's flat positions, tg.y the out-channel
// tile - everything else (input reads, bounds check, MMA, global write) is
// identical to conv2d, so it is bit-exact on the strip.
static void encode_conv2d_window(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvWindowParams* params
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawConvWindowParams) atIndex:4];
    size_t strip_rows = (size_t)params->row1 - (size_t)params->row0;
    size_t strip_pos = strip_rows * (size_t)params->width;
    MTLSize groups = MTLSizeMake((strip_pos + 31u) / 32u,
                                 ((size_t)params->out_ch + 31u) / 32u, 1);
    MTLSize threads = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// x4 variant for 4-simdgroup conv kernels (128 threads, four 32-oc tiles per
// threadgroup). Bench/tuning entry; the chain wires it once a kernel wins.
int zdraw_metal_run_conv2d_window_x4(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvWindowParams* params
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawConvWindowParams) atIndex:4];
    size_t strip_rows = (size_t)params->row1 - (size_t)params->row0;
    size_t strip_pos = strip_rows * (size_t)params->width;
    MTLSize groups = MTLSizeMake((strip_pos + 31u) / 32u,
                                 ((size_t)params->out_ch + 127u) / 128u, 1);
    MTLSize threads = MTLSizeMake(128, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// 64-position variant of the x4 dispatch (h4w64 race/integration).
// Resident batch: one command buffer + encoder across many encodes; ends
// with a single commit+wait. Cuts per-dispatch sync for resident model
// forwards (Klein: ~340 dispatches/step).
typedef struct {
    void* cmd; // retained id<MTLCommandBuffer>
    void* enc; // retained id<MTLComputeCommandEncoder>
} ZdrawBatch;

void* zdraw_metal_batch_begin(void* queue) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return NULL;
    ZdrawBatch* b = malloc(sizeof(ZdrawBatch));
    b->cmd = (__bridge_retained void*)cmd;
    b->enc = (__bridge_retained void*)enc;
    return b;
}

int zdraw_metal_batch_end(void* batch) {
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge_transfer id<MTLComputeCommandEncoder>)b->enc;
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    int ok = (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
    free(b);
    return ok;
}

static inline NSUInteger ceil_div_u32(uint32_t value, uint32_t divisor) {
    return ((NSUInteger)value + (NSUInteger)divisor - 1u) / (NSUInteger)divisor;
}

// Shared encode body for the standalone GEMM runner (queue/batch wrappers).
static void gemm_encode(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* a_buf, void* w_buf, void* c_buf,
    const GemmParams* params,
    uint64_t a_off, uint64_t c_off
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:c_off atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    count_gemm(params);
    [enc dispatchThreadgroups:MTLSizeMake(ceil_div_u32(params->n, 32u),
                                          ceil_div_u32(params->m, 32u), 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

int zdraw_metal_run_gemm_enc(
    void* batch,
    void* pipeline,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    ZdrawBatch* b = (ZdrawBatch*)batch;
    gemm_encode((__bridge id<MTLComputeCommandEncoder>)b->enc,
                (__bridge id<MTLComputePipelineState>)pipeline,
                a_buf, w_buf, c_buf, params, 0, 0);
    return 0;
}

// P6: Klein double-block variant. A (input) and C (output) byte offsets let the
// qkv GEMMs write straight into the shared q/k/v at the stream's row offset, and
// the out-proj read o at the stream offset — removing the staging buffers and
// concat copies. GemmParams ABI is unchanged (offsets are setBuffer:offset:, not
// struct fields), so Z-Image's fused chain is untouched.
int zdraw_metal_run_gemm_off_enc(
    void* batch,
    void* pipeline,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off
) {
    ZdrawBatch* b = (ZdrawBatch*)batch;
    gemm_encode((__bridge id<MTLComputeCommandEncoder>)b->enc,
                (__bridge id<MTLComputePipelineState>)pipeline,
                a_buf, w_buf, c_buf, params, a_off, c_off);
    return 0;
}

// Concurrency micro-bench (Klein denoise headroom): dispatch `n` INDEPENDENT
// GEMMs (same a/w, distinct c[i]) into ONE command buffer via a serial or a
// concurrent compute encoder, then commit+wait. The caller times wall around
// this and takes the min over reps (load-robust) to ask: does overlapping the
// independent qkv GEMMs beat running them serially, or does one GEMM already
// saturate the GPU? Concurrent dispatch carries no automatic hazard tracking,
// which is safe here because the c[i] are distinct and a/w are read-only.
// Bench-only; never on a product path.
int zdraw_metal_bench_gemm_fanout(
    void* queue_,
    void* pipeline_,
    void* a_buf,
    void* w_buf,
    void* const* c_bufs,
    const GemmParams* params,
    uint32_t n,
    int concurrent
) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline_;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = concurrent
        ? [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
        : [cmd computeCommandEncoder];
    for (uint32_t i = 0; i < n; i++) {
        gemm_encode(enc, pipe, a_buf, w_buf, c_bufs[i], params, 0, 0);
    }
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -1;
}

// Shared encode body for the generic glue dispatcher (see the public
// queue/batch wrappers below).
static void glue_encode(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* b0, void* b1, void* b2, void* b3, void* b4,
    const size_t* offsets,
    const void* cbytes, size_t cbytes_len, uint32_t cbytes_index,
    size_t grid, size_t tg_threads, uint32_t grid_is_threadgroups
) {
    [enc setComputePipelineState:pipe];
    void* bufs[5] = {b0, b1, b2, b3, b4};
    for (uint32_t i = 0; i < 5; i++) {
        if (bufs[i]) {
            [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i]
                    offset:(offsets ? offsets[i] : 0)
                   atIndex:i];
        }
    }
    if (cbytes && cbytes_len > 0) {
        [enc setBytes:cbytes length:cbytes_len atIndex:cbytes_index];
    }
    if (grid_is_threadgroups) {
        [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
    }
    g_dispatch++;
}

int zdraw_metal_run_glue_enc(
    void* batch,
    void* pipeline,
    void* b0, void* b1, void* b2, void* b3, void* b4,
    const size_t* offsets,
    const void* cbytes, size_t cbytes_len, uint32_t cbytes_index,
    size_t grid, size_t tg_threads, uint32_t grid_is_threadgroups
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    glue_encode((__bridge id<MTLComputeCommandEncoder>)bb->enc,
                (__bridge id<MTLComputePipelineState>)pipeline,
                b0, b1, b2, b3, b4, offsets,
                cbytes, cbytes_len, cbytes_index,
                grid, tg_threads, grid_is_threadgroups);
    return 0;
}

// Generic small-kernel dispatcher for resident model glue (norms, rope,
// swiglu, gated adds): up to five buffers, one constant blob, thread- or
// threadgroup-grid dispatch.
int zdraw_metal_run_glue(
    void* queue,
    void* pipeline,
    void* b0, void* b1, void* b2, void* b3, void* b4,
    const size_t* offsets,
    const void* cbytes, size_t cbytes_len, uint32_t cbytes_index,
    size_t grid, size_t tg_threads, uint32_t grid_is_threadgroups
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    glue_encode(enc, (__bridge id<MTLComputePipelineState>)pipeline,
                b0, b1, b2, b3, b4, offsets,
                cbytes, cbytes_len, cbytes_index,
                grid, tg_threads, grid_is_threadgroups);
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_conv2d_window(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvWindowParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_conv2d_window(enc, pipe, input, weight, bias, output, params);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Exact-streaming windowed conv with GroupNorm+SiLU fused into the input read.
// Identical dispatch shape to encode_conv2d_window; the extra buffers (stats,
// norm weight/bias) feed the inline norm transform applied to each loaded
// residual element before the MAC. Bit-exact vs apply-then-conv2d_window.
static void encode_conv2d_prenorm_window(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    void* stats,
    void* norm_weight,
    void* norm_bias,
    const ZdrawConvPrenormWindowParams* params,
    int oc_x4  // v3 kernel: 4 simdgroups per TG, grid y covers 128 oc each
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawConvPrenormWindowParams) atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)stats offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)norm_weight offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)norm_bias offset:0 atIndex:7];
    size_t strip_rows = (size_t)params->row1 - (size_t)params->row0;
    size_t strip_pos = strip_rows * (size_t)params->width;
    size_t oc_per_tg = oc_x4 ? 128u : 32u;
    size_t pos_per_tg = oc_x4 == 2 ? 64u : 32u;
    MTLSize groups = MTLSizeMake((strip_pos + pos_per_tg - 1u) / pos_per_tg,
                                 ((size_t)params->out_ch + oc_per_tg - 1u) / oc_per_tg, 1);
    MTLSize threads = MTLSizeMake(32, oc_x4 == 2 ? 8 : (oc_x4 ? 4 : 1), 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// Old-signature shim for the standalone (non-chain) prenorm runner.
static void encode_conv2d_prenorm_window_compat(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    void* stats,
    void* norm_weight,
    void* norm_bias,
    const ZdrawConvPrenormWindowParams* params,
    size_t thread_count
) {
    encode_conv2d_prenorm_window(enc, pipe, input, weight, bias, output, stats,
                                 norm_weight, norm_bias, params, 0);
}

// Exact-streaming windowed conv with nearest 2x upsample fused into the input
// read. Same dispatch shape as encode_conv2d_window but over the 2x output grid.
static void encode_conv2d_upsample_window(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvUpsampleWindowParams* params,
    int oc_x4  // v3 kernel: 4 simdgroups per TG, grid y covers 128 oc each
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawConvUpsampleWindowParams) atIndex:4];
    size_t strip_rows = (size_t)params->row1 - (size_t)params->row0;
    size_t strip_pos = strip_rows * (size_t)params->out_width;
    size_t oc_per_tg = oc_x4 ? 128u : 32u;
    size_t pos_per_tg = oc_x4 == 2 ? 64u : 32u;
    MTLSize groups = MTLSizeMake((strip_pos + pos_per_tg - 1u) / pos_per_tg,
                                 ((size_t)params->channels + oc_per_tg - 1u) / oc_per_tg, 1);
    MTLSize threads = MTLSizeMake(32, oc_x4 == 2 ? 8 : (oc_x4 ? 4 : 1), 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

int zdraw_metal_run_conv2d_upsample_window(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvUpsampleWindowParams* params,
    size_t thread_count,
    int oc_x4
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_conv2d_upsample_window(enc, pipe, input, weight, bias, output,
                                  params, oc_x4);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static size_t dtype_size(uint32_t dtype);
static int mps_dtype(uint32_t dtype, MPSDataType* out);
static int mps_gemm_eligible(const GemmParams* params);




// Graph tensor data cannot carry a byte offset, so offset weight/bias slices
// are copied once into cached buffers keyed by (source, offset, length).
static void* conv_slice(void* buf, uint64_t offset, size_t len) {
    if (offset == 0) return buf;
    NSMutableDictionary<NSString*, id<MTLBuffer>>* cache = conv_slice_cache();
    NSString* key = [NSString stringWithFormat:@"%p:%llu:%zu", buf, offset, len];
    id<MTLBuffer> hit = cache[key];
    if (hit) return (__bridge void*)hit;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)buf;
    id<MTLBuffer> copy = [src.device newBufferWithBytes:(char*)src.contents + offset
                                                 length:len
                                                options:MTLResourceStorageModeShared];
    if (!copy) return NULL;
    metrics_add(len);
    g_conv_slice_bytes += len;
    cache[key] = copy;
    return (__bridge void*)copy;
}







static void encode_vae_norm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawVaeNormParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(params->groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// Standalone dispatch of the fused vae_norm_silu kernel. Used as the exact
// reference in the streaming-split gate; the chain paths inline encode_vae_norm.
int zdraw_metal_run_vae_norm(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_norm(enc, pipe, input, weight, bias, output, params, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Exact-streaming split: global-stats pass. Same dispatch as encode_vae_norm
// (one threadgroup per group), writes [mean, scale] per group into `stats`.
static void encode_vae_norm_stats(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* stats,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)stats offset:0 atIndex:1];
    [enc setBytes:params length:sizeof(ZdrawVaeNormParams) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(params->groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// Exact-streaming split: windowed apply pass over [row0,row1) x [col0,col1).
static void encode_vae_norm_apply_window(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* stats,
    void* weight,
    void* bias,
    void* output,
    const ZdrawVaeNormWindowParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)stats offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:4];
    [enc setBytes:params length:sizeof(ZdrawVaeNormWindowParams) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(params->groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// Standalone dispatch of the stats pass (gate/test harness entry point).
int zdraw_metal_run_vae_norm_stats(
    void* queue,
    void* pipeline,
    void* input,
    void* stats,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_norm_stats(enc, pipe, input, stats, params, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Stats pass over a channel chunk (wall 1 of the memory ladder). The kernel is
// per group: binding the input at a channel-group boundary and the stats at
// the matching group index makes every thread visit the same elements in the
// same order as the whole-map pass, so the chunked statistics are
// bit-identical (ledger memory-ladder-w1-strip-decode-20261006).
static void encode_vae_norm_stats_at(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    size_t input_off,
    void* stats,
    size_t stats_off,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:input_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)stats offset:stats_off atIndex:1];
    [enc setBytes:params length:sizeof(ZdrawVaeNormParams) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(params->groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

int zdraw_metal_run_vae_norm_stats_at(
    void* queue,
    void* pipeline,
    void* input,
    size_t input_off,
    void* stats,
    size_t stats_off,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    encode_vae_norm_stats_at(enc, pipe, input, input_off, stats, stats_off, params, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Standalone dispatch of the windowed apply pass (gate/test harness entry point).
int zdraw_metal_run_vae_norm_apply_window(
    void* queue,
    void* pipeline,
    void* input,
    void* stats,
    void* weight,
    void* bias,
    void* output,
    const ZdrawVaeNormWindowParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_norm_apply_window(enc, pipe, input, stats, weight, bias, output,
                                 params, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Standalone windowed prenorm conv (gemmbench gate harness entry point).
int zdraw_metal_run_conv2d_prenorm_window(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    void* stats,
    void* norm_weight,
    void* norm_bias,
    const ZdrawConvPrenormWindowParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_conv2d_prenorm_window_compat(enc, pipe, input, weight, bias, output, stats,
                                 norm_weight, norm_bias, params, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Bench runner for the 4-/8-simdgroup prenorm kernels (oc_x4: 1 = _h4, 2 = _h8).
int zdraw_metal_run_conv2d_prenorm_window_xn(
    void* queue,
    void* pipeline,
    void* input,
    void* weight,
    void* bias,
    void* output,
    void* stats,
    void* norm_weight,
    void* norm_bias,
    const ZdrawConvPrenormWindowParams* params,
    int oc_x4
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    encode_conv2d_prenorm_window(enc, pipe, input, weight, bias, output, stats,
                                 norm_weight, norm_bias, params, oc_x4);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}
// Winograd F(4x4,3x3) conv (mconv_wino_shader.zig): weight transform once
// (do_weight), then per tile batch the input transform, the 36-plane batched
// GEMM and the output transform, all in one command buffer. tile batches are
// multiples of 64 (the GEMM's n contract); the caller sizes U/V/M for
// batch_tiles.
typedef struct {
    uint32_t in_ch, out_ch, height, width;
    uint32_t tiles_x, tile0, tile_count, groups;
    uint32_t has_bias, bias_dtype, norm_dtype, norm_bias_dtype;
    uint64_t weight_offset, bias_offset, norm_weight_offset, norm_bias_offset;
    // Strip-memory sub-ranges (mirrors MSL WinoParams; memory-ladder wall 1).
    // Whole-map: in_row0 = out_row0 = oc0 = 0, in_rows = out_rows = height,
    // oc_count = out_ch, out_local = 0 (wino_params_from_conv sets these).
    uint32_t in_row0, in_rows, out_row0, out_rows;
    uint32_t oc0, oc_count, out_local;
    uint32_t stash_row0, stash_rows, pad2_, pad3_;  // tier 2 in-place strips
} ZdrawWinoParams;

typedef struct { uint32_t m, k, n, m0; } ZdrawWinoGemmParams;

// The Winograd pipes + scratch the product chain carries (NULL = direct convs).
typedef struct {
    void* weight_pipe;
    void* input_pipe;     // prenorm-fused input transform
    void* input_up_pipe;  // nearest-2x input transform (upsample conv)
    void* gemm_pipe;
    void* output_pipe;
    void* u;              // half [36][out_ch][in_ch]
    void* v;              // half [36][in_ch][batch]
    void* m;              // half [36][out_ch][batch]
    uint32_t v_bytes;     // capacity of v and m (each)
} ZdrawWinoSet;

// Tile batch: V + M for the batch <= 24 MB (SLC-resident), a multiple of 64.
static uint32_t wino_batch_tiles(uint32_t ic, uint32_t oc, uint32_t total, uint32_t cap_bytes) {
    uint64_t per_tile = 36ull * (ic + oc) * 2ull;
    uint64_t b = (24ull * 1024 * 1024) / per_tile;
    uint64_t cap_v = (uint64_t)cap_bytes / (36ull * ic * 2ull);
    uint64_t cap_m = (uint64_t)cap_bytes / (36ull * oc * 2ull);
    if (b > cap_v) b = cap_v;
    if (b > cap_m) b = cap_m;
    b = (b / 64u) * 64u;
    if (b > total) b = total;
    return (uint32_t)b;
}

// Winograd-eligible 3x3 conv: f16 weights, channel multiples of the GEMM
// contract, a 4-aligned map whose tile count is a multiple of 64.
static int wino_eligible(uint32_t in_ch, uint32_t out_ch, uint32_t h, uint32_t w, uint32_t ksize, uint32_t pad, uint32_t dtype) {
    if (ksize != 3 || pad != 1 || dtype != 1) return 0;
    if (in_ch % 32u != 0 || out_ch % 64u != 0 || h % 4u != 0 || w % 4u != 0) return 0;
    uint32_t total = (h / 4u) * (w / 4u);
    return total % 64u == 0;
}

// Encode one Winograd conv over the tile range [tile_begin, tile_end): weight
// transform, then per tile batch the input transform, the 36-plane batched
// GEMM over the output channels [oc0, oc0+oc_count) and the output transform.
// `upsample` selects the nearest-2x input transform (no stats/norm operands).
// Per-tile and per-(output channel, tile) work is independent of the batch
// composition, so any sub-range is bit-identical to the whole-map encode.
static int encode_wino_conv_range(
    id<MTLComputeCommandEncoder> enc,
    const ZdrawWinoSet* ws,
    void* input, void* weight, void* bias, void* stats, void* norm_w, void* norm_b,
    void* output, const ZdrawWinoParams* params, int upsample,
    uint32_t tile_begin, uint32_t tile_end, void* stash
) {
    ZdrawWinoParams p = *params;
    // Zero sub-range fields (callers that build the params by hand, e.g. the
    // upsample and the bench runners) mean the whole map.
    if (p.in_rows == 0) p.in_rows = p.height;
    if (p.out_rows == 0) p.out_rows = p.height;
    if (p.oc_count == 0) p.oc_count = p.out_ch;
    if (tile_end <= tile_begin) return -3;
    uint32_t total = tile_end - tile_begin;
    uint32_t batch = wino_batch_tiles(p.in_ch, p.out_ch, total, ws->v_bytes);
    if (batch == 0 || total % 64u != 0) return -3;
    if (p.oc_count == 0 || p.oc_count % 64u != 0 || p.oc0 + p.oc_count > p.out_ch) return -3;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)ws->weight_pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)ws->u offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    size_t n = (size_t)p.out_ch * p.in_ch;
    [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch++;
    for (uint32_t t0 = tile_begin; t0 < tile_end; t0 += batch) {
        uint32_t count = tile_end - t0 < batch ? tile_end - t0 : batch;
        p.tile0 = t0;
        p.tile_count = count;
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)(upsample ? ws->input_up_pipe : ws->input_pipe)];
        [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)ws->v offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        if (!upsample) {
            [enc setBuffer:(__bridge id<MTLBuffer>)stats offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)norm_w offset:0 atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)norm_b offset:0 atIndex:5];
            // The stash is read only when stash_rows != 0; bind the input otherwise.
            [enc setBuffer:(__bridge id<MTLBuffer>)(stash ? stash : input) offset:0 atIndex:6];
        }
        [enc dispatchThreadgroups:MTLSizeMake((count + 31u) / 32u, (p.in_ch + 3u) / 4u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        ZdrawWinoGemmParams gp = { p.out_ch, p.in_ch, count, p.oc0 };
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)ws->gemm_pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)ws->u offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)ws->v offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)ws->m offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(count / 64u, p.oc_count / 64u, 36)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)ws->output_pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)ws->m offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((count + 31u) / 32u, (p.oc_count + 3u) / 4u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        g_dispatch += 3;
    }
    return 0;
}

// Whole-map Winograd conv (every caller before wall 1).
static int encode_wino_conv(
    id<MTLComputeCommandEncoder> enc,
    const ZdrawWinoSet* ws,
    void* input, void* weight, void* bias, void* stats, void* norm_w, void* norm_b,
    void* output, const ZdrawWinoParams* params, int upsample
) {
    uint32_t total = params->tiles_x * (params->height / 4u);
    return encode_wino_conv_range(enc, ws, input, weight, bias, stats, norm_w, norm_b,
                                  output, params, upsample, 0, total, NULL);
}

static ZdrawWinoParams wino_params_from_conv(const ZdrawConvParams* conv, const ZdrawVaeNormParams* norm) {
    ZdrawWinoParams p;
    memset(&p, 0, sizeof(p));
    p.in_ch = conv->in_ch;
    p.out_ch = conv->out_ch;
    p.height = conv->height;
    p.width = conv->width;
    p.in_rows = conv->height;
    p.out_rows = conv->height;
    p.oc_count = conv->out_ch;
    p.tiles_x = conv->width / 4u;
    p.has_bias = conv->has_bias;
    p.bias_dtype = conv->bias_dtype;
    p.weight_offset = conv->weight_offset;
    p.bias_offset = conv->bias_offset;
    if (norm) {
        p.groups = norm->groups;
        p.norm_dtype = norm->dtype;
        p.norm_bias_dtype = norm->bias_dtype;
        p.norm_weight_offset = norm->weight_offset;
        p.norm_bias_offset = norm->bias_offset;
    }
    return p;
}

// Standalone (bench) runner: one command buffer for the whole conv.
int zdraw_metal_run_wino_conv(
    void* queue,
    void* weight_pipe, void* input_pipe, void* gemm_pipe, void* output_pipe,
    void* input, void* weight, void* bias, void* stats, void* norm_w, void* norm_b,
    void* U, void* V, void* M, void* output,
    const ZdrawWinoParams* params, int do_weight, uint32_t batch_tiles
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    ZdrawWinoSet ws = { weight_pipe, input_pipe, input_pipe, gemm_pipe, output_pipe, U, V, M,
                        (uint32_t)(36ull * params->in_ch * batch_tiles * 2ull) };
    uint32_t m_bytes = (uint32_t)(36ull * params->out_ch * batch_tiles * 2ull);
    if (m_bytes < ws.v_bytes) ws.v_bytes = m_bytes;
    int rc = 0;
    if (do_weight >= 2) {
        // bench: the weight transform alone (2), x10 in one buffer (3), or an
        // empty command buffer (4) to separate the kernel from the round trip.
        int reps = do_weight == 3 ? 10 : (do_weight == 4 ? 0 : 1);
        for (int i = 0; i < reps; i++) {
            [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)weight_pipe];
            [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)U offset:0 atIndex:1];
            [enc setBytes:params length:sizeof(*params) atIndex:2];
            size_t n = (size_t)params->out_ch * params->in_ch;
            [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            g_dispatch++;
        }
    } else {
        rc = encode_wino_conv(enc, &ws, input, weight, bias, stats, norm_w, norm_b, output, params, 0);
    }
    [enc endEncoding];
    if (rc != 0) return rc;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// The fused upsample conv on the Winograd route (whole map, one command buffer).
int zdraw_metal_run_wino_upsample(
    void* queue,
    const ZdrawWinoSet* ws,
    void* input, void* weight, void* bias, void* output,
    const ZdrawConvUpsampleWindowParams* up
) {
    if (!wino_eligible(up->channels, up->channels, up->out_height, up->out_width, up->ksize, up->pad, up->dtype)) return 1;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    ZdrawWinoParams p;
    memset(&p, 0, sizeof(p));
    p.in_ch = up->channels;
    p.out_ch = up->channels;
    p.height = up->out_height;
    p.width = up->out_width;
    p.tiles_x = up->out_width / 4u;
    p.has_bias = up->has_bias;
    p.bias_dtype = up->bias_dtype;
    p.weight_offset = up->weight_offset;
    p.bias_offset = up->bias_offset;
    int rc = encode_wino_conv(enc, ws, input, weight, bias, NULL, NULL, NULL, output, &p, 1);
    [enc endEncoding];
    if (rc != 0) return rc;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Metal 4 tensor-ops pipelines (mgemm_mpp_shader.zig): compiled at runtime
// 1 = MSL 4.0 (macOS 26) and a device exist, 0 = older macOS, -1 = no device.
int zdraw_metal4_available(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return -1;
    if (@available(macOS 26.0, *)) return 1;
    return 0;
}

// with language version 4.0; NULL (with a one-line reason) on systems without
// it, so callers fall back to the simdgroup_matrix kernels.
void* zdraw_metal_compile_mpp(void* device, const char* source, const char* entry) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    if (@available(macOS 26.0, *)) {
        MTLCompileOptions* o = zdraw_compile_options();
        o.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);
        NSError* err = nil;
        NSString* text = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [dev newLibraryWithSource:text options:o error:&err];
        if (!lib) {
            fprintf(stderr, "mpp: compile failed: %s\n", err.localizedDescription.UTF8String);
            return NULL;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
        if (!fn) return NULL;
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) {
            fprintf(stderr, "mpp: pipeline failed: %s\n", err.localizedDescription.UTF8String);
            return NULL;
        }
        return (__bridge_retained void*)pipe;
    }
    return NULL;
}

// One GEMM on an MPP pipeline: grid (n / tn, m / 64), 128 threads.
int zdraw_metal_run_gemm_mpp(
    void* queue, void* pipeline, void* a, void* w, void* c_out,
    const GemmParams* params, uint32_t tn
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_out offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(params->n / tn, params->m / 64, 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Mid-attention prep: GroupNorm stats + the no-SiLU apply fused with the
// NCHW-to-token transpose, one command buffer. The stats pass is the resident
// mid-attention's only value-changing dispatch (tree reduction vs the CPU
// scalar loops); the apply/transpose is a permutation of the same expression.
int zdraw_metal_run_vae_attn_prep(
    void* queue,
    void* stats_pipeline,
    void* apply_pipeline,
    void* feature,
    void* stats,
    void* norm_w,
    void* norm_b,
    void* seq_out,
    const ZdrawVaeNormParams* params,
    size_t stats_threads,
    size_t apply_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_norm_stats(enc, (__bridge id<MTLComputePipelineState>)stats_pipeline,
                          feature, stats, params, stats_threads);
    id<MTLComputePipelineState> apply =
        (__bridge id<MTLComputePipelineState>)apply_pipeline;
    [enc setComputePipelineState:apply];
    [enc setBuffer:(__bridge id<MTLBuffer>)feature offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)stats offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)norm_w offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)norm_b offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)seq_out offset:0 atIndex:4];
    [enc setBytes:params length:sizeof(ZdrawVaeNormParams) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(params->groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(apply_threads, 1, 1)];
    [enc endEncoding];
    g_dispatch += 2;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Mid-attention finish: token-major attention output scatter-added into the
// NCHW feature. One thread per element; exact.
int zdraw_metal_run_vae_attn_finish(
    void* queue,
    void* pipeline,
    void* feature,
    void* seq,
    const ZdrawVaeNormParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)feature offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)seq offset:0 atIndex:1];
    [enc setBytes:params length:sizeof(ZdrawVaeNormParams) atIndex:2];
    uint64_t total = (uint64_t)params->channels * params->height * params->width;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static void encode_vae_add(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* output,
    void* residual,
    uint32_t count,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)residual offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(uint32_t) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// Standalone full-frame residual add (gate/test harness entry point); the
// exact reference the windowed add must match bit-for-bit.
int zdraw_metal_run_vae_add(
    void* queue,
    void* pipeline,
    void* output,
    void* residual,
    uint32_t count,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_add(enc, pipe, output, residual, count, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Exact-streaming residual add over the output row-strip [row0, row1). Same
// per-element `output += residual` as encode_vae_add, restricted to the strip.
static void encode_vae_add_window(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* output,
    void* residual,
    const ZdrawVaeAddWindowParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)residual offset:0 atIndex:1];
    [enc setBytes:params length:sizeof(ZdrawVaeAddWindowParams) atIndex:2];
    size_t strip = ((size_t)params->row1 - (size_t)params->row0) *
                   (size_t)params->width;
    size_t total = (size_t)params->channels * strip;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// Standalone dispatch of the windowed residual add (gate/test harness entry).
int zdraw_metal_run_vae_add_window(
    void* queue,
    void* pipeline,
    void* output,
    void* residual,
    const ZdrawVaeAddWindowParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_add_window(enc, pipe, output, residual, params, thread_count);
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static void encode_vae_up(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* output,
    const ZdrawUpParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:1];
    [enc setBytes:params length:sizeof(ZdrawUpParams) atIndex:2];
    size_t total = (size_t)params->channels * (size_t)params->height *
        (size_t)params->width * 4;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// The owned conv kernel inside a chain function (the MPSGraph splice that
// used to live here is gone: no framework on a render path, 2026-08-27).
static int encode_conv_plain(
    id<MTLComputeCommandEncoder>* enc_io,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvParams* params,
    size_t thread_count
) {
    (void)thread_count;
    encode_conv(*enc_io, pipe, input, weight, bias, output, params);
    return 0;
}

int zdraw_metal_run_vae_up_conv(
    void* queue,
    void* up_pipeline,
    void* conv_pipeline,
    const ZdrawVaeUpBuffers* b,
    const ZdrawVaeUpParams* p,
    size_t up_threads,
    size_t conv_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> up = (__bridge id<MTLComputePipelineState>)up_pipeline;
    id<MTLComputePipelineState> conv = (__bridge id<MTLComputePipelineState>)conv_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_up(enc, up, b->input, b->high, &p->up, up_threads);
    if (encode_conv_plain(&enc, conv, b->high, b->weight, b->bias, b->output,
                         &p->conv, conv_threads) != 0) return -1;
    [enc endEncoding];

    g_dispatch += 2; // upsample + conv
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}


int zdraw_metal_run_vae_final(
    void* queue,
    void* norm_pipeline,
    void* conv_pipeline,
    const ZdrawVaeFinalBuffers* b,
    const ZdrawVaeFinalParams* p,
    size_t norm_threads,
    size_t conv_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> norm = (__bridge id<MTLComputePipelineState>)norm_pipeline;
    id<MTLComputePipelineState> conv = (__bridge id<MTLComputePipelineState>)conv_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_vae_norm(enc, norm, b->input, b->norm_w, b->norm_b, b->norm, &p->norm, norm_threads);
    if (encode_conv_plain(&enc, conv, b->norm, b->conv_w, b->conv_b, b->output,
                         &p->conv, conv_threads) != 0) return -1;
    [enc endEncoding];

    g_dispatch += 2; // fused norm+SiLU + conv
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static int encode_vae_res_block(
    id<MTLCommandBuffer> __strong* cmd_io,
    id<MTLComputeCommandEncoder>* enc_io,
    id<MTLComputePipelineState> norm,
    id<MTLComputePipelineState> conv,
    id<MTLComputePipelineState> add,
    const ZdrawVaeResBuffers* b,
    const ZdrawVaeResParams* p,
    size_t norm_threads,
    size_t conv_threads,
    size_t add_threads
) {
    id<MTLComputeCommandEncoder> enc = *enc_io;
    encode_vae_norm(enc, norm, b->input, b->norm1_w, b->norm1_b, b->norm, &p->norm1, norm_threads);
    if (encode_conv_plain(&enc, conv, b->norm, b->conv1_w, b->conv1_b, b->work,
                         &p->conv1, conv_threads) != 0) return -1;
    encode_vae_norm(enc, norm, b->work, b->norm2_w, b->norm2_b, b->work, &p->norm2, norm_threads);
    if (encode_conv_plain(&enc, conv, b->work, b->conv2_w, b->conv2_b, b->output,
                         &p->conv2, conv_threads) != 0) return -1;
    void* residual = b->input;
    if (p->has_skip != 0) {
        if (encode_conv_plain(&enc, conv, b->input, b->skip_w, b->skip_b, b->skip,
                             &p->skip, conv_threads) != 0) return -1;
        residual = b->skip;
    }
    encode_vae_add(enc, add, b->output, residual, p->out_count, add_threads);
    *enc_io = enc;
    return 0;
}

int zdraw_metal_run_vae_res(
    void* queue,
    void* norm_pipeline,
    void* conv_pipeline,
    void* add_pipeline,
    const ZdrawVaeResBuffers* b,
    const ZdrawVaeResParams* p,
    size_t norm_threads,
    size_t conv_threads,
    size_t add_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> norm = (__bridge id<MTLComputePipelineState>)norm_pipeline;
    id<MTLComputePipelineState> conv = (__bridge id<MTLComputePipelineState>)conv_pipeline;
    id<MTLComputePipelineState> add = (__bridge id<MTLComputePipelineState>)add_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    if (encode_vae_res_block(&cmd, &enc, norm, conv, add, b, p,
                             norm_threads, conv_threads, add_threads) != 0) return -1;
    [enc endEncoding];

    // norm1, conv1, norm2, conv2, add; plus the 1x1 skip conv when present
    g_dispatch += 5 + (p->has_skip != 0 ? 1 : 0);
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_vae_res_chain(
    void* queue,
    void* norm_pipeline,
    void* conv_pipeline,
    void* add_pipeline,
    const ZdrawVaeResBuffers* b,
    const ZdrawVaeResParams* p,
    size_t count,
    size_t norm_threads,
    size_t conv_threads,
    size_t add_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> norm = (__bridge id<MTLComputePipelineState>)norm_pipeline;
    id<MTLComputePipelineState> conv = (__bridge id<MTLComputePipelineState>)conv_pipeline;
    id<MTLComputePipelineState> add = (__bridge id<MTLComputePipelineState>)add_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    uint64_t dispatches = 0;
    for (size_t i = 0; i < count; i++) {
        if (encode_vae_res_block(&cmd, &enc, norm, conv, add, &b[i], &p[i],
                                 norm_threads, conv_threads, add_threads) != 0) return -1;
        // norm1, conv1, norm2, conv2, add; plus the 1x1 skip conv when present
        dispatches += 5 + (p[i].has_skip != 0 ? 1 : 0);
    }
    [enc endEncoding];

    g_dispatch += dispatches;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Build the fused-prenorm windowed conv params for one strip [row0,row1) from a
// conv ZdrawConvParams (the post-norm conv) and the GroupNorm it absorbs.
static ZdrawConvPrenormWindowParams prenorm_window_params(
    const ZdrawConvParams* conv,
    const ZdrawVaeNormParams* norm,
    uint32_t row0,
    uint32_t row1
) {
    ZdrawConvPrenormWindowParams p;
    p.in_ch = conv->in_ch;
    p.out_ch = conv->out_ch;
    p.height = conv->height;
    p.width = conv->width;
    p.ksize = conv->ksize;
    p.pad = conv->pad;
    p.dtype = conv->dtype;
    p.bias_dtype = conv->bias_dtype;
    p.has_bias = conv->has_bias;
    p.groups = norm->groups;
    p.weight_offset = conv->weight_offset;
    p.bias_offset = conv->bias_offset;
    p.row0 = row0;
    p.row1 = row1;
    p.norm_dtype = norm->dtype;
    p.norm_bias_dtype = norm->bias_dtype;
    p.norm_weight_offset = norm->weight_offset;
    p.norm_bias_offset = norm->bias_offset;
    return p;
}

// Emit every [row0,row1) strip covering [0,height) for one fused-prenorm conv.
static void encode_prenorm_strips(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    void* stats,
    void* norm_weight,
    void* norm_bias,
    const ZdrawConvParams* conv,
    const ZdrawVaeNormParams* norm,
    uint32_t strip_rows,
    size_t thread_count,
    int oc_x4
) {
    for (uint32_t row0 = 0; row0 < conv->height; row0 += strip_rows) {
        uint32_t row1 = row0 + strip_rows;
        if (row1 > conv->height) row1 = conv->height;
        ZdrawConvPrenormWindowParams p =
            prenorm_window_params(conv, norm, row0, row1);
        encode_conv2d_prenorm_window(enc, pipe, input, weight, bias, output,
                                     stats, norm_weight, norm_bias, &p, oc_x4);
        g_dispatch++;
    }
}

// Emit every [row0,row1) strip for a plain windowed conv (the 1x1 skip).
static void encode_conv_window_strips(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* bias,
    void* output,
    const ZdrawConvParams* conv,
    uint32_t strip_rows,
    size_t thread_count
) {
    for (uint32_t row0 = 0; row0 < conv->height; row0 += strip_rows) {
        uint32_t row1 = row0 + strip_rows;
        if (row1 > conv->height) row1 = conv->height;
        ZdrawConvWindowParams p;
        p.in_ch = conv->in_ch;
        p.out_ch = conv->out_ch;
        p.height = conv->height;
        p.width = conv->width;
        p.ksize = conv->ksize;
        p.pad = conv->pad;
        p.dtype = conv->dtype;
        p.bias_dtype = conv->bias_dtype;
        p.has_bias = conv->has_bias;
        p.pad1 = 0;
        p.weight_offset = conv->weight_offset;
        p.bias_offset = conv->bias_offset;
        p.row0 = row0;
        p.row1 = row1;
        encode_conv2d_window(enc, pipe, input, weight, bias, output, &p);
        g_dispatch++;
    }
}

// Emit every [row0,row1) strip for the residual add.
static void encode_add_window_strips(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* output,
    void* residual,
    uint32_t channels,
    uint32_t height,
    uint32_t width,
    uint32_t strip_rows,
    size_t thread_count
) {
    for (uint32_t row0 = 0; row0 < height; row0 += strip_rows) {
        uint32_t row1 = row0 + strip_rows;
        if (row1 > height) row1 = height;
        ZdrawVaeAddWindowParams p = { channels, height, width, row0, row1 };
        encode_vae_add_window(enc, pipe, output, residual, &p, thread_count);
        g_dispatch++;
    }
}

// Emit every [row0,row1) full-width strip of the norm+SiLU apply pass
// (unfuse path): writes silu(norm(x)) into the scratch so a PLAIN conv can
// consume it. Identical math to the fused prenorm fill => bit-identical.
static void encode_apply_strips(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* stats,
    void* norm_weight,
    void* norm_bias,
    void* output,
    const ZdrawVaeNormParams* norm,
    uint32_t strip_rows,
    size_t thread_count
) {
    for (uint32_t row0 = 0; row0 < norm->height; row0 += strip_rows) {
        uint32_t row1 = row0 + strip_rows;
        if (row1 > norm->height) row1 = norm->height;
        ZdrawVaeNormWindowParams p;
        p.channels = norm->channels;
        p.height = norm->height;
        p.width = norm->width;
        p.groups = norm->groups;
        p.dtype = norm->dtype;
        p.bias_dtype = norm->bias_dtype;
        p.eps = norm->eps;
        p.weight_offset = norm->weight_offset;
        p.bias_offset = norm->bias_offset;
        p.row0 = row0;
        p.row1 = row1;
        p.col0 = 0;
        p.col1 = norm->width;
        encode_vae_norm_apply_window(enc, pipe, input, stats, norm_weight,
                                     norm_bias, output, &p, thread_count);
        g_dispatch++;
    }
}

// ZDRAW_VAE_UNFUSE=1: run norm+SiLU as a separate apply pass and the conv as a
// PLAIN windowed conv over the normed scratch. Removes the fused prenorm's
// cross-threadgroup norm recompute (one apply per element instead of one per
// oc-tile); values and conv order unchanged => bit-identical.
static int vae_unfuse_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char* s = getenv("ZDRAW_VAE_UNFUSE");
        v = (s && s[0] == '1') ? 1 : 0;
    }
    return v;
}


// One resident streamed resblock, fully on-GPU and bit-identical to the strip
// pipeline that mvres_stream proved (norm1+SiLU -> conv1 -> norm2+SiLU -> conv2
// -> +skip): global stats then fused-prenorm conv per strip, twice, plus the
// optional 1x1 skip and the windowed add. No intermediate readback.
static int encode_vae_res_stream_block(
    id<MTLCommandBuffer> __strong* cmd_io,
    id<MTLComputeCommandEncoder>* enc_io,
    id<MTLComputePipelineState> stats_pipe,
    id<MTLComputePipelineState> prenorm_pipe,
    id<MTLComputePipelineState> conv_pipe,
    id<MTLComputePipelineState> add_pipe,
    id<MTLComputePipelineState> apply_pipe,
    id<MTLComputePipelineState> apply_h_pipe,
    id<MTLComputePipelineState> conv3_pipe,
    id<MTLComputePipelineState> stats2_pipe,
    id<MTLComputePipelineState> prenorm2_pipe,
    const ZdrawVaeResStreamBuffers* b,
    const ZdrawVaeResParams* p,
    uint32_t strip_rows,
    size_t stats_threads,
    size_t conv_threads,
    size_t add_threads,
    int prenorm_x4,
    int hgraph,
    const ZdrawWinoSet* wino
) {
    id<MTLComputeCommandEncoder> enc = *enc_io;
    const int unfuse = vae_unfuse_enabled() && apply_pipe && conv3_pipe && b->norm &&
        p->conv1.dtype == 3 && p->conv2.dtype == 3;
    (void)hgraph;
    (void)apply_h_pipe;
    const int wino1 = wino && !unfuse && wino_eligible(p->conv1.in_ch, p->conv1.out_ch,
        p->conv1.height, p->conv1.width, p->conv1.ksize, p->conv1.pad, p->conv1.dtype);
    const int wino2 = wino && !unfuse && wino_eligible(p->conv2.in_ch, p->conv2.out_ch,
        p->conv2.height, p->conv2.width, p->conv2.ksize, p->conv2.pad, p->conv2.dtype);
    // norm1 global stats over the raw input.
    encode_vae_norm_stats(enc, stats_pipe, b->input, b->stats1, &p->norm1,
                          stats_threads);
    if (unfuse) {
        // norm1+SiLU into the scratch, then a PLAIN conv1 over it.
        encode_apply_strips(enc, apply_pipe, b->input, b->stats1, b->norm1_w,
                            b->norm1_b, b->norm, &p->norm1, strip_rows,
                            stats_threads);
        encode_conv_window_strips(enc, conv3_pipe, b->norm, b->conv1_w,
                                  b->conv1_b, b->conv1_out, &p->conv1,
                                  strip_rows, conv_threads);
    } else if (wino1) {
        ZdrawWinoParams wp = wino_params_from_conv(&p->conv1, &p->norm1);
        if (encode_wino_conv(enc, wino, b->input, b->conv1_w, b->conv1_b, b->stats1,
                             b->norm1_w, b->norm1_b, b->conv1_out, &wp, 0) != 0) return -1;
    } else {
        // conv1 with norm1+SiLU fused, per strip -> conv1_out (full).
        encode_prenorm_strips(enc, prenorm_pipe, b->input, b->conv1_w, b->conv1_b,
                              b->conv1_out, b->stats1, b->norm1_w, b->norm1_b,
                              &p->conv1, &p->norm1, strip_rows, conv_threads,
                              prenorm_x4);
    }
    // norm2 global stats over conv1_out (its own pipe: conv1_out may be a
    // different storage dtype than the block input in hybrid f16 mode).
    encode_vae_norm_stats(enc, stats2_pipe ? stats2_pipe : stats_pipe,
                          b->conv1_out, b->stats2, &p->norm2, stats_threads);
    if (unfuse) {
        encode_apply_strips(enc, apply_pipe, b->conv1_out, b->stats2, b->norm2_w,
                            b->norm2_b, b->norm, &p->norm2, strip_rows,
                            stats_threads);
        encode_conv_window_strips(enc, conv3_pipe, b->norm, b->conv2_w,
                                  b->conv2_b, b->output, &p->conv2,
                                  strip_rows, conv_threads);
    } else if (wino2) {
        ZdrawWinoParams wp = wino_params_from_conv(&p->conv2, &p->norm2);
        if (encode_wino_conv(enc, wino, b->conv1_out, b->conv2_w, b->conv2_b, b->stats2,
                             b->norm2_w, b->norm2_b, b->output, &wp, 0) != 0) return -1;
    } else {
        // conv2 with norm2+SiLU fused, per strip -> output (its own pipe in
        // hybrid mode: reads f32 conv1_out, writes f16).
        encode_prenorm_strips(enc, prenorm2_pipe ? prenorm2_pipe : prenorm_pipe,
                              b->conv1_out, b->conv2_w,
                              b->conv2_b, b->output, b->stats2, b->norm2_w,
                              b->norm2_b, &p->conv2, &p->norm2, strip_rows,
                              conv_threads, prenorm_x4);
    }
    // residual: 1x1 skip projection of the input, or the input itself.
    void* residual = b->input;
    if (p->has_skip != 0) {
        encode_conv_window_strips(enc, conv_pipe, b->input, b->skip_w,
                                  b->skip_b, b->skip, &p->skip, strip_rows,
                                  conv_threads);
        residual = b->skip;
    }
    encode_add_window_strips(enc, add_pipe, b->output, residual,
                             p->conv2.out_ch, p->conv2.height, p->conv2.width,
                             strip_rows, add_threads);
    return 0;
}

// Resident exact-streaming resblock chain: every block (and every strip) is
// encoded into ONE command buffer over caller-owned ping-pong buffers, so the
// whole group decodes on-GPU with a single readback. Bit-identical to running
int zdraw_metal_run_vae_res_stream_chain(
    void* queue,
    void* stats_pipeline,
    void* prenorm_pipeline,
    void* conv_pipeline,
    void* add_pipeline,
    void* apply_pipeline,
    void* apply_h_pipeline,
    void* conv3_pipeline,
    void* stats2_pipeline,
    void* prenorm2_pipeline,
    const ZdrawVaeResStreamBuffers* b,
    const ZdrawVaeResParams* p,
    size_t count,
    uint32_t strip_rows,
    size_t stats_threads,
    size_t conv_threads,
    size_t add_threads,
    int prenorm_x4,
    int hgraph,
    const ZdrawWinoSet* wino
) {
    if (strip_rows == 0) return -1;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> stats = (__bridge id<MTLComputePipelineState>)stats_pipeline;
    id<MTLComputePipelineState> prenorm = (__bridge id<MTLComputePipelineState>)prenorm_pipeline;
    id<MTLComputePipelineState> conv = (__bridge id<MTLComputePipelineState>)conv_pipeline;
    id<MTLComputePipelineState> add = (__bridge id<MTLComputePipelineState>)add_pipeline;
    id<MTLComputePipelineState> applyp = (__bridge id<MTLComputePipelineState>)apply_pipeline;
    id<MTLComputePipelineState> applyh = (__bridge id<MTLComputePipelineState>)apply_h_pipeline;
    id<MTLComputePipelineState> conv3 = (__bridge id<MTLComputePipelineState>)conv3_pipeline;
    id<MTLComputePipelineState> stats2 = (__bridge id<MTLComputePipelineState>)stats2_pipeline;
    id<MTLComputePipelineState> prenorm2 = (__bridge id<MTLComputePipelineState>)prenorm2_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    for (size_t i = 0; i < count; i++) {
        if (encode_vae_res_stream_block(&cmd, &enc, stats, prenorm, conv, add,
                                        applyp, applyh, conv3, stats2, prenorm2,
                                        &b[i], &p[i], strip_rows, stats_threads,
                                        conv_threads, add_threads, prenorm_x4,
                                        hgraph, wino) != 0) {
            return -1;
        }
        // The two norm-stats passes; every strip helper and graph conv counts
        // its own dispatches at its encode site.
        g_dispatch += 2;
    }
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// ---- Strip-memory resblock chain (memory-ladder wall 1) ---------------------
//
// The whole-map conv1_out (and the skip that aliases it) is replaced by two
// small scratch buffers. Per resblock:
//   stats1 over the whole input (unchanged);
//   pass A: conv1 in 64-output-channel chunks into conv1_chunk, each chunk's
//           GroupNorm statistics written at its group offset of stats2 (the
//           stats kernel is per group, so this is the whole-map arithmetic);
//   pass B: per row strip, conv1 recomputed for the strip plus a halo into
//           conv1_strip, conv2 over the strip reading it (prenorm fused with
//           the whole stats2), the 1x1 skip into skip_strip, the residual add.
// Winograd per-tile work does not depend on the batch, so every sub-range is
// bit-identical to the whole-map chain. Winograd-only: the caller routes any
// non-eligible group to the whole-map chain.
typedef struct {
    void* conv1_chunk;   // [chunk_ch][height][width] half
    void* conv1_strip;   // [out_ch][strip_rows + 2 * unit_rows][width] half
    void* skip_strip;    // [out_ch][strip_rows][width] half: the 1x1 skip, or the
                         // conv2 result of a no-skip in-place block (tier 2)
    void* stash;         // [in_ch][stash_rows][width] half: the halo rows the previous
                         // strip overwrote (tier 2, in-place blocks)
    uint32_t chunk_ch;   // multiple of the GroupNorm group width and of 64
    uint32_t strip_rows; // multiple of the tile-row alignment unit
    uint32_t stash_rows; // unit_rows + 1 (0 = no in-place blocks)
    uint32_t pad_;
} ZdrawVaeStripScratch;

// Rows per alignment unit: the smallest run of tile rows whose tile count is
// a multiple of 64 (the Winograd GEMM's n contract), times 4 pixel rows.
static uint32_t strip_unit_rows(uint32_t tiles_x) {
    uint32_t u = 1;
    while (((uint64_t)tiles_x * u) % 64u != 0u) u++;
    return 4u * u;
}

static int encode_vae_res_strip_block(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> stats_pipe,
    id<MTLComputePipelineState> stats2_pipe,
    id<MTLComputePipelineState> add_pipe,
    id<MTLComputePipelineState> add_strip_pipe,
    id<MTLComputePipelineState> add_rev_pipe,
    id<MTLComputePipelineState> skip_strip_pipe,
    id<MTLComputePipelineState> rows_copy_pipe,
    const ZdrawVaeResStreamBuffers* b,
    const ZdrawVaeStripScratch* s,
    const ZdrawVaeResParams* p,
    size_t stats_threads,
    size_t add_threads,
    const ZdrawWinoSet* wino
) {
    if (!wino_eligible(p->conv1.in_ch, p->conv1.out_ch, p->conv1.height, p->conv1.width,
                       p->conv1.ksize, p->conv1.pad, p->conv1.dtype)) return -4;
    if (!wino_eligible(p->conv2.in_ch, p->conv2.out_ch, p->conv2.height, p->conv2.width,
                       p->conv2.ksize, p->conv2.pad, p->conv2.dtype)) return -4;
    const uint32_t height = p->conv1.height;
    const uint32_t width = p->conv1.width;
    const uint32_t tiles_x = width / 4u;
    const uint32_t unit = strip_unit_rows(tiles_x);
    const uint32_t group_ch = p->norm2.channels / p->norm2.groups;
    if (s->strip_rows == 0 || s->strip_rows % unit != 0 || height % unit != 0) return -3;
    if (s->chunk_ch == 0 || s->chunk_ch % 64u != 0 || s->chunk_ch % group_ch != 0 ||
        p->conv1.out_ch % s->chunk_ch != 0) return -3;
    // Tier 2: the block writes its output over its input rows as strips
    // complete (output == input). The halo rows the previous strip overwrote
    // come from the stash; the skip and the no-skip residual read the input
    // rows before they are overwritten.
    const int inplace = (b->output == b->input);
    if (inplace && (s->stash == NULL || s->stash_rows != unit + 1u || s->strip_rows < unit + 1u ||
                    p->conv1.in_ch < p->conv2.out_ch)) return -3;

    // norm1 global stats over the raw input.
    encode_vae_norm_stats(enc, stats_pipe, b->input, b->stats1, &p->norm1, stats_threads);
    g_dispatch++;

    // Pass A: conv1 by output-channel chunk -> chunk-local buffer -> stats2 slice.
    const uint32_t total_tiles = tiles_x * (height / 4u);
    for (uint32_t oc0 = 0; oc0 < p->conv1.out_ch; oc0 += s->chunk_ch) {
        ZdrawWinoParams wp = wino_params_from_conv(&p->conv1, &p->norm1);
        wp.oc0 = oc0;
        wp.oc_count = s->chunk_ch;
        wp.out_local = 1;
        if (encode_wino_conv_range(enc, wino, b->input, b->conv1_w, b->conv1_b, b->stats1,
                                   b->norm1_w, b->norm1_b, s->conv1_chunk, &wp, 0,
                                   0, total_tiles, NULL) != 0) return -1;
        ZdrawVaeNormParams np = p->norm2;
        np.channels = s->chunk_ch;
        np.groups = s->chunk_ch / group_ch;
        size_t stats_off = (size_t)(oc0 / group_ch) * 2u * sizeof(float);
        encode_vae_norm_stats_at(enc, stats2_pipe, s->conv1_chunk, 0, b->stats2, stats_off,
                                 &np, stats_threads);
        g_dispatch++;
    }

    // Pass B: row strips.
    for (uint32_t r0 = 0; r0 < height; r0 += s->strip_rows) {
        uint32_t r1 = r0 + s->strip_rows;
        if (r1 > height) r1 = height;
        uint32_t hr0 = r0 >= unit ? r0 - unit : 0u;
        uint32_t hr1 = r1 + unit;
        if (hr1 > height) hr1 = height;
        // conv1 for the halo'd rows into the strip-local buffer; in place, the
        // rows below r0 that the previous strip overwrote come from the stash.
        ZdrawWinoParams w1 = wino_params_from_conv(&p->conv1, &p->norm1);
        w1.out_local = 1;
        w1.out_row0 = hr0;
        w1.out_rows = hr1 - hr0;
        if (inplace && r0 > 0) {
            w1.stash_row0 = r0 - (unit + 1u);
            w1.stash_rows = unit + 1u;
        }
        if (encode_wino_conv_range(enc, wino, b->input, b->conv1_w, b->conv1_b, b->stats1,
                                   b->norm1_w, b->norm1_b, s->conv1_strip, &w1, 0,
                                   (hr0 / 4u) * tiles_x, (hr1 / 4u) * tiles_x,
                                   inplace ? s->stash : NULL) != 0) return -1;
        if (inplace && r1 < height) {
            // Stash the rows the next strip's halo needs before this strip overwrites them.
            ZdrawVaeAddWindowParams cp = { p->conv1.in_ch, height, width, r1 - (unit + 1u), r1 };
            encode_vae_add_window(enc, rows_copy_pipe, b->input, s->stash, &cp, add_threads);
            g_dispatch++;
        }
        // The 1x1 skip reads the input rows before anything overwrites them.
        ZdrawVaeAddWindowParams ap = { p->conv2.out_ch, height, width, r0, r1 };
        if (p->has_skip != 0) {
            ZdrawConvWindowParams sp;
            sp.in_ch = p->skip.in_ch;
            sp.out_ch = p->skip.out_ch;
            sp.height = p->skip.height;
            sp.width = p->skip.width;
            sp.ksize = p->skip.ksize;
            sp.pad = p->skip.pad;
            sp.dtype = p->skip.dtype;
            sp.bias_dtype = p->skip.bias_dtype;
            sp.has_bias = p->skip.has_bias;
            sp.pad1 = 0;
            sp.weight_offset = p->skip.weight_offset;
            sp.bias_offset = p->skip.bias_offset;
            sp.row0 = r0;
            sp.row1 = r1;
            encode_conv2d_window(enc, skip_strip_pipe, b->input, b->skip_w, b->skip_b,
                                 s->skip_strip, &sp);
            g_dispatch++;
        }
        // conv2 over the strip's own tiles, reading the strip-local conv1. In
        // place without a skip it lands strip-locally (the add still needs the
        // input rows); otherwise it writes the output rows directly.
        ZdrawWinoParams w2 = wino_params_from_conv(&p->conv2, &p->norm2);
        w2.in_row0 = hr0;
        w2.in_rows = hr1 - hr0;
        const int conv2_local = inplace && p->has_skip == 0;
        if (conv2_local) {
            w2.out_local = 1;
            w2.out_row0 = r0;
            w2.out_rows = r1 - r0;
        }
        if (encode_wino_conv_range(enc, wino, s->conv1_strip, b->conv2_w, b->conv2_b, b->stats2,
                                   b->norm2_w, b->norm2_b, conv2_local ? s->skip_strip : b->output,
                                   &w2, 0, (r0 / 4u) * tiles_x, (r1 / 4u) * tiles_x, NULL) != 0) return -1;
        if (conv2_local) {
            // output(=input) rows = conv2 (strip-local) + input rows.
            encode_vae_add_window(enc, add_rev_pipe, b->output, s->skip_strip, &ap, add_threads);
        } else if (p->has_skip != 0) {
            encode_vae_add_window(enc, add_strip_pipe, b->output, s->skip_strip, &ap, add_threads);
        } else {
            encode_vae_add_window(enc, add_pipe, b->output, b->input, &ap, add_threads);
        }
        g_dispatch++;
    }
    return 0;
}

int zdraw_metal_run_vae_res_strip_chain(
    void* queue,
    void* stats_pipeline,
    void* stats2_pipeline,
    void* add_pipeline,
    void* add_strip_pipeline,
    void* add_rev_pipeline,
    void* skip_strip_pipeline,
    void* rows_copy_pipeline,
    const ZdrawVaeResStreamBuffers* b,
    const ZdrawVaeStripScratch* s,
    const ZdrawVaeResParams* p,
    size_t count,
    size_t stats_threads,
    size_t add_threads,
    const ZdrawWinoSet* wino
) {
    if (!wino) return -4;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> stats = (__bridge id<MTLComputePipelineState>)stats_pipeline;
    id<MTLComputePipelineState> stats2 = (__bridge id<MTLComputePipelineState>)stats2_pipeline;
    id<MTLComputePipelineState> add = (__bridge id<MTLComputePipelineState>)add_pipeline;
    id<MTLComputePipelineState> adds = (__bridge id<MTLComputePipelineState>)add_strip_pipeline;
    id<MTLComputePipelineState> addr = (__bridge id<MTLComputePipelineState>)add_rev_pipeline;
    id<MTLComputePipelineState> skips = (__bridge id<MTLComputePipelineState>)skip_strip_pipeline;
    id<MTLComputePipelineState> rows = (__bridge id<MTLComputePipelineState>)rows_copy_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    for (size_t i = 0; i < count; i++) {
        int rc = encode_vae_res_strip_block(enc, stats, stats2, add, adds, addr, skips, rows,
                                            &b[i], s, &p[i], stats_threads, add_threads, wino);
        if (rc != 0) {
            [enc endEncoding];
            return rc;
        }
    }
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

uint32_t zdraw_metal_vae_strip_unit_rows(uint32_t width) {
    return strip_unit_rows(width / 4u);
}

// ---- Strip finish (memory-ladder wall 1, tier 3) --------------------------
//
// The final GroupNorm+SiLU and the 128->3 RGB conv without the whole-map f32
// norm scratch: the two-pass statistics over the whole final feature (the
// verbatim copy of vae_norm_silu_h's reductions), then per row strip the
// apply into a strip-local float buffer for rows [r0-1, r1+1) and conv2d
// over the output rows [r0, r1) reading it. Same values, same MMA order.
typedef struct {
    void* input;       // the final feature, half, whole map
    void* stats;       // [mean, scale] per group
    void* norm_strip;  // [channels][strip_rows + 2][width] float
    void* norm_w; void* norm_b; void* conv_w; void* conv_b;
    void* output;      // 3 x height x width float
} ZdrawVaeFinalStripBuffers;

int zdraw_metal_run_vae_final_strips(
    void* queue,
    void* stats_pipeline,
    void* apply_pipeline,
    void* conv_pipeline,
    const ZdrawVaeFinalStripBuffers* b,
    const ZdrawVaeFinalParams* p,
    uint32_t strip_rows,
    size_t stats_threads,
    size_t apply_threads
) {
    if (strip_rows == 0) return -3;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> stats = (__bridge id<MTLComputePipelineState>)stats_pipeline;
    id<MTLComputePipelineState> apply = (__bridge id<MTLComputePipelineState>)apply_pipeline;
    id<MTLComputePipelineState> conv = (__bridge id<MTLComputePipelineState>)conv_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    const uint32_t height = p->norm.height;
    const uint32_t width = p->norm.width;
    encode_vae_norm_stats(enc, stats, b->input, b->stats, &p->norm, stats_threads);
    g_dispatch++;
    for (uint32_t r0 = 0; r0 < height; r0 += strip_rows) {
        uint32_t r1 = r0 + strip_rows;
        if (r1 > height) r1 = height;
        uint32_t a0 = r0 > 0 ? r0 - 1u : 0u;
        uint32_t a1 = r1 + 1u > height ? height : r1 + 1u;
        ZdrawVaeNormWindowParams np;
        np.channels = p->norm.channels;
        np.height = height;
        np.width = width;
        np.groups = p->norm.groups;
        np.dtype = p->norm.dtype;
        np.bias_dtype = p->norm.bias_dtype;
        np.eps = p->norm.eps;
        np.weight_offset = p->norm.weight_offset;
        np.bias_offset = p->norm.bias_offset;
        np.row0 = a0;
        np.row1 = a1;
        np.col0 = 0;
        np.col1 = width;
        encode_vae_norm_apply_window(enc, apply, b->input, b->stats, b->norm_w, b->norm_b,
                                     b->norm_strip, &np, apply_threads);
        g_dispatch++;
        ZdrawConvStripParams cp;
        cp.in_ch = p->conv.in_ch;
        cp.out_ch = p->conv.out_ch;
        cp.height = p->conv.height;
        cp.width = p->conv.width;
        cp.ksize = p->conv.ksize;
        cp.pad = p->conv.pad;
        cp.dtype = p->conv.dtype;
        cp.bias_dtype = p->conv.bias_dtype;
        cp.has_bias = p->conv.has_bias;
        cp.pad1 = 0;
        cp.weight_offset = p->conv.weight_offset;
        cp.bias_offset = p->conv.bias_offset;
        cp.row0 = r0;
        cp.row1 = r1;
        cp.in_row0 = a0;
        cp.in_rows = a1 - a0;
        [enc setComputePipelineState:conv];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->norm_strip offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->conv_w offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->conv_b offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->output offset:0 atIndex:3];
        [enc setBytes:&cp length:sizeof(cp) atIndex:4];
        size_t strip_pos = (size_t)(r1 - r0) * (size_t)width;
        [enc dispatchThreadgroups:MTLSizeMake((strip_pos + 31u) / 32u, ((size_t)cp.out_ch + 31u) / 32u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        g_dispatch++;
    }
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}



static id<MTLComputePipelineState> headmajor_convert(id<MTLDevice> device);
static id<MTLBuffer> attn_half_scratch(id<MTLDevice> device, int slot, size_t bytes);

// Kernel identity passed explicitly from mattn.Kernel (rows, flash, block,
// wide): the half-input staging and the grid are keyed on it, never on the
// thread count a PSO happens to resolve (the seam of the 07-18 incident).
enum { ATTN_K_ROWS = 0, ATTN_K_FLASH = 1, ATTN_K_BLOCK = 2, ATTN_K_WIDE = 3 };

// Shared encode body for the standalone attention runner: block16 (64
// threads) converts q/k/v to half head-major scratch in-encoder first.
static void attention_encode(
    id<MTLComputeCommandEncoder> enc,
    id<MTLDevice> device,
    id<MTLComputePipelineState> pipe,
    void* q_buf, void* k_buf, void* v_buf, void* output,
    const ZdrawAttnParams* params,
    uint32_t kernel,
    size_t thread_count,
    uint64_t in_off,   // byte offset into q/k/v (batched multi-image segments)
    uint64_t out_off   // byte offset into output
) {
    void* qb = q_buf;
    void* kb = k_buf;
    void* vb = v_buf;
    uint64_t q_off = in_off;
    if (kernel == ATTN_K_BLOCK) { // block16 takes half inputs
        const uint32_t qn = params->tokens * params->heads * params->head_dim;
        const uint32_t kn = params->tokens * params->kv_heads * params->head_dim;
        id<MTLComputePipelineState> conv = headmajor_convert(device);
        void* src[3] = {q_buf, k_buf, v_buf};
        uint32_t count[3] = {qn, kn, kn};
        for (int i = 0; i < 3; i++) {
            id<MTLBuffer> hbuf = attn_half_scratch(device, i, (size_t)count[i] * 2);
            const uint32_t hh = i == 0 ? params->heads : params->kv_heads;
            [enc setComputePipelineState:conv];
            [enc setBuffer:(__bridge id<MTLBuffer>)src[i] offset:(size_t)in_off atIndex:0];
            [enc setBuffer:hbuf offset:0 atIndex:1];
            [enc setBytes:&params->tokens length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&hh length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&params->head_dim length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(count[i], 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            if (i == 0) qb = (__bridge void*)hbuf;
            if (i == 1) kb = (__bridge void*)hbuf;
            if (i == 2) vb = (__bridge void*)hbuf;
        }
        q_off = 0; // converts consumed the offset; scratch is segment-local
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)qb offset:(size_t)q_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)kb offset:(size_t)q_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)vb offset:(size_t)q_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:(size_t)out_off atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawAttnParams) atIndex:4];
    // block16: three head-major converts plus the attention kernel
    g_dispatch += (kernel == ATTN_K_BLOCK) ? 4 : 1;
    size_t pairs = (size_t)params->tokens * (size_t)params->heads;
    size_t groups = pairs;
    if (kernel == ATTN_K_BLOCK) {
        // block16: one group per 16-row Q block per head.
        groups = ((size_t)params->tokens + 15) / 16 * (size_t)params->heads;
    } else if (kernel == ATTN_K_WIDE && thread_count == 512) {
        // flash_wide: 16 (token, head) pairs per group, one per simdgroup.
        groups = (pairs + 15) / 16;
    } else if (kernel == ATTN_K_WIDE && thread_count == 128) {
        // wide MMA (chunked-D): 16 query rows per group, 4 simdgroups each
        // owning a 128-dim D-slice. Keyed on the kernel identity, so a rows
        // PSO that resolves 128 threads can never take this grid.
        groups = ((size_t)params->tokens + 15) / 16 * (size_t)params->heads;
    }
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

int zdraw_metal_run_attention(
    void* queue,
    void* pipeline,
    void* q_buf,
    void* k_buf,
    void* v_buf,
    void* output,
    const ZdrawAttnParams* params,
    uint32_t kernel,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    attention_encode(enc, q.device, (__bridge id<MTLComputePipelineState>)pipeline,
                     q_buf, k_buf, v_buf, output, params, kernel, thread_count, 0, 0);
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Batch-encoder variant of run_attention: same converts + kernel dispatch,
// no own command buffer (rides the resident batch; zero flush boundaries).
int zdraw_metal_run_attention_enc(
    void* batch,
    void* pipeline,
    void* q_buf,
    void* k_buf,
    void* v_buf,
    void* output,
    const ZdrawAttnParams* params,
    uint32_t kernel,
    size_t thread_count
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)bb->cmd;
    attention_encode((__bridge id<MTLComputeCommandEncoder>)bb->enc, cmd.device,
                     (__bridge id<MTLComputePipelineState>)pipeline,
                     q_buf, k_buf, v_buf, output, params, kernel, thread_count, 0, 0);
    return 0;
}

// Offset variant for batched multi-seed denoise: one attention call per image
// segment of the shared q/k/v/o buffers (no cross-image attention).
int zdraw_metal_run_attention_enc_off(
    void* batch,
    void* pipeline,
    void* q_buf,
    void* k_buf,
    void* v_buf,
    void* output,
    const ZdrawAttnParams* params,
    uint32_t kernel,
    size_t thread_count,
    uint64_t in_off,
    uint64_t out_off
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)bb->cmd;
    attention_encode((__bridge id<MTLComputeCommandEncoder>)bb->enc, cmd.device,
                     (__bridge id<MTLComputePipelineState>)pipeline,
                     q_buf, k_buf, v_buf, output, params, kernel, thread_count,
                     in_off, out_off);
    return 0;
}

int zdraw_metal_run_qk_norm_rope(
    void* queue,
    void* pipeline,
    void* q_buf,
    void* k_buf,
    void* q_weight,
    void* k_weight,
    void* pos_buf,
    void* rope_buf,
    const ZdrawQkNormParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)q_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)k_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)q_weight offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)k_weight offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)pos_buf offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)rope_buf offset:0 atIndex:5];
    [enc setBytes:params length:sizeof(ZdrawQkNormParams) atIndex:6];
    size_t groups = (size_t)params->tokens * (params->heads + params->kv_heads);
    MTLSize grid = MTLSizeMake(groups, 1, 1);
    MTLSize threads = MTLSizeMake(thread_count, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_block_kernel(
    void* queue,
    void* pipeline,
    void* input_buf,
    void* weight_buf,
    void* scale_buf,
    void* output_buf,
    const ZdrawBlockParams* params,
    size_t thread_count
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)scale_buf offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output_buf offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawBlockParams) atIndex:4];
    MTLSize grid = MTLSizeMake(params->tokens, 1, 1);
    MTLSize threads = MTLSizeMake(thread_count, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_gemm(
    void* queue,
    void* pipeline,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    gemm_encode(enc, (__bridge id<MTLComputePipelineState>)pipeline,
                a_buf, w_buf, c_buf, params, 0, 0);
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// --- Vendored steel_gemm (MLX, MIT) bench path: load the offline-compiled
// metallib and run gemm<half,32,32,16,2,2,false,true>. Validation only. ---
typedef struct {
    int M; int N; int K;
    int lda; int ldb; int ldd;
    int tiles_n; int tiles_m;
    int64_t batch_stride_a; int64_t batch_stride_b; int64_t batch_stride_d;
    int swizzle_log;
    int gemm_k_iterations_aligned;
    int batch_ndim;
} SteelGEMMParams;

// bm: 32 or 64 (tile size). align_M passed in since M may not divide BM.
void* zdraw_metal_steel_make(void* device_, const char* path, int bm,
                             int align_m, int align_n, int align_k) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
    if (!lib) return NULL;
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool f = false;
    bool am = align_m != 0, an = align_n != 0, ak = align_k != 0;
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:10];   // has_batch
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:100];  // use_out_source
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:110];  // do_axpby
    [fc setConstantValue:&am type:MTLDataTypeBool atIndex:200]; // align_M
    [fc setConstantValue:&an type:MTLDataTypeBool atIndex:201]; // align_N
    [fc setConstantValue:&ak type:MTLDataTypeBool atIndex:202]; // align_K
    NSString* name = (bm == 64) ? @"steel_gemm_h_64" : @"steel_gemm_h_32";
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:fc error:&err];
    if (!fn) return NULL;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) return NULL;
    return (__bridge_retained void*)pso;
}

int zdraw_metal_steel_run(void* queue_, void* pipe_, void* a_, void* b_, void* d_,
                          int M, int N, int K, int bm) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipe_;
    SteelGEMMParams p;
    p.M = M; p.N = N; p.K = K;
    p.lda = K; p.ldb = K; p.ldd = N;             // A[M,K], B[N,K] (transpose_b), D[M,N]
    p.tiles_n = (N + bm - 1) / bm; p.tiles_m = (M + bm - 1) / bm;
    p.batch_stride_a = 0; p.batch_stride_b = 0; p.batch_stride_d = 0;
    p.swizzle_log = 0;
    p.gemm_k_iterations_aligned = K / 16;        // BK = 16
    p.batch_ndim = 0;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)b_ offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)d_ offset:0 atIndex:3];  // C(2) gated off
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    g_dispatch++;
    MTLSize grid = MTLSizeMake((NSUInteger)p.tiles_n, (NSUInteger)p.tiles_m, 1);
    MTLSize tg = MTLSizeMake(32, 2, 2);          // WM*WN*32 = 128 threads
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// --- W6 steel GEMM (zdraw): steel's MMA scheduling + a W6-dequant B-loader.
// Reads OUR W6 weight (codes-then-scales, src/zw6.zig), f16 A, f16 D out.
// Same metallib as the f16 steel kernel; host name "steel_gemm_w6_64". ---
// Config-variant helpers for tile tuning: fn name + BM/BN/BK/WM/WN explicit.
void* zdraw_metal_steel_w6_make_cfg(void* device_, const char* path, const char* fname,
                                    int align_m, int align_n, int align_k) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
    if (!lib) return NULL;
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool am = align_m != 0, an = align_n != 0, ak = align_k != 0;
    [fc setConstantValue:&am type:MTLDataTypeBool atIndex:200];
    [fc setConstantValue:&an type:MTLDataTypeBool atIndex:201];
    [fc setConstantValue:&ak type:MTLDataTypeBool atIndex:202];
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fname]
                                   constantValues:fc error:&err];
    if (!fn) return NULL;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) return NULL;
    return (__bridge_retained void*)pso;
}

int zdraw_metal_steel_w6_run_cfg(void* queue_, void* pipe_, void* a_, void* w_, void* d_,
                                 int M, int N, int K, int bm, int bn, int bk, int wm, int wn) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipe_;
    SteelGEMMParams p;
    p.M = M; p.N = N; p.K = K;
    p.lda = K; p.ldb = K; p.ldd = N;
    p.tiles_n = (N + bn - 1) / bn; p.tiles_m = (M + bm - 1) / bm;
    p.batch_stride_a = 0; p.batch_stride_b = 0; p.batch_stride_d = 0;
    p.swizzle_log = 0;
    p.gemm_k_iterations_aligned = K / bk;
    p.batch_ndim = 0;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_ offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)d_ offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    g_dispatch++;
    MTLSize grid = MTLSizeMake((NSUInteger)p.tiles_n, (NSUInteger)p.tiles_m, 1);
    MTLSize tg = MTLSizeMake(32, (NSUInteger)wm, (NSUInteger)wn);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -1;
}

void* zdraw_metal_steel_w6_make(void* device_, const char* path,
                                int align_m, int align_n, int align_k) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
    if (!lib) return NULL;
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool am = align_m != 0, an = align_n != 0, ak = align_k != 0;
    [fc setConstantValue:&am type:MTLDataTypeBool atIndex:200]; // align_M
    [fc setConstantValue:&an type:MTLDataTypeBool atIndex:201]; // align_N
    [fc setConstantValue:&ak type:MTLDataTypeBool atIndex:202]; // align_K
    id<MTLFunction> fn = [lib newFunctionWithName:@"steel_gemm_w6_64" constantValues:fc error:&err];
    if (!fn) return NULL;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) return NULL;
    return (__bridge_retained void*)pso;
}

// a_: f16 A [M,K]; w_: W6 weight buffer; d_: f16 D [M,N]. K-aligned to 16,
// N to 64 for the FFN shapes; M may be unaligned (align_m passed at make).
int zdraw_metal_steel_w6_run(void* queue_, void* pipe_, void* a_, void* w_, void* d_,
                             int M, int N, int K) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipe_;
    SteelGEMMParams p;
    p.M = M; p.N = N; p.K = K;
    p.lda = K; p.ldb = K; p.ldd = N;             // A[M,K], W6[N,K], D[M,N]
    p.tiles_n = (N + 63) / 64; p.tiles_m = (M + 63) / 64;
    p.batch_stride_a = 0; p.batch_stride_b = 0; p.batch_stride_d = 0;
    p.swizzle_log = 0;
    p.gemm_k_iterations_aligned = K / 16;        // BK = 16
    p.batch_ndim = 0;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_ offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)d_ offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    g_dispatch++;
    MTLSize grid = MTLSizeMake((NSUInteger)p.tiles_n, (NSUInteger)p.tiles_m, 1);
    MTLSize tg = MTLSizeMake(32, 2, 2);          // WM*WN*32 = 128 threads
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Resident steel path: load the metallib once and encode gemm<half,64,...> into
// the live chain encoder. align_M=false (token count M varies by image size);
// N (model dims) and K are always 64/16-aligned. f16 A, f16 W, f16 D out.
// Where the steel metallib lives: ZDRAW_STEEL_LIB, else lib/steel.metallib
// next to the running binary (build.zig installs it there on macOS), else
// the legacy /tmp path tools/build_steel_lib.sh writes.
static NSString* steel_lib_path(void) {
    static NSString* cached = nil;
    if (cached) return cached;
    const char* env = getenv("ZDRAW_STEEL_LIB");
    if (env && env[0]) {
        cached = [NSString stringWithUTF8String:env];
        return cached;
    }
    char exe[4096];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) == 0) {
        NSString* dir = [[[NSString stringWithUTF8String:exe] stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent];
        // <dir>/lib (the release archive and the Homebrew cask: zdraw beside
        // lib/) before <dir>/../lib (zig-out/bin beside zig-out/lib).
        NSArray<NSString*>* candidates = @[ @"lib/steel.metallib", @"../lib/steel.metallib" ];
        for (NSString* rel in candidates) {
            NSString* beside = [[dir stringByAppendingPathComponent:rel] stringByStandardizingPath];
            if ([[NSFileManager defaultManager] fileExistsAtPath:beside]) {
                cached = beside;
                return cached;
            }
        }
    }
    cached = @"/tmp/steel_gemm.metallib";
    return cached;
}

// Doctor probe: where the steel metallib resolves and whether it carries the
// kernels the default route needs. 0 ok, 1 missing, 2 unusable (err_out),
// 3 stale (err_out names the first missing kernel).
int zdraw_metal_steel_probe(char* path_out, size_t path_len, char* err_out, size_t err_len) {
    NSString* path = steel_lib_path();
    strlcpy(path_out, path.UTF8String, path_len);
    if (err_len) err_out[0] = 0;
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) return 1;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { strlcpy(err_out, "no Metal device", err_len); return 2; }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) { strlcpy(err_out, err.localizedDescription.UTF8String, err_len); return 2; }
    NSArray<NSString*>* names = [lib functionNames];
    const char* need[] = { "steel_gemm_h_64", "steel_attn_h128" };
    for (size_t i = 0; i < sizeof(need) / sizeof(need[0]); i++) {
        if (![names containsObject:[NSString stringWithUTF8String:need[i]]]) {
            strlcpy(err_out, need[i], err_len);
            return 3;
        }
    }
    return 0;
}

static id<MTLComputePipelineState> steel_chain_pipe(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    static int tried = 0;
    if (tried) return pipe;
    tried = 1;
    NSString* path = steel_lib_path();
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) {
        fprintf(stderr, "zdraw: WARNING steel metallib unusable at %s (%s); "
                        "falling back to MPS f16-output GEMMs\n",
                [path UTF8String],
                err ? [[err localizedDescription] UTF8String] : "no error");
        return nil;
    }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool f = false, t = true;
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:10];   // has_batch
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:100];  // use_out_source
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:110];  // do_axpby
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:200];  // align_M (M varies)
    [fc setConstantValue:&t type:MTLDataTypeBool atIndex:201];  // align_N
    [fc setConstantValue:&t type:MTLDataTypeBool atIndex:202];  // align_K
    id<MTLFunction> fn = [lib newFunctionWithName:@"steel_gemm_h_64" constantValues:fc error:&err];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static SteelGEMMParams steel_params_for(const GemmParams* gp, int tile) {
    SteelGEMMParams p;
    p.M = (int)gp->m; p.N = (int)gp->n; p.K = (int)gp->k;
    p.lda = (int)gp->k; p.ldb = (int)gp->k; p.ldd = (int)gp->n;
    p.tiles_n = ((int)gp->n + tile - 1) / tile;
    p.tiles_m = ((int)gp->m + tile - 1) / tile;
    p.batch_stride_a = 0; p.batch_stride_b = 0; p.batch_stride_d = 0;
    p.swizzle_log = 0;
    p.gemm_k_iterations_aligned = (int)gp->k / 16;
    p.batch_ndim = 0;
    return p;
}

// Encode a Steel GEMM into the open encoder. The pipeline decides whether D is
// f16 or f32; gp carries M/K/N and the byte weight offset.
static int encode_steel_off(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                            id<MTLBuffer> a16, void* w, void* d, const GemmParams* gp,
                            uint64_t a_off, uint64_t c_off) {
    SteelGEMMParams p = steel_params_for(gp, 64);
    [enc setComputePipelineState:pipe];
    [enc setBuffer:a16 offset:(NSUInteger)a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w offset:gp->weight_offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)d offset:(NSUInteger)c_off atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.tiles_n, (NSUInteger)p.tiles_m, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
    return 1;
}

static int encode_steel(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                        id<MTLBuffer> a16, void* w, void* d, const GemmParams* gp) {
    return encode_steel_off(enc, pipe, a16, w, d, gp, 0, 0);
}

// Resident steel W6 path: load steel_gemm_w6_64 from the same metallib and encode
// into the live chain encoder. The W6 dequant loader (loader_w6.h) reads OUR
// packed layout (codes-then-scales, group 64 — identical to gemm_w6_staged), so
// only the weight buffer dtype differs from the f16 steel path: A is f16, W is the
// W6 uchar buffer, D is f16. Same function constants (align_M off, N/K on).
static id<MTLComputePipelineState> steel_w6_chain_pipe(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    static int tried = 0;
    if (tried) return pipe;
    tried = 1;
    NSString* path = steel_lib_path();
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) {
        // Loud fallback: a missing/corrupt metallib silently cost ~3x on every
        // W6 GEMM before this warning existed. Rebuild: tools/build_steel_lib.sh
        fprintf(stderr, "zdraw: WARNING steel W6 metallib unusable at %s (%s); "
                        "falling back to the slow staged W6 kernel\n",
                [path UTF8String],
                err ? [[err localizedDescription] UTF8String] : "no error");
        return nil;
    }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool f = false, t = true;
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:200];  // align_M (M varies)
    [fc setConstantValue:&t type:MTLDataTypeBool atIndex:201];  // align_N
    [fc setConstantValue:&t type:MTLDataTypeBool atIndex:202];  // align_K
    id<MTLFunction> fn = [lib newFunctionWithName:@"steel_gemm_w6_64" constantValues:fc error:&err];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// Encode a steel W6 GEMM (f16 A x W6 W^T -> f16 D) into the open encoder. The W6
// kernel's bindings match encode_steel exactly (A@0, W@1 at byte weight_offset,
// D@3, params@4); W is the packed uchar buffer instead of f16, dequantized in the
// threadgroup load. gp->weight_offset is the byte offset into the W6 buffer.
static int encode_steel_w6(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                           id<MTLBuffer> a16, void* w, void* d, const GemmParams* gp) {
    SteelGEMMParams p;
    p.M = (int)gp->m; p.N = (int)gp->n; p.K = (int)gp->k;
    p.lda = (int)gp->k; p.ldb = (int)gp->k; p.ldd = (int)gp->n;
    p.tiles_n = ((int)gp->n + 63) / 64; p.tiles_m = ((int)gp->m + 63) / 64;
    p.batch_stride_a = 0; p.batch_stride_b = 0; p.batch_stride_d = 0;
    p.swizzle_log = 0; p.gemm_k_iterations_aligned = (int)gp->k / 16; p.batch_ndim = 0;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:a16 offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w offset:gp->weight_offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)d offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.tiles_n, (NSUInteger)p.tiles_m, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
    return 1;
}

// ZDRAW_STEEL=1/"all" -> every verified f16 GEMM; or a comma list like
// "gateup,qkv,proj" to scope. Parts: gateup, qkv, proj, down.
static int steel_scope(const char* part) {
    const char* s = getenv("ZDRAW_STEEL");
    if (!s) return 0;
    if (strcmp(s, "1") == 0 || strcmp(s, "all") == 0) return 1;  // all GEMMs (proj/down fixed)
    return strstr(s, part) != NULL;
}

// ZDRAW_STEEL_W6 gates the steel W6 path (low-memory W6 weights at steel speed).
// "1"/"all" -> every W6 GEMM; or a comma list (qkv,proj,gateup,down) to scope.
// Default OFF: when unset the chain keeps the slow gemm_w6_staged kernel.
static int steel_w6_scope(const char* part) {
    const char* s = getenv("ZDRAW_STEEL_W6");
    if (!s) return 0;
    if (strcmp(s, "1") == 0 || strcmp(s, "all") == 0) return 1;
    return strstr(s, part) != NULL;
}

static id<MTLComputePipelineState> f16_to_f32_pipe(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void f16_to_f32(const device half* in [[buffer(0)]],"
        "device float* out [[buffer(1)]], constant uint& count [[buffer(2)]],"
        "constant float& scale [[buffer(3)]],"
        "uint id [[thread_position_in_grid]]) { if (id < count) out[id] = float(in[id]) * scale; }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"f16_to_f32"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// Convert an f16 buffer to f32 (scale 1). Used at the f16-VAE boundary so the
// finish pass keeps reading f32 features.
int zdraw_metal_run_f16_to_f32(void* queue_, void* device_, void* in_, void* out_, uint32_t count) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pipe = f16_to_f32_pipe((__bridge id<MTLDevice>)device_);
    if (!pipe) return -1;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    float scale = 1.0f;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)in_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_ offset:0 atIndex:1];
    [enc setBytes:&count length:4 atIndex:2];
    [enc setBytes:&scale length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    g_dispatch++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}


// Steel output scratch (f16), slot-keyed so q/k/v can overlap (distinct buffers).
static id<MTLBuffer> steel_scratch(id<MTLDevice> device, int slot, size_t bytes) {
    static id<MTLBuffer> bufs[5] = {nil, nil, nil, nil, nil};
    if (slot < 0 || slot > 4) return nil;
    if (!bufs[slot] || bufs[slot].length < bytes)
        bufs[slot] = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return bufs[slot];
}

// Steel with a pre-computed f16 A, output cast f16->f32 into dst (for consumers
// that read f32: qk-norm, residual). a16 is shared across q/k/v (same b->norm).
static int encode_steel_a16(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> sp,
                            id<MTLBuffer> a16, void* w, void* dst_f32, int scratch_slot,
                            void* scratch_override, float out_scale, const GemmParams* gp) {
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_f32;
    size_t out_count = (size_t)gp->m * gp->n;
    // A chain-managed f16 buffer (scratch_override) survives the MPS-attention
    // command-buffer boundary; the static steel_scratch does not (proj/down bug).
    id<MTLBuffer> scratch = scratch_override
        ? (__bridge id<MTLBuffer>)scratch_override
        : steel_scratch(dst.device, scratch_slot, out_count * 2);
    id<MTLComputePipelineState> cast = f16_to_f32_pipe(dst.device);
    if (!scratch || !cast) return 0;
    encode_steel(enc, sp, a16, w, (__bridge void*)scratch, gp);
    uint32_t n = (uint32_t)out_count;
    [enc setComputePipelineState:cast];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];  // steel write -> cast read
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&out_scale length:sizeof(float) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch++;
    return 1;
}

// Steel W6 with a pre-computed f16 A, output cast f16->f32 into dst. Mirrors
// encode_steel_a16 exactly (same f16 scratch + f16->f32 cast + the 1/256-in /
// 256-out overflow handling for proj/down), but the GEMM is the W6 dequant
// kernel reading the packed uchar weight. w is the W6 weight buffer; gp carries
// M/K/N and the byte weight_offset.
static int encode_steel_w6_a16(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> sp,
                               id<MTLBuffer> a16, void* w, void* dst_f32, int scratch_slot,
                               void* scratch_override, float out_scale, const GemmParams* gp) {
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_f32;
    size_t out_count = (size_t)gp->m * gp->n;
    id<MTLBuffer> scratch = scratch_override
        ? (__bridge id<MTLBuffer>)scratch_override
        : steel_scratch(dst.device, scratch_slot, out_count * 2);
    id<MTLComputePipelineState> cast = f16_to_f32_pipe(dst.device);
    if (!scratch || !cast) return 0;
    encode_steel_w6(enc, sp, a16, w, (__bridge void*)scratch, gp);
    uint32_t n = (uint32_t)out_count;
    [enc setComputePipelineState:cast];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];  // steel write -> cast read
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&out_scale length:sizeof(float) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch++;
    return 1;
}

static void encode_gemm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params);

int zdraw_metal_run_gemm_pair(
    void* queue,
    void* pipeline,
    void* a_buf,
    void* w0_buf,
    void* c0_buf,
    const GemmParams* params0,
    void* w1_buf,
    void* c1_buf,
    const GemmParams* params1
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_gemm(enc, pipe, a_buf, w0_buf, c0_buf, params0);
    encode_gemm(enc, pipe, a_buf, w1_buf, c1_buf, params1);
    [enc endEncoding];

    count_gemm(params0);
    count_gemm(params1);

    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static void encode_gemm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    MTLSize groups = MTLSizeMake(ceil_div_u32(params->n, 32u),
                                 ceil_div_u32(params->m, 32u), 1);
    MTLSize threads = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static int exact_staged_enabled(void);
static int exact_staged_ok(const GemmParams* params);
static id<MTLComputePipelineState> exact_staged_pipeline(id<MTLDevice> device);

static void encode_gemm_auto(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> exact,
    id<MTLComputePipelineState> half,
    id<MTLComputePipelineState> w8,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    if (exact_staged_enabled() && exact_staged_ok(params)) {
        id<MTLBuffer> ab = (__bridge id<MTLBuffer>)a_buf;
        id<MTLComputePipelineState> staged = exact_staged_pipeline(ab.device);
        static int reported = 0;
        if (!reported) {
            reported = 1;
            fprintf(stderr, "exact-staged: %s\n", staged ? "ACTIVE" : "PIPELINE-NIL");
        }
        if (staged) {
            [enc setComputePipelineState:staged];
            [enc setBuffer:ab offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((params->n + 63) / 64,
                                                  (params->m + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
    }
    id<MTLComputePipelineState> pipe = (params->mode == 3) ? w8 : (params->mode == 2) ? half : exact;
    encode_gemm(enc, pipe, a_buf, w_buf, c_buf, params);
}

static void count_gemm(const GemmParams* params) {
    g_dispatch++;
    g_gemm++;
    if (params->mode == 3) g_gemm_w8++;
    else if (params->mode == 2) g_gemm_half++;
    else if (params->mode == 4) g_gemm_w6++;
    else g_gemm_exact++;
}

static void count_mps_gemm(void) {
    g_dispatch++;
    g_gemm++;
    g_gemm_mps++;
}

static int dense_mps_gateup_enabled(void) {
    const char* raw = getenv("ZDRAW_DENSE");
    if (!raw) return 0;
    return strcmp(raw, "mps-ffn-gateup") == 0 ||
           strcmp(raw, "mps-ffn-qkv") == 0 ||
           strcmp(raw, "mps-ffn-proj") == 0 ||
           strcmp(raw, "mps-ffn-qkvo") == 0 ||
           strcmp(raw, "mps-dense") == 0 ||
           strcmp(raw, "mps-dense-full") == 0;
}

static int dense_mps_qkv_enabled(void) {
    const char* raw = getenv("ZDRAW_DENSE");
    if (!raw) return 0;
    return strcmp(raw, "mps-qkv") == 0 ||
           strcmp(raw, "mps-qkvo") == 0 ||
           strcmp(raw, "mps-ffn-qkv") == 0 ||
           strcmp(raw, "mps-ffn-qkvo") == 0 ||
           strcmp(raw, "mps-dense") == 0 ||
           strcmp(raw, "mps-dense-full") == 0;
}

static int dense_mps_proj_enabled(void) {
    const char* raw = getenv("ZDRAW_DENSE");
    if (!raw) return 0;
    return strcmp(raw, "mps-proj") == 0 ||
           strcmp(raw, "mps-qkvo") == 0 ||
           strcmp(raw, "mps-ffn-proj") == 0 ||
           strcmp(raw, "mps-ffn-qkvo") == 0 ||
           strcmp(raw, "mps-dense") == 0 ||
           strcmp(raw, "mps-dense-full") == 0;
}

static int dense_mps_down_enabled(void) {
    const char* raw = getenv("ZDRAW_DENSE");
    if (!raw) return 0;
    return strcmp(raw, "mps-ffn-down") == 0 ||
           strcmp(raw, "mps-dense-full") == 0;
}

static int dense_mps_any_enabled(void) {
    const char* raw = getenv("ZDRAW_DENSE");
    if (!raw) return 1;
    return strcmp(raw, "mps-qkvo") == 0 ||
           strcmp(raw, "mps-qkv") == 0 ||
           strcmp(raw, "mps-proj") == 0 ||
           strcmp(raw, "mps-ffn-gateup") == 0 ||
           strcmp(raw, "mps-ffn-qkv") == 0 ||
           strcmp(raw, "mps-ffn-proj") == 0 ||
           strcmp(raw, "mps-ffn-qkvo") == 0 ||
           strcmp(raw, "mps-ffn-down") == 0 ||
           strcmp(raw, "mps-dense") == 0 ||
           strcmp(raw, "mps-dense-full") == 0 ||
           strcmp(raw, "ours-f16") == 0 ||
           strcmp(raw, "ours-v2") == 0;
}

static size_t dtype_size(uint32_t dtype) {
    if (dtype == 3) return 4;
    return 2;
}

static int mps_dtype(uint32_t dtype, MPSDataType* out) {
    switch (dtype) {
        case 1: *out = MPSDataTypeFloat16; return 1;
        case 2: *out = MPSDataTypeBFloat16; return 1;
        case 3: *out = MPSDataTypeFloat32; return 1;
        default: return 0;
    }
}

static int mps_gemm_eligible(const GemmParams* params) {
    if (params->mode != 2) return 0;
    if (params->m == 0 || params->k == 0 || params->n == 0) return 0;
    if (params->m % 32 != 0 || params->n % 32 != 0) return 0;
    MPSDataType tmp;
    return mps_dtype(params->dtype, &tmp);
}

// --- ZDRAW_MPS_F16A probe: stage f32 activations to f16 so MPS runs its
// all-f16 GEMM kernels in-chain. Self-contained: own convert pipeline and
// staging buffers, no Zig-side changes. Diagnostic, not a product path.
static int mps_f16a_enabled(void) {
    const char* raw = getenv("ZDRAW_MPS_F16A");
    return raw && strcmp(raw, "0") != 0;
}

static id<MTLComputePipelineState> f16a_convert(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void f32_to_f16(const device float* in [[buffer(0)]],"
                     "device half* out [[buffer(1)]],"
                     "constant uint& count [[buffer(2)]],"
                     "constant float& scale [[buffer(3)]],"
                     "uint id [[thread_position_in_grid]]) {"
                     "if (id < count) out[id] = half(clamp(in[id] * scale, -65504.0f, 65504.0f)); }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"f32_to_f16"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// Staging buffers keyed by slot (norm, mix, gate), grown on demand.
static id<MTLBuffer> f16a_stage(id<MTLDevice> device, int slot, size_t bytes) {
    static id<MTLBuffer> stages[3] = {nil, nil, nil};
    if (slot < 0 || slot > 2) return nil;
    if (!stages[slot] || stages[slot].length < bytes) {
        stages[slot] = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    return stages[slot];
}

// Convert `count` f32 values into the slot's f16 stage inside the current
// encoder; returns the stage buffer (or nil to fall back to f32 A).
static id<MTLBuffer> f16a_encode_convert_scaled(
    id<MTLComputeCommandEncoder> enc,
    void* src_buf,
    int slot,
    uint32_t count,
    float scale
) {
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_buf;
    id<MTLComputePipelineState> pipe = f16a_convert(src.device);
    if (!pipe) return nil;
    id<MTLBuffer> stage = f16a_stage(src.device, slot, (size_t)count * 2);
    if (!stage) return nil;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:stage offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&scale length:sizeof(float) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch++;
    return stage;
}

static id<MTLBuffer> f16a_encode_convert(
    id<MTLComputeCommandEncoder> enc,
    void* src_buf,
    int slot,
    uint32_t count
) {
    return f16a_encode_convert_scaled(enc, src_buf, slot, count, 1.0f);
}

static int swiglu_f16a_fuse_enabled(void) {
    const char* raw = getenv("ZDRAW_SWIGLU_F16A_FUSE");
    return !raw || strcmp(raw, "0") != 0;
}

// Product-path fold: the W6 down GEMM wants f16 A. The old path wrote SwiGLU
// into a full f32 buffer, then immediately read it back to cast/scale to f16.
// This kernel computes the exact same f32 expression and stores the same scaled
// half values into the A-stage, removing one dispatch and one f32 round trip.
static id<MTLComputePipelineState> f16a_swiglu_scaled(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void swiglu_to_f16(const device float* gate [[buffer(0)]],"
        "const device float* up [[buffer(1)]],"
        "device half* out [[buffer(2)]],"
        "constant uint& count [[buffer(3)]],"
        "constant float& scale [[buffer(4)]],"
        "uint id [[thread_position_in_grid]]) {"
        "if (id >= count) return;"
        "float g = gate[id];"
        "float v = (g / (1.0f + exp(-g))) * up[id];"
        "out[id] = half(clamp(v * scale, -65504.0f, 65504.0f)); }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"swiglu_to_f16"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static id<MTLBuffer> f16a_encode_swiglu_scaled(
    id<MTLComputeCommandEncoder> enc,
    void* gate_buf,
    void* up_buf,
    int slot,
    uint32_t count,
    float scale
) {
    id<MTLBuffer> gate = (__bridge id<MTLBuffer>)gate_buf;
    id<MTLComputePipelineState> pipe = f16a_swiglu_scaled(gate.device);
    if (!pipe) return nil;
    id<MTLBuffer> stage = f16a_stage(gate.device, slot, (size_t)count * 2);
    if (!stage) return nil;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_buf offset:0 atIndex:1];
    [enc setBuffer:stage offset:0 atIndex:2];
    [enc setBytes:&count length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&scale length:sizeof(float) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch++;
    return stage;
}

static int norm_f16a_fuse_enabled(void) {
    const char* raw = getenv("ZDRAW_NORM_F16A_FUSE");
    return !raw || strcmp(raw, "0") != 0;
}

// Fold block_norm_scale + f32->f16 A staging. The normalized f32 `b->norm`
// produced before qkv is consumed only by q/k/v and overwritten later by the
// attention-residual norm, so the P0 W6-steel qkv path can stage half A
// directly without materializing the intermediate f32 feature map.
static id<MTLComputePipelineState> block_norm_to_f16(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Params { uint tokens; uint hidden; uint dtype; uint has_scale;"
        "float eps; ulong weight_offset; };"
        "static inline float weight_at(const device uchar* base, uint index, uint dtype) {"
        "if (dtype == 3) return ((const device float*)base)[index];"
        "ushort bits = ((const device ushort*)base)[index];"
        "if (dtype == 2) return as_type<float>(uint(bits) << 16);"
        "return float(as_type<half>(bits)); }"
        "kernel void norm_to_f16(const device float* input [[buffer(0)]],"
        "const device uchar* weight_bytes [[buffer(1)]],"
        "const device float* scale [[buffer(2)]],"
        "device half* output [[buffer(3)]],"
        "constant Params& p [[buffer(4)]],"
        "uint tok [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint tg_size [[threads_per_threadgroup]]) {"
        "threadgroup float reduce[256];"
        "uint base = tok * p.hidden;"
        "const device uchar* weight = weight_bytes + p.weight_offset;"
        "float local = 0.0f;"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = input[base + dim]; local += value * value; }"
        "reduce[tid] = local;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float norm = rsqrt(reduce[0] / float(p.hidden) + p.eps);"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = input[base + dim] * norm * weight_at(weight, dim, p.dtype);"
        "if (p.has_scale != 0) value *= scale[dim];"
        "output[base + dim] = half(clamp(value, -65504.0f, 65504.0f)); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"norm_to_f16"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static id<MTLBuffer> f16a_encode_norm(
    id<MTLComputeCommandEncoder> enc,
    void* input,
    void* weight,
    void* scale,
    int slot,
    const ZdrawBlockParams* params,
    size_t thread_count
) {
    id<MTLBuffer> in = (__bridge id<MTLBuffer>)input;
    id<MTLComputePipelineState> pipe = block_norm_to_f16(in.device);
    if (!pipe) return nil;
    const uint32_t count = params->tokens * params->hidden;
    id<MTLBuffer> stage = f16a_stage(in.device, slot, (size_t)count * 2);
    if (!stage) return nil;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:2];
    [enc setBuffer:stage offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawBlockParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(params->tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
    // No count: this fused kernel replaces the attn-norm slot already covered
    // by count_chain_layer's non-GEMM total (and the qkv convert it absorbs).
    return stage;
}

static int ffn_resid_f16_fuse_enabled(void) {
    const char* raw = getenv("ZDRAW_FFN_RESID_F16_FUSE");
    return !raw || strcmp(raw, "0") != 0;
}

// Fold f16-down-output cast + FFN residual. The W6 down path writes f16 D then
// used to cast D to f32 before block_residual_norm. Reading the same half values
// here and multiplying by the same power-of-two scale produces the identical f32
// input values, then runs the same RMSNorm/residual math as block_residual_norm.
static id<MTLComputePipelineState> block_residual_norm_f16_scaled(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Params { uint tokens; uint hidden; uint dtype; uint has_scale;"
        "float eps; ulong weight_offset; };"
        "static inline float weight_at(const device uchar* base, uint index, uint dtype) {"
        "if (dtype == 3) return ((const device float*)base)[index];"
        "ushort bits = ((const device ushort*)base)[index];"
        "if (dtype == 2) return as_type<float>(uint(bits) << 16);"
        "return float(as_type<half>(bits)); }"
        "kernel void residual_f16_scaled(const device half* input [[buffer(0)]],"
        "const device uchar* weight_bytes [[buffer(1)]],"
        "const device float* gate [[buffer(2)]],"
        "device float* state [[buffer(3)]],"
        "constant Params& p [[buffer(4)]],"
        "constant float& input_scale [[buffer(5)]],"
        "uint tok [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint tg_size [[threads_per_threadgroup]]) {"
        "threadgroup float reduce[256];"
        "uint base = tok * p.hidden;"
        "const device uchar* weight = weight_bytes + p.weight_offset;"
        "float local = 0.0f;"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = float(input[base + dim]) * input_scale;"
        "local += value * value; }"
        "reduce[tid] = local;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float norm = rsqrt(reduce[0] / float(p.hidden) + p.eps);"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = float(input[base + dim]) * input_scale;"
        "float inc = value * norm * weight_at(weight, dim, p.dtype);"
        "if (p.has_scale != 0) inc *= gate[dim];"
        "state[base + dim] += inc; }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"residual_f16_scaled"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static int encode_resid_f16_scaled(
    id<MTLComputeCommandEncoder> enc,
    void* input_f16,
    void* weight,
    void* gate,
    void* state,
    const ZdrawBlockParams* params,
    float input_scale,
    size_t thread_count
) {
    id<MTLBuffer> in = (__bridge id<MTLBuffer>)input_f16;
    id<MTLComputePipelineState> pipe = block_residual_norm_f16_scaled(in.device);
    if (!pipe) return 0;
    [enc setComputePipelineState:pipe];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)state offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawBlockParams) atIndex:4];
    [enc setBytes:&input_scale length:sizeof(float) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(params->tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
    g_dispatch++;
    return 1;
}

static int mps_f16c_enabled(void) {
    const char* raw = getenv("ZDRAW_MPS_F16C");
    return raw && strcmp(raw, "0") != 0;
}

// Half-input fused swiglu for the f16-C probe (bridge-compiled, no Zig churn).
static id<MTLComputePipelineState> f16c_swiglu(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void swiglu_fused_f16in(const device half* gateup [[buffer(0)]],"
                     "device float* out [[buffer(1)]],"
                     "constant uint& count [[buffer(2)]],"
                     "constant uint& inner [[buffer(3)]],"
                     "uint id [[thread_position_in_grid]]) {"
                     "if (id >= count) return;"
                     "uint row = id / inner; uint col = id - row * inner;"
                     "float g = float(gateup[row * 2 * inner + col]);"
                     "float u = float(gateup[row * 2 * inner + inner + col]);"
                     "out[id] = (g / (1.0f + exp(-g))) * u; }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"swiglu_fused_f16in"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// --- ZDRAW_GPU_TRACE=1: per-family GPU time inside the real chain.
// Apple GPUs sample counters at stage boundaries only, so the traced chain
// splits into per-family encoders whose start/end timestamps land in one
// MTLCounterSampleBuffer, resolved and bucketed into the SP_ families.
static int gpu_trace_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_GPU_TRACE");
        cached = (raw && strcmp(raw, "0") != 0) ? 1 : 0;
    }
    return cached;
}

#define TRACE_MAX_SAMPLES 4096
static id<MTLCounterSampleBuffer> g_trace_buf = nil;
static uint32_t g_trace_used = 0;
static uint8_t g_trace_kind[TRACE_MAX_SAMPLES / 2];

static id<MTLCounterSampleBuffer> trace_buffer(id<MTLDevice> device) {
    if (g_trace_buf) return g_trace_buf;
    id<MTLCounterSet> set = nil;
    for (id<MTLCounterSet> s in device.counterSets) {
        if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) set = s;
    }
    if (!set) return nil;
    MTLCounterSampleBufferDescriptor* d = [[MTLCounterSampleBufferDescriptor alloc] init];
    d.counterSet = set;
    d.sampleCount = TRACE_MAX_SAMPLES;
    d.storageMode = MTLStorageModeShared;
    NSError* err = nil;
    g_trace_buf = [device newCounterSampleBufferWithDescriptor:d error:&err];
    return g_trace_buf;
}

// Open a traced compute encoder whose start/end timestamps are tagged `kind`.
static id<MTLComputeCommandEncoder> trace_encoder(
    id<MTLCommandBuffer> cmd,
    uint64_t kind
) {
    if (!gpu_trace_enabled() || g_trace_used + 2 > TRACE_MAX_SAMPLES) {
        return [cmd computeCommandEncoder];
    }
    id<MTLCounterSampleBuffer> buf = trace_buffer(cmd.device);
    if (!buf) return [cmd computeCommandEncoder];
    MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
    pass.sampleBufferAttachments[0].sampleBuffer = buf;
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = g_trace_used;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = g_trace_used + 1;
    g_trace_kind[g_trace_used / 2] = (uint8_t)kind;
    g_trace_used += 2;
    return [cmd computeCommandEncoderWithDescriptor:pass];
}

// Resolve all samples into the per-family GPU accumulators and reset.
static void zdraw_metal_trace_resolve(void* device_) {
    if (!g_trace_buf || g_trace_used == 0) return;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    MTLTimestamp cpu0, gpu0, cpu1, gpu1;
    [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    NSData* data = [g_trace_buf resolveCounterRange:NSMakeRange(0, g_trace_used)];
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    if (!data) { g_trace_used = 0; return; }
    const double scale = gpu1 > gpu0
        ? (double)(cpu1 - cpu0) / (double)(gpu1 - gpu0)
        : 1.0;
    const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)data.bytes;
    for (uint32_t i = 0; i + 1 < g_trace_used; i += 2) {
        const uint64_t a = ts[i].timestamp;
        const uint64_t b = ts[i + 1].timestamp;
        if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b <= a) continue;
        const uint8_t kind = g_trace_kind[i / 2];
        if (kind < SP_COUNT) {
            g_stack_profile_gpu[kind] += (double)(b - a) * scale / 1e9;
            g_stack_profile_samples[kind]++;
        }
    }
    g_trace_used = 0;
}









static int dense_ours16_enabled(void) {
    // Default ON (the production dense path); other ZDRAW_DENSE values or
    // "off" select the legacy substrates.
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_DENSE");
        // "ours-v2" rides the same route with the direct-W pipeline swapped
        // in at encode time (dense_ours_v2_enabled).
        cached = (!raw || strcmp(raw, "ours-f16") == 0 || strcmp(raw, "ours-v2") == 0) ? 1 : 0;
    }
    return cached;
}

// ZDRAW_DENSE=ours-v2: the MPP-informed variant of gemm_f16_direct — same
// f32->f16 A staging, W simdgroup_loaded DIRECT from device (no Ws staging).
// Bench-proven faster on every 1024-class production shape with bit-equal
// output (ledger gemm-v2-occupancy-probe); needs n % 64 == 0 for the
// unguarded W loads, which every production N satisfies (the 288-class fat
// op keeps the staged kernel by measurement).
static int dense_ours_v2_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_DENSE");
        cached = (raw && strcmp(raw, "ours-v2") == 0) ? 1 : 0;
    }
    return cached;
}

static void ours_v2_report(void) {
    static int reported = 0;
    if (!reported) {
        reported = 1;
        fprintf(stderr, "dense: ours-v2 direct-W kernel active\n");
    }
}

static id<MTLComputePipelineState> ours_v2_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "kernel void gemm_f16_v2(const device float* A [[buffer(0)]],"
        "const device half* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device half* Wp = W + (p.weight_offset >> 1);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "uint cur = 0;"
        "const uint ar = tile_m + row;"
        "const device float4* Arow = (const device float4*)(A + ar * p.k);"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++)"
        "    ad[q] = ar < p.m ? half4(clamp(Arow[(seg >> 2) + q], -65504.0f, 65504.0f)) : half4(0.0h); }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "const uint wr = tile_n + sn;"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++)"
        "      ad[q] = ar < p.m ? half4(clamp(Arow[base + q], -65504.0f, 65504.0f)) : half4(0.0h); }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], Wp + (wr + j * 8) * p.k + k0 + kk, p.k, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f16_v2"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// P1 stage-1 (f16-act-v2-program): half-A twin of gemm_f16_v2. A is already
// f16 in device memory (f16 activation buffers), so the threadgroup A staging
// disappears and BOTH operands simdgroup_load direct from device — the
// geometry the v2a bench kernel measured at 0.88-0.92x MPS. Contract:
// m >= 64, n >= 64, k % 8 == 0; edge tiles clamp-overlap (identical
// rewrites). Dormant until the ActModes policy routes ops here.
static id<MTLComputePipelineState> f16a_v2_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "kernel void gemm_f16a_v2(const device half* A [[buffer(0)]],"
        "const device half* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device half* Wp = W + (p.weight_offset >> 1);"
        "const uint tile_m = min(tg.y * 64, p.m - 64);"
        "const uint tile_n = min(tg.x * 64, p.n - 64);"
        "const uint sm = tile_m + (sgid / 2) * 32;"
        "const uint sn = tile_n + (sgid % 2) * 32;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "for (uint k0 = 0; k0 < p.k; k0 += 8) {"
        "  simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "  for (uint i = 0; i < 4; i++)"
        "    simdgroup_load(a[i], A + (sm + i * 8) * p.k + k0, p.k);"
        "  for (uint j = 0; j < 4; j++)"
        "    simdgroup_load(b[j], Wp + (sn + j * 8) * p.k + k0, p.k, ulong2(0, 0), true);"
        "  for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  simdgroup_store(acc[i][j], C + (sm + i * 8) * p.n + (sn + j * 8), p.n);"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f16a_v2"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// The weight-typed tokens below are macro'd so one source serves both the
// f16 checkpoint path and the bf16 one (Klein's text encoder ships bf16, which
// otherwise cannot reach this kernel at all). bf16 -> half happens during the
// threadgroup staging copy the kernel already performs, so it is free: the MMA
// sees identical half fragments either way.
static id<MTLComputePipelineState> ours16_pipeline_typed(id<MTLDevice> device, int bf16) {
    NSString* prelude = bf16
        // The macro parameter is _w, not x: the preprocessor would also
        // substitute the x in ".x", turning (x).x into (Wrow[i]).Wrow[i].
        ? @"#define WT ushort\n#define WT4 ushort4\n"
           "#define WCVT(_w) half4(clamp(float4("
           "as_type<float>(uint((_w).x) << 16), as_type<float>(uint((_w).y) << 16),"
           "as_type<float>(uint((_w).z) << 16), as_type<float>(uint((_w).w) << 16)),"
           " -65504.0f, 65504.0f))\n"
        : @"#define WT half\n#define WT4 half4\n#define WCVT(_w) (_w)\n";
    NSString* src = [prelude stringByAppendingString:
        @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        // 64x64 macro-tile, 4 simdgroups (2x2 of 32x32), K in steps of 16 with
        // ping-pong threadgroup staging so device latency hides behind MMA.
        "kernel void gemm_f16_direct(const device float* A [[buffer(0)]],"
        "const device WT* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device WT* Wp = W + (p.weight_offset >> 1);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32]; threadgroup half Ws[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "uint cur = 0;"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device float4* Arow = (const device float4*)(A + ar * p.k);"
        "const device WT4* Wrow = (const device WT4*)(Wp + wr * p.k);"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  threadgroup half4* wd = (threadgroup half4*)&Ws[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++) {"
        "    ad[q] = ar < p.m ? half4(clamp(Arow[(seg >> 2) + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "    wd[q] = wr < p.n ? WCVT(Wrow[(seg >> 2) + q]) : half4(0.0h); } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    threadgroup half4* wd = (threadgroup half4*)&Ws[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++) {"
        "      ad[q] = ar < p.m ? half4(clamp(Arow[base + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "      wd[q] = wr < p.n ? WCVT(Wrow[base + q]) : half4(0.0h); } }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * 32 + kk], 32, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}"];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f16_direct"];
    if (!fn) return nil;
    return [device newComputePipelineStateWithFunction:fn error:&err];
}

static id<MTLComputePipelineState> ours16_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (!pipe) pipe = ours16_pipeline_typed(device, 0);
    return pipe;
}

static id<MTLComputePipelineState> ours16_bf16_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (!pipe) pipe = ours16_pipeline_typed(device, 1);
    return pipe;
}

static id<MTLComputePipelineState> ours16_f16a_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "kernel void gemm_f16a_direct(const device half* A [[buffer(0)]],"
        "const device half* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device half* Wp = W + (p.weight_offset >> 1);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32]; threadgroup half Ws[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "uint cur = 0;"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device half4* Arow = (const device half4*)(A + ar * p.k);"
        "const device half4* Wrow = (const device half4*)(Wp + wr * p.k);"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  threadgroup half4* wd = (threadgroup half4*)&Ws[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++) {"
        "    ad[q] = ar < p.m ? Arow[(seg >> 2) + q] : half4(0.0h);"
        "    wd[q] = wr < p.n ? Wrow[(seg >> 2) + q] : half4(0.0h); } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    threadgroup half4* wd = (threadgroup half4*)&Ws[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++) {"
        "      ad[q] = ar < p.m ? Arow[base + q] : half4(0.0h);"
        "      wd[q] = wr < p.n ? Wrow[base + q] : half4(0.0h); } }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * 32 + kk], 32, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f16a_direct"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// Custom kernel (steel-informed, ours): gemm_f16_direct + three techniques —
// (1) f16 A reads (half the activation bandwidth; also the f16-activation
// memory win), (2) +8-half padded staging (LD=40, bank-conflict-free
// simdgroup loads), (3) serpentine MMA traversal (reuse the b register across
// the i boundary). 64x64, 4 simdgroups, BK=32, double-buffered.
static id<MTLComputePipelineState> ours_custom_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "kernel void gemm_f16_custom(const device half* A [[buffer(0)]],"
        "const device half* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device half* Wp = W + (p.weight_offset >> 1);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "const uint LD = 40;"
        "threadgroup half As[2][64 * 40]; threadgroup half Ws[2][64 * 40];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "uint cur = 0;"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device half4* Arow = (const device half4*)(A + ar * p.k);"
        "const device half4* Wrow = (const device half4*)(Wp + wr * p.k);"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * LD + seg];"
        "  threadgroup half4* wd = (threadgroup half4*)&Ws[0][row * LD + seg];"
        "  for (uint q = 0; q < 4; q++) {"
        "    ad[q] = ar < p.m ? Arow[(seg >> 2) + q] : half4(0.0h);"
        "    wd[q] = wr < p.n ? Wrow[(seg >> 2) + q] : half4(0.0h); } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * LD + seg];"
        "    threadgroup half4* wd = (threadgroup half4*)&Ws[1 - cur][row * LD + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++) {"
        "      ad[q] = ar < p.m ? Arow[base + q] : half4(0.0h);"
        "      wd[q] = wr < p.n ? Wrow[base + q] : half4(0.0h); } }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * LD + kk], LD);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * LD + kk], LD, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "      uint jj = (i & 1) ? (3 - j) : j;"
        "      simdgroup_multiply_accumulate(acc[i][jj], a[i], b[jj], acc[i][jj]); } }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f16_custom"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static int klein_custom_gemm_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_KLEIN_CUSTOM_GEMM");
        cached = (raw && raw[0] != '\0' && strcmp(raw, "0") != 0) ? 1 : 0;
    }
    return cached;
}

void* zdraw_metal_ours16_make(void* device_) {
    return (__bridge_retained void*)ours16_pipeline((__bridge id<MTLDevice>)device_);
}
void* zdraw_metal_ourscustom_make(void* device_) {
    return (__bridge_retained void*)ours_custom_pipeline((__bridge id<MTLDevice>)device_);
}
void* zdraw_metal_f16a_v2_make(void* device_) {
    return (__bridge_retained void*)f16a_v2_pipeline((__bridge id<MTLDevice>)device_);
}
// Bench handle for the f16-A kernel Klein's resident path dispatches
// (run_gemm_f16a_enc), so the headroom bench races the production kernel.
void* zdraw_metal_f16a_direct_make(void* device_) {
    return (__bridge_retained void*)ours16_f16a_pipeline((__bridge id<MTLDevice>)device_);
}
// Standalone bench dispatch for our 64x64 kernels (A is f32 for ours16/direct,
// f16 for custom — the caller supplies the matching buffer).
int zdraw_metal_ours64_run(void* queue_, void* pipe_, void* a_, void* w_, void* c_,
                           int M, int N, int K) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipe_;
    GemmParams p;
    p.m = M; p.k = K; p.n = N; p.dtype = 1; p.mode = 0; p.weight_offset = 0;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_ offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_ offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(GemmParams) atIndex:3];
    g_dispatch++;
    [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}


// 1D-grid twin of ours64_run for the v2c locality-walk bench kernel: the
// kernel derives its own (bx, by) from the linear group index over 8-wide
// column bands, so the grid is bands*8*tiles_y groups of 128 threads.
int zdraw_metal_ours64_run_1d(void* queue_, void* pipe_, void* a_, void* w_, void* c_,
                              int M, int N, int K) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipe_;
    GemmParams p;
    p.m = M; p.k = K; p.n = N; p.dtype = 1; p.mode = 0; p.weight_offset = 0;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_ offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_ offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(GemmParams) atIndex:3];
    g_dispatch++;
    size_t tiles_x = ((size_t)N + 63) / 64;
    size_t tiles_y = ((size_t)M + 63) / 64;
    size_t groups = (tiles_x + 7) / 8 * 8 * tiles_y;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Staged f32 exact kernel: the proven ping-pong staging at f32 width.
// K-step 16 keeps threadgroup memory at 16 KB (2 x (64x16 A + 64x16 W) f32).
static int exact_staged_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_EXACT_STAGED");
        cached = (raw && strcmp(raw, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static id<MTLComputePipelineState> exact_staged_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "kernel void gemm_f32_staged(const device float* A [[buffer(0)]],"
        "const device float* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device float* Wp = W + (p.weight_offset >> 2);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup float As[2][64 * 16]; threadgroup float Ws[2][64 * 16];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 8;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device float4* Arow = (const device float4*)(A + ar * p.k);"
        "const device float4* Wrow = (const device float4*)(Wp + wr * p.k);"
        "uint cur = 0;"
        "{ threadgroup float4* ad = (threadgroup float4*)&As[0][row * 16 + seg];"
        "  threadgroup float4* wd = (threadgroup float4*)&Ws[0][row * 16 + seg];"
        "  for (uint q = 0; q < 2; q++) {"
        "    ad[q] = ar < p.m ? Arow[(seg >> 2) + q] : float4(0.0f);"
        "    wd[q] = wr < p.n ? Wrow[(seg >> 2) + q] : float4(0.0f); } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 16) {"
        "  const uint nk = k0 + 16;"
        "  if (nk < p.k) {"
        "    threadgroup float4* ad = (threadgroup float4*)&As[1 - cur][row * 16 + seg];"
        "    threadgroup float4* wd = (threadgroup float4*)&Ws[1 - cur][row * 16 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 2; q++) {"
        "      ad[q] = ar < p.m ? Arow[base + q] : float4(0.0f);"
        "      wd[q] = wr < p.n ? Wrow[base + q] : float4(0.0f); } }"
        "  for (uint kk = 0; kk < 16; kk += 8) {"
        "    simdgroup_float8x8 a[4]; simdgroup_float8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 16 + kk], 16);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * 16 + kk], 16, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f32_staged"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static int exact_staged_ok(const GemmParams* params) {
    // exact mode == 1, f32 weights dtype == 3 in the chain param scheme
    return params->mode == 1 && params->dtype == 3 &&
           params->m % 32 == 0 && params->n % 32 == 0 && params->k % 16 == 0;
}

// W6 staged GEMM: 6-bit grouped weights dequantized during the W-stage
// load (4 codes per 3 bytes, f16 scale per 64-col group; layout per
// src/zw6.zig). Same 64x64 ping-pong shape as gemm_f16_direct.
static id<MTLComputePipelineState> w6_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "constant uint W6G = 64;\n"
        "kernel void gemm_w6_staged(const device float* A [[buffer(0)]],"
        "const device uchar* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device uchar* Wp = W + p.weight_offset;"
        "const uint gpr = (p.k + W6G - 1) / W6G;"
        "const uint cbpr = gpr * W6G / 4 * 3;"
        "const device half* scales = (const device half*)(Wp + ulong(p.n) * cbpr);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32]; threadgroup half Ws[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device float4* Arow = (const device float4*)(A + ar * p.k);"
        "uint cur = 0;"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++)"
        "    ad[q] = ar < p.m ? half4(clamp(Arow[(seg >> 2) + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "  for (uint q = 0; q < 16; q++) {"
        "    const uint col = seg + q;"
        "    const half s = wr < p.n ? scales[wr * gpr + col / W6G] : half(0.0h);"
        "    const uint quad = col / 4;"
        "    const uint at = wr * cbpr + quad * 3;"
        "    const uint word = uint(Wp[at]) | (uint(Wp[at+1]) << 8) | (uint(Wp[at+2]) << 16);"
        "    const uint raw = (word >> ((col % 4) * 6)) & 0x3F;"
        "    const int code = raw < 32 ? int(raw) : int(raw) - 64;"
        "    Ws[0][row * 32 + seg + q] = wr < p.n ? half(code) * s : half(0.0h);"
        "  } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++)"
        "      ad[q] = ar < p.m ? half4(clamp(Arow[base + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "    for (uint q = 0; q < 16; q++) {"
        "      const uint col = nk + seg + q;"
        "      const half s = wr < p.n ? scales[wr * gpr + col / W6G] : half(0.0h);"
        "      const uint quad = col / 4;"
        "      const uint at = wr * cbpr + quad * 3;"
        "      const uint word = uint(Wp[at]) | (uint(Wp[at+1]) << 8) | (uint(Wp[at+2]) << 16);"
        "      const uint raw = (word >> ((col % 4) * 6)) & 0x3F;"
        "      const int code = raw < 32 ? int(raw) : int(raw) - 64;"
        "      Ws[1 - cur][row * 32 + seg + q] = wr < p.n ? half(code) * s : half(0.0h);"
        "    } }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * 32 + kk], 32, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_w6_staged"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// W6 split-scales variant: identical math to gemm_w6_staged, but the f16
// scales come from buffer(4) instead of being computed as Wp + n*cbpr. That
// makes row-offset sub-binding into a fused matrix legal (the fused matrix's
// scales live after ALL its rows' codes, so a sub-matrix cannot derive its
// scales base from its own n).
static id<MTLComputePipelineState> w6_split_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "constant uint W6G = 64;\n"
        "kernel void gemm_w6_split(const device float* A [[buffer(0)]],"
        "const device uchar* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "const device half* S [[buffer(4)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device uchar* Wp = W + p.weight_offset;"
        "const uint gpr = (p.k + W6G - 1) / W6G;"
        "const uint cbpr = gpr * W6G / 4 * 3;"
        "const device half* scales = S;"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32]; threadgroup half Ws[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device float4* Arow = (const device float4*)(A + ar * p.k);"
        "uint cur = 0;"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++)"
        "    ad[q] = ar < p.m ? half4(clamp(Arow[(seg >> 2) + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "  for (uint q = 0; q < 16; q++) {"
        "    const uint col = seg + q;"
        "    const half s = wr < p.n ? scales[wr * gpr + col / W6G] : half(0.0h);"
        "    const uint quad = col / 4;"
        "    const uint at = wr * cbpr + quad * 3;"
        "    const uint word = uint(Wp[at]) | (uint(Wp[at+1]) << 8) | (uint(Wp[at+2]) << 16);"
        "    const uint raw = (word >> ((col % 4) * 6)) & 0x3F;"
        "    const int code = raw < 32 ? int(raw) : int(raw) - 64;"
        "    Ws[0][row * 32 + seg + q] = wr < p.n ? half(code) * s : half(0.0h);"
        "  } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++)"
        "      ad[q] = ar < p.m ? half4(clamp(Arow[base + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "    for (uint q = 0; q < 16; q++) {"
        "      const uint col = nk + seg + q;"
        "      const half s = wr < p.n ? scales[wr * gpr + col / W6G] : half(0.0h);"
        "      const uint quad = col / 4;"
        "      const uint at = wr * cbpr + quad * 3;"
        "      const uint word = uint(Wp[at]) | (uint(Wp[at+1]) << 8) | (uint(Wp[at+2]) << 16);"
        "      const uint raw = (word >> ((col % 4) * 6)) & 0x3F;"
        "      const int code = raw < 32 ? int(raw) : int(raw) - 64;"
        "      Ws[1 - cur][row * 32 + seg + q] = wr < p.n ? half(code) * s : half(0.0h);"
        "    } }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * 32 + kk], 32, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_w6_split"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// The W4 twin of w6_split_pipeline: two 4-bit codes per byte, same tiles,
// same scales binding, same per-element decode (int code -> half * half scale).
static id<MTLComputePipelineState> w4_split_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "constant uint W4G = 64;\n"
        "kernel void gemm_w4_split(const device float* A [[buffer(0)]],"
        "const device uchar* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "const device half* S [[buffer(4)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device uchar* Wp = W + p.weight_offset;"
        "const uint gpr = (p.k + W4G - 1) / W4G;"
        "const uint cbpr = gpr * W4G / 2;"
        "const device half* scales = S;"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32]; threadgroup half Ws[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "const uint ar = tile_m + row; const uint wr = tile_n + row;"
        "const device float4* Arow = (const device float4*)(A + ar * p.k);"
        "uint cur = 0;"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++)"
        "    ad[q] = ar < p.m ? half4(clamp(Arow[(seg >> 2) + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "  for (uint q = 0; q < 16; q++) {"
        "    const uint col = seg + q;"
        "    const half s = wr < p.n ? scales[wr * gpr + col / W4G] : half(0.0h);"
        "    const uint at = wr * cbpr + col / 2;"
        "    const uint raw = (uint(Wp[at]) >> ((col % 2) * 4)) & 0xF;"
        "    const int code = raw < 8 ? int(raw) : int(raw) - 16;"
        "    Ws[0][row * 32 + seg + q] = wr < p.n ? half(code) * s : half(0.0h);"
        "  } }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++)"
        "      ad[q] = ar < p.m ? half4(clamp(Arow[base + q], -65504.0f, 65504.0f)) : half4(0.0h);"
        "    for (uint q = 0; q < 16; q++) {"
        "      const uint col = nk + seg + q;"
        "      const half s = wr < p.n ? scales[wr * gpr + col / W4G] : half(0.0h);"
        "      const uint at = wr * cbpr + col / 2;"
        "      const uint raw = (uint(Wp[at]) >> ((col % 2) * 4)) & 0xF;"
        "      const int code = raw < 8 ? int(raw) : int(raw) - 16;"
        "      Ws[1 - cur][row * 32 + seg + q] = wr < p.n ? half(code) * s : half(0.0h);"
        "    } }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], &Ws[cur][(sn + j * 8) * 32 + kk], 32, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_w4_split"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

// W must be f16 (dtype 1, the W16 sidecar) and shapes tile-aligned.
// bf16 checkpoints reach gemm_f16_direct through the bf16 staging variant.
// Default ON: measured byte-identical on both Klein and Z-Image (the staging
// conversion feeds the MMA the same half fragments gemm_half would have, and
// both kernels walk K in the same order), for -17% encoder GPU time.
// ZDRAW_GEMM_BF16=0 opts out.
static int ours16_bf16_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* s = getenv("ZDRAW_GEMM_BF16");
        cached = (s && s[0] == '0') ? 0 : 1;
    }
    return cached;
}

static int ours16_ok(const GemmParams* params) {
    return params->mode == 2 && params->dtype == 1 &&
           params->m % 32 == 0 && params->n % 32 == 0 && params->k % 32 == 0;
}

// Same contract, but also admits bf16 weights. Deliberately NOT folded into
// ours16_ok: the f16a (half-A) entry shares that predicate and has no bf16
// kernel, so widening it there would feed bf16 bytes to a half-typed pointer.
static int ours16_ok_w16(const GemmParams* params) {
    if (ours16_ok(params)) return 1;
    return params->mode == 2 && params->dtype == 2 && ours16_bf16_enabled() &&
           params->m % 32 == 0 && params->n % 32 == 0 && params->k % 32 == 0;
}

// Both weight layouts are 2 bytes per element, so only the staging conversion
// differs; the dispatch geometry and MMA are identical.
static id<MTLComputePipelineState> ours16_pipeline_for(
    id<MTLDevice> device,
    const GemmParams* params
) {
    return params->dtype == 2 ? ours16_bf16_pipeline(device) : ours16_pipeline(device);
}

// Docs §6.3: two independent GEMMs into separate buffers overlap (ramp/tail
// amortized); the fused single op pays its ramp alone. A/B via env.
static int unfuse_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_UNFUSE");
        cached = (raw && strcmp(raw, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static void ours16_dispatch(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> a16,
    void* w_buf,
    void* c_buf,
    const GemmParams* op,
    size_t a_off,
    size_t c_off
) {
    [enc setBuffer:a16 offset:a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:c_off atIndex:2];
    [enc setBytes:op length:sizeof(GemmParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((op->n + 63) / 64, (op->m + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static int encode_ours16(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> a16,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    if (!pipe || !a16) return 0;
    if (dense_ours_v2_enabled() && (params->n & 63u) == 0u) {
        id<MTLComputePipelineState> v2 = ours_v2_pipeline(enc.device);
        if (v2) {
            pipe = v2;
            ours_v2_report();
        }
    }
    [enc setComputePipelineState:pipe];
    ours16_dispatch(enc, a16, w_buf, c_buf, params, 0, 0);
    return 1;
}

// Klein f16-A GEMM on the Metal 4 tensor path inside the resident batch
// (ZDRAW_KLEIN_GEMM_MPP=1, an A/B arm; klein-gemm-mpp-bench-20260827): same
// operands and byte offsets as zdraw_metal_run_gemm_f16a_enc, the pipeline
// compiled by the caller via zdraw_metal_compile_mpp (gemm_mpp64: m % 64,
// n % 64, k % 32). Returns 1 without encoding on any other shape.
int zdraw_metal_run_gemm_mpp_enc(
    void* batch, void* pipeline, void* a_buf, void* w_buf, void* c_buf,
    const GemmParams* params, uint64_t a_off, uint64_t c_off
) {
    if (!batch || !pipeline || !a_buf || !w_buf || !c_buf || !params) return 1;
    if (params->mode != 2 || params->dtype != 1) return 1;
    if (params->m % 64 || params->n % 64 || params->k % 32) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)b->enc;
    if (!enc) return 1;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:(NSUInteger)a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:(NSUInteger)c_off atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(params->n / 64, params->m / 64, 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    count_gemm(params);
    return 0;
}

// Klein resident fast path: encode one W16 GEMM through the native 64x64 direct
// kernel while preserving the resident ABI (byte offsets for A/C, byte
// weight_offset inside GemmParams). Non-eligible shapes return 1 so callers can
// fall back to gemm_half without crashing a generation.
int zdraw_metal_run_gemm_ours16_enc(
    void* batch,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off
) {
    if (!params || !batch || !a_buf || !w_buf || !c_buf || !ours16_ok_w16(params)) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)b->enc;
    if (!cmd || !enc) return 1;
    id<MTLComputePipelineState> pipe = ours16_pipeline_for(cmd.device, params);
    if (!pipe) return 1;
    if (dense_ours_v2_enabled() && (params->n & 63u) == 0u) {
        id<MTLComputePipelineState> v2 = ours_v2_pipeline(cmd.device);
        if (v2) {
            pipe = v2;
            ours_v2_report();
        }
    }
    [enc setComputePipelineState:pipe];
    ours16_dispatch(
        enc,
        (__bridge id<MTLBuffer>)a_buf,
        w_buf,
        c_buf,
        params,
        (size_t)a_off,
        (size_t)c_off
    );
    count_gemm(params); // mode 2: dispatch + gemm + half tier
    return 0;
}

// Klein W6 resident path: encode one W6 GEMM (codes at params->weight_offset,
// f16 scales bound separately at scales_off) into the live batch. The split
// scales make row-offset sub-binding into the fused single-block matrix legal.
int zdraw_metal_run_gemm_w6_enc(
    void* batch,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off,
    uint64_t scales_off
) {
    if (!params || !batch || !a_buf || !w_buf || !c_buf) return 1;
    if (params->weight_offset % 4 != 0 || scales_off % 4 != 0) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)b->enc;
    if (!cmd || !enc) return 1;
    id<MTLComputePipelineState> pipe = w6_split_pipeline(cmd.device);
    if (!pipe) return 1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:(size_t)a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:(size_t)c_off atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:(size_t)scales_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((params->n + 63) / 64, (params->m + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    g_dispatch++;
    g_gemm++;
    g_gemm_w6++;
    return 0;
}

// Klein packed tiers on the f16-A route: the steel dequant kernels
// (steel_gemm_w6_64 / steel_gemm_w4_64 from the metallib, f16 A x packed W^T
// -> f16 D) with the f16->f32 cast into the f32 C the executor's consumers
// read. Replaces the f32-A split kernel for W6/W4 whenever ZDRAW_KLEIN_ACT is
// not f32. The loader derives the scales pointer from N and K, so only the
// weight_offset (codes base) is bound.
static id<MTLComputePipelineState> steel_packed_pipe(id<MTLDevice> device, uint32_t dtype) {
    static id<MTLComputePipelineState> pipes[3] = {nil, nil, nil};
    static int tried[3] = {0, 0, 0};
    const int slot = (dtype == 5) ? 1 : (dtype == 6) ? 2 : 0;
    if (tried[slot]) return pipes[slot];
    tried[slot] = 1;
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:steel_lib_path()] error:&err];
    if (!lib) {
        fprintf(stderr, "zdraw: steel metallib unusable at %s (%s); packed tiers need ZDRAW_KLEIN_ACT=f32\n",
                [steel_lib_path() UTF8String], err ? [[err localizedDescription] UTF8String] : "no error");
        return nil;
    }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool f = false, t = true;
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:200];  // align_M (M varies)
    [fc setConstantValue:&t type:MTLDataTypeBool atIndex:201];  // align_N
    [fc setConstantValue:&t type:MTLDataTypeBool atIndex:202];  // align_K
    NSString* name = (dtype == 5) ? @"steel_gemm_w4s_64" : (dtype == 6) ? @"steel_gemm_w2s_64" : @"steel_gemm_w6s_64";
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:fc error:&err];
    if (!fn) {
        fprintf(stderr, "zdraw: %s missing from the steel metallib (rebuild: tools/build_steel_lib.sh)\n",
                [name UTF8String]);
        return nil;
    }
    pipes[slot] = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipes[slot];
}

int zdraw_metal_run_gemm_steel_enc(
    void* batch,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off,
    uint64_t scales_off
) {
    if (!params || !batch || !a_buf || !w_buf || !c_buf) return 1;
    if (params->k % 16 != 0 || params->n % 64 != 0) return 1;
    if (params->weight_offset % 4 != 0 || scales_off % 2 != 0) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)b->enc;
    if (!cmd || !enc) return 1;
    id<MTLComputePipelineState> pipe = steel_packed_pipe(cmd.device, params->dtype);
    id<MTLComputePipelineState> cast = f16_to_f32_pipe(cmd.device);
    if (!pipe || !cast) return 1;
    size_t out_count = (size_t)params->m * params->n;
    id<MTLBuffer> scratch = steel_scratch(cmd.device, 0, out_count * 2);
    if (!scratch) return 1;
    SteelGEMMParams p;
    p.M = (int)params->m; p.N = (int)params->n; p.K = (int)params->k;
    p.lda = (int)params->k; p.ldb = (int)params->k; p.ldd = (int)params->n;
    p.tiles_n = ((int)params->n + 63) / 64; p.tiles_m = ((int)params->m + 63) / 64;
    p.batch_stride_a = 0; p.batch_stride_b = 0; p.batch_stride_d = 0;
    p.swizzle_log = 0; p.gemm_k_iterations_aligned = (int)params->k / 16; p.batch_ndim = 0;
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];  // the previous cast read the scratch
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:(size_t)a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:(size_t)params->weight_offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:(size_t)scales_off atIndex:2];
    [enc setBuffer:scratch offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.tiles_n, (NSUInteger)p.tiles_m, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
    uint32_t n = (uint32_t)out_count;
    float one = 1.0f;
    [enc setComputePipelineState:cast];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];  // steel write -> cast read
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:(size_t)c_off atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&one length:sizeof(float) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch += 2;
    g_gemm++;
    if (params->dtype == 5) g_gemm_w4++; else if (params->dtype == 6) g_gemm_w2++; else g_gemm_w6++;
    return 0;
}

// Klein W4 resident path: the W6 entry with the W4 split kernel and counter.
int zdraw_metal_run_gemm_w4_enc(
    void* batch,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off,
    uint64_t scales_off
) {
    if (!params || !batch || !a_buf || !w_buf || !c_buf) return 1;
    if (params->weight_offset % 4 != 0 || scales_off % 4 != 0) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)b->enc;
    if (!cmd || !enc) return 1;
    id<MTLComputePipelineState> pipe = w4_split_pipeline(cmd.device);
    if (!pipe) return 1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:(size_t)a_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:(size_t)c_off atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:(size_t)scales_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((params->n + 63) / 64, (params->m + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    g_dispatch++;
    g_gemm++;
    g_gemm_w4++;
    return 0;
}

int zdraw_metal_run_gemm_f16a_enc(
    void* batch,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off
) {
    if (!params || !batch || !a_buf || !w_buf || !c_buf || !ours16_ok(params)) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)b->enc;
    if (!cmd || !enc) return 1;
    id<MTLComputePipelineState> pipe = klein_custom_gemm_enabled()
        ? ours_custom_pipeline(cmd.device)
        : ours16_f16a_pipeline(cmd.device);
    if (!pipe) return 1;
    [enc setComputePipelineState:pipe];
    ours16_dispatch(
        enc,
        (__bridge id<MTLBuffer>)a_buf,
        w_buf,
        c_buf,
        params,
        (size_t)a_off,
        (size_t)c_off
    );
    count_gemm(params); // mode 2: dispatch + gemm + half tier
    return 0;
}

static int encode_mps_gemm_a_off(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16,
    int c_f16,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off
);
int zdraw_metal_run_gemm_mps_enc(
    void* batch,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    if (!params || !batch || !a_buf || !w_buf || !c_buf || !mps_gemm_eligible(params)) return 1;
    ZdrawBatch* b = (ZdrawBatch*)batch;
    if (!b || !b->cmd || !b->enc) return 1;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)b->cmd;
    id<MTLComputeCommandEncoder> old = (__bridge_transfer id<MTLComputeCommandEncoder>)b->enc;
    b->enc = NULL;
    [old endEncoding];

    const int ok = encode_mps_gemm_a_off(
        cmd,
        nil,
        0,
        a_buf,
        w_buf,
        c_buf,
        params,
        0,
        0
    );
    if (ok) count_mps_gemm();

    id<MTLComputeCommandEncoder> next = [cmd computeCommandEncoder];
    if (!next) return 1;
    b->enc = (__bridge_retained void*)next;
    return ok ? 0 : 1;
}

static MPSMatrixMultiplication* mps_mul_for(
    id<MTLDevice> device,
    const GemmParams* params,
    int a_f16,
    int c_f16
) {
    static const NSUInteger max_cache = 64;
    static NSMutableDictionary<NSString*, MPSMatrixMultiplication*>* cache = nil;
    if (!cache) cache = [[NSMutableDictionary alloc] init];
    NSString* key = [NSString stringWithFormat:@"%u:%u:%u:%u:%d:%d",
                                               params->m, params->k, params->n, params->dtype,
                                               a_f16, c_f16];
    MPSMatrixMultiplication* mul = cache[key];
    if (mul) return mul;
    if ([cache count] >= max_cache) [cache removeAllObjects];
    mul = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
         transposeLeft:NO
        transposeRight:YES
            resultRows:params->m
         resultColumns:params->n
       interiorColumns:params->k
                 alpha:1.0
                  beta:0.0];
    cache[key] = mul;
    return mul;
}

static int encode_mps_gemm_a(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16, // staged f16 activations, or nil for the f32 buffer
    int c_f16, // write the result as f16 (consumer must read half)
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    return encode_mps_gemm_a_off(cmd, a16, c_f16, a_buf, w_buf, c_buf, params, 0, 0);
}

static int encode_mps_gemm_a_off(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16,
    int c_f16,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t a_off,
    uint64_t c_off
) {
    if (!mps_gemm_eligible(params)) return 0;
    @autoreleasepool {
        MPSDataType weight_type;
        if (!mps_dtype(params->dtype, &weight_type)) return 0;
        id<MTLBuffer> a = a16 ? a16 : (__bridge id<MTLBuffer>)a_buf;
        id<MTLBuffer> w = (__bridge id<MTLBuffer>)w_buf;
        id<MTLBuffer> c = (__bridge id<MTLBuffer>)c_buf;
        MPSMatrixMultiplication* mul = mps_mul_for(a.device, params, a16 != nil, c_f16);
        if (!mul) return 0;

        MPSMatrixDescriptor* da =
            [MPSMatrixDescriptor matrixDescriptorWithRows:params->m
                                                  columns:params->k
                                                 rowBytes:(NSUInteger)params->k *
                                                          (a16 ? 2 : sizeof(float))
                                                 dataType:a16 ? MPSDataTypeFloat16
                                                              : MPSDataTypeFloat32];
        MPSMatrixDescriptor* dw =
            [MPSMatrixDescriptor matrixDescriptorWithRows:params->n
                                                  columns:params->k
                                                 rowBytes:(NSUInteger)params->k * dtype_size(params->dtype)
                                                 dataType:weight_type];
        MPSMatrixDescriptor* dc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:params->m
                                                  columns:params->n
                                                 rowBytes:(NSUInteger)params->n *
                                                          (c_f16 ? 2 : sizeof(float))
                                                 dataType:c_f16 ? MPSDataTypeFloat16
                                                                : MPSDataTypeFloat32];
        MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:a
                                                   offset:(NSUInteger)a_off
                                               descriptor:da];
        MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:w
                                                   offset:(NSUInteger)params->weight_offset
                                               descriptor:dw];
        MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:c
                                                   offset:(NSUInteger)c_off
                                               descriptor:dc];
        @try {
            [mul encodeToCommandBuffer:cmd leftMatrix:ma rightMatrix:mw resultMatrix:mc];
        } @catch (NSException* exception) {
            g_gemm_mps_fallback++;
            return 0;
        }
        return 1;
    }
}

static int encode_mps_gemm(
    id<MTLCommandBuffer> cmd,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    return encode_mps_gemm_a(cmd, nil, 0, a_buf, w_buf, c_buf, params);
}

// Single-shot MPS GEMM for benchmarking the mixed f32-activation /
// low-precision-weight path outside the resident chain.
int zdraw_metal_run_gemm_mps(
    void* queue,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    if (!cmd) return -1;
    if (!encode_mps_gemm(cmd, a_buf, w_buf, c_buf, params)) return -3;
    count_mps_gemm();
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Interleaved A/B kernel comparison: variants alternate inside ONE command
// buffer so both see identical clock/thermal state; per-op times come from
// stage-boundary timestamps. Valid on a hot machine by construction.
int zdraw_metal_ab_probe(
    void* device_, void* queue_,
    uint32_t m, uint32_t k, uint32_t n, int pairs,
    uint64_t* a_ns, uint64_t* b_ns
) {
    const int chain_style = pairs < 0;
    if (chain_style) pairs = -pairs;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> base = ours16_pipeline(device);
    id<MTLCounterSampleBuffer> buf = trace_buffer(device);
    if (!base || !buf) return -1;
    enum { kW = 8 };
    id<MTLBuffer> A = [device newBufferWithLength:(size_t)m * k * 2
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:(size_t)m * n * 4
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> W[kW];
    for (int s = 0; s < kW; s++) {
        W[s] = [device newBufferWithLength:(size_t)n * k * 2
                                   options:MTLResourceStorageModeShared];
        if (!W[s]) return -1;
    }
    if (!A || !C) return -1;
    const int total = pairs * 2;
    if (total * 2 > TRACE_MAX_SAMPLES) return -1;
    GemmParams params = { m, k, n, 1, 2, 0 };
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    if (!cmd) return -1;
    for (int i = 0; i < total; i++) {
        MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = buf;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = i * 2;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = i * 2 + 1;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDescriptor:pass];
        if (!enc) return -1;
        const int is_b = i & 1;
        if (is_b && chain_style) {
            // chain-style operand state: A freshly written by the convert
            id<MTLComputePipelineState> conv = f16a_convert(device);
            const uint32_t cc = m * k;
            [enc setComputePipelineState:conv];
            [enc setBuffer:C offset:0 atIndex:0];
            [enc setBuffer:A offset:0 atIndex:1];
            [enc setBytes:&cc length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(cc, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc setComputePipelineState:base];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:W[i % kW] offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(GemmParams) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (m + 63) / 64, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
    }
    [cmd commit];
    [cmd waitUntilCompleted];
    MTLTimestamp cpu0, gpu0, cpu1, gpu1;
    [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    NSData* data = [buf resolveCounterRange:NSMakeRange(0, total * 2)];
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    if (!data) return -1;
    const double scale = gpu1 > gpu0
        ? (double)(cpu1 - cpu0) / (double)(gpu1 - gpu0) : 1.0;
    const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)data.bytes;
    double sum[2] = {0, 0};
    int cnt[2] = {0, 0};
    for (int i = 0; i < total; i++) {
        const uint64_t s0 = ts[i * 2].timestamp, s1 = ts[i * 2 + 1].timestamp;
        if (s0 == MTLCounterErrorValue || s1 == MTLCounterErrorValue || s1 <= s0) continue;
        sum[i & 1] += (double)(s1 - s0) * scale;
        cnt[i & 1]++;
    }
    if (!cnt[0] || !cnt[1]) return -1;
    *a_ns = (uint64_t)(sum[0] / cnt[0]);
    *b_ns = (uint64_t)(sum[1] / cnt[1]);
    g_trace_used = 0;
    return 0;
}

// Real per-layer GEMM mix (qkv x3, proj, fused gate+up, down), 30 layers,
// cold weights, staged kernel. Reproduces or acquits the in-chain GEMM cost.
uint64_t zdraw_metal_mix_probe(void* device_, void* queue_, int layers) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pipe = ours16_pipeline(device);
    if (!pipe) return 0;
    const uint32_t m = 288, h = 3840, inner = 10240;
    enum { kLayerSets = 4 };
    typedef struct { uint32_t k, n; } Shape;
    const Shape shapes[6] = {
        {h, h}, {h, h}, {h, h},      // q, k, v
        {h, h},                      // proj
        {h, 2 * inner},              // fused gate+up
        {inner, h},                  // down
    };
    id<MTLBuffer> W[kLayerSets][6];
    id<MTLBuffer> A = [device newBufferWithLength:(size_t)m * inner * 2
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:(size_t)m * 2 * inner * 4
                                          options:MTLResourceStorageModeShared];
    if (!A || !C) return 0;
    for (int s = 0; s < kLayerSets; s++) {
        for (int o = 0; o < 6; o++) {
            W[s][o] = [device newBufferWithLength:(size_t)shapes[o].k * shapes[o].n * 2
                                          options:MTLResourceStorageModeShared];
            if (!W[s][o]) return 0;
        }
    }
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return 0;
    [enc setComputePipelineState:pipe];
    for (int l = 0; l < layers; l++) {
        const int s = l % kLayerSets;
        for (int o = 0; o < 6; o++) {
            GemmParams op = { m, shapes[o].k, shapes[o].n, 1, 2, 0 };
            encode_ours16(enc, pipe, A, (__bridge void*)W[s][o], (__bridge void*)C, &op);
            count_gemm(&op);
        }
    }
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    const double span = cmd.GPUEndTime - cmd.GPUStartTime;
    return span > 0 ? (uint64_t)(span * 1e9) : 0;
}

// Empirically determine the simdgroup 8x8 fragment thread->element mapping
// (undocumented but fixed per GPU). Each thread reports its two elements of
// a matrix loaded with value = row*8 + col. out: 64 floats [tid*2 + i].
int zdraw_metal_frag_map_probe(void* device_, void* queue_, float* out64) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "kernel void frag_map(device float* out [[buffer(0)]],"
        "uint tid [[thread_index_in_threadgroup]]) {"
        "threadgroup float m[64];"
        "if (tid < 64) m[tid] = float(tid);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "simdgroup_float8x8 f;"
        "simdgroup_load(f, m, 8);"
        "thread auto& te = f.thread_elements();"
        "if (tid < 32) { out[tid * 2] = te[0]; out[tid * 2 + 1] = te[1]; }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return -1;
    id<MTLFunction> fn = [lib newFunctionWithName:@"frag_map"];
    id<MTLComputePipelineState> pipe =
        fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!pipe) return -1;
    id<MTLBuffer> buf = [device newBufferWithLength:64 * 4
                                            options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!buf || !cmd || !enc) return -1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    memcpy(out64, buf.contents, 64 * 4);
    return 0;
}

// Chain-structure probe: N identical direct-f16 GEMMs in one encoder.
// dependent=1 serializes via write-after-write on one C; dependent=0 uses
// 8 disjoint buffer sets. Returns GPU nanoseconds (0 on failure).
uint64_t zdraw_metal_chain_probe(void* device_, void* queue_, int dependent, int ops) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pipe = ours16_pipeline(device);
    if (!pipe) return 0;
    const uint32_t m = 288, k = 3840, n = 3840;
    GemmParams params = { m, k, n, 1, 2, 0 };
    enum { kSets = 8, kWeights = 24 }; // 24 x 28 MB of weights defeats the SLC
    id<MTLBuffer> A[kSets]; id<MTLBuffer> W[kWeights]; id<MTLBuffer> C[kSets];
    for (int s = 0; s < kSets; s++) {
        A[s] = [device newBufferWithLength:(size_t)m * k * 2 options:MTLResourceStorageModeShared];
        C[s] = [device newBufferWithLength:(size_t)m * n * 4 options:MTLResourceStorageModeShared];
        if (!A[s] || !C[s]) return 0;
    }
    for (int s = 0; s < kWeights; s++) {
        W[s] = [device newBufferWithLength:(size_t)n * k * 2 options:MTLResourceStorageModeShared];
        if (!W[s]) return 0;
    }
    id<MTLBuffer> bigW = nil;
    if (dependent == 4) {
        bigW = [device newBufferWithLength:(size_t)n * k * 2 * kWeights
                                   options:MTLResourceStorageModeShared];
        if (!bigW) return 0;
    }
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return 0;
    [enc setComputePipelineState:pipe];
    [enc setBytes:&params length:sizeof(GemmParams) atIndex:3];
    // mode: 0 cold weights, 1 hot weights, 2 independent interleave,
    // 3 dependent sandwich (convert reads this C, writes the next A).
    id<MTLComputePipelineState> conv = f16a_convert(device);
    const uint32_t conv_count = m * k;
    for (int i = 0; i < ops; i++) {
        const int s = i % kSets;
        const int nxt = (i + 1) % kSets;
        if (dependent >= 2 && conv) {
            [enc setComputePipelineState:conv];
            [enc setBuffer:C[dependent == 3 ? s : (s + 3) % kSets] offset:0 atIndex:0];
            [enc setBuffer:A[dependent == 3 ? nxt : (s + 3) % kSets] offset:0 atIndex:1];
            [enc setBytes:&conv_count length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(conv_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setComputePipelineState:pipe];
            [enc setBytes:&params length:sizeof(GemmParams) atIndex:3];
        }
        [enc setBuffer:A[s] offset:0 atIndex:0];
        if (dependent == 4) {
            GemmParams op = params;
            op.weight_offset = (uint64_t)(i % kWeights) * n * k * 2;
            [enc setBytes:&op length:sizeof(GemmParams) atIndex:3];
            [enc setBuffer:bigW offset:0 atIndex:1];
        } else {
            [enc setBuffer:W[dependent == 1 ? 0 : (i % kWeights)] offset:0 atIndex:1];
        }
        [enc setBuffer:C[s] offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (m + 63) / 64, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    const double span = cmd.GPUEndTime - cmd.GPUStartTime;
    return span > 0 ? (uint64_t)(span * 1e9) : 0;
}

// One-shot W6 GEMM for the gemmbench correctness gate.
int zdraw_metal_run_gemm_w6(
    void* queue,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = w6_pipeline(q.device);
    if (!pipe) return -3;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    count_gemm(params); // callers pass mode 4, so this banks the w6 tier
    [enc dispatchThreadgroups:MTLSizeMake((params->n + 63) / 64, (params->m + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

// Queue (non-batch) driver for the split-scales W6 kernel — the gemmbench
// correctness gate for sub-binding into fused matrices.
int zdraw_metal_run_gemm_w6_split(
    void* queue,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params,
    uint64_t scales_off
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = w6_split_pipeline(q.device);
    if (!pipe) return -3;
    if (params->weight_offset % 4 != 0 || scales_off % 4 != 0) return -4;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:(size_t)scales_off atIndex:4];
    count_gemm(params); // callers pass mode 4, so this banks the w6 tier
    [enc dispatchThreadgroups:MTLSizeMake((params->n + 63) / 64, (params->m + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_gemm_bias(
    void* queue,
    void* gemm_pipeline,
    void* bias_pipeline,
    void* a_buf,
    void* w_buf,
    void* bias_buf,
    void* c_buf,
    const GemmParams* params,
    const BiasParams* bias_params
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> gemm = (__bridge id<MTLComputePipelineState>)gemm_pipeline;
    id<MTLComputePipelineState> bias = (__bridge id<MTLComputePipelineState>)bias_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_gemm(enc, gemm, a_buf, w_buf, c_buf, params);
    [enc setComputePipelineState:bias];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias_buf offset:0 atIndex:1];
    [enc setBytes:bias_params length:sizeof(BiasParams) atIndex:2];
    size_t total = (size_t)params->m * (size_t)params->n;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];

    count_gemm(params);
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

typedef struct {
    void* state; void* norm; void* attn; void* q; void* k;
    void* v; void* mix; void* ffn; void* gate; void* up;
    void* gateup;
} ZdrawBlockBuffers;

typedef struct {
    void* attn_in; void* attn_scale; void* q; void* k; void* v;
    void* q_norm; void* k_norm; void* pos; void* rope; void* proj;
    void* attn_out; void* attn_gate; void* ffn_in; void* mlp_scale;
    void* ffn_gate; void* ffn_up; void* ffn_down; void* ffn_out; void* mlp_gate;
    void* ffn_fused;
} ZdrawBlockWeights;

typedef struct {
    ZdrawBlockParams attn_norm;
    GemmParams q; GemmParams k; GemmParams v;
    ZdrawQkNormParams qk;
    ZdrawAttnParams attn;
    GemmParams proj;
    ZdrawBlockParams attn_resid;
    ZdrawBlockParams ffn_norm;
    GemmParams ffn_gate; GemmParams ffn_up; GemmParams ffn_down;
    ZdrawBlockParams ffn_resid;
    GemmParams ffn_fused;
} ZdrawBlockChainParams;

typedef struct {
    size_t block; size_t qk; size_t attn; size_t swiglu;
    size_t attn_kernel; // mattn.Kernel ordinal: the attention grid is keyed on it
} ZdrawBlockThreads;

// ABI contract with src/abi_assert.zig: both sides assert the same numbers,
// so layout drift on either side fails the build instead of corrupting GPU
// memory. Update both files in the same commit.
_Static_assert(sizeof(GemmParams) == ZDRAW_ABI_GEMM_PARAMS_SIZE, "GemmParams ABI");
_Static_assert(sizeof(ZdrawAttnParams) == ZDRAW_ABI_ATTN_PARAMS_SIZE, "AttnParams ABI");
_Static_assert(sizeof(ZdrawQkNormParams) == ZDRAW_ABI_QK_NORM_PARAMS_SIZE, "QkNormParams ABI");
_Static_assert(sizeof(ZdrawLinearParams) == ZDRAW_ABI_LINEAR_PARAMS_SIZE, "LinearParams ABI");
_Static_assert(sizeof(ZdrawConvParams) == ZDRAW_ABI_CONV_PARAMS_SIZE, "ConvParams ABI");
_Static_assert(sizeof(ZdrawConvWindowParams) == ZDRAW_ABI_CONV_WINDOW_PARAMS_SIZE, "ConvWindowParams ABI");
_Static_assert(sizeof(ZdrawVaeAddWindowParams) == ZDRAW_ABI_ADD_WINDOW_PARAMS_SIZE, "VaeAddWindowParams ABI");
_Static_assert(sizeof(ZdrawBlockParams) == ZDRAW_ABI_BLOCK_PARAMS_SIZE, "BlockParams ABI");
_Static_assert(sizeof(ZdrawBlockBuffers) == ZDRAW_ABI_CHAIN_BUFFERS_SIZE, "BlockBuffers ABI");
_Static_assert(sizeof(ZdrawBlockWeights) == ZDRAW_ABI_CHAIN_WEIGHTS_SIZE, "BlockWeights ABI");
_Static_assert(sizeof(ZdrawBlockChainParams) == ZDRAW_ABI_CHAIN_PARAMS_SIZE, "ChainParams ABI");
_Static_assert(offsetof(ZdrawBlockChainParams, ffn_fused) == 456, "ChainParams.ffn_fused ABI");
_Static_assert(offsetof(ZdrawBlockChainParams, qk) == 128, "ChainParams.qk ABI");
_Static_assert(offsetof(ZdrawBlockChainParams, proj) == 232, "ChainParams.proj ABI");
_Static_assert(offsetof(ZdrawBlockChainParams, ffn_resid) == 424, "ChainParams.ffn_resid ABI");
_Static_assert(sizeof(ZdrawBlockThreads) == ZDRAW_ABI_CHAIN_THREADS_SIZE, "BlockThreads ABI");
_Static_assert(sizeof(FinalBuffers) == ZDRAW_ABI_FINAL_BUFFERS_SIZE, "FinalBuffers ABI");
_Static_assert(sizeof(FinalNormParams) == ZDRAW_ABI_FINAL_NORM_PARAMS_SIZE, "FinalNormParams ABI");
_Static_assert(sizeof(BiasParams) == ZDRAW_ABI_BIAS_PARAMS_SIZE, "BiasParams ABI");
_Static_assert(sizeof(ZdrawVaeNormParams) == ZDRAW_ABI_VAE_NORM_PARAMS_SIZE, "VaeNormParams ABI");
_Static_assert(sizeof(ZdrawVaeNormWindowParams) == ZDRAW_ABI_NORM_WINDOW_PARAMS_SIZE, "VaeNormWindowParams ABI");
_Static_assert(sizeof(ZdrawConvUpsampleWindowParams) == ZDRAW_ABI_CONV_UPSAMPLE_WINDOW_PARAMS_SIZE, "ConvUpsampleWindowParams ABI");
_Static_assert(sizeof(ZdrawVaeResParams) == ZDRAW_ABI_VAE_RES_PARAMS_SIZE, "VaeResParams ABI");
_Static_assert(sizeof(ZdrawVaeResBuffers) == ZDRAW_ABI_VAE_RES_BUFFERS_SIZE, "VaeResBuffers ABI");
_Static_assert(sizeof(ZdrawVaeResStreamBuffers) == ZDRAW_ABI_STREAM_BUFFERS_SIZE, "VaeResStreamBuffers ABI");

static int encode_mps_gateup_pair(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p
) {
    if (!mps_gemm_eligible(&p->ffn_gate) || !mps_gemm_eligible(&p->ffn_up)) return 0;
    if (!encode_mps_gemm_a(cmd, a16, 0, b->norm, w->ffn_gate, b->gate, &p->ffn_gate)) return 0;
    if (!encode_mps_gemm_a(cmd, a16, 0, b->norm, w->ffn_up, b->up, &p->ffn_up)) return 0;
    return 1;
}

static int mps_gateup_pair_eligible(const ZdrawBlockChainParams* p) {
    return mps_gemm_eligible(&p->ffn_gate) && mps_gemm_eligible(&p->ffn_up);
}

static int encode_mps_qkv_triple(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p
) {
    if (!mps_gemm_eligible(&p->q) || !mps_gemm_eligible(&p->k) ||
        !mps_gemm_eligible(&p->v)) return 0;
    if (!encode_mps_gemm_a(cmd, a16, 0, b->norm, w->q, b->q, &p->q)) return 0;
    if (!encode_mps_gemm_a(cmd, a16, 0, b->norm, w->k, b->k, &p->k)) return 0;
    if (!encode_mps_gemm_a(cmd, a16, 0, b->norm, w->v, b->v, &p->v)) return 0;
    return 1;
}

static int mps_qkv_triple_eligible(const ZdrawBlockChainParams* p) {
    return mps_gemm_eligible(&p->q) && mps_gemm_eligible(&p->k) &&
           mps_gemm_eligible(&p->v);
}

static int encode_mps_proj(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p
) {
    if (!mps_gemm_eligible(&p->proj)) return 0;
    return encode_mps_gemm_a(cmd, a16, 0, b->mix, w->proj, b->attn, &p->proj);
}

static int encode_mps_down(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> a16,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p
) {
    // swiglu writes the gated product into b->gate in place; down reads it.
    if (!mps_gemm_eligible(&p->ffn_down)) return 0;
    return encode_mps_gemm_a(cmd, a16, 0, b->gate, w->ffn_down, b->ffn, &p->ffn_down);
}

static void encode_block_kernel(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* scale,
    void* output,
    const ZdrawBlockParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawBlockParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(params->tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

static void encode_qk_norm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawQkNormParams* params,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->q offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->k offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->q_norm offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->k_norm offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->pos offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->rope offset:0 atIndex:5];
    [enc setBytes:params length:sizeof(ZdrawQkNormParams) atIndex:6];
    size_t groups = (size_t)params->tokens * (params->heads + params->kv_heads);
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// block16 wants half q/k/v; convert f32 buffers into persistent bridge
// scratch (converts measured ~free; sandwich probe 0.998).
static id<MTLComputePipelineState> headmajor_convert(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void to_headmajor_f16(const device float* in [[buffer(0)]],"
        "device half* out [[buffer(1)]],"
        "constant uint& tokens [[buffer(2)]],"
        "constant uint& heads [[buffer(3)]],"
        "constant uint& dim [[buffer(4)]],"
        "uint id [[thread_position_in_grid]]) {"
        "if (id >= tokens * heads * dim) return;"
        "const uint d = id % dim;"
        "const uint h = (id / dim) % heads;"
        "const uint t = id / (dim * heads);"
        "out[(h * tokens + t) * dim + d] = half(clamp(in[id], -65504.0f, 65504.0f)); }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"to_headmajor_f16"];
    pipe = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    return pipe;
}

static id<MTLBuffer> attn_half_scratch(id<MTLDevice> device, int slot, size_t bytes) {
    static id<MTLBuffer> bufs[5] = {nil, nil, nil, nil, nil};
    if (!bufs[slot] || bufs[slot].length < bytes) {
        bufs[slot] = [device newBufferWithLength:bytes
                                         options:MTLResourceStorageModePrivate];
    }
    return bufs[slot];
}

static int qk_hm_enabled(void) {
    const char* raw = getenv("ZDRAW_QK_HM");
    return !raw || raw[0] != '0';
}

static id<MTLComputePipelineState> qk_hm_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Pair { float c; float s; };"
        "struct Params { uint tokens; uint heads; uint kv_heads; uint head_dim;"
        "uint q_dtype; uint k_dtype; uint dim0; uint dim1; uint dim2; float eps;"
        "ulong q_offset; ulong k_offset; ulong base0; ulong base1; ulong base2; };"
        "static inline float weight_at(const device uchar* base, uint index, uint dtype) {"
        "if (dtype == 3) return ((const device float*)base)[index];"
        "ushort bits = ((const device ushort*)base)[index];"
        "if (dtype == 2) return as_type<float>(uint(bits) << 16);"
        "return float(as_type<half>(bits)); }"
        "static inline ulong axis_base(constant Params& p, uint axis) {"
        "return axis == 0 ? p.base0 : (axis == 1 ? p.base1 : p.base2); }"
        "static inline uint axis_dim(constant Params& p, uint axis) {"
        "return axis == 0 ? p.dim0 : (axis == 1 ? p.dim1 : p.dim2); }"
        "kernel void qk_norm_rope_hm_f16("
        "device float* q [[buffer(0)]],"
        "device float* k [[buffer(1)]],"
        "const device uchar* q_weight [[buffer(2)]],"
        "const device uchar* k_weight [[buffer(3)]],"
        "const device ulong* pos [[buffer(4)]],"
        "const device Pair* rope [[buffer(5)]],"
        "device half* q_out [[buffer(6)]],"
        "device half* k_out [[buffer(7)]],"
        "constant Params& p [[buffer(8)]],"
        "uint group [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint tg_size [[threads_per_threadgroup]]) {"
        "uint q_groups = p.tokens * p.heads;"
        "bool is_q = group < q_groups;"
        "uint local = is_q ? group : group - q_groups;"
        "uint heads = is_q ? p.heads : p.kv_heads;"
        "uint tok = local / heads;"
        "uint head = local - tok * heads;"
        "device float* data = is_q ? q : k;"
        "device half* out = is_q ? q_out : k_out;"
        "const device uchar* wb = (is_q ? q_weight + p.q_offset : k_weight + p.k_offset);"
        "uint dtype = is_q ? p.q_dtype : p.k_dtype;"
        "uint base = (tok * heads + head) * p.head_dim;"
        "threadgroup float reduce[256];"
        "float local_sum = 0.0f;"
        "for (uint dim = tid; dim < p.head_dim; dim += tg_size) {"
        "float value = data[base + dim]; local_sum += value * value; }"
        "reduce[tid] = local_sum;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float scale = rsqrt(reduce[0] / float(p.head_dim) + p.eps);"
        "uint pair_count = p.head_dim / 2;"
        "for (uint pair = tid; pair < pair_count; pair += tg_size) {"
        "uint axis = pair * 2 < p.dim0 ? 0 : (pair * 2 < p.dim0 + p.dim1 ? 1 : 2);"
        "uint axis_off = axis == 0 ? 0 : (axis == 1 ? p.dim0 : p.dim0 + p.dim1);"
        "uint local_pair = pair - axis_off / 2;"
        "uint axis_pairs = axis_dim(p, axis) / 2;"
        "ulong rope_pos = pos[tok * 3 + axis];"
        "Pair rot = rope[axis_base(p, axis) + rope_pos * axis_pairs + local_pair];"
        "uint i = base + pair * 2;"
        "float aw = scale * weight_at(wb, pair * 2, dtype);"
        "float bw = scale * weight_at(wb, pair * 2 + 1, dtype);"
        "float a = data[i] * aw;"
        "float b = data[i + 1] * bw;"
        "uint oi = (head * p.tokens + tok) * p.head_dim + pair * 2;"
        "out[oi] = half(clamp(a * rot.c - b * rot.s, -65504.0f, 65504.0f));"
        "out[oi + 1] = half(clamp(b * rot.c + a * rot.s, -65504.0f, 65504.0f));"
        "}"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qk_norm_rope_hm_f16"];
    if (!fn) return nil;
    pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    return pipe;
}

static int encode_qk_norm_hm(
    id<MTLComputeCommandEncoder> enc,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawQkNormParams* params,
    id<MTLBuffer>* qhm_out,
    id<MTLBuffer>* khm_out
) {
    if (!qk_hm_enabled()) return 0;
    if (params->heads != params->kv_heads || params->head_dim != 128) return 0;
    id<MTLBuffer> qbuf = (__bridge id<MTLBuffer>)b->q;
    id<MTLComputePipelineState> pipe = qk_hm_pipeline(qbuf.device);
    if (!pipe) return 0;
    const size_t n = (size_t)params->tokens * params->heads * params->head_dim;
    id<MTLBuffer> qhm = attn_half_scratch(qbuf.device, 0, n * 2);
    id<MTLBuffer> khm = attn_half_scratch(qbuf.device, 1, n * 2);
    if (!qhm || !khm) return 0;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:qbuf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->k offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->q_norm offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->k_norm offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->pos offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->rope offset:0 atIndex:5];
    [enc setBuffer:qhm offset:0 atIndex:6];
    [enc setBuffer:khm offset:0 atIndex:7];
    [enc setBytes:params length:sizeof(ZdrawQkNormParams) atIndex:8];
    size_t groups = (size_t)params->tokens * (params->heads + params->kv_heads);
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    *qhm_out = qhm;
    *khm_out = khm;
    g_dispatch++;
    return 1;
}

// Vendored MFA attention is the production default at long sequences
// (wins ~6.7 s sampling at 1024px, neutral at 512px; SSIM 1.0 vs SDPA).
// ZDRAW_ATTN_MFA=0 opts out; threshold tunable for certify sweeps.
static int mfa_attn_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("ZDRAW_ATTN_MFA");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached;
}

static uint32_t mfa_min_tokens(void) {
    const char* env = getenv("ZDRAW_MFA_MIN_TOKENS");
    return env ? (uint32_t)atoi(env) : 2048;
}

// GPU-family split: apple9 (M3/M4) streams K/V from device with 64-thread TGs;
// older families (M1/M2) stage K/V via threadgroup with 128-thread TGs. The
// 1024px Klein shape is faster on Apple9 with 32-row tiles than the original
// 16-row Apple9 default (measured 2026-06 on M4 Max: 10.7/10.8s warm vs
// 11.7/12.2s retained baseline), while keeping the 64-thread Apple9 body.
static int mfa_is_apple9(id<MTLDevice> device) {
    static int cached = -1;
    if (cached < 0) {
        cached = [device supportsFamily:MTLGPUFamilyApple9] ? 1 : 0;
    }
    return cached;
}

// The block size is BAKED INTO the generated kernel body (16-row on Apple9,
// 32-row on M1/M2); the dispatch must match it. Forcing 32 rows on Apple9
// (8e9cfc1, a Klein-tuned change) halved row coverage and produced the
// 2026-07-18 lower-half checkerboard corruption on every >=2048-token
// Z-Image render. Never re-tile a generated MFA kernel from the host side.
static uint32_t mfa_block_rows(id<MTLDevice> device) {
    return mfa_is_apple9(device) ? 16 : 32;
}

static uint32_t mfa_tg_threads(id<MTLDevice> device) {
    return mfa_is_apple9(device) ? 64 : 128;
}

static int mfa_multihead_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("ZDRAW_MFA_MULTIHEAD");
        cached = (env && (env[0] == '0' || strcmp(env, "false") == 0 || strcmp(env, "off") == 0)) ? 0 : 1;
    }
    return cached;
}

// Defined by src/mfa_embed.zig in every compile that imports the bridge
// declarations; this weak default covers test roots and lab tools that link
// the bridge without them, which then fall back to the source file.
__attribute__((weak)) const char* zdraw_mfa_source(int apple9, size_t* len) {
    (void)apple9;
    *len = 0;
    return NULL;
}

static id<MTLLibrary> mfa_library(id<MTLDevice> device) {
    static id<MTLLibrary> lib = nil;
    static int failed = 0;
    if (lib || failed) return lib;
    // The kernel source is embedded at build time (src/mfa_embed.zig);
    // ZDRAW_MFA_METAL points at a file for lab variants only.
    const char* override = getenv("ZDRAW_MFA_METAL");
    NSError* err = nil;
    NSString* code = nil;
    if (override) {
        code = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:override]
                                         encoding:NSUTF8StringEncoding
                                            error:&err];
    } else {
        size_t len = 0;
        const char* src = zdraw_mfa_source(mfa_is_apple9(device) ? 1 : 0, &len);
        if (src && len) {
            code = [[NSString alloc] initWithBytes:src length:len encoding:NSUTF8StringEncoding];
        } else {
            // No embedded source in this compile: read the vendored file
            // (relative to the working directory, dev/test only).
            NSString* path = mfa_is_apple9(device) ? @"vendor/mfa/mfa-fwd-d128.metal"
                                                   : @"vendor/mfa/mfa-fwd-d128-m1m2.metal";
            code = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
        }
    }
    if (!code) { failed = 1; return nil; }
    lib = [device newLibraryWithSource:code options:zdraw_compile_options() error:&err];
    if (!lib) failed = 1;
    return lib;
}

// PSO cache for (R, C) pairs — a generation uses exactly one shape, plus
// possibly the refiner shape; four slots cover practice.
static id<MTLComputePipelineState> mfa_pipeline_for(id<MTLDevice> device, uint32_t tokens) {
    static uint32_t keys[4] = {0, 0, 0, 0};
    static id<MTLComputePipelineState> psos[4] = {nil, nil, nil, nil};
    static int next_replace = 0;
    for (int i = 0; i < 4; i++) {
        if (keys[i] == tokens) return psos[i];
    }
    id<MTLLibrary> lib = mfa_library(device);
    if (!lib) return nil;
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    uint32_t R = tokens;
    uint32_t C = tokens;
    [fc setConstantValue:&R type:MTLDataTypeUInt atIndex:0];
    [fc setConstantValue:&C type:MTLDataTypeUInt atIndex:1];
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"attention" constantValues:fc error:&err];
    if (!fn) return nil;
    MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn;
    pd.maxTotalThreadsPerThreadgroup = 1024;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&err];
    if (!pso) return nil;
    for (int i = 0; i < 4; i++) {
        if (keys[i] == 0) {
            keys[i] = tokens;
            psos[i] = pso;
            next_replace = (i + 1) % 4;
            return pso;
        }
    }
    keys[next_replace] = tokens;
    psos[next_replace] = pso;
    next_replace = (next_replace + 1) % 4;
    return pso;
}

// token-major f32 gather back from head-major f32 (the O un-permute).
static id<MTLComputePipelineState> from_headmajor_f32(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void from_headmajor_f32(const device float* in [[buffer(0)]],"
        "device float* out [[buffer(1)]],"
        "constant uint& tokens [[buffer(2)]],"
        "constant uint& heads [[buffer(3)]],"
        "constant uint& dim [[buffer(4)]],"
        "uint id [[thread_position_in_grid]]) {"
        "if (id >= tokens * heads * dim) return;"
        "const uint d = id % dim;"
        "const uint h = (id / dim) % heads;"
        "const uint t = id / (dim * heads);"
        "out[id] = in[(h * tokens + t) * dim + d]; }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"from_headmajor_f32"];
    pipe = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    return pipe;
}

// Encode the vendored MFA forward attention inside the current encoder:
// q/k/v f32 token-major -> head-major half scratch -> one attention dispatch
// per all heads -> O head-major f32 -> un-permute into out. Returns 0 on
// success.
static int encode_attention_mfa_raw(
    id<MTLComputeCommandEncoder> enc,
    id<MTLDevice> device,
    void* q_buf,
    void* k_buf,
    void* v_buf,
    void* out_buf,
    const ZdrawAttnParams* params,
    id<MTLBuffer> qhm_in,
    id<MTLBuffer> khm_in,
    id<MTLBuffer> vhm_in,
    size_t in_off,
    size_t out_off,
    int keep_o_hm      // leave O head-major f32 in scratch slot 3, skip the un-permute
) {
    const uint32_t tokens = params->tokens;
    const uint32_t heads = params->heads;
    const uint32_t dim = params->head_dim;
    id<MTLComputePipelineState> pso = mfa_pipeline_for(device, tokens);
    id<MTLComputePipelineState> conv = headmajor_convert(device);
    id<MTLComputePipelineState> unperm = from_headmajor_f32(device);
    if (!pso || !conv || !unperm) return -1;
    const uint32_t n = tokens * heads * dim;
    id<MTLBuffer> hm[3];
    void* src[3] = {q_buf, k_buf, v_buf};
    uint32_t convert_dispatches = 0;
    for (int i = 0; i < 3; i++) {
        if (i == 0 && qhm_in) {
            hm[i] = qhm_in;
            continue;
        }
        if (i == 1 && khm_in) {
            hm[i] = khm_in;
            continue;
        }
        if (i == 2 && vhm_in) {
            hm[i] = vhm_in;
            continue;
        }
        hm[i] = attn_half_scratch(device, i, (size_t)n * 2);
        if (!hm[i]) return -1;
        [enc setComputePipelineState:conv];
        [enc setBuffer:(__bridge id<MTLBuffer>)src[i] offset:in_off atIndex:0];
        [enc setBuffer:hm[i] offset:0 atIndex:1];
        [enc setBytes:&tokens length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&heads length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        convert_dispatches++;
    }
    id<MTLBuffer> obuf = attn_half_scratch(device, 3, (size_t)n * 4);
    id<MTLBuffer> lbuf = attn_half_scratch(device, 4, (size_t)tokens * heads * 2);
    if (!obuf || !lbuf) return -1;
    [enc setComputePipelineState:pso];
    [enc setThreadgroupMemoryLength:8192 atIndex:0];
    const size_t head_qkv = (size_t)tokens * dim * 2;
    const size_t head_o = (size_t)tokens * dim * 4;
    const uint32_t rows = mfa_block_rows(device);
    const uint32_t tgt = mfa_tg_threads(device);
    if (mfa_multihead_enabled()) {
        [enc setBuffer:hm[0] offset:0 atIndex:0];
        [enc setBuffer:hm[1] offset:0 atIndex:1];
        [enc setBuffer:hm[2] offset:0 atIndex:2];
        [enc setBuffer:obuf offset:0 atIndex:3];
        [enc setBuffer:lbuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((tokens + rows - 1) / rows, heads, 1)
            threadsPerThreadgroup:MTLSizeMake(tgt, 1, 1)];
        g_dispatch += 1;
    } else {
        for (uint32_t h = 0; h < heads; h++) {
            [enc setBuffer:hm[0] offset:h * head_qkv atIndex:0];
            [enc setBuffer:hm[1] offset:h * head_qkv atIndex:1];
            [enc setBuffer:hm[2] offset:h * head_qkv atIndex:2];
            [enc setBuffer:obuf offset:h * head_o atIndex:3];
            [enc setBuffer:lbuf offset:(size_t)h * tokens * 2 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((tokens + rows - 1) / rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tgt, 1, 1)];
        }
        g_dispatch += heads;
    }
    if (keep_o_hm) {
        g_dispatch += convert_dispatches;
        return 0;
    }
    [enc setComputePipelineState:unperm];
    [enc setBuffer:obuf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:out_off atIndex:1];
    [enc setBytes:&tokens length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&heads length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch += convert_dispatches + 1;
    return 0;
}

static int encode_attention_mfa(
    id<MTLComputeCommandEncoder> enc,
    const ZdrawBlockBuffers* b,
    const ZdrawAttnParams* params,
    id<MTLBuffer> qhm_in,
    id<MTLBuffer> khm_in
) {
    id<MTLBuffer> some = (__bridge id<MTLBuffer>)b->q;
    return encode_attention_mfa_raw(
        enc,
        some.device,
        b->q,
        b->k,
        b->v,
        b->mix,
        params,
        qhm_in,
        khm_in,
        nil,
        0,
        0,
        0
    );
}

int zdraw_metal_run_attention_mfa_enc(
    void* batch,
    void* q_buf,
    void* k_buf,
    void* v_buf,
    void* output,
    const ZdrawAttnParams* params,
    size_t in_off,
    size_t out_off
) {
    if (!batch || !q_buf || !k_buf || !v_buf || !output || !params) return 1;
    if (params->causal || params->heads != params->kv_heads || params->head_dim != 128) {
        return 1;
    }
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)bb->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!cmd || !enc) return 1;
    return encode_attention_mfa_raw(
        enc,
        cmd.device,
        q_buf,
        k_buf,
        v_buf,
        output,
        params,
        nil,
        nil,
        nil,
        in_off,
        out_off,
        0
    );
}

// ---- MLX steel attention (vendor/steel/attn_entry.metal): f16 Q/K/V/O in
// [B, H, L, D] (head-major, the MFA scratch layout), f32 accumulate, BQ 32 /
// BK 16 / D 128 / 128 threads. Mirrors mlx::steel::AttnParams exactly.
typedef struct {
    int B, H, D, qL, kL, gqa_factor;
    float scale;
    int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
    int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
} SteelAttnParams;

// Bench-only variant selector (ZDRAW_KLEIN_ATTN_VARIANT = bk32); the product
// route uses the default instantiation.
static const char* steel_attn_variant(int* bq, int* bk) {
    const char* v = getenv("ZDRAW_KLEIN_ATTN_VARIANT");
    *bq = 32; *bk = 16;
    if (!v || !v[0]) return "steel_attn_h128";
    if (strcmp(v, "bk32") == 0) { *bk = 32; return "steel_attn_h128_bk32"; }
    return "steel_attn_h128";
}

static id<MTLComputePipelineState> steel_attn_pipe(id<MTLDevice> device, bool align_q, bool align_k) {
    static id<MTLComputePipelineState> pipes[4] = {nil, nil, nil, nil};
    static int tried[4] = {0, 0, 0, 0};
    const int key = (align_q ? 2 : 0) | (align_k ? 1 : 0);
    if (tried[key]) return pipes[key];
    tried[key] = 1;
    int bq = 32, bk = 16;
    const char* fname = steel_attn_variant(&bq, &bk);
    NSString* path = steel_lib_path();
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) {
        fprintf(stderr, "zdraw: WARNING steel metallib unusable at %s (%s); steel attention unavailable\n",
                path.UTF8String, err.localizedDescription.UTF8String);
        return nil;
    }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    bool aq = align_q, ak = align_k, f = false;
    [fc setConstantValue:&aq type:MTLDataTypeBool atIndex:200]; // align_Q
    [fc setConstantValue:&ak type:MTLDataTypeBool atIndex:201]; // align_K
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:300];  // has_mask
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:301];  // do_causal
    [fc setConstantValue:&f type:MTLDataTypeBool atIndex:302];  // has_sinks
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fname] constantValues:fc error:&err];
    if (!fn) {
        fprintf(stderr, "zdraw: WARNING %s missing from %s (rebuild with tools/build_steel_lib.sh); "
                "attention falls back to MFA: NOT the default route\n", fname, path.UTF8String);
        return nil;
    }
    pipes[key] = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipes[key]) fprintf(stderr, "zdraw: WARNING steel attention pipeline failed: %s\n", err.localizedDescription.UTF8String);
    return pipes[key];
}

// Encode one non-causal self-attention over head-major half Q/K/V into
// head-major half O. Returns 0 on success, 1 when the kernel is unavailable.
static int encode_steel_attn(
    id<MTLComputeCommandEncoder> enc,
    id<MTLDevice> device,
    id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v, id<MTLBuffer> o,
    uint32_t tokens, uint32_t heads, uint32_t dim
) {
    if (dim != 128) return 1;
    int bq = 32, bk = 16;
    (void)steel_attn_variant(&bq, &bk);
    const bool align_q = (tokens % (uint32_t)bq) == 0u;
    const bool align_k = (tokens % (uint32_t)bk) == 0u;
    id<MTLComputePipelineState> pso = steel_attn_pipe(device, align_q, align_k);
    if (!pso) return 1;
    SteelAttnParams p;
    memset(&p, 0, sizeof(p));
    p.B = 1; p.H = (int)heads; p.D = (int)dim; p.qL = (int)tokens; p.kL = (int)tokens;
    p.gqa_factor = 1;
    p.scale = 1.0f / sqrtf((float)dim);
    p.NQ = (int)((tokens + bq - 1) / bq); p.NK = (int)((tokens + bk - 1) / bk);
    p.NQ_aligned = (int)(tokens / bq); p.NK_aligned = (int)(tokens / bk);
    p.qL_rem = (int)(tokens % bq); p.kL_rem = (int)(tokens % bk); p.qL_off = 0;
    const int64_t hs = (int64_t)tokens * dim;
    for (int i = 0; i < 3; i++) {
        const int64_t s = (i == 0) ? hs * heads : (i == 1) ? hs : (int64_t)dim;
        p.Q_strides[i] = s; p.K_strides[i] = s; p.V_strides[i] = s; p.O_strides[i] = s;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:q offset:0 atIndex:0];
    [enc setBuffer:k offset:0 atIndex:1];
    [enc setBuffer:v offset:0 atIndex:2];
    [enc setBuffer:o offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.NQ, heads, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    g_dispatch += 1;
    return 0;
}

// Bench/route entry: q/k/v head-major half (the MFA scratch slots or any
// caller buffers), O head-major half into `o_hm`.
int zdraw_metal_run_attention_steel_hm_enc(
    void* batch, void* qhm, void* khm, void* vhm, void* o_hm, const ZdrawAttnParams* params
) {
    if (!batch || !qhm || !khm || !vhm || !o_hm || !params) return 1;
    if (params->causal || params->heads != params->kv_heads || params->head_dim != 128) return 1;
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)bb->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!cmd || !enc) return 1;
    return encode_steel_attn(enc, cmd.device,
        (__bridge id<MTLBuffer>)qhm, (__bridge id<MTLBuffer>)khm,
        (__bridge id<MTLBuffer>)vhm, (__bridge id<MTLBuffer>)o_hm,
        params->tokens, params->heads, params->head_dim);
}

// token-major f32 gather back from head-major HALF (the steel O un-permute).
static id<MTLComputePipelineState> from_headmajor_h32(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipe = nil;
    if (pipe) return pipe;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void from_headmajor_h32(const device half* in [[buffer(0)]],"
        "device float* out [[buffer(1)]],"
        "constant uint& tokens [[buffer(2)]],"
        "constant uint& heads [[buffer(3)]],"
        "constant uint& dim [[buffer(4)]],"
        "uint id [[thread_position_in_grid]]) {"
        "if (id >= tokens * heads * dim) return;"
        "const uint d = id % dim;"
        "const uint h = (id / dim) % heads;"
        "const uint t = id / (dim * heads);"
        "out[id] = float(in[(h * tokens + t) * dim + d]); }";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"from_headmajor_h32"];
    pipe = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    return pipe;
}

// Klein route: q/k head-major half already in slots 0/1 (krms_rope_hm); v
// converted here into slot 2; steel attention writes O head-major HALF into
// slot 3; keep_o_hm leaves it there (the single block's kunperm_hm_hh reads
// it), else it is un-permuted to token-major f32 into output. Returns 1
// (nothing encoded) when the steel kernel is unavailable so the caller can
// take the MFA route instead.
// in_off / out_off: byte offsets into v_buf and output (one image's segment
// of a --seeds batch); qhm/khm are per-image scratch, always at 0.
int zdraw_metal_run_attention_steel_route_enc(
    void* batch, void* qhm, void* khm, void* v_buf, void* output,
    const ZdrawAttnParams* params, int keep_o_hm, size_t in_off, size_t out_off
) {
    if (!batch || !qhm || !khm || !v_buf || !output || !params) return 1;
    if (params->causal || params->heads != params->kv_heads || params->head_dim != 128) return 1;
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)bb->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!cmd || !enc) return 1;
    id<MTLDevice> device = cmd.device;
    const uint32_t tokens = params->tokens, heads = params->heads, dim = params->head_dim;
    const bool align_q = (tokens % 32u) == 0u, align_k = (tokens % 16u) == 0u;
    if (!steel_attn_pipe(device, align_q, align_k)) { g_attn_steel_fallback++; return 1; }
    id<MTLComputePipelineState> conv = headmajor_convert(device);
    id<MTLComputePipelineState> unperm = from_headmajor_h32(device);
    if (!conv || !unperm) return 1;
    const uint32_t n = tokens * heads * dim;
    id<MTLBuffer> vhm = attn_half_scratch(device, 2, (size_t)n * 2);
    id<MTLBuffer> ohm = attn_half_scratch(device, 3, (size_t)n * 4);
    if (!vhm || !ohm) return 1;
    [enc setComputePipelineState:conv];
    [enc setBuffer:(__bridge id<MTLBuffer>)v_buf offset:in_off atIndex:0];
    [enc setBuffer:vhm offset:0 atIndex:1];
    [enc setBytes:&tokens length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&heads length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (encode_steel_attn(enc, device, (__bridge id<MTLBuffer>)qhm, (__bridge id<MTLBuffer>)khm, vhm, ohm,
                          tokens, heads, dim) != 0) return 1;
    if (keep_o_hm) {
        g_dispatch += 1;
        return 0;
    }
    [enc setComputePipelineState:unperm];
    [enc setBuffer:ohm offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:out_off atIndex:1];
    [enc setBytes:&tokens length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&heads length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch += 2;
    return 0;
}

// ZDRAW_ATTN_STEEL=0 restores the MFA / owned-kernel order for the Z-Image
// chain (A/B); default on since 2026-08-27 (ledger klein-attn-steel-route,
// the same kernel and gate for both models).
static int attn_steel_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_ATTN_STEEL");
        cached = (raw && raw[0] == '0') ? 0 : 1;
    }
    return cached;
}

// Steel attention for the Z-Image chain over token-major f32 q/k/v (any
// operand already head-major half may be passed in): converts what is
// missing into the scratch slots, runs the kernel, un-permutes O into out.
// Returns 1 without encoding when the kernel is unavailable.
static int encode_attention_steel_any(
    id<MTLComputeCommandEncoder> enc,
    id<MTLDevice> device,
    void* q_buf, void* k_buf, void* v_buf, void* out_buf,
    const ZdrawAttnParams* params,
    id<MTLBuffer> qhm_in, id<MTLBuffer> khm_in
) {
    if (!attn_steel_enabled()) return 1;
    const uint32_t tokens = params->tokens, heads = params->heads, dim = params->head_dim;
    if (params->causal || heads != params->kv_heads || dim != 128) return 1;
    const bool align_q = (tokens % 32u) == 0u, align_k = (tokens % 16u) == 0u;
    if (!steel_attn_pipe(device, align_q, align_k)) { g_attn_steel_fallback++; return 1; }
    id<MTLComputePipelineState> conv = headmajor_convert(device);
    id<MTLComputePipelineState> unperm = from_headmajor_h32(device);
    if (!conv || !unperm) return 1;
    const uint32_t n = tokens * heads * dim;
    id<MTLBuffer> hm[3] = {qhm_in, khm_in, nil};
    void* src[3] = {q_buf, k_buf, v_buf};
    for (int i = 0; i < 3; i++) {
        if (hm[i]) continue;
        hm[i] = attn_half_scratch(device, i, (size_t)n * 2);
        if (!hm[i]) return 1;
        [enc setComputePipelineState:conv];
        [enc setBuffer:(__bridge id<MTLBuffer>)src[i] offset:0 atIndex:0];
        [enc setBuffer:hm[i] offset:0 atIndex:1];
        [enc setBytes:&tokens length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&heads length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        g_dispatch += 1;
    }
    id<MTLBuffer> ohm = attn_half_scratch(device, 3, (size_t)n * 4);
    if (!ohm) return 1;
    if (encode_steel_attn(enc, device, hm[0], hm[1], hm[2], ohm, tokens, heads, dim) != 0) return 1;
    [enc setComputePipelineState:unperm];
    [enc setBuffer:ohm offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:1];
    [enc setBytes:&tokens length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&heads length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch += 1;
    return 0;
}

// Head-major half scratch handles for callers that fill the MFA q/k inputs
// themselves (Klein's krms_rope_hm). Static lifetime, never released.
void* zdraw_metal_attn_hm_scratch(void* device_, int slot, size_t bytes) {
    if (slot < 0 || slot > 4) return NULL;
    return (__bridge void*)attn_half_scratch((__bridge id<MTLDevice>)device_, slot, bytes);
}

// MFA attention with q and k already head-major half in the scratch slots
// (0 and 1); only v is converted, O is un-permuted into output as before.
int zdraw_metal_run_attention_mfa_hm_enc(
    void* batch,
    void* qhm,
    void* khm,
    void* v_buf,
    void* output,
    const ZdrawAttnParams* params,
    size_t in_off,
    size_t out_off,
    int keep_o_hm
) {
    if (!batch || !qhm || !khm || !v_buf || !output || !params) return 1;
    if (params->causal || params->heads != params->kv_heads || params->head_dim != 128) {
        return 1;
    }
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)bb->cmd;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!cmd || !enc) return 1;
    return encode_attention_mfa_raw(
        enc,
        cmd.device,
        NULL,
        NULL,
        v_buf,
        output,
        params,
        (__bridge id<MTLBuffer>)qhm,
        (__bridge id<MTLBuffer>)khm,
        nil,
        in_off,
        out_off,
        keep_o_hm
    );
}

static int encode_w6(
    id<MTLComputeCommandEncoder> enc,
    id<MTLDevice> device,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    id<MTLComputePipelineState> w6p = w6_pipeline(device);
    if (!w6p) return 0;
    [enc setComputePipelineState:w6p];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(GemmParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((params->n + 63) / 64, (params->m + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

static void encode_attention(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawAttnParams* params,
    uint32_t kernel,
    size_t thread_count
) {
    void* qb = b->q;
    void* kb = b->k;
    void* vb = b->v;
    if (kernel == ATTN_K_BLOCK) { // block16: stage half copies of q/k/v
        const uint32_t qn = params->tokens * params->heads * params->head_dim;
        const uint32_t kn = params->tokens * params->kv_heads * params->head_dim;
        id<MTLBuffer> some = (__bridge id<MTLBuffer>)b->q;
        id<MTLComputePipelineState> conv = headmajor_convert(some.device);
        void* src[3] = {b->q, b->k, b->v};
        uint32_t count[3] = {qn, kn, kn};
        void* dst[3];
        for (int i = 0; i < 3; i++) {
            id<MTLBuffer> hbuf = attn_half_scratch(some.device, i, (size_t)count[i] * 2);
            if (!conv || !hbuf) { break; }
            const uint32_t hh = i == 0 ? params->heads : params->kv_heads;
            [enc setComputePipelineState:conv];
            [enc setBuffer:(__bridge id<MTLBuffer>)src[i] offset:0 atIndex:0];
            [enc setBuffer:hbuf offset:0 atIndex:1];
            [enc setBytes:&params->tokens length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&hh length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&params->head_dim length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(count[i], 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            dst[i] = (__bridge void*)hbuf;
        }
        qb = dst[0];
        kb = dst[1];
        vb = dst[2];
        g_dispatch += 3;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)qb offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)kb offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)vb offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->mix offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(ZdrawAttnParams) atIndex:4];
    // block16: one threadgroup per 16-query block per head (rows/flash use
    // one threadgroup per token per head).
    size_t groups = kernel == ATTN_K_BLOCK
        ? ((size_t)params->tokens + 15) / 16 * (size_t)params->heads
        : (size_t)params->tokens * (size_t)params->heads;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

static void encode_swiglu(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* gate,
    void* up,
    uint32_t count,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)up offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(uint32_t) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

// Reads the fused [M, 2*inner] gate+up GEMM output and writes the gated
// product compactly so the FFN-down path is unchanged.
static void encode_swiglu_fused(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* gateup,
    void* out,
    uint32_t count,
    uint32_t inner,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)gateup offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&inner length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
    g_dispatch++;
}

static void encode_resid_next_norm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    size_t thread_count
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->attn offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->attn_out offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->attn_gate offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->state offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->ffn_in offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->mlp_scale offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->norm offset:0 atIndex:6];
    [enc setBytes:&p->attn_resid length:sizeof(ZdrawBlockParams) atIndex:7];
    [enc setBytes:&p->ffn_norm length:sizeof(ZdrawBlockParams) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(p->attn_resid.tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

static void encode_chain_layer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> norm,
    id<MTLComputePipelineState> resid,
    id<MTLComputePipelineState> gemm_exact,
    id<MTLComputePipelineState> gemm_half,
    id<MTLComputePipelineState> gemm_w8,
    id<MTLComputePipelineState> qk,
    id<MTLComputePipelineState> attn,
    id<MTLComputePipelineState> swiglu,
    id<MTLComputePipelineState> resid_norm,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    encode_block_kernel(enc, norm, b->state, w->attn_in, w->attn_scale, b->norm, &p->attn_norm, t->block);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->q, b->q, &p->q);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->k, b->k, &p->k);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->v, b->v, &p->v);
    encode_qk_norm(enc, qk, b, w, &p->qk, t->qk);
    encode_attention(enc, attn, b, &p->attn, (uint32_t)t->attn_kernel, t->attn);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->mix, w->proj, b->attn, &p->proj);
    encode_resid_next_norm(enc, resid_norm, b, w, p, t->block);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->ffn_gate, b->gate, &p->ffn_gate);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->ffn_up, b->up, &p->ffn_up);
    encode_swiglu(enc, swiglu, b->gate, b->up, p->ffn_gate.m * p->ffn_gate.n, t->swiglu);
    encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->gate, w->ffn_down, b->ffn, &p->ffn_down);
    encode_block_kernel(enc, resid, b->ffn, w->ffn_out, w->mlp_gate, b->state, &p->ffn_resid, t->block);
}

// Whole layer on our direct-f16 kernel: f16 A staged in-encoder, W16 f16
// weights, f32 outputs, raw kernels for everything else. One encoder, no MPS.
static int encode_chain_layer_ours16(
    id<MTLCommandBuffer> cmd,
    id<MTLComputeCommandEncoder>* enc_io,
    id<MTLComputePipelineState> norm,
    id<MTLComputePipelineState> resid,
    id<MTLComputePipelineState> gemm_exact,
    id<MTLComputePipelineState> gemm_half,
    id<MTLComputePipelineState> gemm_w8,
    id<MTLComputePipelineState> qk,
    id<MTLComputePipelineState> attn,
    id<MTLComputePipelineState> swiglu,
    id<MTLComputePipelineState> swiglu_fused,
    id<MTLComputePipelineState> resid_norm,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t,
    id<MTLBuffer> qkv_a16,
    int* used_mps_qkv,
    int* used_mps_proj,
    int* used_mps_gateup,
    int* used_mps_down
) {
    id<MTLBuffer> nb = (__bridge id<MTLBuffer>)b->norm;
    id<MTLComputePipelineState> direct = ours16_pipeline(nb.device);
    const int trace = gpu_trace_enabled();
    id<MTLComputeCommandEncoder> enc = *enc_io;
#define TRACE_GROUP(kind) \
    if (trace) { [enc endEncoding]; enc = trace_encoder(cmd, kind); if (!enc) return -1; }

    TRACE_GROUP(SP_QKV)
    if (p->q.mode == 4 && p->k.mode == 4 && p->v.mode == 4) {
        // W6 qkv: steel dequant GEMM when gated on (b->norm post-RMSNorm is
        // bounded, so scale 1.0 like the f16-steel qkv path); else slow staged.
        id<MTLComputePipelineState> wp = steel_w6_scope("qkv")
            ? steel_w6_chain_pipe(((__bridge id<MTLBuffer>)b->q).device) : nil;
        id<MTLBuffer> a16 = wp ? (qkv_a16 ? qkv_a16 :
            f16a_encode_convert(enc, b->norm, 0, p->q.m * p->q.k)) : nil;
        if (a16) {
            if (!encode_steel_w6_a16(enc, wp, a16, w->q, b->q, 0, NULL, 1.0f, &p->q)) return -1;
            if (!encode_steel_w6_a16(enc, wp, a16, w->k, b->k, 1, NULL, 1.0f, &p->k)) return -1;
            if (!encode_steel_w6_a16(enc, wp, a16, w->v, b->v, 2, NULL, 1.0f, &p->v)) return -1;
        } else {
            if (!encode_w6(enc, nb.device, b->norm, w->q, b->q, &p->q)) return -1;
            if (!encode_w6(enc, nb.device, b->norm, w->k, b->k, &p->k)) return -1;
            if (!encode_w6(enc, nb.device, b->norm, w->v, b->v, &p->v)) return -1;
        }
        *used_mps_qkv = 1;
    } else if (ours16_ok(&p->q) && direct) {
        id<MTLBuffer> a32 = (__bridge id<MTLBuffer>)b->norm;
        id<MTLComputePipelineState> sp = steel_scope("qkv")
            ? steel_chain_pipe(((__bridge id<MTLBuffer>)b->q).device) : nil;
        id<MTLBuffer> a16 = sp ? f16a_encode_convert(enc, b->norm, 0, p->q.m * p->q.k) : nil;
        int ok;
        if (a16) {
            ok = encode_steel_a16(enc, sp, a16, w->q, b->q, 0, NULL, 1.0f, &p->q) &&
                 encode_steel_a16(enc, sp, a16, w->k, b->k, 1, NULL, 1.0f, &p->k) &&
                 encode_steel_a16(enc, sp, a16, w->v, b->v, 2, NULL, 1.0f, &p->v);
        } else {
            ok = encode_ours16(enc, direct, a32, w->q, b->q, &p->q);
            ok = ok && encode_ours16(enc, direct, a32, w->k, b->k, &p->k);
            ok = ok && encode_ours16(enc, direct, a32, w->v, b->v, &p->v);
        }
        if (ok) *used_mps_qkv = 1;
        if (!ok) return -1;
    } else {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->q, b->q, &p->q);
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->k, b->k, &p->k);
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->v, b->v, &p->v);
    }
    const int mfa_shape_ok = mfa_attn_enabled() && p->attn.causal == 0 &&
        p->attn.heads == p->attn.kv_heads && p->attn.head_dim == 128 &&
        p->attn.tokens >= mfa_min_tokens();
    id<MTLBuffer> qhm = nil;
    id<MTLBuffer> khm = nil;
    int qk_hm = 0;
    TRACE_GROUP(SP_QK_ROPE)
    if (mfa_shape_ok) {
        qk_hm = encode_qk_norm_hm(enc, b, w, &p->qk, &qhm, &khm);
    }
    if (!qk_hm) {
        encode_qk_norm(enc, qk, b, w, &p->qk, t->qk);
    }
    TRACE_GROUP(SP_ATTENTION)
    if (encode_attention_steel_any(enc, ((__bridge id<MTLBuffer>)b->q).device, b->q, b->k, b->v,
                                   b->mix, &p->attn, qhm, khm) == 0) {
        // vendored steel attention (every head_dim-128 shape; no framework)
    } else if (mfa_shape_ok &&
        encode_attention_mfa(enc, b, &p->attn, qhm, khm) == 0) {
        // vendored MFA path encoded
    } else if (qk_hm) {
        return -1;
    } else {
        encode_attention(enc, attn, b, &p->attn, (uint32_t)t->attn_kernel, t->attn);
    }

    TRACE_GROUP(SP_PROJ)
    if (p->proj.mode == 4) {
        // W6 proj: post-attention b->mix can exceed f16 range, so apply the same
        // 1/256-in + 256-out lossless scale as the f16-steel proj path (white-
        // image fix). scratch_override b->gateup survives the attn cmd boundary.
        id<MTLComputePipelineState> wp = steel_w6_scope("proj")
            ? steel_w6_chain_pipe(((__bridge id<MTLBuffer>)b->attn).device) : nil;
        id<MTLBuffer> a16 = wp ? f16a_encode_convert_scaled(enc, b->mix, 0, p->proj.m * p->proj.k, 1.0f / 256.0f) : nil;
        if (a16) {
            if (!encode_steel_w6_a16(enc, wp, a16, w->proj, b->attn, 3, b->gateup, 256.0f, &p->proj)) return -1;
        } else if (!encode_w6(enc, nb.device, b->mix, w->proj, b->attn, &p->proj)) {
            return -1;
        }
        *used_mps_proj = 1;
    } else if (ours16_ok(&p->proj) && direct) {
        id<MTLBuffer> a32 = (__bridge id<MTLBuffer>)b->mix;
        id<MTLComputePipelineState> sp = steel_scope("proj")
            ? steel_chain_pipe(((__bridge id<MTLBuffer>)b->attn).device) : nil;
        id<MTLBuffer> a16 = sp ? f16a_encode_convert_scaled(enc, b->mix, 0, p->proj.m * p->proj.k, 1.0f / 256.0f) : nil;
        if (a16) {
            if (!encode_steel_a16(enc, sp, a16, w->proj, b->attn, 3, b->gateup, 256.0f, &p->proj)) return -1;
        } else if (!encode_ours16(enc, direct, a32, w->proj, b->attn, &p->proj)) {
            return -1;
        }
        *used_mps_proj = 1;
    } else {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->mix, w->proj, b->attn, &p->proj);
    }
    TRACE_GROUP(SP_ATTN_RESID_NORM)
    encode_resid_next_norm(enc, resid_norm, b, w, p, t->block);

    TRACE_GROUP(SP_FFN_GATEUP)
    int fused = 0;
    int pair16 = 0;
    if (p->ffn_gate.mode == 4 && p->ffn_up.mode == 4) {
        // W6 gate/up: steel dequant GEMM when gated on (b->norm bounded -> scale
        // 1.0); outputs cast f16->f32 into b->gate/b->up so swiglu is unchanged.
        id<MTLComputePipelineState> wp = steel_w6_scope("gateup")
            ? steel_w6_chain_pipe(((__bridge id<MTLBuffer>)b->gate).device) : nil;
        id<MTLBuffer> a16 = wp ? f16a_encode_convert(enc, b->norm, 0, p->ffn_gate.m * p->ffn_gate.k) : nil;
        if (a16) {
            if (!encode_steel_w6_a16(enc, wp, a16, w->ffn_gate, b->gate, 0, NULL, 1.0f, &p->ffn_gate)) return -1;
            if (!encode_steel_w6_a16(enc, wp, a16, w->ffn_up, b->up, 1, NULL, 1.0f, &p->ffn_up)) return -1;
        } else {
            if (!encode_w6(enc, nb.device, b->norm, w->ffn_gate, b->gate, &p->ffn_gate)) return -1;
            if (!encode_w6(enc, nb.device, b->norm, w->ffn_up, b->up, &p->ffn_up)) return -1;
        }
        pair16 = 1;
        *used_mps_gateup = 1;
    } else if (unfuse_enabled() && ours16_ok(&p->ffn_gate) && ours16_ok(&p->ffn_up) && direct) {
        id<MTLBuffer> a32 = (__bridge id<MTLBuffer>)b->norm;
        pair16 = encode_ours16(enc, direct, a32, w->ffn_gate, b->gate, &p->ffn_gate) &&
                 encode_ours16(enc, direct, a32, w->ffn_up, b->up, &p->ffn_up);
        if (pair16) *used_mps_gateup = 1;
    }
    if (!pair16 && p->ffn_fused.mode != 0 && ours16_ok(&p->ffn_fused) && direct && swiglu_fused) {
        id<MTLBuffer> gate_buf = (__bridge id<MTLBuffer>)b->gateup;
        id<MTLComputePipelineState> sp = steel_scope("gateup") ? steel_chain_pipe(gate_buf.device) : nil;
        id<MTLComputePipelineState> hs = sp ? f16c_swiglu(gate_buf.device) : nil;
        id<MTLBuffer> a16 = (sp && hs)
            ? f16a_encode_convert(enc, b->norm, 0, p->ffn_fused.m * p->ffn_fused.k)
            : nil;
        if (a16) {
            // steel f16 A -> f16 gateup -> f16 swiglu (all in-encoder).
            encode_steel(enc, sp, a16, w->ffn_fused, b->gateup, &p->ffn_fused);
            fused = 1;
            *used_mps_gateup = 1;
            TRACE_GROUP(SP_SWIGLU)
            encode_swiglu_fused(enc, hs, b->gateup, b->gate,
                                p->ffn_gate.m * p->ffn_gate.n, p->ffn_gate.n, t->swiglu);
        } else {
            fused = encode_ours16(enc, direct, (__bridge id<MTLBuffer>)b->norm,
                                  w->ffn_fused, b->gateup, &p->ffn_fused);
            if (fused) {
                *used_mps_gateup = 1;
                TRACE_GROUP(SP_SWIGLU)
                encode_swiglu_fused(enc, swiglu_fused, b->gateup, b->gate,
                                    p->ffn_gate.m * p->ffn_gate.n, p->ffn_gate.n, t->swiglu);
            }
        }
    }
    if (!fused && !pair16) {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->ffn_gate, b->gate, &p->ffn_gate);
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->ffn_up, b->up, &p->ffn_up);
    }
    id<MTLComputePipelineState> fused_down_w6_pipe = nil;
    if (!fused && pair16 && p->ffn_down.mode == 4 && swiglu_f16a_fuse_enabled() &&
        steel_w6_scope("down")) {
        fused_down_w6_pipe = steel_w6_chain_pipe(((__bridge id<MTLBuffer>)b->ffn).device);
    }
    const int fuse_swiglu_down = fused_down_w6_pipe != nil;
    if (!fused && !fuse_swiglu_down) {
        encode_swiglu(enc, swiglu, b->gate, b->up, p->ffn_gate.m * p->ffn_gate.n, t->swiglu);
    }
    id<MTLBuffer> ffn_resid_f16 = nil;
    float ffn_resid_f16_scale = 1.0f;
    TRACE_GROUP(SP_FFN_DOWN)
    if (p->ffn_down.mode == 4) { // W6: dequant GEMM (steel when gated, else staged)
        // W6 down: post-swiglu b->gate can exceed f16 range -> same 1/256-in +
        // 256-out lossless scale as the f16-steel down path. b->gateup is free
        // here (gate/up are unfused in the W6 path) and survives as scratch.
        id<MTLComputePipelineState> wp = fused_down_w6_pipe ? fused_down_w6_pipe :
            (steel_w6_scope("down") ? steel_w6_chain_pipe(((__bridge id<MTLBuffer>)b->ffn).device) : nil);
        id<MTLBuffer> a16 = nil;
        if (wp) {
            a16 = fuse_swiglu_down
                ? f16a_encode_swiglu_scaled(enc, b->gate, b->up, 0, p->ffn_down.m * p->ffn_down.k, 1.0f / 256.0f)
                : f16a_encode_convert_scaled(enc, b->gate, 0, p->ffn_down.m * p->ffn_down.k, 1.0f / 256.0f);
        }
        if (a16 && ffn_resid_f16_fuse_enabled()) {
            id<MTLBuffer> scratch = (__bridge id<MTLBuffer>)b->gateup;
            encode_steel_w6(enc, wp, a16, w->ffn_down, (__bridge void*)scratch, &p->ffn_down);
            ffn_resid_f16 = scratch;
            ffn_resid_f16_scale = 256.0f;
        } else if (a16) {
            if (!encode_steel_w6_a16(enc, wp, a16, w->ffn_down, b->ffn, 4, b->gateup, 256.0f, &p->ffn_down)) return -1;
        } else if (!encode_w6(enc, nb.device, b->gate, w->ffn_down, b->ffn, &p->ffn_down)) {
            return -1;
        }
        *used_mps_down = 1;
    } else if (ours16_ok(&p->ffn_down) && direct) {
        id<MTLBuffer> a32 = (__bridge id<MTLBuffer>)b->gate;
        id<MTLComputePipelineState> sp = steel_scope("down")
            ? steel_chain_pipe(((__bridge id<MTLBuffer>)b->ffn).device) : nil;
        id<MTLBuffer> a16 = sp ? f16a_encode_convert_scaled(enc, b->gate, 0, p->ffn_down.m * p->ffn_down.k, 1.0f / 256.0f) : nil;
        if (a16) {
            if (!encode_steel_a16(enc, sp, a16, w->ffn_down, b->ffn, 4, b->gateup, 256.0f, &p->ffn_down)) return -1;
        } else if (!encode_ours16(enc, direct, a32, w->ffn_down, b->ffn, &p->ffn_down)) {
            return -1;
        }
        *used_mps_down = 1;
    } else {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->gate, w->ffn_down, b->ffn, &p->ffn_down);
    }
    TRACE_GROUP(SP_FFN_RESID)
    if (ffn_resid_f16) {
        if (!encode_resid_f16_scaled(enc, (__bridge void*)ffn_resid_f16,
                                     w->ffn_out, w->mlp_gate, b->state,
                                     &p->ffn_resid, ffn_resid_f16_scale, t->block)) {
            return -1;
        }
    } else {
        encode_block_kernel(enc, resid, b->ffn, w->ffn_out, w->mlp_gate, b->state, &p->ffn_resid, t->block);
    }
#undef TRACE_GROUP
    *enc_io = enc;
    return 0;
}

// ============== f16 activation tier (P1 stage 1-2, ZDRAW_STACK_ACT=f16) ==============
// Purely additive regime: when every layer is eligible, the chain runs with
// f16-resident activations - half state (living in b->up, which the fused FFN
// route never touches), half norm, half gate (1/256-scaled, the banked range
// idiom) - and the half-A direct GEMM (gemm_f16a_v2) replaces the in-kernel
// f32->f16 A staging for qkv/fused/down. q/k/v/mix/attn/gateup/ffn stay f32,
// so attention (MFA/SDPA/rows) and proj are untouched. Any ineligible layer
// keeps the whole chain on the default f32 regime; default behavior is
// byte-identical by construction (no legacy path is edited).

static int act16_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = getenv("ZDRAW_STACK_ACT");
        cached = (raw && strcmp(raw, "f16") == 0) ? 1 : 0;
    }
    return cached;
}

static void act16_report(void) {
    static int reported = 0;
    if (!reported) {
        reported = 1;
        fprintf(stderr, "stack: f16 activation tier active\n");
    }
}

typedef struct {
    id<MTLComputePipelineState> state_in;
    id<MTLComputePipelineState> norm_h;
    id<MTLComputePipelineState> resid_next_h;
    id<MTLComputePipelineState> resid_h;
    id<MTLComputePipelineState> swiglu_h;
    id<MTLComputePipelineState> final_h;
    id<MTLComputePipelineState> gemm;
} Act16Pipes;

static id<MTLComputePipelineState> f16a_direct_pipeline(id<MTLDevice> device);

static Act16Pipes* act16_pipes(id<MTLDevice> device) {
    static Act16Pipes pipes;
    static int state = 0; // 0 unbuilt, 1 ok, -1 failed
    if (state) return state > 0 ? &pipes : NULL;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct Params { uint tokens; uint hidden; uint dtype; uint has_scale;"
        "float eps; ulong weight_offset; };\n"
        "struct FinalNormParams { uint tokens; uint hidden; float eps; uint pad; };\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "static inline float weight_at(const device uchar* base, uint index, uint dtype) {"
        "if (dtype == 3) return ((const device float*)base)[index];"
        "ushort bits = ((const device ushort*)base)[index];"
        "if (dtype == 2) return as_type<float>(uint(bits) << 16);"
        "return float(as_type<half>(bits)); }\n"
        "static inline half sat_h(float v) { return half(clamp(v, -65504.0f, 65504.0f)); }\n"
        "kernel void act_state_in(const device float* src [[buffer(0)]],"
        "device half* dst [[buffer(1)]], constant uint& count [[buffer(2)]],"
        "uint i [[thread_position_in_grid]]) {"
        "if (i < count) dst[i] = sat_h(src[i]); }\n"
        "kernel void act_norm_h(const device half* input [[buffer(0)]],"
        "const device uchar* weight_bytes [[buffer(1)]],"
        "const device float* scale [[buffer(2)]],"
        "device half* output [[buffer(3)]],"
        "constant Params& p [[buffer(4)]],"
        "uint tok [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint tg_size [[threads_per_threadgroup]]) {"
        "threadgroup float reduce[256];"
        "uint base = tok * p.hidden;"
        "const device uchar* weight = weight_bytes + p.weight_offset;"
        "float local = 0.0f;"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = float(input[base + dim]); local += value * value; }"
        "reduce[tid] = local;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float norm = rsqrt(reduce[0] / float(p.hidden) + p.eps);"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = float(input[base + dim]) * norm * weight_at(weight, dim, p.dtype);"
        "if (p.has_scale != 0) value *= scale[dim];"
        "output[base + dim] = sat_h(value); }"
        "}\n"
        "kernel void act_resid_next_h(const device float* input [[buffer(0)]],"
        "const device uchar* resid_weight_bytes [[buffer(1)]],"
        "const device float* gate [[buffer(2)]],"
        "device half* state [[buffer(3)]],"
        "const device uchar* norm_weight_bytes [[buffer(4)]],"
        "const device float* scale [[buffer(5)]],"
        "device half* output [[buffer(6)]],"
        "constant Params& resid [[buffer(7)]],"
        "constant Params& norm_p [[buffer(8)]],"
        "uint tok [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint tg_size [[threads_per_threadgroup]]) {"
        "threadgroup float reduce[256];"
        "uint base = tok * resid.hidden;"
        "const device uchar* rw = resid_weight_bytes + resid.weight_offset;"
        "float local = 0.0f;"
        "for (uint dim = tid; dim < resid.hidden; dim += tg_size) {"
        "float value = input[base + dim]; local += value * value; }"
        "reduce[tid] = local;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float resid_norm = rsqrt(reduce[0] / float(resid.hidden) + resid.eps);"
        "float state_sum = 0.0f;"
        "for (uint dim = tid; dim < resid.hidden; dim += tg_size) {"
        "float inc = input[base + dim] * resid_norm * weight_at(rw, dim, resid.dtype);"
        "if (resid.has_scale != 0) inc *= gate[dim];"
        "float next = float(state[base + dim]) + inc;"
        "state[base + dim] = sat_h(next);"
        "state_sum += next * next; }"
        "reduce[tid] = state_sum;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float norm = rsqrt(reduce[0] / float(norm_p.hidden) + norm_p.eps);"
        "const device uchar* nw = norm_weight_bytes + norm_p.weight_offset;"
        "for (uint dim = tid; dim < norm_p.hidden; dim += tg_size) {"
        "float value = float(state[base + dim]) * norm * weight_at(nw, dim, norm_p.dtype);"
        "if (norm_p.has_scale != 0) value *= scale[dim];"
        "output[base + dim] = sat_h(value); }"
        "}\n"
        "kernel void act_resid_h(const device float* input [[buffer(0)]],"
        "const device uchar* weight_bytes [[buffer(1)]],"
        "const device float* gate [[buffer(2)]],"
        "device half* state [[buffer(3)]],"
        "constant Params& p [[buffer(4)]],"
        "uint tok [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint tg_size [[threads_per_threadgroup]]) {"
        "threadgroup float reduce[256];"
        "uint base = tok * p.hidden;"
        "const device uchar* weight = weight_bytes + p.weight_offset;"
        "float local = 0.0f;"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = input[base + dim] * 256.0f; local += value * value; }"
        "reduce[tid] = local;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {"
        "if (tid < stride) reduce[tid] += reduce[tid + stride];"
        "threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "float norm = rsqrt(reduce[0] / float(p.hidden) + p.eps);"
        "for (uint dim = tid; dim < p.hidden; dim += tg_size) {"
        "float value = input[base + dim] * 256.0f * norm * weight_at(weight, dim, p.dtype);"
        "if (p.has_scale != 0) value *= gate[dim];"
        "float next = float(state[base + dim]) + value;"
        "state[base + dim] = sat_h(next); }"
        "}\n"
        "kernel void act_swiglu_h(const device float* gateup [[buffer(0)]],"
        "device half* out [[buffer(1)]],"
        "constant uint& count [[buffer(2)]],"
        "constant uint& inner [[buffer(3)]],"
        "uint id [[thread_position_in_grid]]) {"
        "if (id >= count) return;"
        "uint row = id / inner;"
        "uint col = id - row * inner;"
        "float g = gateup[row * 2 * inner + col];"
        "float u = gateup[row * 2 * inner + inner + col];"
        "out[id] = sat_h((g / (1.0f + exp(-g))) * u * (1.0f / 256.0f)); }\n"
        "kernel void act_final_h(const device half* input [[buffer(0)]],"
        "const device float* scale [[buffer(1)]],"
        "device float* output [[buffer(2)]],"
        "constant FinalNormParams& p [[buffer(3)]],"
        "uint token [[thread_position_in_grid]]) {"
        "if (token >= p.tokens) return;"
        "const uint base = token * p.hidden;"
        "float mean = 0.0f;"
        "for (uint i = 0; i < p.hidden; i++) mean += float(input[base + i]);"
        "mean /= float(p.hidden);"
        "float var_sum = 0.0f;"
        "for (uint i = 0; i < p.hidden; i++) {"
        "const float diff = float(input[base + i]) - mean;"
        "var_sum += diff * diff; }"
        "const float inv = 1.0f / sqrt(var_sum / float(p.hidden) + p.eps);"
        "for (uint i = 0; i < p.hidden; i++) {"
        "output[base + i] = (float(input[base + i]) - mean) * inv * scale[i]; }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) { state = -1; return NULL; }
#define ACT16_FN(field, name) \
    { id<MTLFunction> fn = [lib newFunctionWithName:@name]; \
      pipes.field = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil; \
      if (!pipes.field) { state = -1; return NULL; } }
    ACT16_FN(state_in, "act_state_in")
    ACT16_FN(norm_h, "act_norm_h")
    ACT16_FN(resid_next_h, "act_resid_next_h")
    ACT16_FN(resid_h, "act_resid_h")
    ACT16_FN(swiglu_h, "act_swiglu_h")
    ACT16_FN(final_h, "act_final_h")
#undef ACT16_FN
    // ZDRAW_ACT_GEMM=v2 keeps the flat-loop kernels: the in-chain A/B that
    // discriminates cold-weight latency structure from everything else.
    const char* ag = getenv("ZDRAW_ACT_GEMM");
    pipes.gemm = (ag && strcmp(ag, "v2") == 0)
        ? f16a_v2_pipeline(device)
        : f16a_direct_pipeline(device);
    if (!pipes.gemm) { state = -1; return NULL; }
    state = 1;
    return &pipes;
}

// Chain-grade half-A GEMM: the ours-v2 body (direct W, double-buffered
// threadgroup A staging) with the A stage reduced to a pure half4 copy. No
// epilogue scaling: a thread_elements() epilogue measured 4.6x slower on the
// down op (it forces the accumulators out of matrix registers), so the 1/256
// gate range idiom is restored where the down C is consumed (act_resid_h).
static id<MTLComputePipelineState> g_f16a_direct = nil;

static id<MTLComputePipelineState> f16a_direct_pipeline(id<MTLDevice> device) {
    if (g_f16a_direct) return g_f16a_direct;
    NSString* src = @"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };\n"
        "kernel void gemm_f16a_d(const device half* A [[buffer(0)]],"
        "const device half* W [[buffer(1)]],"
        "device float* C [[buffer(2)]],"
        "constant GemmParams& p [[buffer(3)]],"
        "uint2 tg [[threadgroup_position_in_grid]],"
        "uint tid [[thread_index_in_threadgroup]],"
        "uint sgid [[simdgroup_index_in_threadgroup]]) {"
        "const device half* Wp = W + (p.weight_offset >> 1);"
        "const uint tile_m = tg.y * 64; const uint tile_n = tg.x * 64;"
        "const uint sm = (sgid / 2) * 32; const uint sn = (sgid % 2) * 32;"
        "threadgroup half As[2][64 * 32];"
        "const uint row = tid >> 1; const uint seg = (tid & 1) * 16;"
        "simdgroup_float8x8 acc[4][4];"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "  acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
        "uint cur = 0;"
        "const uint ar = tile_m + row;"
        "const device half4* Arow = (const device half4*)(A + ar * p.k);"
        "{ threadgroup half4* ad = (threadgroup half4*)&As[0][row * 32 + seg];"
        "  for (uint q = 0; q < 4; q++)"
        "    ad[q] = ar < p.m ? Arow[(seg >> 2) + q] : half4(0.0h); }"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "const uint wr = tile_n + sn;"
        "for (uint k0 = 0; k0 < p.k; k0 += 32) {"
        "  const uint nk = k0 + 32;"
        "  if (nk < p.k) {"
        "    threadgroup half4* ad = (threadgroup half4*)&As[1 - cur][row * 32 + seg];"
        "    const uint base = (nk + seg) >> 2;"
        "    for (uint q = 0; q < 4; q++)"
        "      ad[q] = ar < p.m ? Arow[base + q] : half4(0.0h); }"
        "  for (uint kk = 0; kk < 32; kk += 8) {"
        "    simdgroup_half8x8 a[4]; simdgroup_half8x8 b[4];"
        "    for (uint i = 0; i < 4; i++)"
        "      simdgroup_load(a[i], &As[cur][(sm + i * 8) * 32 + kk], 32);"
        "    for (uint j = 0; j < 4; j++)"
        "      simdgroup_load(b[j], Wp + (wr + j * 8) * p.k + k0 + kk, p.k, ulong2(0, 0), true);"
        "    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)"
        "      simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); }"
        "  cur = 1 - cur;"
        "  threadgroup_barrier(mem_flags::mem_threadgroup); }"
        "const uint cm = tile_m + sm; const uint cn = tile_n + sn;"
        "for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {"
        "  if (cm + i * 8 < p.m && cn + j * 8 < p.n)"
        "    simdgroup_store(acc[i][j], C + (cm + i * 8) * p.n + (cn + j * 8), p.n); }"
        "}";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:zdraw_compile_options() error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_f16a_d"];
    if (!fn) return nil;
    g_f16a_direct = [device newComputePipelineStateWithFunction:fn error:&err];
    return g_f16a_direct;
}

// gemm_f16a_v2 contract on top of ours16_ok: min-clamped 64x64 tiles need
// m,n >= 64 and n % 64 == 0 for the unguarded direct-W loads.
static int act16_gemm_ok(const GemmParams* p) {
    return ours16_ok(p) && p->m >= 64 && p->n >= 64 && (p->n & 63u) == 0u;
}

static int act16_layer_ok(const ZdrawBlockChainParams* p) {
    return !unfuse_enabled() &&
           act16_gemm_ok(&p->q) && act16_gemm_ok(&p->k) && act16_gemm_ok(&p->v) &&
           ours16_ok(&p->proj) &&
           p->ffn_fused.mode != 0 && act16_gemm_ok(&p->ffn_fused) &&
           act16_gemm_ok(&p->ffn_down);
}

static int g_act16_step = 0;

static int encode_f16a(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    const GemmParams* params
) {
    if (!pipe) return 0;
    [enc setComputePipelineState:pipe];
    ours16_dispatch(enc, (__bridge id<MTLBuffer>)a_buf, w_buf, c_buf, params, 0, 0);
    return 1;
}

static int encode_chain_layer_act16(
    id<MTLCommandBuffer> cmd,
    id<MTLComputeCommandEncoder>* enc_io,
    id<MTLComputePipelineState> qk,
    id<MTLComputePipelineState> attn,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    id<MTLComputeCommandEncoder> enc = *enc_io;
    id<MTLDevice> dev = ((__bridge id<MTLBuffer>)b->norm).device;
    Act16Pipes* a = act16_pipes(dev);
    if (!a) return -1;
    const int trace = gpu_trace_enabled();
#define TRACE_GROUP(kind) \
    if (trace) { [enc endEncoding]; enc = trace_encoder(cmd, kind); if (!enc) return -1; }

    TRACE_GROUP(SP_QKV)
    // Half state (b->up) -> half norm, then half-A direct qkv.
    [enc setComputePipelineState:a->norm_h];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->up offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->attn_in offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->attn_scale offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->norm offset:0 atIndex:3];
    [enc setBytes:&p->attn_norm length:sizeof(ZdrawBlockParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(p->attn_norm.tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(t->block, 1, 1)];
    g_dispatch++;
    if (!encode_f16a(enc, a->gemm, b->norm, w->q, b->q, &p->q)) return -1;
    if (!encode_f16a(enc, a->gemm, b->norm, w->k, b->k, &p->k)) return -1;
    if (!encode_f16a(enc, a->gemm, b->norm, w->v, b->v, &p->v)) return -1;

    // Attention: same route logic as the ours16 layer (MFA -> SDPA -> rows);
    // q/k/v/mix stay f32 so this section is precisely the certified path.
    const int mfa_shape_ok = mfa_attn_enabled() && p->attn.causal == 0 &&
        p->attn.heads == p->attn.kv_heads && p->attn.head_dim == 128 &&
        p->attn.tokens >= mfa_min_tokens();
    id<MTLBuffer> qhm = nil;
    id<MTLBuffer> khm = nil;
    int qk_hm = 0;
    TRACE_GROUP(SP_QK_ROPE)
    if (mfa_shape_ok) {
        qk_hm = encode_qk_norm_hm(enc, b, w, &p->qk, &qhm, &khm);
    }
    if (!qk_hm) {
        encode_qk_norm(enc, qk, b, w, &p->qk, t->qk);
    }
    TRACE_GROUP(SP_ATTENTION)
    if (encode_attention_steel_any(enc, ((__bridge id<MTLBuffer>)b->q).device, b->q, b->k, b->v,
                                   b->mix, &p->attn, qhm, khm) == 0) {
        // vendored steel attention (every head_dim-128 shape; no framework)
    } else if (mfa_shape_ok &&
        encode_attention_mfa(enc, b, &p->attn, qhm, khm) == 0) {
        // vendored MFA path encoded
    } else if (qk_hm) {
        return -1;
    } else {
        encode_attention(enc, attn, b, &p->attn, (uint32_t)t->attn_kernel, t->attn);
    }

    TRACE_GROUP(SP_PROJ)
    // proj keeps the f32-A direct kernel (b->mix stays f32 in this tier).
    if (!encode_ours16(enc, ours16_pipeline(dev), (__bridge id<MTLBuffer>)b->mix,
                       w->proj, b->attn, &p->proj)) {
        return -1;
    }

    TRACE_GROUP(SP_ATTN_RESID_NORM)
    [enc setComputePipelineState:a->resid_next_h];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->attn offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->attn_out offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->attn_gate offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->up offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->ffn_in offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->mlp_scale offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->norm offset:0 atIndex:6];
    [enc setBytes:&p->attn_resid length:sizeof(ZdrawBlockParams) atIndex:7];
    [enc setBytes:&p->ffn_norm length:sizeof(ZdrawBlockParams) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(p->attn_resid.tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(t->block, 1, 1)];
    g_dispatch++;

    TRACE_GROUP(SP_FFN_GATEUP)
    if (!encode_f16a(enc, a->gemm, b->norm, w->ffn_fused, b->gateup, &p->ffn_fused)) return -1;
    TRACE_GROUP(SP_SWIGLU)
    {
        uint32_t count = p->ffn_gate.m * p->ffn_gate.n;
        uint32_t inner = p->ffn_gate.n;
        [enc setComputePipelineState:a->swiglu_h];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->gateup offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->gate offset:0 atIndex:1];
        [enc setBytes:&count length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&inner length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(t->swiglu, 1, 1)];
        g_dispatch++;
    }
    TRACE_GROUP(SP_FFN_DOWN)
    // Half gate carries 1/256; act_resid_h restores 256x at its input reads.
    if (!encode_f16a(enc, a->gemm, b->gate, w->ffn_down, b->ffn, &p->ffn_down)) {
        return -1;
    }
    TRACE_GROUP(SP_FFN_RESID)
    [enc setComputePipelineState:a->resid_h];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->ffn offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->ffn_out offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w->mlp_gate offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b->up offset:0 atIndex:3];
    [enc setBytes:&p->ffn_resid length:sizeof(ZdrawBlockParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(p->ffn_resid.tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(t->block, 1, 1)];
    g_dispatch++;
#undef TRACE_GROUP
    *enc_io = enc;
    return 0;
}

static int encode_chain_layer_mps_dense(
    id<MTLCommandBuffer> cmd,
    id<MTLComputeCommandEncoder>* enc_io,
    id<MTLComputePipelineState> norm,
    id<MTLComputePipelineState> resid,
    id<MTLComputePipelineState> gemm_exact,
    id<MTLComputePipelineState> gemm_half,
    id<MTLComputePipelineState> gemm_w8,
    id<MTLComputePipelineState> qk,
    id<MTLComputePipelineState> attn,
    id<MTLComputePipelineState> swiglu,
    id<MTLComputePipelineState> swiglu_fused,
    id<MTLComputePipelineState> resid_norm,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t,
    int use_mps_qkv,
    int use_mps_proj,
    int use_mps_gateup,
    int use_mps_down,
    int* used_mps_qkv,
    int* used_mps_proj,
    int* used_mps_gateup,
    int* used_mps_down
) {
    id<MTLComputeCommandEncoder> enc = *enc_io;
    *used_mps_qkv = 0;
    *used_mps_proj = 0;
    *used_mps_gateup = 0;
    *used_mps_down = 0;

    id<MTLBuffer> qkv_a16 = nil;
    if (dense_ours16_enabled() && norm_f16a_fuse_enabled() &&
        p->q.mode == 4 && p->k.mode == 4 && p->v.mode == 4 && steel_w6_scope("qkv") &&
        steel_w6_chain_pipe(((__bridge id<MTLBuffer>)b->q).device)) {
        qkv_a16 = f16a_encode_norm(enc, b->state, w->attn_in, w->attn_scale, 0,
                                   &p->attn_norm, t->block);
    }
    if (!qkv_a16) {
        encode_block_kernel(enc, norm, b->state, w->attn_in, w->attn_scale, b->norm, &p->attn_norm, t->block);
    }
    if (dense_ours16_enabled()) {
        const int rc = encode_chain_layer_ours16(
            cmd, &enc, norm, resid, gemm_exact, gemm_half, gemm_w8, qk, attn, swiglu,
            swiglu_fused, resid_norm, b, w, p, t, qkv_a16,
            used_mps_qkv, used_mps_proj, used_mps_gateup, used_mps_down);
        *enc_io = enc;
        return rc;
    }
    const int f16a = mps_f16a_enabled();
    int used_qkv = 0;
    if (use_mps_qkv && mps_qkv_triple_eligible(p)) {
        id<MTLBuffer> a16 = f16a
            ? f16a_encode_convert(enc, b->norm, 0, p->q.m * p->q.k)
            : nil;
        [enc endEncoding];
        used_qkv = encode_mps_qkv_triple(cmd, a16, b, w, p);
        if (used_qkv) *used_mps_qkv = 1;
        enc = [cmd computeCommandEncoder];
        if (!enc) return -1;
    }
    if (!used_qkv) {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->q, b->q, &p->q);
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->k, b->k, &p->k);
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->v, b->v, &p->v);
    }
    encode_qk_norm(enc, qk, b, w, &p->qk, t->qk);
    encode_attention(enc, attn, b, &p->attn, (uint32_t)t->attn_kernel, t->attn);

    int used_proj = 0;
    if (use_mps_proj && mps_gemm_eligible(&p->proj)) {
        id<MTLBuffer> a16 = f16a
            ? f16a_encode_convert(enc, b->mix, 1, p->proj.m * p->proj.k)
            : nil;
        [enc endEncoding];
        used_proj = encode_mps_proj(cmd, a16, b, w, p);
        if (used_proj) *used_mps_proj = 1;
        enc = [cmd computeCommandEncoder];
        if (!enc) return -1;
    }
    if (!used_proj) {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->mix, w->proj, b->attn, &p->proj);
    }
    encode_resid_next_norm(enc, resid_norm, b, w, p, t->block);

    int used_gateup = 0;
    int fused = 0;
    // Resident steel GEMM for the FFN gate/up (the biggest 1024px GEMM): f16 A
    // -> f16 gateup -> f16 swiglu, all in the open encoder (no MPS break).
    if (0 && f16a && swiglu_fused && p->ffn_fused.mode != 0) {  // redundant: steel is in the ours path
        id<MTLBuffer> gate_buf = (__bridge id<MTLBuffer>)b->gateup;
        id<MTLComputePipelineState> sp = steel_chain_pipe(gate_buf.device);
        id<MTLComputePipelineState> hs = f16c_swiglu(gate_buf.device);
        id<MTLBuffer> a16 = (sp && hs)
            ? f16a_encode_convert(enc, b->norm, 0, p->ffn_fused.m * p->ffn_fused.k)
            : nil;
        if (a16) {
            encode_steel(enc, sp, a16, w->ffn_fused, b->gateup, &p->ffn_fused);
            encode_swiglu_fused(enc, hs, b->gateup, b->gate,
                                p->ffn_gate.m * p->ffn_gate.n, p->ffn_gate.n, t->swiglu);
            used_gateup = 1;
            *used_mps_gateup = 1;
        }
    }
    if (!used_gateup && use_mps_gateup && swiglu_fused && p->ffn_fused.mode != 0 &&
        mps_gemm_eligible(&p->ffn_fused)) {
        id<MTLBuffer> a16 = f16a
            ? f16a_encode_convert(enc, b->norm, 0, p->ffn_fused.m * p->ffn_fused.k)
            : nil;
        // MPSMatrix only allows f16 C when A is also f16 (its sole mixed
        // mode is f32A x f16B -> f32C), so the probe requires both flags.
        const int c16 = mps_f16c_enabled() && a16 != nil;
        id<MTLBuffer> gate_buf = (__bridge id<MTLBuffer>)b->gateup;
        id<MTLComputePipelineState> half_swiglu = c16 ? f16c_swiglu(gate_buf.device) : nil;
        [enc endEncoding];
        fused = encode_mps_gemm_a(cmd, a16, c16 && half_swiglu, b->norm, w->ffn_fused,
                                  b->gateup, &p->ffn_fused);
        enc = [cmd computeCommandEncoder];
        if (!enc) return -1;
        if (fused) {
            used_gateup = 1;
            *used_mps_gateup = 1;
            encode_swiglu_fused(enc, (c16 && half_swiglu) ? half_swiglu : swiglu_fused,
                                b->gateup, b->gate,
                                p->ffn_gate.m * p->ffn_gate.n, p->ffn_gate.n, t->swiglu);
        }
    }
    if (!used_gateup && use_mps_gateup && mps_gateup_pair_eligible(p)) {
        id<MTLBuffer> a16 = f16a
            ? f16a_encode_convert(enc, b->norm, 0, p->ffn_gate.m * p->ffn_gate.k)
            : nil;
        [enc endEncoding];
        used_gateup = encode_mps_gateup_pair(cmd, a16, b, w, p);
        if (used_gateup) {
            *used_mps_gateup = 1;
        }
        enc = [cmd computeCommandEncoder];
        if (!enc) return -1;
    }
    if (!used_gateup) {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->ffn_gate, b->gate, &p->ffn_gate);
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->norm, w->ffn_up, b->up, &p->ffn_up);
    }
    if (!fused) {
        encode_swiglu(enc, swiglu, b->gate, b->up, p->ffn_gate.m * p->ffn_gate.n, t->swiglu);
    }
    int used_down = 0;
    if (use_mps_down && mps_gemm_eligible(&p->ffn_down)) {
        id<MTLBuffer> a16 = f16a
            ? f16a_encode_convert(enc, b->gate, 2, p->ffn_down.m * p->ffn_down.k)
            : nil;
        [enc endEncoding];
        used_down = encode_mps_down(cmd, a16, b, w, p);
        if (used_down) *used_mps_down = 1;
        enc = [cmd computeCommandEncoder];
        if (!enc) return -1;
    }
    if (!used_down) {
        encode_gemm_auto(enc, gemm_exact, gemm_half, gemm_w8, b->gate, w->ffn_down, b->ffn, &p->ffn_down);
    }
    encode_block_kernel(enc, resid, b->ffn, w->ffn_out, w->mlp_gate, b->state, &p->ffn_resid, t->block);
    *enc_io = enc;
    return 0;
}

static void count_chain_layer(const ZdrawBlockChainParams* p) {
    count_gemm(&p->q); count_gemm(&p->k); count_gemm(&p->v); count_gemm(&p->proj);
    count_gemm(&p->ffn_gate); count_gemm(&p->ffn_up); count_gemm(&p->ffn_down);
    g_dispatch += 6;
}

static void count_chain_layer_mps_dense(
    const ZdrawBlockChainParams* p,
    int used_mps_qkv,
    int used_mps_proj,
    int used_mps_gateup,
    int used_mps_down
) {
    if (dense_ours16_enabled()) {
        count_chain_layer(p);
        return;
    }
    if (used_mps_qkv) {
        count_mps_gemm();
        count_mps_gemm();
        count_mps_gemm();
    } else {
        count_gemm(&p->q);
        count_gemm(&p->k);
        count_gemm(&p->v);
    }
    if (used_mps_proj) {
        count_mps_gemm();
    } else {
        count_gemm(&p->proj);
    }
    if (used_mps_gateup) {
        count_mps_gemm();
        count_mps_gemm();
    } else {
        count_gemm(&p->ffn_gate);
        count_gemm(&p->ffn_up);
    }
    if (used_mps_down) {
        count_mps_gemm();
    } else {
        count_gemm(&p->ffn_down);
    }
    g_dispatch += 6;
}

int zdraw_metal_encode_chain_layer(
    void* batch,
    void* norm_pipeline,
    void* resid_pipeline,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* qk_pipeline,
    void* attn_pipeline,
    void* swiglu_pipeline,
    void* swiglu_fused_pipeline,
    void* resid_norm_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    if (dense_mps_any_enabled()) {
        id<MTLCommandBuffer> cmd =
            (__bridge id<MTLCommandBuffer>)bb->cmd;
        id<MTLComputeCommandEncoder> initial = enc;
        int used_qkv = 0, used_proj = 0, used_gateup = 0, used_down = 0;
        int rc = encode_chain_layer_mps_dense(
            cmd,
            &enc,
            (__bridge id<MTLComputePipelineState>)norm_pipeline,
            (__bridge id<MTLComputePipelineState>)resid_pipeline,
            (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline,
            (__bridge id<MTLComputePipelineState>)gemm_half_pipeline,
            (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline,
            (__bridge id<MTLComputePipelineState>)qk_pipeline,
            (__bridge id<MTLComputePipelineState>)attn_pipeline,
            (__bridge id<MTLComputePipelineState>)swiglu_pipeline,
            (__bridge id<MTLComputePipelineState>)swiglu_fused_pipeline,
            (__bridge id<MTLComputePipelineState>)resid_norm_pipeline,
            b,
            w,
            p,
            t,
            dense_mps_qkv_enabled(),
            dense_mps_proj_enabled(),
            dense_mps_gateup_enabled(),
            dense_mps_down_enabled(),
            &used_qkv,
            &used_proj,
            &used_gateup,
            &used_down);
        if (enc != initial) {
            id previous __attribute__((unused)) =
                (__bridge_transfer id<MTLComputeCommandEncoder>)bb->enc;
            bb->enc = (__bridge_retained void*)enc;
        }
        if (rc != 0) return rc;
        count_chain_layer_mps_dense(
            p,
            used_qkv,
            used_proj,
            used_gateup,
            used_down);
        return 0;
    }
    encode_chain_layer(
        enc,
        (__bridge id<MTLComputePipelineState>)norm_pipeline,
        (__bridge id<MTLComputePipelineState>)resid_pipeline,
        (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline,
        (__bridge id<MTLComputePipelineState>)gemm_half_pipeline,
        (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline,
        (__bridge id<MTLComputePipelineState>)qk_pipeline,
        (__bridge id<MTLComputePipelineState>)attn_pipeline,
        (__bridge id<MTLComputePipelineState>)swiglu_pipeline,
        (__bridge id<MTLComputePipelineState>)resid_norm_pipeline,
        b,
        w,
        p,
        t);
    count_chain_layer(p);
    return 0;
}

// ToMA splits a block at merge boundaries, but its GEMMs must retain the same
// production backend selection as the unsplit resident chain. Falling back to
// gemm_half here makes a reduced block slower than the full-token ours-f16
// path and invalidates the experiment.
static int encode_toma_gemm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> exact,
    id<MTLComputePipelineState> half,
    id<MTLComputePipelineState> w8,
    void* input,
    void* weight,
    void* output,
    const GemmParams* p
) {
    id<MTLBuffer> a = (__bridge id<MTLBuffer>)input;
    if (p->mode == 4) {
        if (!encode_w6(enc, a.device, input, weight, output, p)) return -1;
        count_gemm(p);
        return 0;
    }
    if (ours16_ok(p)) {
        id<MTLComputePipelineState> direct = ours16_pipeline(a.device);
        if (encode_ours16(enc, direct, a, weight, output, p)) {
            count_gemm(p);
            return 0;
        }
    }
    encode_gemm_auto(enc, exact, half, w8, input, weight, output, p);
    count_gemm(p);
    return 0;
}

int zdraw_metal_encode_toma_attn_norm(
    void* batch,
    void* norm_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    encode_block_kernel(
        enc,
        (__bridge id<MTLComputePipelineState>)norm_pipeline,
        b->state,
        w->attn_in,
        w->attn_scale,
        b->norm,
        &p->attn_norm,
        t->block);
    g_dispatch++;
    return 0;
}

int zdraw_metal_encode_toma_attn_core(
    void* batch,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* qk_pipeline,
    void* attn_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    id<MTLComputePipelineState> exact =
        (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline;
    id<MTLComputePipelineState> half =
        (__bridge id<MTLComputePipelineState>)gemm_half_pipeline;
    id<MTLComputePipelineState> w8 =
        (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline;
    if (encode_toma_gemm(enc, exact, half, w8, b->norm, w->q, b->q, &p->q) ||
        encode_toma_gemm(enc, exact, half, w8, b->norm, w->k, b->k, &p->k) ||
        encode_toma_gemm(enc, exact, half, w8, b->norm, w->v, b->v, &p->v)) {
        return -1;
    }
    const int mfa_shape_ok = mfa_attn_enabled() && p->attn.causal == 0 &&
        p->attn.heads == p->attn.kv_heads && p->attn.head_dim == 128 &&
        p->attn.tokens >= mfa_min_tokens();
    id<MTLBuffer> qhm = nil;
    id<MTLBuffer> khm = nil;
    int qk_hm = 0;
    if (mfa_shape_ok) {
        qk_hm = encode_qk_norm_hm(enc, b, w, &p->qk, &qhm, &khm);
    }
    if (!qk_hm) {
        encode_qk_norm(
            enc,
            (__bridge id<MTLComputePipelineState>)qk_pipeline,
            b,
            w,
            &p->qk,
            t->qk);
        g_dispatch++;
    }
    if (mfa_shape_ok && encode_attention_mfa(enc, b, &p->attn, qhm, khm) == 0) {
        return 0;
    }
    if (qk_hm) return -1;
    encode_attention(
        enc,
        (__bridge id<MTLComputePipelineState>)attn_pipeline,
        b,
        &p->attn,
        (uint32_t)t->attn_kernel,
        t->attn);
    g_dispatch++;
    return 0;
}

int zdraw_metal_encode_toma_attn_finish(
    void* batch,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* resid_norm_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    if (encode_toma_gemm(
        enc,
        (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline,
        (__bridge id<MTLComputePipelineState>)gemm_half_pipeline,
        (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline,
        b->mix,
        w->proj,
        b->attn,
        &p->proj
    )) return -1;
    encode_resid_next_norm(
        enc,
        (__bridge id<MTLComputePipelineState>)resid_norm_pipeline,
        b,
        w,
        p,
        t->block);
    g_dispatch++;
    return 0;
}

int zdraw_metal_encode_toma_ffn_core(
    void* batch,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* swiglu_pipeline,
    void* swiglu_fused_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    id<MTLComputePipelineState> exact =
        (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline;
    id<MTLComputePipelineState> half =
        (__bridge id<MTLComputePipelineState>)gemm_half_pipeline;
    id<MTLComputePipelineState> w8 =
        (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline;
    id<MTLComputePipelineState> direct =
        ours16_pipeline(((__bridge id<MTLBuffer>)b->norm).device);
    id<MTLComputePipelineState> fused_pipe =
        (__bridge id<MTLComputePipelineState>)swiglu_fused_pipeline;
    int fused = 0;
    if (!unfuse_enabled() && p->ffn_fused.mode != 0 &&
        ours16_ok(&p->ffn_fused) && direct && fused_pipe) {
        fused = encode_ours16(
            enc,
            direct,
            (__bridge id<MTLBuffer>)b->norm,
            w->ffn_fused,
            b->gateup,
            &p->ffn_fused);
        if (fused) {
            count_gemm(&p->ffn_fused);
            encode_swiglu_fused(
                enc,
                fused_pipe,
                b->gateup,
                b->gate,
                p->ffn_gate.m * p->ffn_gate.n,
                p->ffn_gate.n,
                t->swiglu);
        }
    }
    if (!fused) {
        if (encode_toma_gemm(
                enc, exact, half, w8, b->norm, w->ffn_gate, b->gate, &p->ffn_gate) ||
            encode_toma_gemm(
                enc, exact, half, w8, b->norm, w->ffn_up, b->up, &p->ffn_up)) {
            return -1;
        }
        encode_swiglu(
            enc,
            (__bridge id<MTLComputePipelineState>)swiglu_pipeline,
            b->gate,
            b->up,
            p->ffn_gate.m * p->ffn_gate.n,
            t->swiglu);
        g_dispatch++;
    }
    if (encode_toma_gemm(
        enc, exact, half, w8, b->gate, w->ffn_down, b->ffn, &p->ffn_down
    )) return -1;
    return 0;
}

int zdraw_metal_encode_toma_ffn_finish(
    void* batch,
    void* resid_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    encode_block_kernel(
        enc,
        (__bridge id<MTLComputePipelineState>)resid_pipeline,
        b->ffn,
        w->ffn_out,
        w->mlp_gate,
        b->state,
        &p->ffn_resid,
        t->block);
    g_dispatch++;
    return 0;
}

int zdraw_metal_run_block_chain(
    void* queue,
    void* norm_pipeline,
    void* resid_pipeline,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* qk_pipeline,
    void* attn_pipeline,
    void* swiglu_pipeline,
    void* swiglu_fused_pipeline,
    void* resid_norm_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLComputePipelineState> norm =
            (__bridge id<MTLComputePipelineState>)norm_pipeline;
        id<MTLComputePipelineState> resid =
            (__bridge id<MTLComputePipelineState>)resid_pipeline;
        id<MTLComputePipelineState> gemm_exact =
            (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline;
        id<MTLComputePipelineState> gemm_half =
            (__bridge id<MTLComputePipelineState>)gemm_half_pipeline;
        id<MTLComputePipelineState> gemm_w8 =
            (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline;
        id<MTLComputePipelineState> qk =
            (__bridge id<MTLComputePipelineState>)qk_pipeline;
        id<MTLComputePipelineState> attn =
            (__bridge id<MTLComputePipelineState>)attn_pipeline;
        id<MTLComputePipelineState> swiglu =
            (__bridge id<MTLComputePipelineState>)swiglu_pipeline;
        id<MTLComputePipelineState> swiglu_fused =
            (__bridge id<MTLComputePipelineState>)swiglu_fused_pipeline;
        id<MTLComputePipelineState> resid_norm =
            (__bridge id<MTLComputePipelineState>)resid_norm_pipeline;
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) return -1;

        encode_chain_layer(enc, norm, resid, gemm_exact, gemm_half, gemm_w8,
                           qk, attn, swiglu, resid_norm, b, w, p, t);
        [enc endEncoding];

        count_chain_layer(p);
        g_command++;
        [cmd commit];
        wait_cmd(cmd);
        return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
    }
}

int zdraw_metal_run_stack_chain(
    void* queue,
    void* norm_pipeline,
    void* resid_pipeline,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* qk_pipeline,
    void* attn_pipeline,
    void* swiglu_pipeline,
    void* swiglu_fused_pipeline,
    void* resid_norm_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* weights,
    const ZdrawBlockChainParams* params,
    size_t layer_count,
    const ZdrawBlockThreads* t
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> norm = (__bridge id<MTLComputePipelineState>)norm_pipeline;
    id<MTLComputePipelineState> resid = (__bridge id<MTLComputePipelineState>)resid_pipeline;
    id<MTLComputePipelineState> gemm_exact = (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline;
    id<MTLComputePipelineState> gemm_half = (__bridge id<MTLComputePipelineState>)gemm_half_pipeline;
    id<MTLComputePipelineState> gemm_w8 = (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline;
    id<MTLComputePipelineState> qk = (__bridge id<MTLComputePipelineState>)qk_pipeline;
    id<MTLComputePipelineState> attn = (__bridge id<MTLComputePipelineState>)attn_pipeline;
    id<MTLComputePipelineState> swiglu = (__bridge id<MTLComputePipelineState>)swiglu_pipeline;
    id<MTLComputePipelineState> swiglu_fused =
        (__bridge id<MTLComputePipelineState>)swiglu_fused_pipeline;
    id<MTLComputePipelineState> resid_norm = (__bridge id<MTLComputePipelineState>)resid_norm_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    const int use_mps_gateup = dense_mps_gateup_enabled();
    const int use_mps_qkv = dense_mps_qkv_enabled();
    const int use_mps_proj = dense_mps_proj_enabled();
    const int use_mps_down = dense_mps_down_enabled();
    const int use_mps_any = dense_mps_any_enabled();
    for (size_t i = 0; i < layer_count; i++) {
        if (use_mps_any) {
            int used_qkv = 0, used_proj = 0, used_gateup = 0, used_down = 0;
            int rc = encode_chain_layer_mps_dense(
                cmd, &enc, norm, resid, gemm_exact, gemm_half, gemm_w8,
                qk, attn, swiglu, swiglu_fused, resid_norm, b, &weights[i], &params[i], t,
                use_mps_qkv, use_mps_proj, use_mps_gateup, use_mps_down,
                &used_qkv, &used_proj, &used_gateup, &used_down);
            if (rc != 0) return rc;
            count_chain_layer_mps_dense(&params[i], used_qkv, used_proj, used_gateup, used_down);
        } else {
            encode_chain_layer(enc, norm, resid, gemm_exact, gemm_half, gemm_w8,
                               qk, attn, swiglu, resid_norm, b, &weights[i], &params[i], t);
            count_chain_layer(&params[i]);
        }
    }
    [enc endEncoding];

    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    if (gpu_trace_enabled()) zdraw_metal_trace_resolve((__bridge void*)q.device);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static void encode_final_norm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    void* state,
    void* scale,
    void* output,
    const FinalNormParams* params,
    size_t threads
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)state offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(FinalNormParams) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(params->tokens, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static int stack_profile_enabled(void) {
    const char* raw = getenv("ZDRAW_STACK_PROFILE");
    return raw && raw[0] != '\0' && strcmp(raw, "0") != 0;
}

static void stack_profile_add(uint64_t kind, uint64_t start) {
    if (kind >= SP_COUNT) return;
    g_stack_profile_ns[kind] += zdraw_now_ns() - start;
    g_stack_profile_samples[kind]++;
}

static int begin_profile_cmd(
    id<MTLCommandQueue> q,
    id<MTLCommandBuffer>* cmd,
    id<MTLComputeCommandEncoder>* enc
) {
    *cmd = [q commandBuffer];
    *enc = [*cmd computeCommandEncoder];
    return (*cmd && *enc) ? 0 : -1;
}

static int finish_profile_cmd(
    id<MTLCommandBuffer> cmd,
    id<MTLComputeCommandEncoder> enc,
    uint64_t kind,
    uint64_t start
) {
    [enc endEncoding];
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    stack_profile_add(kind, start);
    g_stack_profile_gpu[kind] += cmd_gpu_seconds(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

static int profile_block_cmd(
    id<MTLCommandQueue> q,
    uint64_t kind,
    id<MTLComputePipelineState> pipe,
    void* input,
    void* weight,
    void* scale,
    void* output,
    const ZdrawBlockParams* params,
    size_t threads
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_block_kernel(enc, pipe, input, weight, scale, output, params, threads);
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, kind, start);
}

static int profile_resid_norm_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    size_t threads
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_resid_next_norm(enc, pipe, b, w, p, threads);
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, SP_ATTN_RESID_NORM, start);
}

static int profile_qk_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    size_t threads
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_qk_norm(enc, pipe, b, w, &p->qk, threads);
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, SP_QK_ROPE, start);
}

static int profile_attention_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockChainParams* p,
    uint32_t kernel,
    size_t threads
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_attention(enc, pipe, b, &p->attn, kernel, threads);
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, SP_ATTENTION, start);
}

static int profile_gemm1_cmd(
    id<MTLCommandQueue> q,
    uint64_t kind,
    id<MTLComputePipelineState> exact,
    id<MTLComputePipelineState> half,
    id<MTLComputePipelineState> w8,
    void* a,
    void* weight,
    void* out,
    const GemmParams* params
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_gemm_auto(enc, exact, half, w8, a, weight, out, params);
    count_gemm(params);
    return finish_profile_cmd(cmd, enc, kind, start);
}

static int profile_gemm2_cmd(
    id<MTLCommandQueue> q,
    uint64_t kind,
    id<MTLComputePipelineState> exact,
    id<MTLComputePipelineState> half,
    id<MTLComputePipelineState> w8,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_gemm_auto(enc, exact, half, w8, b->norm, w->ffn_gate, b->gate, &p->ffn_gate);
    encode_gemm_auto(enc, exact, half, w8, b->norm, w->ffn_up, b->up, &p->ffn_up);
    count_gemm(&p->ffn_gate);
    count_gemm(&p->ffn_up);
    return finish_profile_cmd(cmd, enc, kind, start);
}

static int profile_gemm3_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> exact,
    id<MTLComputePipelineState> half,
    id<MTLComputePipelineState> w8,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_gemm_auto(enc, exact, half, w8, b->norm, w->q, b->q, &p->q);
    encode_gemm_auto(enc, exact, half, w8, b->norm, w->k, b->k, &p->k);
    encode_gemm_auto(enc, exact, half, w8, b->norm, w->v, b->v, &p->v);
    count_gemm(&p->q);
    count_gemm(&p->k);
    count_gemm(&p->v);
    return finish_profile_cmd(cmd, enc, SP_QKV, start);
}

static int profile_swiglu_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockChainParams* p,
    size_t threads
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_swiglu(enc, pipe, b->gate, b->up, p->ffn_gate.m * p->ffn_gate.n, threads);
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, SP_SWIGLU, start);
}

static int profile_bias_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pipe,
    const FinalBuffers* buffers,
    const BiasParams* params,
    const GemmParams* gemm
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:(__bridge id<MTLBuffer>)buffers->output offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)buffers->bias offset:0 atIndex:1];
    [enc setBytes:params length:sizeof(BiasParams) atIndex:2];
    size_t total = (size_t)gemm->m * (size_t)gemm->n;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, SP_FINAL_BIAS, start);
}

static int profile_final_norm_cmd(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pipe,
    const ZdrawBlockBuffers* b,
    const FinalBuffers* buffers,
    const FinalNormParams* params,
    size_t threads
) {
    uint64_t start = zdraw_now_ns();
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    if (begin_profile_cmd(q, &cmd, &enc) != 0) return -1;
    encode_final_norm(enc, pipe, b->state, buffers->scale, buffers->batch, params, threads);
    g_dispatch++;
    return finish_profile_cmd(cmd, enc, SP_FINAL_NORM, start);
}

static int run_profiled_layer(
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> norm,
    id<MTLComputePipelineState> resid,
    id<MTLComputePipelineState> gemm_exact,
    id<MTLComputePipelineState> gemm_half,
    id<MTLComputePipelineState> gemm_w8,
    id<MTLComputePipelineState> qk,
    id<MTLComputePipelineState> attn,
    id<MTLComputePipelineState> swiglu,
    id<MTLComputePipelineState> resid_norm,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* w,
    const ZdrawBlockChainParams* p,
    const ZdrawBlockThreads* t
) {
    int rc = profile_block_cmd(q, SP_ATTN_NORM, norm, b->state, w->attn_in,
                               w->attn_scale, b->norm, &p->attn_norm, t->block);
    if (rc != 0) return rc;
    rc = profile_gemm3_cmd(q, gemm_exact, gemm_half, gemm_w8, b, w, p);
    if (rc != 0) return rc;
    rc = profile_qk_cmd(q, qk, b, w, p, t->qk);
    if (rc != 0) return rc;
    rc = profile_attention_cmd(q, attn, b, p, (uint32_t)t->attn_kernel, t->attn);
    if (rc != 0) return rc;
    rc = profile_gemm1_cmd(q, SP_PROJ, gemm_exact, gemm_half, gemm_w8,
                           b->mix, w->proj, b->attn, &p->proj);
    if (rc != 0) return rc;
    rc = profile_resid_norm_cmd(q, resid_norm, b, w, p, t->block);
    if (rc != 0) return rc;
    rc = profile_gemm2_cmd(q, SP_FFN_GATEUP, gemm_exact, gemm_half, gemm_w8, b, w, p);
    if (rc != 0) return rc;
    rc = profile_swiglu_cmd(q, swiglu, b, p, t->swiglu);
    if (rc != 0) return rc;
    rc = profile_gemm1_cmd(q, SP_FFN_DOWN, gemm_exact, gemm_half, gemm_w8,
                           b->gate, w->ffn_down, b->ffn, &p->ffn_down);
    if (rc != 0) return rc;
    return profile_block_cmd(q, SP_FFN_RESID, resid, b->ffn, w->ffn_out,
                             w->mlp_gate, b->state, &p->ffn_resid, t->block);
}

int zdraw_metal_run_stack_final(
    void* queue,
    void* norm_pipeline,
    void* resid_pipeline,
    void* gemm_exact_pipeline,
    void* gemm_half_pipeline,
    void* gemm_w8_pipeline,
    void* qk_pipeline,
    void* attn_pipeline,
    void* swiglu_pipeline,
    void* swiglu_fused_pipeline,
    void* resid_norm_pipeline,
    void* final_pipeline,
    void* bias_pipeline,
    const ZdrawBlockBuffers* b,
    const ZdrawBlockWeights* weights,
    const ZdrawBlockChainParams* params,
    size_t layer_count,
    const FinalBuffers* final_buffers,
    const FinalNormParams* final_norm,
    const GemmParams* final_gemm,
    const BiasParams* final_bias,
    const ZdrawBlockThreads* t,
    size_t final_threads
) {
    // ZDRAW_MIX_PROBE=1: once, run the GEMM-only probe with the chain's OWN
    // weights+params (synthetic A/C) on this queue. Discriminates real
    // weight views vs synthetic buffers as the source of the chain gap.
    static int probe_done = 0;
    if (!probe_done && layer_count >= 16 && getenv("ZDRAW_MIX_PROBE")) {
        probe_done = 1;
        id<MTLCommandQueue> q0 = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q0.device;
        id<MTLComputePipelineState> direct = ours16_pipeline(dev);
        const uint32_t m0 = params[0].q.m;
        id<MTLBuffer> a32 = [dev newBufferWithLength:(size_t)m0 * 10240 * 4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> c1 = [dev newBufferWithLength:(size_t)m0 * 20480 * 4
                                            options:MTLResourceStorageModeShared];
        if (direct && a32 && c1) {
            memset(a32.contents, 0x3C, a32.length);
            uint64_t best = ~0ull;
            int n_ours = 0, n_skip = 0, n_fused = 0, n_down = 0;
            for (int r = 0; r < 6; r++) {
                n_ours = 0;
                n_skip = 0;
                n_fused = 0;
                n_down = 0;
                id<MTLCommandBuffer> pc = [q0 commandBuffer];
                id<MTLComputeCommandEncoder> pe = [pc computeCommandEncoder];
                if (!pc || !pe) break;
                // ZDRAW_MIX_PROBE=2: bind the chain's OWN A/C buffers
                // (identity + content); =1 keeps synthetic a32/c1.
                const int real_bufs = getenv("ZDRAW_MIX_PROBE")[0] == '2';
                id<MTLBuffer> an = real_bufs ? (__bridge id<MTLBuffer>)b->norm : a32;
                id<MTLBuffer> am = real_bufs ? (__bridge id<MTLBuffer>)b->mix : a32;
                id<MTLBuffer> ag = real_bufs ? (__bridge id<MTLBuffer>)b->gate : a32;
                void* cv = (__bridge void*)c1;
                void* cq = real_bufs ? b->q : cv;
                void* ck = real_bufs ? b->k : cv;
                void* cx = real_bufs ? b->v : cv;
                void* ca = real_bufs ? b->attn : cv;
                void* cg = real_bufs ? b->gateup : cv;
                void* cf = real_bufs ? b->ffn : cv;
                for (size_t l = 0; l < layer_count; l++) {
                    const ZdrawBlockWeights* w0 = &weights[l];
                    const ZdrawBlockChainParams* p0 = &params[l];
                    if (!ours16_ok(&p0->q)) { n_skip++; continue; }
                    n_ours++;
                    encode_ours16(pe, direct, an, w0->q, cq, &p0->q);
                    count_gemm(&p0->q);
                    encode_ours16(pe, direct, an, w0->k, ck, &p0->k);
                    count_gemm(&p0->k);
                    encode_ours16(pe, direct, an, w0->v, cx, &p0->v);
                    count_gemm(&p0->v);
                    encode_ours16(pe, direct, am, w0->proj, ca, &p0->proj);
                    count_gemm(&p0->proj);
                    if (p0->ffn_fused.mode != 0 && ours16_ok(&p0->ffn_fused)) {
                        encode_ours16(pe, direct, an, w0->ffn_fused, cg, &p0->ffn_fused);
                        count_gemm(&p0->ffn_fused);
                        n_fused++;
                    }
                    if (ours16_ok(&p0->ffn_down)) {
                        encode_ours16(pe, direct, ag, w0->ffn_down, cf, &p0->ffn_down);
                        count_gemm(&p0->ffn_down);
                        n_down++;
                    }
                }
                [pe endEncoding];
                [pc commit];
                [pc waitUntilCompleted];
                const double span = pc.GPUEndTime - pc.GPUStartTime;
                const uint64_t ns = span > 0 ? (uint64_t)(span * 1e9) : 0;
                if (r > 0 && ns && ns < best) best = ns;
            }
            fprintf(stderr,
                    "real-weights probe: %.2f ms/layer, %.0f ms/step "
                    "(q %d, skipped %d, fused %d, down %d)\n",
                    (double)best / (double)layer_count / 1e6, (double)best / 1e6,
                    n_ours, n_skip, n_fused, n_down);
        }
    }

    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> norm = (__bridge id<MTLComputePipelineState>)norm_pipeline;
    id<MTLComputePipelineState> resid = (__bridge id<MTLComputePipelineState>)resid_pipeline;
    id<MTLComputePipelineState> gemm_exact = (__bridge id<MTLComputePipelineState>)gemm_exact_pipeline;
    id<MTLComputePipelineState> gemm_half = (__bridge id<MTLComputePipelineState>)gemm_half_pipeline;
    id<MTLComputePipelineState> gemm_w8 = (__bridge id<MTLComputePipelineState>)gemm_w8_pipeline;
    id<MTLComputePipelineState> qk = (__bridge id<MTLComputePipelineState>)qk_pipeline;
    id<MTLComputePipelineState> attn = (__bridge id<MTLComputePipelineState>)attn_pipeline;
    id<MTLComputePipelineState> swiglu = (__bridge id<MTLComputePipelineState>)swiglu_pipeline;
    id<MTLComputePipelineState> swiglu_fused =
        (__bridge id<MTLComputePipelineState>)swiglu_fused_pipeline;
    id<MTLComputePipelineState> resid_norm = (__bridge id<MTLComputePipelineState>)resid_norm_pipeline;
    id<MTLComputePipelineState> final_pipe = (__bridge id<MTLComputePipelineState>)final_pipeline;
    id<MTLComputePipelineState> bias = (__bridge id<MTLComputePipelineState>)bias_pipeline;
    if (stack_profile_enabled()) {
        for (size_t i = 0; i < layer_count; i++) {
            int rc = run_profiled_layer(q, norm, resid, gemm_exact, gemm_half, gemm_w8,
                                        qk, attn, swiglu, resid_norm,
                                        b, &weights[i], &params[i], t);
            if (rc != 0) return rc;
        }
        int rc = profile_final_norm_cmd(q, final_pipe, b, final_buffers, final_norm, final_threads);
        if (rc != 0) return rc;
        rc = profile_gemm1_cmd(q, SP_FINAL_PROJ, gemm_exact, gemm_exact, gemm_exact,
                               final_buffers->batch, final_buffers->weight,
                               final_buffers->output, final_gemm);
        if (rc != 0) return rc;
        return profile_bias_cmd(q, bias, final_buffers, final_bias, final_gemm);
    }
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    const int use_mps_gateup = dense_mps_gateup_enabled();
    const int use_mps_qkv = dense_mps_qkv_enabled();
    const int use_mps_proj = dense_mps_proj_enabled();
    const int use_mps_down = dense_mps_down_enabled();
    const int use_mps_any = dense_mps_any_enabled();
    g_act16_step = 0;
    if (act16_enabled() && layer_count > 0 && dense_ours16_enabled()) {
        id<MTLDevice> adev = ((__bridge id<MTLBuffer>)b->norm).device;
        Act16Pipes* ap = act16_pipes(adev);
        int ok = ap != NULL;
        for (size_t i = 0; ok && i < layer_count; i++) ok = act16_layer_ok(&params[i]);
        if (ok) {
            g_act16_step = 1;
            act16_report();
            uint32_t n = params[0].attn_norm.tokens * params[0].attn_norm.hidden;
            [enc setComputePipelineState:ap->state_in];
            [enc setBuffer:(__bridge id<MTLBuffer>)b->state offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)b->up offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            g_dispatch++;
        }
    }
    for (size_t i = 0; i < layer_count; i++) {
        if (g_act16_step) {
            int rc = encode_chain_layer_act16(cmd, &enc, qk, attn, b,
                                              &weights[i], &params[i], t);
            if (rc != 0) return rc;
            count_chain_layer(&params[i]);
        } else if (use_mps_any) {
            int used_qkv = 0, used_proj = 0, used_gateup = 0, used_down = 0;
            int rc = encode_chain_layer_mps_dense(
                cmd, &enc, norm, resid, gemm_exact, gemm_half, gemm_w8,
                qk, attn, swiglu, swiglu_fused, resid_norm, b, &weights[i], &params[i], t,
                use_mps_qkv, use_mps_proj, use_mps_gateup, use_mps_down,
                &used_qkv, &used_proj, &used_gateup, &used_down);
            if (rc != 0) return rc;
            count_chain_layer_mps_dense(&params[i], used_qkv, used_proj, used_gateup, used_down);
        } else {
            encode_chain_layer(enc, norm, resid, gemm_exact, gemm_half, gemm_w8,
                               qk, attn, swiglu, resid_norm, b, &weights[i], &params[i], t);
            count_chain_layer(&params[i]);
        }
    }
    if (g_act16_step) {
        Act16Pipes* ap = act16_pipes(((__bridge id<MTLBuffer>)b->norm).device);
        [enc setComputePipelineState:ap->final_h];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->up offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)final_buffers->scale offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)final_buffers->batch offset:0 atIndex:2];
        [enc setBytes:final_norm length:sizeof(FinalNormParams) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(final_norm->tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(final_threads, 1, 1)];
    } else {
        encode_final_norm(enc, final_pipe, b->state, final_buffers->scale,
                          final_buffers->batch, final_norm, final_threads);
    }
    encode_gemm(enc, gemm_exact, final_buffers->batch, final_buffers->weight,
                final_buffers->output, final_gemm);
    [enc setComputePipelineState:bias];
    [enc setBuffer:(__bridge id<MTLBuffer>)final_buffers->output offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)final_buffers->bias offset:0 atIndex:1];
    [enc setBytes:final_bias length:sizeof(BiasParams) atIndex:2];
    size_t total = (size_t)final_gemm->m * (size_t)final_gemm->n;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];

    g_dispatch += 2;
    count_gemm(final_gemm);
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    if (gpu_trace_enabled()) zdraw_metal_trace_resolve((__bridge void*)q.device);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_encode_final(
    void* batch,
    void* final_pipeline,
    void* gemm_pipeline,
    void* bias_pipeline,
    void* state,
    const FinalBuffers* buffers,
    const FinalNormParams* norm_params,
    const GemmParams* gemm_params,
    const BiasParams* bias_params,
    size_t norm_threads
) {
    ZdrawBatch* bb = (ZdrawBatch*)batch;
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)bb->enc;
    if (!enc) return -1;
    encode_final_norm(
        enc,
        (__bridge id<MTLComputePipelineState>)final_pipeline,
        state,
        buffers->scale,
        buffers->batch,
        norm_params,
        norm_threads);
    encode_gemm(
        enc,
        (__bridge id<MTLComputePipelineState>)gemm_pipeline,
        buffers->batch,
        buffers->weight,
        buffers->output,
        gemm_params);
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)bias_pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)buffers->output offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)buffers->bias offset:0 atIndex:1];
    [enc setBytes:bias_params length:sizeof(BiasParams) atIndex:2];
    size_t total = (size_t)gemm_params->m * (size_t)gemm_params->n;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_dispatch += 2;
    count_gemm(gemm_params);
    return 0;
}

int zdraw_metal_run_final_proj(
    void* queue,
    void* norm_pipeline,
    void* gemm_pipeline,
    void* bias_pipeline,
    void* state,
    void* scale,
    void* weight,
    void* bias_buf,
    void* batch,
    void* output,
    const FinalNormParams* norm_params,
    const GemmParams* gemm_params,
    const BiasParams* bias_params,
    size_t norm_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> norm = (__bridge id<MTLComputePipelineState>)norm_pipeline;
    id<MTLComputePipelineState> gemm = (__bridge id<MTLComputePipelineState>)gemm_pipeline;
    id<MTLComputePipelineState> bias = (__bridge id<MTLComputePipelineState>)bias_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_final_norm(enc, norm, state, scale, batch, norm_params, norm_threads);
    encode_gemm(enc, gemm, batch, weight, output, gemm_params);
    [enc setComputePipelineState:bias];
    [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias_buf offset:0 atIndex:1];
    [enc setBytes:bias_params length:sizeof(BiasParams) atIndex:2];
    size_t total = (size_t)gemm_params->m * (size_t)gemm_params->n;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];

    g_dispatch += 2;
    count_gemm(gemm_params);
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_gemm_triple(
    void* queue,
    void* pipeline,
    void* a_buf,
    void* w0_buf,
    void* c0_buf,
    const GemmParams* params0,
    void* w1_buf,
    void* c1_buf,
    const GemmParams* params1,
    void* w2_buf,
    void* c2_buf,
    const GemmParams* params2
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_gemm(enc, pipe, a_buf, w0_buf, c0_buf, params0);
    encode_gemm(enc, pipe, a_buf, w1_buf, c1_buf, params1);
    encode_gemm(enc, pipe, a_buf, w2_buf, c2_buf, params2);
    [enc endEncoding];

    count_gemm(params0);
    count_gemm(params1);
    count_gemm(params2);
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

int zdraw_metal_run_ffn(
    void* queue,
    void* gemm_pipeline,
    void* swiglu_pipeline,
    void* input_buf,
    void* gate_weight,
    void* up_weight,
    void* down_weight,
    void* gate_buf,
    void* up_buf,
    void* out_buf,
    const GemmParams* gate_params,
    const GemmParams* up_params,
    const GemmParams* down_params,
    uint32_t swiglu_count,
    size_t swiglu_threads
) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLComputePipelineState> gemm = (__bridge id<MTLComputePipelineState>)gemm_pipeline;
    id<MTLComputePipelineState> swiglu = (__bridge id<MTLComputePipelineState>)swiglu_pipeline;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return -1;

    encode_gemm(enc, gemm, input_buf, gate_weight, gate_buf, gate_params);
    encode_gemm(enc, gemm, input_buf, up_weight, up_buf, up_params);

    [enc setComputePipelineState:swiglu];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_buf offset:0 atIndex:1];
    [enc setBytes:&swiglu_count length:sizeof(uint32_t) atIndex:2];
    MTLSize grid = MTLSizeMake(swiglu_count, 1, 1);
    MTLSize threads = MTLSizeMake(swiglu_threads, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];

    encode_gemm(enc, gemm, gate_buf, down_weight, out_buf, down_params);
    [enc endEncoding];

    count_gemm(gate_params);
    count_gemm(up_params);
    count_gemm(down_params);
    g_dispatch++;
    g_command++;
    [cmd commit];
    wait_cmd(cmd);
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}
