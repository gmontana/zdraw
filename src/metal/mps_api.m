// Bench-only MPS oracle for GEMM comparisons.
//
// This file is intentionally not linked into the zdraw runtime. It lets
// gemmbench compare the custom Metal kernel against Apple's MPS implementation.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <stdint.h>
#import <stdlib.h>

typedef struct {
    void* mul;
    void* ma;
    void* mw;
    void* mc;
} MpsGemm;

void* zdraw_mps_gemm_make(
    void* device,
    void* a_buf,
    void* w_buf,
    void* c_buf,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k
        rowBytes:(NSUInteger)k * 2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* dw = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k
        rowBytes:(NSUInteger)k * 2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
        rowBytes:(NSUInteger)n * 2 dataType:MPSDataTypeFloat16];
    MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a_buf descriptor:da];
    MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)w_buf descriptor:dw];
    MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c_buf descriptor:dc];
    MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
        initWithDevice:dev transposeLeft:NO transposeRight:YES
        resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
    MpsGemm* g = (MpsGemm*)malloc(sizeof(MpsGemm));
    if (!g) return nil;
    g->mul = (__bridge_retained void*)mul;
    g->ma = (__bridge_retained void*)ma;
    g->mw = (__bridge_retained void*)mw;
    g->mc = (__bridge_retained void*)mc;
    return g;
}

int zdraw_mps_gemm_run(void* queue, void* ctx) {
    MpsGemm* g = (MpsGemm*)ctx;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    MPSMatrixMultiplication* mul = (__bridge MPSMatrixMultiplication*)g->mul;
    MPSMatrix* ma = (__bridge MPSMatrix*)g->ma;
    MPSMatrix* mw = (__bridge MPSMatrix*)g->mw;
    MPSMatrix* mc = (__bridge MPSMatrix*)g->mc;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    [mul encodeToCommandBuffer:cmd leftMatrix:ma rightMatrix:mw resultMatrix:mc];
    [cmd commit];
    [cmd waitUntilCompleted];
    return (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : -2;
}

void zdraw_mps_gemm_free(void* ctx) {
    MpsGemm* g = (MpsGemm*)ctx;
    (void)(__bridge_transfer MPSMatrixMultiplication*)g->mul;
    (void)(__bridge_transfer MPSMatrix*)g->ma;
    (void)(__bridge_transfer MPSMatrix*)g->mw;
    (void)(__bridge_transfer MPSMatrix*)g->mc;
    free(g);
}
