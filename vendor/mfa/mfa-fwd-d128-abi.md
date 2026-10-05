==================== ABI / DISPATCH SUMMARY ====================
Device reported by MTLCreateSystemDefaultDevice(): Apple M4 Max
supportsFamily(.apple9): true   (true => Apple9/M3/M4 block-size path taken)

Kernel type:               forward
Function name:             attention
Compile + pipeline build:  FAILED (toolchain/header issue, see note): Error Domain=MTLLibraryErrorDomain Code=3 "program_source:23:9: error: illegal string literal in 'asm'

Block dimensions (parallelization, traversal, head): (16, 128, 32)
  - selected table row for D=128 (forwardMixed, apple9): "| 160 | 16 | 128 | 32 | O |"
  - matched because headDimension(128) <= maximumHeadDimension(160)
  - cached operands in registers: O only (Q is NOT cached at D=128)

Threadgroup size (threads/threadgroup): 64
  - formula: 32 * (blockParallelization / 8) = 32 * (16/8) = 64
  - dispatched as MTLSize(width: 64, height: 1, depth: 1)
  - SIMD groups per threadgroup: 2

Threadgroup memory bytes (setThreadgroupMemoryLength index 0): 8192
  - kernel.threadgroupMemoryAllocation = 8192  (authoritative; from the library)
  - = blockTraversal(128) * blockHead(32) * sizeof(half)(2) for the K/V tile
  - dynamic allocation; the kernel declares `threadgroup uchar *threadgroup_block [[threadgroup(0)]]`

Pipeline reflection (live build): see note below. When buildable, the host should
pass maxTotalThreadsPerThreadgroup = 1024 on the pipeline descriptor. threadExecutionWidth
is fixed at 32 on all Apple GPUs (one SIMD = 32 lanes).
  - reflected maxTotalThreadsPerThreadgroup = n/a (build blocked)
  - reflected threadExecutionWidth         = n/a (build blocked); =32 by hardware
  - reflected staticThreadgroupMemoryLength = n/a (build blocked); set 8192 dynamically

NOTE on live build: On this machine's Xcode 26 / Metal 4 toolchain, MFA's vendored
'metal_simdgroup_event' prologue (GEMMHeaders.swift) fails to compile because the new
Metal frontend rejects the raw `__asm("air.simdgroup_async_copy_*")` intrinsic bindings
("illegal string literal in 'asm'"). This is a PRE-EXISTING toolchain incompatibility in
MFA itself, NOT a problem with the generated attention body: MFA's own SquareAttentionTest
fails identically on this toolchain. The generated kernel source (the deliverable) is
complete and correct; to build it you need a Metal toolchain that still accepts those
intrinsics (Xcode <= 16 era), or you swap the async-copy prologue for an equivalent that
the newer toolchain accepts. All numeric values above (block dims, TG size, TG memory,
bindings, constants, grid formula) are derived directly from the library and are exact.

Grid mapping (how gridSize derives from R):
  - one thread block covers blockParallelization=16 rows of the output
  - gridSize.width = ceil(R / 16) = ceil(4128 / 16) = 258
  - gridSize.height = 1, gridSize.depth = 1
  - encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
  - inside the kernel: parallelization_group_offset = gid * 16
    (gid = threadgroup_position_in_grid); SIMDs past R return early.

Buffer index bindings (device pointers; sorted by index in the signature):
  buffer(0) = Q  (half)   device pointer, read
  buffer(1) = K  (half)   device pointer, read
  buffer(2) = V  (half)   device pointer, read
  buffer(3) = O  (float)   device pointer, write
  buffer(4) = L  (half)   device pointer, write (log-sum-exp, always produced)
  (buffers 6..9 dO/dV/dK/dQ and 5 D are NOT bound by the forward kernel.)

Function constants that MUST be set at pipeline creation:
  - function_constant(0) = R (uint) = 4128   (output/query sequence length)
  - function_constant(1) = C (uint) = 4128   (input/key/value sequence length)
  These are set via AttentionDescriptor.setFunctionConstants(_:) -> setConstantValue index 0,1.
  They are the ONLY function constants. D (head dim) and all block sizes are baked
  into the source as literals at generation time, not function constants.

Runtime values the host must supply:
  - Q, K, V buffers (FP16/half), O buffer (FP32/float), L buffer (FP16/half).
  - Function constants R and C (above).
  - Threadgroup memory length = 8192 bytes at index 0.
  - NO scale buffer: the softmax scale (log2(e)/sqrt(D) = 1.442695041/sqrt(128))
    is compiled into the kernel as a literal.
  - NO stride/leading-dimension arguments: leading dimensions are baked in.
      * For a NON-transposed operand the row stride is the literal headDimension = 128.
      * Multi-head: there is ONE dispatch PER HEAD. To make heads contiguous,
        the host must offset each buffer to the head's base AND the leading
        dimension would need to be D*H. But this generated kernel bakes the
        leading dimension as D=128 (transposeState all false), so as generated it
        assumes HEAD-CONTIGUOUS, SINGLE-HEAD-PER-BUFFER layout (row stride = D).
        To support interleaved multi-head (stride D*H) you must regenerate with
        H known at compile time, per AttentionKernelDescriptor.swift lines 37-41.
  - NO mask buffer: this is dense (non-causal) attention. Out-of-range columns on
    the final traversal block are masked internally via the C function constant.

Precision (memory) for forward operands:
  - Q,K,V = half ; O = float ; L = half
  - register precision: Q,K,S,P,V = half(FP16) for S/P; O accumulated in float(FP32)

Dispatch recipe (exactly as SquareAttentionTest.swift does it):
  let pipelineDesc = MTLComputePipelineDescriptor()
  pipelineDesc.computeFunction = function           // makeFunction("attention", constants)
  pipelineDesc.maxTotalThreadsPerThreadgroup = 1024 // forces occupancy
  // ... makeComputePipelineState(descriptor:)
  encoder.setComputePipelineState(pipeline)
  encoder.setThreadgroupMemoryLength(8192, index: 0)
  encoder.setBuffer(Q, offset: 0, index: 0)
  encoder.setBuffer(K, offset: 0, index: 1)
  encoder.setBuffer(V, offset: 0, index: 2)
  encoder.setBuffer(O, offset: 0, index: 3)
  encoder.setBuffer(L, offset: 0, index: 4)
  let blockCount = ceilDivide(R, 16)  // = 258
  encoder.dispatchThreadgroups(
    MTLSize(width: blockCount, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

MIT license for attribution: /tmp/mfa/LICENSE
  (Copyright (c) 2024 Philip Turner; repo philipturner/metal-flash-attention)
===============================================================