//! Compile-time ABI guards for structs shared with the Obj-C Metal bridge.
//!
//! src/metal_api.m mirrors these layouts with hand-written typedefs (and
//! _Static_asserts on the same numbers). Drift between the two sides means
//! silent GPU corruption, so any layout change must update both files and
//! these expected sizes in the same commit.

const c = @import("../metal/metal_c.zig");
const mblock = @import("../metal/mblock_c.zig");
const chain = @import("../metal/mblock_chain_c.zig");
const final_c = @import("../metal/mstack_final_c.zig");
const mfinal = @import("../metal/mfinal.zig");
const vbufs = @import("../metal/mvres_buf.zig");
const vparam = @import("../metal/mvres_param.zig");

fn size(comptime T: type, comptime bytes: usize) void {
    if (@sizeOf(T) != bytes) @compileError("ABI size drift: " ++ @typeName(T));
}

fn at(comptime T: type, comptime field: []const u8, comptime offset: usize) void {
    if (@offsetOf(T, field) != offset)
        @compileError("ABI offset drift: " ++ @typeName(T) ++ "." ++ field);
}

comptime {
    size(c.GemmParams, 32);
    at(c.GemmParams, "mode", 16);
    at(c.GemmParams, "weight_offset", 24);
    size(c.AttnParams, 24);
    size(c.QkNormParams, 80);
    at(c.QkNormParams, "eps", 36);
    at(c.QkNormParams, "q_offset", 40);
    at(c.QkNormParams, "base2", 72);
    size(c.LinearParams, 48);
    at(c.LinearParams, "weight_offset", 32);
    size(c.ConvParams, 56);
    at(c.ConvParams, "weight_offset", 40);
    size(mblock.Params, 32);
    at(mblock.Params, "eps", 16);
    at(mblock.Params, "weight_offset", 24);
}

comptime {
    size(chain.Buffers, 88);
    size(chain.Weights, 160);
    size(chain.Threads, 40);
    size(chain.Params, 488);
    at(chain.Params, "q", 32);
    at(chain.Params, "qk", 128);
    at(chain.Params, "attn", 208);
    at(chain.Params, "proj", 232);
    at(chain.Params, "attn_resid", 264);
    at(chain.Params, "ffn_gate", 328);
    at(chain.Params, "ffn_resid", 424);
    at(chain.Params, "ffn_fused", 456);
    size(final_c.FinalBufs, 40);
    size(mfinal.NormParams, 16);
    size(mfinal.BiasParams, 16);
    at(mfinal.BiasParams, "bias_offset", 8);
}

comptime {
    size(vparam.NormParams, 48);
    at(vparam.NormParams, "weight_offset", 32);
    size(vparam.ResParams, 272);
    at(vparam.ResParams, "conv1", 48);
    at(vparam.ResParams, "skip", 208);
    at(vparam.ResParams, "has_skip", 264);
    size(vbufs.ResBuffers, 120);
}

test "abi layouts hold" {}
