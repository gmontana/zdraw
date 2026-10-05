//! C declarations for the opt-in resident stack profiler.

pub extern fn zdraw_metal_stack_profile_count() u64;
pub extern fn zdraw_metal_stack_profile_name(index: u64) [*:0]const u8;
pub extern fn zdraw_metal_stack_profile_ns(index: u64) u64;
pub extern fn zdraw_metal_stack_profile_samples(index: u64) u64;
pub extern fn zdraw_metal_stack_profile_gpu_ns(index: u64) u64;
