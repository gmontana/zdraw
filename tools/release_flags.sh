#!/bin/bash
# Zig flags for every shipped binary (CLI, engine archive, xcframework):
# minimum macOS 14, the M1 instruction set, the SDK given explicitly because
# an explicit -Dtarget switches zig's native SDK detection off.  Sourced by
# tools/release_assets.sh and tools/build_xcframework.sh.
ZIG_RELEASE_FLAGS=(--sysroot "$(xcrun --show-sdk-path)" -Dtarget=aarch64-macos.14.0 -Dcpu=apple_m1)
