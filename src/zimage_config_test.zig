const std = @import("std");

const config = @import("zimage_config.zig");

test "load z-image config values" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try rootPath(std.testing.allocator, tmp);
    defer std.testing.allocator.free(root);
    try makeDirs(root);

    try writeAt(root, "text_encoder/config.json",
        \\{"hidden_size":2560,"num_hidden_layers":36,"num_attention_heads":32,
        \\"num_key_value_heads":8,"head_dim":128,"intermediate_size":9728,
        \\"rms_norm_eps":0.000001,"vocab_size":151936,"rope_theta":1000000}
    );
    try writeAt(root, "transformer/config.json",
        \\{"dim":3840,"n_layers":30,"n_refiner_layers":2,"n_heads":30,
        \\"n_kv_heads":30,"cap_feat_dim":2560,"in_channels":16,
        \\"axes_dims":[32,48,48],"axes_lens":[1536,512,512],
        \\"norm_eps":0.00001,"rope_theta":256.0,"t_scale":1000,"qk_norm":true}
    );
    try writeAt(root, "vae/config.json",
        \\{"latent_channels":16,"scaling_factor":0.3611,"shift_factor":0.1159}
    );
    try writeAt(root, "scheduler/scheduler_config.json",
        \\{"num_train_timesteps":1000,"shift":3.0}
    );

    const loaded = try config.load(std.testing.io, std.testing.allocator, root);
    const text_hidden: u32 = 2560;
    const text_mid: u32 = 9728;
    const model_dim: u32 = 3840;
    const axis0: u32 = 32;
    const latent_channels: u32 = 16;
    const shift: f64 = 3.0;

    try std.testing.expectEqual(text_hidden, loaded.text.hidden_size);
    try std.testing.expectEqual(text_mid, loaded.text.intermediate_size);
    try std.testing.expectEqual(model_dim, loaded.transformer.dim);
    try std.testing.expectEqual(axis0, loaded.transformer.axes_dims[0]);
    try std.testing.expect(loaded.transformer.qk_norm);
    try std.testing.expectEqual(latent_channels, loaded.vae.latent_channels);
    try std.testing.expectApproxEqAbs(shift, loaded.scheduler.shift, 0.001);
}

fn rootPath(allocator: std.mem.Allocator, tmp: std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/model", .{tmp.sub_path});
}

fn makeDirs(root: []const u8) !void {
    for ([_][]const u8{ "text_encoder", "transformer", "vae", "scheduler" }) |dir| {
        const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/{s}", .{ root, dir });
        defer std.testing.allocator.free(path);
        try std.Io.Dir.cwd().createDirPath(std.testing.io, path);
    }
}

fn writeAt(root: []const u8, name: []const u8, text: []const u8) !void {
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/{s}", .{ root, name });
    defer std.testing.allocator.free(path);

    const file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
    defer file.close(std.testing.io);
    var buffer: [512]u8 = undefined;
    var writer = file.writerStreaming(std.testing.io, &buffer);
    try writer.interface.writeAll(text);
    try writer.interface.flush();
}
