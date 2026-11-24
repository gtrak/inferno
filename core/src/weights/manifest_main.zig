//! `zig build manifest` entry point: types the checkpoint, writes the report.
//!
//! Reads `<model-directory>/config.json` and every `*.safetensors` file next to
//! it, then writes `manifest.json` to the path the build step asks for. The
//! exit status carries the verdict: a tensor that no role claims, a shape that
//! contradicts the config, or byte totals that do not add up fail the build.

const std = @import("std");

const manifest = @import("manifest.zig");

const assert = std.debug.assert;

pub fn main() !void {
    var general_purpose = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = general_purpose.deinit();
    var arena = std.heap.ArenaAllocator.init(general_purpose.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    const arguments = try std.process.argsAlloc(allocator);
    assert(arguments.len >= 1);
    if (arguments.len != 3) {
        std.debug.print(
            "usage: {s} <model-dir> <manifest.json>\n",
            .{arguments[0]},
        );
        return error.bad_arguments;
    } else {
        assert(arguments.len == 3);
    }
    const model_directory = arguments[1];
    const output_path = arguments[2];

    var loaded = try manifest.build(allocator, model_directory);
    defer manifest.unload(&loaded);
    assert(loaded.files.len > 0);

    const document = try manifest.renderJson(allocator, &loaded);
    try std.fs.cwd().writeFile(.{ .sub_path = output_path, .data = document });

    printSummary(&loaded, @intCast(document.len));
    assert(document.len > 0);
}

/// One block per role plus the file totals, so the build log is the evidence.
fn printSummary(loaded: *const manifest.Loaded, document_bytes: u64) void {
    assert(document_bytes > 0);
    assert(loaded.files.len > 0);

    const compiled = &loaded.manifest;
    var tensor_count: u64 = 0;
    for (compiled.files) |report| {
        tensor_count += report.tensors;
    }
    std.debug.print(
        "manifest: {d} files, {d} tensors, {d} bytes of weights\n",
        .{
            loaded.files.len,
            tensor_count,
            loaded.tensorBytesTotal(),
        },
    );
    assert(tensor_count > 0);
    for (manifest.all_roles) |role| {
        const total = compiled.total(role);
        if (total.count == 0) {
            continue;
        } else {
            assert(total.count > 0);
        }
        std.debug.print("  {s: <24} {d: >8} tensors {d: >14} bytes\n", .{
            @tagName(role),
            total.count,
            total.bytes,
        });
    }
    std.debug.print("manifest: wrote {d} bytes\n", .{document_bytes});
    assert(compiled.deferred_bytes > 0);
}
