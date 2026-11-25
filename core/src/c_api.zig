//! The C ABI the Rust server links against.
//!
//! Everything exported here is a plain C symbol: no Zig-only types cross the
//! boundary, buffers carry their capacity, and the JSON document is truncated
//! rather than overflowed.

const std = @import("std");

const cuda_driver = @import("hw/api.zig").cuda_driver;
const generated_cubins = @import("generated_cubins");
const handles = @import("model/handles.zig");

const Cubin = generated_cubins.Cubin;

const assert = std.debug.assert;

/// Reported by `inferno_version`; kept in step with the Cargo workspace.
const version_text = "inferno 0.1.0";

/// Rendered documents are smaller than this; the probe buffer is much larger.
const document_bytes_max: u32 = 16 * 1024;

/// Below this a document cannot describe even one device.
const document_bytes_min: u32 = 512;

/// Every cubin embedded in the static library, with its length.
///
/// Zig only emits data something references, so the C accessors below are what
/// keeps every `@embedFile`d cubin inside the archive the module loader reads
/// from (no run-time file lookups).
pub fn cubin_table() *const [generated_cubins.cubins.len]Cubin {
    assert(generated_cubins.cubins.len > 0);
    assert(@sizeOf(Cubin) > 0);
    return &generated_cubins.cubins;
}

/// Number of cubins in the table, so no exported symbol lacks its length.
export fn inferno_cubin_count() u32 {
    assert(generated_cubins.cubins.len > 0);
    assert(generated_cubins.cubins.len <= std.math.maxInt(u32));
    return generated_cubins.cubins.len;
}

/// One embedded cubin, or null when `index` is past the end. Each entry carries
/// its own length, and `inferno_cubin_count` gives the entry count.
export fn inferno_cubin_at(index: u32) ?*const Cubin {
    assert(@sizeOf(Cubin) > 0);
    assert(generated_cubins.cubins.len > 0);

    if (index < generated_cubins.cubins.len) {
        return &generated_cubins.cubins[index];
    } else {
        return null;
    }
}

/// Version string of the core library, as a NUL terminated C string.
export fn inferno_version() [*:0]const u8 {
    assert(version_text.len > 0);
    assert(version_text.len < 32);
    return version_text;
}

/// Writes a JSON description of every CUDA device into `buffer`.
///
/// Returns the number of bytes written, excluding the NUL terminator, or 0 if
/// the driver could not be queried or `capacity` is 0. A zero capacity returns
/// 0 without touching `buffer`: asserting here would abort a C caller across
/// the FFI boundary. The document is always NUL terminated and never writes
/// past `capacity`.
export fn inferno_probe_devices(buffer: [*]u8, capacity: u32) u32 {
    if (capacity == 0) {
        return 0;
    } else {
        assert(@intFromPtr(buffer) != 0);
    }

    buffer[0] = 0;
    var document: [document_bytes_max]u8 = undefined;
    const document_length = render_probe_document(document[0..]) catch |err| {
        std.log.err("device probe failed: {s}", .{@errorName(err)});
        return 0;
    };
    const rendered = document[0..document_length];
    const written = copy_truncated(rendered, buffer[0..capacity]);
    assert(written < capacity);
    return written;
}

/// Bounded JSON writer: every append either fits or fails.
const JsonWriter = struct {
    buffer: []u8,
    length: u32,

    fn init(buffer: []u8) JsonWriter {
        assert(buffer.len > 0);
        assert(buffer.len <= document_bytes_max);
        return .{ .buffer = buffer, .length = 0 };
    }

    fn append(writer: *JsonWriter, text: []const u8) !void {
        assert(writer.length <= writer.buffer.len);
        assert(text.len <= document_bytes_max);

        const end: usize = writer.length + text.len;
        if (end <= writer.buffer.len) {
            @memcpy(writer.buffer[writer.length..end], text);
            writer.length = @intCast(end);
        } else {
            return error.document_too_long;
        }
    }

    fn append_format(
        writer: *JsonWriter,
        comptime format: []const u8,
        args: anytype,
    ) !void {
        assert(writer.length <= writer.buffer.len);
        assert(format.len > 0);

        const remaining = writer.buffer[writer.length..];
        if (std.fmt.bufPrint(remaining, format, args)) |text| {
            writer.length += @intCast(text.len);
        } else |_| {
            return error.document_too_long;
        }
        assert(writer.length <= writer.buffer.len);
    }

    /// Appends `text` as a JSON string. Names come from a foreign driver, so
    /// quotes, backslashes and control bytes are escaped.
    fn append_json_string(writer: *JsonWriter, text: []const u8) !void {
        assert(writer.length < writer.buffer.len);
        assert(text.len <= cuda_driver.device_name_bytes_max);

        try writer.append("\"");
        for (text) |byte| {
            switch (byte) {
                '"' => try writer.append("\\\""),
                '\\' => try writer.append("\\\\"),
                0x00...0x1f => try writer.append_format("\\u{x:0>4}", .{byte}),
                else => try writer.append(&.{byte}),
            }
        }
        try writer.append("\"");
        assert(writer.length >= 2);
    }
};

const ProbeError = cuda_driver.DriverError || error{document_too_long};

/// Queries the driver and renders the probe document into `buffer`.
fn render_probe_document(buffer: []u8) ProbeError!u32 {
    assert(buffer.len >= document_bytes_min);
    assert(buffer.len <= document_bytes_max);

    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const device_count = try cuda_driver.query_devices(devices[0..]);
    const cuda_api_version = try cuda_driver.cuda_api_version();
    // procfs is the only source of the display driver version; when it is
    // unreadable the document reports "unknown" instead of failing the probe.
    var driver_version = cuda_driver.DriverVersion.unknown();
    cuda_driver.display_driver_version(&driver_version) catch {
        driver_version = cuda_driver.DriverVersion.unknown();
    };

    return render_document(
        devices[0..device_count],
        &driver_version,
        cuda_api_version,
        buffer,
    );
}

/// Renders devices that have already been queried; kept free of driver calls
/// so it can be tested without a GPU.
fn render_document(
    devices: []const cuda_driver.Device,
    driver_version: *const cuda_driver.DriverVersion,
    cuda_api_version: u32,
    buffer: []u8,
) ProbeError!u32 {
    assert(devices.len <= cuda_driver.device_count_max);
    assert(buffer.len >= document_bytes_min);

    var writer = JsonWriter.init(buffer);
    try writer.append("{\"driver\":");
    try writer.append_json_string(driver_version.text_slice());
    try writer.append_format(",\"cuda\":{d}", .{cuda_api_version});
    try writer.append(",\"devices\":[");
    for (devices, 0..) |*device, index| {
        if (index == 0) {
            try render_device(&writer, device);
        } else {
            try writer.append(",");
            try render_device(&writer, device);
        }
    }
    try writer.append("]}");
    assert(writer.length > 0);
    return writer.length;
}

/// Renders one device object.
fn render_device(writer: *JsonWriter, device: *const cuda_driver.Device) !void {
    assert(device.name_length <= cuda_driver.device_name_bytes_max);
    assert(device.name_length > 0);

    try writer.append_format("{{\"idx\":{d},\"name\":", .{device.index});
    try writer.append_json_string(device.name_slice());
    try writer.append_format(
        ",\"cc\":\"{d}.{d}\",\"vram_bytes\":{d}}}",
        .{
            device.compute_capability_major,
            device.compute_capability_minor,
            device.vram_bytes,
        },
    );
    assert(writer.length > 0);
}

/// Copies as much of `source` as fits, always NUL terminating `target`.
fn copy_truncated(source: []const u8, target: []u8) u32 {
    assert(target.len > 0);
    assert(source.len <= document_bytes_max);

    const room = target.len - 1;
    const length = @min(room, source.len);
    @memcpy(target[0..length], source[0..length]);
    target[length] = 0;
    assert(length <= room);
    return @intCast(length);
}

// ---------------------------------------------------------------------------
// Model and sequence handles (I-036).
//
// Rust owns the lifetime policy; these seven symbols are the mechanism it
// drives. Every one of them answers a refusal with a code rather than an
// assertion: a panic here would abort the Rust process across the ABI (plan.md
// §8 rule 1b). The codes are `handles.Code`, and `handles.codeFor` is the only
// place an error becomes one.
// ---------------------------------------------------------------------------

/// Opens the model `manifest_json` describes, or returns null.
///
/// The document is a JSON object with a `model` key; `{"model":"tiny"}` names
/// the committed tiny fixtures, which is the only forward this build has until
/// I-017c lands the real checkpoint. The ABI has no error out-parameter, so the
/// reason for a null is logged here instead of returned.
export fn inferno_model_open(
    manifest_json: ?[*]const u8,
    length: u32,
) ?*handles.Cell {
    assert(handles.models_max > 0);
    assert(handles.manifest_bytes_max > 0);

    return handles.openModel(manifest_json, length) catch |err| null_after(
        "inferno_model_open",
        err,
    );
}

/// Closes `handle`, freeing the model and every buffer it owns. Idempotent: a
/// stale handle frees nothing.
export fn inferno_model_close(handle: ?*handles.Cell) void {
    assert(handles.models_max > 0);
    assert(handles.model_cells_max > 0);

    handles.closeModel(handle);
}

/// Opens a sequence over `model` and prefills `ids`, or returns null.
///
/// A sequence that outlives its model is impossible in Rust (the lifetime
/// parameter), but the C ABI cannot forbid it, so the sequence is orphaned by
/// `inferno_model_close` and answers `model_closed` afterwards.
export fn inferno_seq_open(
    model: ?*const handles.Cell,
    ids: ?[*]const u32,
    count: u32,
) ?*handles.Cell {
    assert(handles.sequences_max > 0);
    assert(handles.prompt_ids_max > 0);

    return handles.openSequence(model, ids, count) catch |err| null_after(
        "inferno_seq_open",
        err,
    );
}

/// Closes `handle`, returning its slot to the table. Idempotent.
export fn inferno_seq_close(handle: ?*handles.Cell) void {
    assert(handles.sequence_cells_max > 0);
    assert(handles.sequences_max > 0);

    handles.closeSequence(handle);
}

/// Returns the sequence to start-of-sequence and replays its prompt.
export fn inferno_seq_reset(handle: ?*const handles.Cell) i32 {
    assert(handles.sequences_max > 0);
    assert(handles.prompt_ids_max > 0);

    handles.resetSequence(handle) catch |err| {
        return handles.codeFor(err);
    };
    return @intFromEnum(handles.Code.ok);
}

/// Produces the next id into `out_token`; see `handles.stats_len` for
/// `out_stats`, which may be null or shorter than the vector.
export fn inferno_seq_step(
    handle: ?*const handles.Cell,
    out_token: ?*u32,
    out_stats: ?[*]f32,
    stats_cap: u32,
) i32 {
    assert(handles.stats_len > 0);
    assert(handles.sequences_max > 0);

    handles.stepSequence(handle, out_token, out_stats, stats_cap) catch |err| {
        return handles.codeFor(err);
    };
    return @intFromEnum(handles.Code.ok);
}

/// Fills `out` with the handle tables' counters; 0 on success, -1 on null.
///
/// The Rust side reads this to prove that a dropped `Sequence` and a closed
/// `Model` left no slot occupied and no allocation behind.
export fn inferno_handle_counts(out: ?*handles.Counts) i32 {
    assert(handles.models_max > 0);
    assert(handles.sequences_max > 0);

    if (out) |counts| {
        handles.readCounts(handles.instance(), counts);
        return @intFromEnum(handles.Code.ok);
    } else {
        return @intFromEnum(handles.Code.invalid_argument);
    }
}

/// Logs why a handle could not be issued and returns the ABI's null.
fn null_after(
    entry_point: []const u8,
    err: anyerror,
) ?*handles.Cell {
    assert(entry_point.len > 0);
    assert(err != error.NoError);

    std.log.err("{s}: {s} (code {d})", .{
        entry_point,
        @errorName(err),
        handles.codeFor(err),
    });
    return null;
}

// One merged import list. Each lane added its own `hw/kernel.zig` line, so the
// merges stacked four copies of it; kept deduplicated and grouped by directory.
test {
    _ = @import("hw/bench.zig");
    _ = @import("model/config_bridge.zig");
    _ = @import("testing_alloc.zig");
    _ = @import("hw/api.zig").cuda_context;
    _ = @import("hw/api.zig").cuda_driver;
    _ = @import("hw/dl.zig");
    _ = @import("hw/kernel.zig");
    _ = @import("hw/pcie_link.zig");
    _ = @import("weights/config.zig");
    _ = @import("weights/manifest.zig");
    _ = @import("weights/quant.zig");
    _ = @import("weights/safetensors.zig");
    _ = @import("weights/shard.zig");
    _ = @import("kernels/attention.zig");
    _ = @import("kernels/elementwise.zig");
    _ = @import("kernels/gdn.zig");
    _ = @import("kernels/gemm.zig");
    _ = @import("kernels/gemv.zig");
    _ = @import("kernels/moe.zig");
    _ = @import("kernels/ple.zig");
    _ = @import("model/handles.zig");
    _ = @import("model/mod.zig");
    _ = @import("model/multi.zig");
}

test "inferno_version is a NUL terminated C string" {
    const text = std.mem.span(inferno_version());
    try std.testing.expectEqualStrings(version_text, text);
}

test "the cubin table holds at least the probe kernel" {
    const table = cubin_table();
    try std.testing.expect(table.len > 0);
    const probe_cubin = table[0];
    try std.testing.expectEqualStrings("probe", std.mem.span(probe_cubin.name));
    try std.testing.expect(probe_cubin.length > 0);
    try std.testing.expectEqual(
        @as(u32, @intCast(table.len)),
        inferno_cubin_count(),
    );
}

test "inferno_cubin_at bounds its index" {
    const first = inferno_cubin_at(0) orelse return error.TestExpectedCubin;
    try std.testing.expect(first.length > 0);
    try std.testing.expect(inferno_cubin_at(inferno_cubin_count()) == null);
}

test "inferno_probe_devices with zero capacity writes nothing" {
    var buffer: [8]u8 = [_]u8{0xaa} ** 8;
    const written = inferno_probe_devices(&buffer, 0);
    try std.testing.expectEqual(@as(u32, 0), written);
    try std.testing.expectEqual([_]u8{0xaa} ** 8, buffer);
}

test "copy_truncated NUL terminates a short buffer" {
    const source = "{\"devices\":[]}";
    var target: [8]u8 = undefined;
    const written = copy_truncated(source, target[0..]);
    try std.testing.expectEqual(@as(u32, 7), written);
    try std.testing.expectEqualStrings("{\"devic", target[0..written]);
    try std.testing.expectEqual(@as(u8, 0), target[written]);
}

test "render_document produces the documented JSON" {
    var name: [cuda_driver.device_name_bytes_max]u8 =
        [_]u8{0} ** cuda_driver.device_name_bytes_max;
    const name_text = "NVIDIA GeForce RTX 5060 Ti";
    @memcpy(name[0..name_text.len], name_text);
    const devices = [_]cuda_driver.Device{.{
        .index = 0,
        .name = name,
        .name_length = name_text.len,
        .compute_capability_major = 12,
        .compute_capability_minor = 0,
        .vram_bytes = 16660430848,
        .multiprocessor_count = 36,
        .l2_cache_bytes = 32 * 1024 * 1024,
        .clock_khz = 2602000,
        .memory_clock_khz = 14001000,
        .pci_bus_id = blk: {
            var bus_id: [cuda_driver.pci_bus_id_bytes_max]u8 =
                [_]u8{0} ** cuda_driver.pci_bus_id_bytes_max;
            @memcpy(bus_id[0.."0000:05:00.0".len], "0000:05:00.0");
            break :blk bus_id;
        },
        .pci_bus_id_length = "0000:05:00.0".len,
        .handle = 0,
    }};
    const driver_version = cuda_driver.DriverVersion{
        .text = blk: {
            var text: [cuda_driver.driver_version_bytes_max]u8 =
                [_]u8{0} ** cuda_driver.driver_version_bytes_max;
            @memcpy(text[0.."610.57.04".len], "610.57.04");
            break :blk text;
        },
        .length = "610.57.04".len,
    };

    var buffer: [document_bytes_max]u8 = undefined;
    const length = try render_document(
        devices[0..],
        &driver_version,
        13030,
        buffer[0..],
    );
    try std.testing.expectEqualStrings(
        "{\"driver\":\"610.57.04\",\"cuda\":13030,\"devices\":[{\"idx\":0," ++
            "\"name\":\"NVIDIA GeForce RTX 5060 Ti\",\"cc\":\"12.0\"," ++
            "\"vram_bytes\":16660430848}]}",
        buffer[0..length],
    );
}

test "inferno_probe_devices writes a whole document into a small buffer" {
    var buffer: [8 * 1024]u8 = undefined;
    const written = inferno_probe_devices(&buffer, @intCast(buffer.len));
    if (written == 0) return error.SkipZigTest;

    const document = buffer[0..written];
    try std.testing.expect(std.mem.startsWith(u8, document, "{\"driver\":"));
    try std.testing.expect(std.mem.endsWith(u8, document, "]}"));
    try std.testing.expectEqual(@as(u8, 0), buffer[written]);
}
