//! The C ABI the Rust server links against.
//!
//! Everything exported here is a plain C symbol: no Zig-only types cross the
//! boundary, buffers carry their capacity, and the JSON document is truncated
//! rather than overflowed.

const std = @import("std");

const cuda_driver = @import("hw/cuda_driver.zig");
const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// Reported by `inferno_version`; kept in step with the Cargo workspace.
const version_text = "inferno 0.1.0";

/// Rendered documents are smaller than this; the probe buffer is much larger.
const document_bytes_max: u32 = 16 * 1024;

/// Below this a document cannot describe even one device.
const document_bytes_min: u32 = 512;

// Zig only emits data something references, so the cubin table is exported:
// this keeps every `@embedFile`d cubin inside the static library, which is
// where the module loader reads it from (no run-time file lookups).
comptime {
    assert(generated_cubins.cubins.len > 0);
    @export(&generated_cubins.cubins, .{ .name = "inferno_cubin_table" });
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
/// the driver could not be queried. The document is always NUL terminated and
/// never writes past `capacity`.
export fn inferno_probe_devices(buffer: [*]u8, capacity: u32) u32 {
    assert(@intFromPtr(buffer) != 0);
    assert(capacity > 0);

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

test {
    _ = @import("hw/dl.zig");
    _ = @import("hw/cuda_driver.zig");
}

test "inferno_version is a NUL terminated C string" {
    const text = std.mem.span(inferno_version());
    try std.testing.expectEqualStrings(version_text, text);
}

test "the cubin table holds at least the probe kernel" {
    try std.testing.expect(generated_cubins.cubins.len > 0);
    const probe_cubin = generated_cubins.cubins[0];
    try std.testing.expectEqualStrings("probe", std.mem.span(probe_cubin.name));
    try std.testing.expect(probe_cubin.length > 0);
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
