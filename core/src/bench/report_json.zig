//! Renders the probe report as `bench/hw-probe.json`.
//!
//! The field names are the contract between the probe and the analysis scripts
//! (THEORY.md, `tools/gate.sh`), so they are spelled out here once and never
//! abbreviated.

const std = @import("std");

const bench = @import("hw");
const gemm_peak = bench.gemm_peak;
const cuda_driver = bench.cuda_driver;

const assert = std.debug.assert;

/// The document is a few kilobytes; the bound keeps rendering allocation-free.
pub const document_bytes_max: u32 = 64 * 1024;

/// Below this a document cannot describe four devices.
pub const document_bytes_min: u32 = 1024;

/// "1970-01-01T00:00:00Z" is 20 bytes.
pub const timestamp_bytes_max: u32 = 32;

pub const RenderError = error{document_too_long};

/// Renders `report` as the probe document, returning the byte length.
pub fn render(
    report: *const bench.Report,
    generated: []const u8,
    buffer: []u8,
) RenderError!u32 {
    assert(report.device_count > 0);
    assert(generated.len > 0);
    assert(buffer.len >= document_bytes_min);

    var stream = std.io.fixedBufferStream(buffer);
    const rendered = render_document(report, generated, stream.writer());
    rendered catch |err| switch (err) {
        error.NoSpaceLeft => return RenderError.document_too_long,
    };
    assert(stream.pos <= buffer.len);
    assert(stream.pos > 0);
    return @intCast(stream.pos);
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn iso8601_now(buffer: []u8) RenderError![]const u8 {
    assert(buffer.len >= timestamp_bytes_max);
    assert(timestamp_bytes_max > 0);

    const seconds = std.time.timestamp();
    if (seconds >= 0) {
        return format_iso8601(@intCast(seconds), buffer);
    } else {
        return RenderError.document_too_long;
    }
}

/// Formats a Unix timestamp as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn format_iso8601(
    seconds_since_epoch: u64,
    buffer: []u8,
) RenderError![]const u8 {
    assert(seconds_since_epoch <= std.math.maxInt(u32));
    assert(buffer.len >= timestamp_bytes_max);

    const epoch_seconds = std.time.epoch.EpochSeconds{
        .secs = seconds_since_epoch,
    };
    const epoch_day = epoch_seconds.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_seconds = epoch_seconds.getDaySeconds();
    const text = std.fmt.bufPrint(
        buffer,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z",
        .{
            year_day.year,
            month_day.month.numeric(),
            month_day.day_index + 1,
            day_seconds.getHoursIntoDay(),
            day_seconds.getMinutesIntoHour(),
            day_seconds.getSecondsIntoMinute(),
        },
    ) catch return RenderError.document_too_long;
    assert(text.len == 20);
    return text;
}

fn render_document(
    report: *const bench.Report,
    generated: []const u8,
    writer: anytype,
) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(count <= bench.device_count_max);

    try writer.writeAll("{\"generated\":");
    try write_json_string(generated, writer);
    try writer.print(",\"cuda\":{d},\"driver\":", .{report.cuda_api_version});
    try write_json_string(report.driver_version.text_slice(), writer);
    try writer.writeAll(",\n  \"device_info\":");
    try render_devices(report, writer);
    try writer.writeAll(",\n  \"h2d_bw_gib_s\":");
    try render_f64_array(report.h2d_bw_gib_s[0..count], writer);
    try writer.writeAll(",\n  \"d2h_bw_gib_s\":");
    try render_f64_array(report.d2h_bw_gib_s[0..count], writer);
    // Two explicit bases for the device-local copy, never one number with a
    // basis string: `copied` is what the link would have carried, `hbm` is the
    // read + write traffic that feeds THEORY.md's decode ceiling.
    try writer.writeAll(",\n  \"d2d_copied_gib_s\":");
    try render_f64_array(report.d2d_copied_gib_s[0..count], writer);
    try writer.writeAll(",\n  \"d2d_hbm_gib_s\":");
    try render_f64_array(report.d2d_hbm_gib_s[0..count], writer);
    try writer.writeAll(",\n  \"transfer_size_bytes\":");
    try render_u64_array(bench.transfer_sizes_bytes[0..], writer);
    try writer.writeAll(",\n  \"h2d_bw_gib_s_by_size\":");
    try render_by_size(&report.h2d_bw_by_size_gib_s, count, writer);
    try writer.writeAll(",\n  \"d2h_bw_gib_s_by_size\":");
    try render_by_size(&report.d2h_bw_by_size_gib_s, count, writer);
    try writer.writeAll(",\n  \"p2p\":");
    try render_p2p(report, writer);
    try writer.writeAll(",\n  \"aggregate_h2d_gib_s\":");
    try write_number(report.aggregate_h2d_gib_s, writer);
    try writer.print(
        ",\n  \"aggregate_h2d_bytes_per_device\":{d}",
        .{bench.aggregate_bytes_per_device},
    );
    // Tensor peaks: measured, not guessed, so THEORY.md §3's prefill ceiling
    // and the per-kernel ratios in the GEMM harness both have one basis.
    try writer.writeAll(",\n  \"gemm_peak_tflops\":{");
    try writer.writeAll("\"bf16\":");
    try write_number(report.gemm_peak_tflops.bf16, writer);
    try writer.writeAll(",\"fp8\":");
    try write_number(report.gemm_peak_tflops.fp8, writer);
    try writer.writeAll(",\"nvfp4_mma\":");
    try write_number(report.gemm_peak_tflops.nvfp4_mma, writer);
    try writer.writeAll(",\"nvfp4_dequant_bf16\":");
    try write_number(report.gemm_peak_tflops.nvfp4_dequant_bf16, writer);
    try writer.print(",\"side\":{d}", .{gemm_peak.side});
    try writer.writeAll("}");
    try writer.writeAll(",\n  \"host\":{\"memcpy_gib_s\":");
    try write_number(report.host_memcpy_bw_gib_s, writer);
    try writer.print(
        ",\"memcpy_bytes\":{d},\"cpu_count\":{d},\"round_trip_bytes\":{d},",
        .{
            bench.host_copy_bytes,
            report.cpu_count,
            bench.host_device_latency_bytes,
        },
    );
    try writer.writeAll("\"round_trip_us\":");
    try render_f64_array(report.host_device_latency_us[0..count], writer);
    try writer.writeAll("},\n  \"h2d_achieved_percent\":");
    try render_f64_array(report.h2d_achieved_percent[0..count], writer);
    try writer.writeAll(",\n  \"vram_leak_bytes\":");
    try render_u64_array(report.vram_leak_bytes[0..count], writer);
    try writer.writeAll(",\n  \"basis\":{");
    try render_basis(writer);
    // Closes the basis object, then the document.
    try writer.writeAll("}\n}\n");
    assert(count == report.device_count);
}

/// One line per bandwidth field stating what it counts, so no number in this
/// document depends on the reader guessing its basis.
fn render_basis(writer: anytype) !void {
    assert(bench.transfer_size_count == 3);
    assert(bench.device_count_max >= 4);

    try writer.writeAll("\n    \"h2d_bw_gib_s\":");
    try write_json_string("bytes moved across the link, counted once", writer);
    try writer.writeAll(",\n    \"d2h_bw_gib_s\":");
    try write_json_string("bytes moved across the link, counted once", writer);
    try writer.writeAll(",\n    \"aggregate_h2d_gib_s\":");
    try write_json_string(
        "bytes moved across the link, counted once, 4 GPUs on 4 streams",
        writer,
    );
    try writer.writeAll(",\n    \"d2d_copied_gib_s\":");
    try write_json_string("device-local copy, bytes copied once", writer);
    try writer.writeAll(",\n    \"d2d_hbm_gib_s\":");
    try write_json_string(
        "device-local copy, HBM traffic = read + write = 2x copied",
        writer,
    );
    try writer.writeAll(",\n    \"p2p_bw_gib_s\":");
    try write_json_string("bytes copied once, over the P2P link", writer);
    try writer.writeAll(",\n    \"p2p_latency_us\":");
    try write_json_string(
        "median of 200, 4 KiB round trip A->B->A",
        writer,
    );
    try writer.writeAll(",\n    \"host_memcpy_gib_s\":");
    try write_json_string("host memcpy, read + write", writer);
    try writer.writeAll(",\n    \"link_gib_s_theoretical\":");
    try write_json_string(
        "PCIe generation x width, raw rate from the kernel's link state",
        writer,
    );
    try writer.writeAll("\n  ");
}

fn render_devices(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(count <= bench.device_count_max);

    try writer.writeAll("[");
    var index: u32 = 0;
    while (index < count) : (index += 1) {
        if (index == 0) {
            try render_device(report, index, writer);
        } else {
            try writer.writeAll(",");
            try render_device(report, index, writer);
        }
    }
    try writer.writeAll("]");
    assert(index == count);
}

fn render_device(
    report: *const bench.Report,
    index: u32,
    writer: anytype,
) !void {
    const device = &report.devices[index];
    assert(index < report.device_count);
    assert(device.name_length > 0);

    try writer.print("{{\"idx\":{d},\"name\":", .{device.index});
    try write_json_string(device.name_slice(), writer);
    try writer.print(
        ",\"cc\":\"{d}.{d}\",\"vram_bytes\":{d},\"sm_count\":{d}," ++
            "\"l2_bytes\":{d},\"clock_khz\":{d}," ++
            "\"memory_clock_khz\":{d},\"pcie_bus_id\":",
        .{
            device.compute_capability_major,
            device.compute_capability_minor,
            device.vram_bytes,
            device.multiprocessor_count,
            device.l2_cache_bytes,
            device.clock_khz,
            device.memory_clock_khz,
        },
    );
    try write_json_string(device.pci_bus_id_slice(), writer);
    try writer.print(
        ",\"pcie_link_gen\":{d},\"pcie_link_width\":{d}," ++
            "\"link_gib_s_theoretical\":",
        .{
            report.pcie_link_state[index].generation,
            report.pcie_link_state[index].width,
        },
    );
    try write_number(report.link_gib_s_theoretical[index], writer);
    try writer.writeAll("}");
    assert(device.index == index);
}

/// One row per transfer size, one column per device.
fn render_by_size(
    rows: *const [bench.transfer_size_count][bench.device_count_max]f64,
    count: u32,
    writer: anytype,
) !void {
    assert(count > 0);
    assert(bench.transfer_size_count > 0);

    try writer.writeAll("[");
    var size_index: u32 = 0;
    while (size_index < bench.transfer_size_count) : (size_index += 1) {
        if (size_index == 0) {
            try render_f64_array(rows[size_index][0..count], writer);
        } else {
            try writer.writeAll(",");
            try render_f64_array(rows[size_index][0..count], writer);
        }
    }
    try writer.writeAll("]");
    assert(size_index == bench.transfer_size_count);
}

fn render_p2p(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(count <= bench.device_count_max);

    try writer.writeAll("[");
    var flat: u32 = 0;
    while (flat < count * count) : (flat += 1) {
        const from = @divFloor(flat, count);
        const to = @mod(flat, count);
        if (flat == 0) {
            try render_p2p_entry(&report.p2p[from][to], writer);
        } else {
            try writer.writeAll(",");
            try render_p2p_entry(&report.p2p[from][to], writer);
        }
    }
    try writer.writeAll("]");
    assert(flat == count * count);
}

fn render_p2p_entry(entry: *const bench.P2pEntry, writer: anytype) !void {
    assert(entry.from < bench.device_count_max);
    assert(entry.to < bench.device_count_max);

    try writer.print(
        "{{\"from\":{d},\"to\":{d},\"kind\":\"{s}\",\"can_access\":{}," ++
            "\"enabled\":{},\"bw_gib_s\":",
        .{
            entry.from,
            entry.to,
            @tagName(entry.kind),
            entry.can_access,
            entry.enabled,
        },
    );
    try write_number(entry.bw_gib_s, writer);
    try writer.writeAll(",\"latency_us\":");
    try write_number(entry.latency_us, writer);
    try writer.writeAll("}");
    assert(entry.from <= entry.to or entry.kind == .peer);
}

fn render_f64_array(values: []const f64, writer: anytype) !void {
    assert(values.len > 0);
    assert(values.len <= bench.device_count_max);

    try writer.writeAll("[");
    for (values, 0..) |value, index| {
        if (index == 0) {
            try write_number(value, writer);
        } else {
            try writer.writeAll(",");
            try write_number(value, writer);
        }
    }
    try writer.writeAll("]");
    assert(values.len > 0);
}

fn render_u64_array(values: []const u64, writer: anytype) !void {
    assert(values.len > 0);
    assert(values.len <= bench.device_count_max or
        values.len == bench.transfer_size_count);

    try writer.writeAll("[");
    for (values, 0..) |value, index| {
        if (index == 0) {
            try writer.print("{d}", .{value});
        } else {
            try writer.print(",{d}", .{value});
        }
    }
    try writer.writeAll("]");
    assert(values.len > 0);
}

/// Two decimal places: the probe's own noise is far wider than that.
fn write_number(value: f64, writer: anytype) !void {
    assert(std.math.isFinite(value));
    assert(value >= 0);

    try writer.print("{d:.2}", .{value});
}

/// Names and versions come from a foreign driver, so quotes and control bytes
/// are escaped rather than copied through.
fn write_json_string(text: []const u8, writer: anytype) !void {
    assert(text.len > 0);
    assert(text.len < bench.mib);

    try writer.writeAll("\"");
    for (text) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            0x00...0x1f => try writer.print("\\u{x:0>4}", .{byte}),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeAll("\"");
    assert(text.len > 0);
}

/// A report describing one device, filled with round numbers.
fn test_report() bench.Report {
    assert(bench.device_count_max >= 4);
    assert(bench.transfer_size_count == 3);

    var report: bench.Report = std.mem.zeroes(bench.Report);
    const name = "NVIDIA GeForce RTX 5060 Ti";

    report.device_count = 1;
    report.cuda_api_version = 13030;
    report.driver_version = .{
        .text = [_]u8{0} ** cuda_driver.driver_version_bytes_max,
        .length = "610.57.04".len,
    };
    @memcpy(
        report.driver_version.text[0.."610.57.04".len],
        "610.57.04",
    );
    report.devices[0] = std.mem.zeroes(cuda_driver.Device);
    @memcpy(report.devices[0].name[0..name.len], name);
    report.devices[0].name_length = name.len;
    report.devices[0].compute_capability_major = 12;
    report.devices[0].vram_bytes = 16311 * bench.mib;
    report.devices[0].multiprocessor_count = 36;
    report.devices[0].l2_cache_bytes = 32 * bench.mib;
    report.devices[0].clock_khz = 2602000;
    report.devices[0].memory_clock_khz = 15501000;
    @memcpy(
        report.devices[0].pci_bus_id[0.."0000:05:00.0".len],
        "0000:05:00.0",
    );
    report.devices[0].pci_bus_id_length = "0000:05:00.0".len;
    report.h2d_bw_gib_s[0] = 12.5;
    report.d2h_bw_gib_s[0] = 12.25;
    report.d2d_copied_gib_s[0] = 199.0;
    report.d2d_hbm_gib_s[0] = 398.0;
    report.pcie_link_state[0] = .{ .generation = 4, .width = 4 };
    report.link_gib_s_theoretical[0] = 7.336;
    report.h2d_achieved_percent[0] = 91.7;
    report.h2d_bw_by_size_gib_s = std.mem.zeroes(
        [bench.transfer_size_count][bench.device_count_max]f64,
    );
    report.d2h_bw_by_size_gib_s = report.h2d_bw_by_size_gib_s;
    report.p2p = std.mem.zeroes(
        [bench.device_count_max][bench.device_count_max]bench.P2pEntry,
    );
    report.p2p[0][0] = .{
        .from = 0,
        .to = 0,
        .kind = .self,
        .can_access = true,
        .enabled = true,
        .bw_gib_s = 199.0,
        .latency_us = 3.0,
    };
    report.aggregate_h2d_gib_s = 21.75;
    report.host_memcpy_bw_gib_s = 34.5;
    report.cpu_count = 24;
    report.host_device_latency_us[0] = 12.5;
    return report;
}

test "format_iso8601 renders a known Unix timestamp" {
    var buffer: [timestamp_bytes_max]u8 = undefined;
    const text = try format_iso8601(1_700_000_000, buffer[0..]);
    try std.testing.expectEqualStrings("2023-11-14T22:13:20Z", text);
}

test "iso8601_now renders the current UTC time" {
    var buffer: [timestamp_bytes_max]u8 = undefined;
    const text = try iso8601_now(buffer[0..]);
    try std.testing.expectEqual(@as(usize, 20), text.len);
    try std.testing.expectEqual(@as(u8, 'Z'), text[text.len - 1]);
}

test "render produces valid JSON with the documented schema" {
    const report = test_report();
    var buffer: [document_bytes_max]u8 = undefined;
    const length = try render(&report, "2026-11-24T04:00:00Z", buffer[0..]);
    const document = buffer[0..length];

    var parsed = try std.json.parseFromSlice(
        std.json.Value,
        std.testing.allocator,
        document,
        .{},
    );
    defer parsed.deinit();

    const root = parsed.value.object;
    try std.testing.expectEqualStrings(
        "2026-11-24T04:00:00Z",
        root.get("generated").?.string,
    );
    try std.testing.expectEqual(@as(i64, 13030), root.get("cuda").?.integer);
    try std.testing.expectEqualStrings(
        "610.57.04",
        root.get("driver").?.string,
    );

    const devices = root.get("device_info").?.array;
    try std.testing.expectEqual(@as(usize, 1), devices.items.len);
    const device = devices.items[0].object;
    try std.testing.expectEqualStrings(
        "NVIDIA GeForce RTX 5060 Ti",
        device.get("name").?.string,
    );
    try std.testing.expectEqual(@as(i64, 36), device.get("sm_count").?.integer);
    try std.testing.expectEqual(
        @as(i64, 15501000),
        device.get("memory_clock_khz").?.integer,
    );

    try std.testing.expectEqual(
        @as(usize, 1),
        root.get("h2d_bw_gib_s").?.array.items.len,
    );
    try std.testing.expectEqual(
        @as(usize, bench.transfer_size_count),
        root.get("h2d_bw_gib_s_by_size").?.array.items.len,
    );
    try std.testing.expectEqual(
        @as(usize, 1),
        root.get("p2p").?.array.items.len,
    );
    const cell = root.get("p2p").?.array.items[0].object;
    try std.testing.expectEqualStrings("self", cell.get("kind").?.string);
    try std.testing.expectEqual(true, cell.get("enabled").?.bool);

    try std.testing.expectEqual(
        @as(f64, 398.0),
        root.get("d2d_hbm_gib_s").?.array.items[0].float,
    );
    try std.testing.expectEqual(
        @as(i64, 4),
        device.get("pcie_link_gen").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 4),
        device.get("pcie_link_width").?.integer,
    );
    const basis = root.get("basis").?.object;
    try std.testing.expectEqualStrings(
        "device-local copy, HBM traffic = read + write = 2x copied",
        basis.get("d2d_hbm_gib_s").?.string,
    );
    try std.testing.expectEqualStrings(
        "median of 200, 4 KiB round trip A->B->A",
        basis.get("p2p_latency_us").?.string,
    );

    const host = root.get("host").?.object;
    try std.testing.expectEqual(@as(i64, 24), host.get("cpu_count").?.integer);
    try std.testing.expectEqual(
        @as(usize, 1),
        root.get("vram_leak_bytes").?.array.items.len,
    );
}
