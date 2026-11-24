//! Renders the probe report as the markdown tables THEORY.md pastes in.
//!
//! Every table names its unit and its sample count, because a bandwidth number
//! without either is not a measurement (`inference-optimisation` methodology).

const std = @import("std");

const bench = @import("hw");

const assert = std.debug.assert;

/// Six tables plus headers fit comfortably in this.
pub const table_bytes_max: u32 = 16 * 1024;

/// The one-line summary is far shorter than this.
pub const summary_bytes_max: u32 = 256;

pub const RenderError = error{document_too_long};

/// Renders every table, returning the byte length written to `buffer`.
pub fn render(
    report: *const bench.Report,
    buffer: []u8,
) RenderError!u32 {
    assert(report.device_count > 0);
    assert(buffer.len >= table_bytes_max);

    var stream = std.io.fixedBufferStream(buffer);
    render_tables(report, stream.writer()) catch |err| switch (err) {
        error.NoSpaceLeft => return RenderError.document_too_long,
    };
    assert(stream.pos <= buffer.len);
    return @intCast(stream.pos);
}

/// The three numbers that decide the engine's design: per-GPU HBM bandwidth,
/// the aggregate host<->device bandwidth all four GPUs share, and the worst
/// peer-to-peer link.
pub fn render_summary(
    report: *const bench.Report,
    buffer: []u8,
) RenderError!u32 {
    assert(report.device_count > 0);
    assert(buffer.len >= summary_bytes_max);

    const hbm = median_bandwidth(report.d2d_hbm_gib_s[0..report.device_count]);
    const aggregate = report.aggregate_h2d_gib_s;
    const worst_peer = worst_peer_bandwidth(report);
    const text = std.fmt.bufPrint(
        buffer,
        "per-GPU HBM {d:.1} GiB/s | aggregate host<->device {d:.1} GiB/s" ++
            " | worst P2P {d:.1} GiB/s\n",
        .{ hbm, aggregate, worst_peer },
    ) catch return RenderError.document_too_long;
    assert(text.len > 0);
    return @intCast(text.len);
}

fn render_tables(report: *const bench.Report, writer: anytype) !void {
    assert(report.device_count > 0);
    assert(report.device_count <= bench.device_count_max);

    try render_inventory(report, writer);
    try render_bandwidth(report, writer);
    try render_p2p_bandwidth(report, writer);
    try render_p2p_latency(report, writer);
    try render_host(report, writer);
    try render_stability(report, writer);
    assert(report.device_count <= bench.device_count_max);
}

/// The stability line: measured, or explicitly not measured. A prose spread
/// that no run produced is how this document acquired the last one.
fn render_stability(report: *const bench.Report, writer: anytype) !void {
    if (report.stability_percent_max) |percent| {
        assert(percent >= 0);
        try writer.print(
            "\nstability: worst run-to-run spread {d:.3} %% over every " ++
                "measured number\n",
            .{percent},
        );
    } else {
        try writer.writeAll(
            "\nstability: not measured (one pass cannot say); re-run with\n" ++
                "INFERNO_PROBE_REPEATS=<n>, n >= 2, on an idle machine\n",
        );
    }
}

fn render_inventory(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(count <= bench.device_count_max);

    try writer.writeAll(
        "### GPU inventory (Q, cuDeviceGetAttribute; link from sysfs)\n" ++
            "| idx | name | cc | VRAM GiB | SMs | L2 MiB | SM MHz |" ++
            " mem MHz | link | raw GiB/s |\n" ++
            "|---:|---|---|---:|---:|---:|---:|---:|---|---:|\n",
    );
    var index: u32 = 0;
    while (index < count) : (index += 1) {
        const device = &report.devices[index];
        assert(device.name_length > 0);
        const link = report.pcie_link_state[index];
        try writer.print(
            "| {d} | {s} | {d}.{d} | {d:.2} | {d} | {d:.1} | {d} |" ++
                " {d} | Gen{d} x{d} | {d:.2} |\n",
            .{
                device.index,
                device.name_slice(),
                device.compute_capability_major,
                device.compute_capability_minor,
                @as(f64, @floatFromInt(device.vram_bytes)) /
                    @as(f64, @floatFromInt(bench.gib)),
                device.multiprocessor_count,
                @as(f64, @floatFromInt(device.l2_cache_bytes)) /
                    @as(f64, @floatFromInt(bench.mib)),
                @divFloor(device.clock_khz, 1000),
                @divFloor(device.memory_clock_khz, 1000),
                link.generation,
                link.width,
                report.link_gib_s_theoretical[index],
            },
        );
    }
    assert(index == count);
}

fn render_bandwidth(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(bench.transfer_size_count == 3);

    try writer.writeAll(
        "### Host<->device by transfer size (GiB/s, median of 6," ++
            " one transfer at a time)\n" ++
            "Bytes moved across the link, counted once.\n\n" ++
            "| idx | H2D 64 MiB | H2D 256 MiB | H2D 1 GiB |" ++
            " D2H 64 MiB | D2H 256 MiB | D2H 1 GiB |\n" ++
            "|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    var index: u32 = 0;
    while (index < count) : (index += 1) {
        try writer.print(
            "| {d} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} |" ++
                " {d:.2} |\n",
            .{
                index,
                report.h2d_bw_by_size_gib_s[0][index],
                report.h2d_bw_by_size_gib_s[1][index],
                report.h2d_bw_by_size_gib_s[2][index],
                report.d2h_bw_by_size_gib_s[0][index],
                report.d2h_bw_by_size_gib_s[1][index],
                report.d2h_bw_by_size_gib_s[2][index],
            },
        );
    }
    try render_device_local_copy(report, writer);
    assert(index == count);
}

/// The intra-device copy on both bases, plus the link audit.
fn render_device_local_copy(
    report: *const bench.Report,
    writer: anytype,
) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(bench.intra_device_copy_bytes == bench.gib);

    try writer.writeAll(
        "### Device-local 1 GiB copy, both bases (GiB/s, median of 6)\n" ++
            "`copied once` is what the link would have carried;" ++
            " `HBM r+w` is the memory traffic, 2x copied.\n\n" ++
            "| idx | copied once | HBM r+w | H2D % of raw link |\n" ++
            "|---:|---:|---:|---:|\n",
    );
    var index: u32 = 0;
    while (index < count) : (index += 1) {
        try writer.print(
            "| {d} | {d:.2} | {d:.2} | {d:.1}% |\n",
            .{
                index,
                report.d2d_copied_gib_s[index],
                report.d2d_hbm_gib_s[index],
                report.h2d_achieved_percent[index],
            },
        );
    }
    try writer.writeAll("\n");
    assert(index == count);
}

fn render_p2p_bandwidth(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(bench.p2p_bandwidth_bytes == bench.transfer_256_mib_bytes);

    try writer.writeAll(
        "### P2P bandwidth (GiB/s, 256 MiB copy, median of 6)\n" ++
            "Bytes copied once over the link; the diagonal is the" ++
            " device-local reference on the same basis.\n\n",
    );
    try render_matrix_header(report, writer);
    var from: u32 = 0;
    while (from < count) : (from += 1) {
        try writer.print("| {d} |", .{from});
        var to: u32 = 0;
        while (to < count) : (to += 1) {
            try writer.print(
                " {d:.2} |",
                .{report.p2p[from][to].bw_gib_s},
            );
        }
        try writer.writeAll("\n");
    }
    try writer.writeAll(
        "\n(diagonal = device-local reference copy)\n\n",
    );
    assert(from == count);
}

fn render_p2p_latency(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(bench.p2p_latency_bytes > 0);

    try writer.writeAll(
        "### P2P latency (us, 4 KiB round trip A->B->A, median of " ++
            "200, cuEventElapsedTime)\n",
    );
    try render_matrix_header(report, writer);
    var from: u32 = 0;
    while (from < count) : (from += 1) {
        try writer.print("| {d} |", .{from});
        var to: u32 = 0;
        while (to < count) : (to += 1) {
            try writer.print(
                " {d:.2} |",
                .{report.p2p[from][to].latency_us},
            );
        }
        try writer.writeAll("\n");
    }
    try writer.writeAll(
        "\n(event timer resolves ~0.5 us; a round trip pays both hops)\n\n",
    );
    assert(from == count);
}

fn render_matrix_header(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(count <= bench.device_count_max);

    try writer.writeAll("| from\\to |");
    var to: u32 = 0;
    while (to < count) : (to += 1) {
        try writer.print(" {d} |", .{to});
    }
    try writer.writeAll("\n|---|");
    to = 0;
    while (to < count) : (to += 1) {
        try writer.writeAll("---:|");
    }
    try writer.writeAll("\n");
    assert(to == count);
}

fn render_host(report: *const bench.Report, writer: anytype) !void {
    const count: u32 = report.device_count;
    assert(count > 0);
    assert(report.cpu_count > 0);

    try writer.writeAll(
        "### Host and aggregate\n| item | value |\n|---|---|\n",
    );
    try writer.print(
        "| memcpy 2 GiB (read+write) | {d:.2} GiB/s |\n",
        .{report.host_memcpy_bw_gib_s},
    );
    try writer.print("| CPU threads | {d} |\n", .{report.cpu_count});
    try writer.print(
        "| host<->device 64 B round trip |",
        .{},
    );
    var index: u32 = 0;
    while (index < count) : (index += 1) {
        try writer.print(
            " {d:.2} us |",
            .{report.host_device_latency_us[index]},
        );
    }
    try writer.writeAll("\n");
    try writer.print(
        "| aggregate H2D, {d} GPUs x 256 MiB on {d} streams |" ++
            " {d:.2} GiB/s |\n",
        .{ count, count, report.aggregate_h2d_gib_s },
    );
    try writer.print(
        "| VRAM leak after freeing | {d} bytes max |\n\n",
        .{max_leak_bytes(report)},
    );
    assert(index == count);
}

/// Median of the per-device device-local copy bandwidth.
fn median_bandwidth(values: []const f64) f64 {
    assert(values.len > 0);
    assert(values.len <= bench.device_count_max);

    var sorted: [bench.device_count_max]f64 = undefined;
    for (values, 0..) |value, index| {
        sorted[index] = value;
    }
    std.mem.sort(
        f64,
        sorted[0..values.len],
        {},
        std.sort.asc(f64),
    );
    const middle = @divFloor(values.len, 2);
    assert(middle < values.len);
    return sorted[middle];
}

/// Smallest peer-to-peer bandwidth across every ordered pair that has one.
fn worst_peer_bandwidth(report: *const bench.Report) f64 {
    assert(report.device_count > 0);
    assert(report.device_count <= bench.device_count_max);

    var worst = std.math.floatMax(f64);
    var found = false;
    var from: u32 = 0;
    while (from < report.device_count) : (from += 1) {
        var to: u32 = 0;
        while (to < report.device_count) : (to += 1) {
            const entry = &report.p2p[from][to];
            if (entry.enabled and entry.kind == .peer) {
                worst = @min(worst, entry.bw_gib_s);
                found = true;
            } else {
                assert(entry.from == from);
            }
        }
    }
    if (found) {
        return worst;
    } else {
        return 0;
    }
}

fn max_leak_bytes(report: *const bench.Report) u64 {
    assert(report.device_count > 0);
    assert(report.device_count <= bench.device_count_max);

    var worst: u64 = 0;
    for (report.vram_leak_bytes[0..report.device_count]) |leaked| {
        worst = @max(worst, leaked);
    }
    assert(worst < std.math.maxInt(u32));
    return worst;
}

/// A report describing four GPUs, filled with round numbers.
fn test_report() bench.Report {
    assert(bench.device_count_max >= 4);
    assert(bench.transfer_size_count == 3);

    var report: bench.Report = std.mem.zeroes(bench.Report);
    report.device_count = 4;
    report.cpu_count = 24;
    report.aggregate_h2d_gib_s = 21.5;
    report.host_memcpy_bw_gib_s = 34.5;
    var index: u32 = 0;
    while (index < report.device_count) : (index += 1) {
        const name = "NVIDIA GeForce RTX 5060 Ti";
        @memcpy(report.devices[index].name[0..name.len], name);
        report.devices[index].index = index;
        report.devices[index].name_length = name.len;
        report.devices[index].compute_capability_major = 12;
        report.devices[index].vram_bytes = 16311 * bench.mib;
        report.devices[index].multiprocessor_count = 36;
        report.devices[index].l2_cache_bytes = 32 * bench.mib;
        report.devices[index].clock_khz = 2602000;
        report.devices[index].memory_clock_khz = 15501000;
        report.h2d_bw_gib_s[index] = 12.0;
        report.d2h_bw_gib_s[index] = 12.0;
        report.d2d_copied_gib_s[index] = 199.0;
        report.d2d_hbm_gib_s[index] = 398.0;
        report.pcie_link_state[index] = .{ .generation = 4, .width = 4 };
        report.link_gib_s_theoretical[index] = 7.336;
        report.h2d_achieved_percent[index] = 91.7;
        report.host_device_latency_us[index] = 12.5;
        var size_index: u32 = 0;
        while (size_index < bench.transfer_size_count) : (size_index += 1) {
            report.h2d_bw_by_size_gib_s[size_index][index] = 12.0;
            report.d2h_bw_by_size_gib_s[size_index][index] = 12.0;
        }
        var to: u32 = 0;
        while (to < report.device_count) : (to += 1) {
            report.p2p[index][to] = .{
                .from = index,
                .to = to,
                .kind = if (index == to) .self else .peer,
                .can_access = true,
                .enabled = true,
                .bw_gib_s = if (index == to) 199.0 else 11.0,
                .latency_us = if (index == to) 1.5 else 9.0,
            };
        }
    }
    return report;
}

test "render produces the band table for four GPUs" {
    const report = test_report();
    var buffer: [table_bytes_max]u8 = undefined;
    const length = try render(&report, buffer[0..]);
    const text = buffer[0..length];

    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "### GPU inventory",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "| 3 | NVIDIA GeForce RTX 5060 Ti | 12.0 |",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "| Gen4 x4 | 7.34 |",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "### P2P latency",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "| aggregate H2D, 4 GPUs x 256 MiB on 4 streams | 21.50 GiB/s |",
    ) != null);
}

test "render_summary reports the three decisive numbers" {
    const report = test_report();
    var buffer: [summary_bytes_max]u8 = undefined;
    const length = try render_summary(&report, buffer[0..]);
    const text = buffer[0..length];

    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "per-GPU HBM 398.0",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "aggregate host<->device 21.5",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        text,
        "worst P2P 11.0",
    ) != null);
}

test "the tables say whether stability was measured" {
    var report = test_report();
    var buffer: [table_bytes_max]u8 = undefined;

    report.stability_percent_max = null;
    const unmeasured_length = try render(&report, buffer[0..]);
    const unmeasured = buffer[0..unmeasured_length];
    try std.testing.expect(std.mem.indexOf(
        u8,
        unmeasured,
        "stability: not measured",
    ) != null);

    report.stability_percent_max = 1.25;
    const measured_length = try render(&report, buffer[0..]);
    const measured = buffer[0..measured_length];
    try std.testing.expect(std.mem.indexOf(
        u8,
        measured,
        "stability: worst run-to-run spread 1.250 %",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        measured,
        "stability: not measured",
    ) == null);
}

test "worst_peer_bandwidth ignores the device-local diagonal" {
    var report = test_report();
    report.p2p[2][3].bw_gib_s = 3.5;

    try std.testing.expectApproxEqAbs(
        @as(f64, 3.5),
        worst_peer_bandwidth(&report),
        1e-9,
    );
}
