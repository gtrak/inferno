//! `inferno-probe`: measures this machine and writes `bench/hw-probe.json`.
//!
//! Run it from anywhere: the output path is resolved from the binary's own
//! location (`core/zig-out/bin/inferno-probe`), so the repository root is three
//! directories up. The exit status is non-zero when a GPU fails to initialize
//! or when any ordered GPU pair cannot reach each other over P2P.

const std = @import("std");

const bench = @import("hw");
const report_json = @import("report_json.zig");
const report_markdown = @import("report_markdown.zig");

const assert = std.debug.assert;

/// Directory and path of the document, relative to the repository root.
const output_directory = "bench";
const output_relative_path = "bench/hw-probe.json";

/// Levels from `core/zig-out/bin` up to the repository root.
const repository_root_levels = "../../..";

pub fn main() u8 {
    assert(output_relative_path.len > output_directory.len);
    assert(bench.device_count_max >= 4);

    return run() catch |err| {
        std.debug.print("inferno-probe failed: {s}\n", .{@errorName(err)});
        return 1;
    };
}

fn run() !u8 {
    assert(output_directory.len > 0);
    assert(output_relative_path.len > 0);

    var probe: bench.Probe = undefined;
    try probe.init();
    defer probe.deinit();

    try probe.measure_all();
    try probe.verify_no_leak();
    try write_json(&probe);
    try print_tables(&probe);

    const failures = probe.report.p2p_failure_count;
    if (failures == 0) {
        return 0;
    } else {
        print_failures(&probe);
        return 1;
    }
}

/// Writes `bench/hw-probe.json` into this repository.
fn write_json(probe: *const bench.Probe) !void {
    assert(probe.report.device_count > 0);
    assert(report_json.document_bytes_max > report_json.document_bytes_min);

    var timestamp: [report_json.timestamp_bytes_max]u8 = undefined;
    const generated = try report_json.iso8601_now(timestamp[0..]);
    var document: [report_json.document_bytes_max]u8 = undefined;
    const length = try report_json.render(
        &probe.report,
        generated,
        document[0..],
    );
    try write_repository_file(document[0..length]);
}

/// Writes `contents` to the repository's `bench/` directory.
///
/// The probe may be started from any working directory, so the root is derived
/// from the executable path instead of the cwd.
fn write_repository_file(contents: []const u8) !void {
    assert(contents.len > 0);
    assert(repository_root_levels.len > 0);

    var executable_directory: [std.fs.max_path_bytes]u8 = undefined;
    const executable = try std.fs.selfExeDirPath(executable_directory[0..]);
    var root_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const root = try std.fmt.bufPrint(
        root_buffer[0..],
        "{s}/{s}",
        .{ executable, repository_root_levels },
    );
    var root_directory = try std.fs.openDirAbsolute(root, .{});
    defer root_directory.close();
    try root_directory.makePath(output_directory);
    const file = try root_directory.createFile(output_relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
    assert(contents.len > 0);
}

/// Prints the paste-ready markdown tables plus the one-line summary.
fn print_tables(probe: *const bench.Probe) !void {
    assert(probe.report.device_count > 0);
    assert(report_markdown.table_bytes_max > 0);

    const stdout = std.io.getStdOut();
    var tables: [report_markdown.table_bytes_max]u8 = undefined;
    const table_length = try report_markdown.render(
        &probe.report,
        tables[0..],
    );
    try stdout.writeAll(tables[0..table_length]);

    var summary: [report_markdown.summary_bytes_max]u8 = undefined;
    const summary_length = try report_markdown.render_summary(
        &probe.report,
        summary[0..],
    );
    try stdout.writeAll(summary[0..summary_length]);
}

/// Names every ordered pair without peer access, so the failure is actionable.
fn print_failures(probe: *const bench.Probe) void {
    assert(probe.report.p2p_failure_count > 0);
    assert(probe.report.device_count <= bench.device_count_max);

    var from: u32 = 0;
    while (from < probe.report.device_count) : (from += 1) {
        var to: u32 = 0;
        while (to < probe.report.device_count) : (to += 1) {
            const entry = &probe.report.p2p[from][to];
            if (entry.kind == .peer) {
                print_peer_failure(entry);
            } else {
                // The diagonal never needs peer access.
                assert(entry.from == entry.to);
            }
        }
    }
}

fn print_peer_failure(entry: *const bench.P2pEntry) void {
    assert(entry.kind == .peer);
    assert(entry.from != entry.to);

    if (entry.enabled) {
        return;
    } else {
        std.debug.print(
            "P2P unavailable from GPU{d} to GPU{d}" ++
                " (cuDeviceCanAccessPeer={})\n",
            .{ entry.from, entry.to, entry.can_access },
        );
    }
}

test {
    // Referenced so `zig build test` also runs the renderers' tests.
    _ = @import("report_json.zig");
    _ = @import("report_markdown.zig");
}

test "the output path lives in the output directory" {
    try std.testing.expect(std.mem.startsWith(
        u8,
        output_relative_path,
        output_directory,
    ));
    try std.testing.expect(std.mem.endsWith(u8, output_relative_path, ".json"));
}
