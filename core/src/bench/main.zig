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
const stability = @import("stability.zig");
const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// Directory and path of the document, relative to the repository root.
const output_directory = "bench";
const output_relative_path = "bench/hw-probe.json";

/// Levels from `core/zig-out/bin` up to the repository root.
const repository_root_levels = "../../..";

/// How many times to run the measurement sequence, read from the environment.
///
/// One pass cannot say anything about run-to-run spread, so the default leaves
/// the document's `stability_percent_max` null. Filling it in takes this many
/// passes on an otherwise idle machine.
const repeat_variable = "INFERNO_PROBE_REPEATS";
const repeat_default: usize = 1;

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
    assert(repeat_default <= stability.repeat_max);

    const repeats = repeat_count() catch |err| {
        std.debug.print(
            "inferno-probe: {s} must be a pass count from 1 to {d}\n",
            .{ repeat_variable, stability.repeat_max },
        );
        return err;
    };

    var probe: bench.Probe = undefined;
    try probe.init();
    defer probe.deinit();

    var spread = stability.MaxSpread.init();
    var pass: usize = 0;
    while (pass < repeats) : (pass += 1) {
        try probe.measure_all();
        // The tensor-core peaks come from the prefill GEMM kernels
        // themselves, so the prefill ceiling in THEORY.md §3 is measured
        // rather than assumed.
        try probe.measure_gemm_peaks(try gemm_cubin());
        spread.observe(&probe.report);
        if (repeats > 1) {
            std.debug.print(
                "inferno-probe: pass {d}/{d}\n",
                .{ pass + 1, repeats },
            );
        }
    }
    // Null until a second pass made a spread meaningful.
    probe.report.stability_percent_max = spread.percent_max();

    try probe.verify_no_leak();
    try write_json(&probe);
    try print_tables(&probe);
    print_gemm_peaks(&probe);

    const failures = probe.report.p2p_failure_count;
    if (failures == 0) {
        return 0;
    } else {
        print_failures(&probe);
        return 1;
    }
}

/// `INFERNO_PROBE_REPEATS`, or a single pass when it is unset.
fn repeat_count() !usize {
    return parse_repeats(std.posix.getenv(repeat_variable));
}

/// The repeat count behind `repeat_variable`. A value that is present but not a
/// usable pass count fails: measuring once while believing the number said six
/// is worse than refusing to run.
fn parse_repeats(text: ?[]const u8) !usize {
    const value = text orelse return repeat_default;
    const count = std.fmt.parseInt(usize, value, 10) catch
        return error.invalid_repeats;
    if (count < 1 or count > stability.repeat_max) {
        return error.invalid_repeats;
    } else {
        assert(count >= 1);
        return count;
    }
}

/// Finds the `gemm` cubin among the kernels embedded in this binary.
fn gemm_cubin() ![]const u8 {
    assert(generated_cubins.cubins.len > 0);
    assert(@sizeOf(generated_cubins.Cubin) > 0);

    for (generated_cubins.cubins) |entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), "gemm")) {
            const bytes = entry.bytes[0..entry.length];
            assert(bytes.len > 0);
            return bytes;
        }
    }
    return error.GemmCubinMissing;
}

/// Prints the measured tensor peaks and the payoff of native FP4 MMA.
fn print_gemm_peaks(probe: *const bench.Probe) void {
    assert(probe.report.device_count > 0);
    assert(probe.report.gemm_peak_tflops.nvfp4_mma > 0);

    const peaks = probe.report.gemm_peak_tflops;
    std.debug.print(
        "gemm peaks at M=N=K={d}: bf16 {d:.2} TFLOPS, fp8 {d:.2}, " ++
            "nvfp4_mma {d:.2}, nvfp4_dequant_bf16 {d:.2}; " ++
            "native FP4 payoff {d:.2}x\n",
        .{
            bench.gemm_peak.side,
            peaks.bf16,
            peaks.fp8,
            peaks.nvfp4_mma,
            peaks.nvfp4_dequant_bf16,
            peaks.nvfp4_mma / peaks.nvfp4_dequant_bf16,
        },
    );
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
    _ = @import("stability.zig");
}

test "the repeat count defaults to one pass" {
    try std.testing.expectEqual(repeat_default, try parse_repeats(null));
}

test "the repeat count is taken from the environment" {
    try std.testing.expectEqual(@as(usize, 6), try parse_repeats("6"));
    try std.testing.expectEqual(
        stability.repeat_max,
        try parse_repeats("32"),
    );
}

test "an unusable repeat count fails" {
    for ([_][]const u8{ "0", "33", "", "six", "-1", "6.0" }) |value| {
        try std.testing.expectError(
            error.invalid_repeats,
            parse_repeats(value),
        );
    }
}

test "the output path lives in the output directory" {
    try std.testing.expect(std.mem.startsWith(
        u8,
        output_relative_path,
        output_directory,
    ));
    try std.testing.expect(std.mem.endsWith(u8, output_relative_path, ".json"));
}
