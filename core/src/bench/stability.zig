//! How far the probe's numbers move between passes over the same machine.
//!
//! Run-to-run stability is a claim the analysis documents make about this host,
//! so it has to come from repeated passes rather than from anyone's confidence.
//! This folds each full pass of a report into a per-number minimum and maximum
//! and reports the worst relative spread across all of them, as a percentage of
//! the largest pass value.
//!
//! One pass says nothing: `percent_max` stays `null` until at least two passes
//! have been observed, so an unmeasured stability can never be read as a stable
//! machine.

const std = @import("std");

const bench = @import("hw");

const assert = std.debug.assert;

/// Upper bound on passes, so a mistyped repeat count cannot run forever.
pub const repeat_max: usize = 32;

/// Number of `f64` measurements in a report. `walk` visits them in declaration
/// order, so the two must agree; `observe` asserts that they do.
pub const measurement_count: usize = count_measurements(bench.Report);

/// Counts every `f64` leaf of `T`. Optionals are skipped on purpose: the only
/// optional in the report is the stability number itself, and the spread of a
/// value that summarizes spreads would be circular.
pub fn count_measurements(comptime T: type) usize {
    switch (@typeInfo(T)) {
        .@"struct" => {
            var total: usize = 0;
            inline for (@typeInfo(T).@"struct".fields) |field| {
                total += count_measurements(field.type);
            }
            return total;
        },
        .array => {
            const element = @typeInfo(T).array.child;
            return @typeInfo(T).array.len * count_measurements(element);
        },
        // Only f64 counts: every measured number in the report is one, and the
        // device descriptors beside it are not measurements.
        .float => return if (@typeInfo(T).float.bits == 64) 1 else 0,
        else => return 0,
    }
}

/// The worst spread seen across passes of a report.
pub const MaxSpread = struct {
    passes: usize,
    minimum: [measurement_count]f64,
    maximum: [measurement_count]f64,
    worst_percent: f64,

    pub fn init() MaxSpread {
        assert(measurement_count > 0);

        return .{
            .passes = 0,
            .minimum = undefined,
            .maximum = undefined,
            .worst_percent = 0,
        };
    }

    /// Folds one full pass into the spread. `report` must be fully measured.
    pub fn observe(self: *MaxSpread, report: *const bench.Report) void {
        assert(self.passes < repeat_max);
        assert(self.worst_percent >= 0);

        var visited: usize = 0;
        walk(bench.Report, report, self, &visited);
        // A mismatch means the report gained a measured number this walk does
        // not see, which would silently shrink the reported spread.
        assert(visited == measurement_count);
        self.passes += 1;
    }

    /// The worst spread, or null while a single pass could say nothing.
    pub fn percent_max(self: *const MaxSpread) ?f64 {
        assert(self.worst_percent >= 0);
        assert(std.math.isFinite(self.worst_percent));

        if (self.passes >= 2) {
            return self.worst_percent;
        } else {
            return null;
        }
    }

    /// Records the `index`th measured number of the pass being folded in.
    fn add(self: *MaxSpread, index: usize, value: f64) void {
        assert(index < measurement_count);
        assert(std.math.isFinite(value));
        assert(value >= 0);

        if (self.passes == 0) {
            self.minimum[index] = value;
            self.maximum[index] = value;
        } else {
            if (value < self.minimum[index]) {
                self.minimum[index] = value;
            }
            if (value > self.maximum[index]) {
                self.maximum[index] = value;
            }
        }
        // The spread is relative to the largest pass, so a number that starts
        // at zero yields 100 % rather than a division by zero.
        const peak = self.maximum[index];
        if (peak > 0) {
            const percent = (peak - self.minimum[index]) / peak * 100;
            if (percent > self.worst_percent) {
                self.worst_percent = percent;
            }
        }
    }
};

/// Visits every `f64` leaf of `value` in declaration order.
fn walk(
    comptime T: type,
    value: anytype,
    spread: *MaxSpread,
    visited: *usize,
) void {
    switch (@typeInfo(T)) {
        .@"struct" => {
            inline for (@typeInfo(T).@"struct".fields) |field| {
                walk(field.type, @field(value, field.name), spread, visited);
            }
        },
        .array => {
            var index: usize = 0;
            while (index < @typeInfo(T).array.len) : (index += 1) {
                walk(@typeInfo(T).array.child, value[index], spread, visited);
            }
        },
        .float => {
            if (@typeInfo(T).float.bits == 64) {
                spread.add(visited.*, @as(f64, value));
                visited.* += 1;
            }
        },
        else => {},
    }
}

test "one pass is not a measurement" {
    var spread = MaxSpread.init();
    var report = std.mem.zeroes(bench.Report);
    report.device_count = 1;
    report.h2d_bw_gib_s[0] = 12.5;

    spread.observe(&report);
    try std.testing.expectEqual(@as(usize, 1), spread.passes);
    try std.testing.expectEqual(@as(?f64, null), spread.percent_max());
}

test "identical passes spread nothing" {
    var spread = MaxSpread.init();
    var report = std.mem.zeroes(bench.Report);
    report.device_count = 1;
    report.h2d_bw_gib_s[0] = 12.5;
    report.p2p[0][0].latency_us = 3.0;

    spread.observe(&report);
    spread.observe(&report);
    try expect_percent(0, spread.percent_max());
}

test "the worst measured number sets the spread" {
    var spread = MaxSpread.init();
    var report = std.mem.zeroes(bench.Report);
    report.device_count = 1;
    report.h2d_bw_gib_s[0] = 100;
    report.d2d_hbm_gib_s[0] = 400;

    spread.observe(&report);
    report.h2d_bw_gib_s[0] = 99.9;
    spread.observe(&report);
    // A 0.1 % move on h2d, nothing anywhere else.
    try expect_percent(0.1, spread.percent_max());
}

test "numbers nested in arrays of structs are observed" {
    var spread = MaxSpread.init();
    var report = std.mem.zeroes(bench.Report);
    report.device_count = 4;
    report.p2p[1][0].bw_gib_s = 20;
    report.p2p[3][2].latency_us = 5;

    spread.observe(&report);
    report.p2p[3][2].latency_us = 4;
    spread.observe(&report);
    // 20 % on one ordered pair's latency, inside [4][4]P2pEntry.
    try expect_percent(20, spread.percent_max());
}

/// Asserts the reported spread, failing loudly if nothing was measured at all.
fn expect_percent(expected: f64, measured: ?f64) !void {
    try std.testing.expect(measured != null);
    try std.testing.expectApproxEqAbs(expected, measured.?, 1e-9);
}

test "the report holds the measurements the walk counts" {
    // A floor and a ceiling, not a pinned layout: the walk must cover every
    // number, and a count this small means the walk stopped finding them.
    try std.testing.expect(measurement_count > 40);
    try std.testing.expect(measurement_count < 1024);
}
