//! PCIe link state, read from sysfs.
//!
//! The CUDA driver API has no link-speed attribute and NVML on this host
//! enumerates zero devices, so the probe audits its own host<->device numbers
//! against the kernel's view in `/sys/bus/pci/devices/<bus id>/`. A link drops
//! to a low-power generation while idle, so the caller reads it while a copy
//! keeps the link at full speed - an idle reading says Gen1 and would make the
//! achieved percentage meaningless.

const std = @import("std");

const assert = std.debug.assert;

/// "0000:05:00.0" plus slack; `cuDeviceGetPCIBusId` documents a 16 byte buffer.
pub const pci_bus_id_bytes_max: u32 = 16;

/// Longer than any sysfs link line: "32.0 GT/s PCIe\n".
const link_text_bytes_max: u32 = 64;

pub const LinkError = error{
    link_state_unavailable,
};

/// Negotiated link state: a generation (1-5) and a lane count.
pub const LinkState = struct {
    generation: u32,
    width: u32,
};

/// One PCIe generation and its raw transfer rate.
const LinkSpeed = struct {
    gigatransfers_per_second: f64,
    generation: u32,
};

/// Raw rates as the kernel reports them, oldest first.
const link_speeds = [_]LinkSpeed{
    .{ .gigatransfers_per_second = 2.5, .generation = 1 },
    .{ .gigatransfers_per_second = 5.0, .generation = 2 },
    .{ .gigatransfers_per_second = 8.0, .generation = 3 },
    .{ .gigatransfers_per_second = 16.0, .generation = 4 },
    .{ .gigatransfers_per_second = 32.0, .generation = 5 },
};

/// Reads the negotiated generation and width of one device.
pub fn link_state(pci_bus_id: []const u8) LinkError!LinkState {
    assert(pci_bus_id.len > 0);
    assert(pci_bus_id.len < pci_bus_id_bytes_max);

    var speed_buffer: [link_text_bytes_max]u8 = undefined;
    const speed_text = try read_attribute(
        pci_bus_id,
        "current_link_speed",
        speed_buffer[0..],
    );
    var width_buffer: [link_text_bytes_max]u8 = undefined;
    const width_text = try read_attribute(
        pci_bus_id,
        "current_link_width",
        width_buffer[0..],
    );

    const generation = try generation_from_text(speed_text);
    const width = try width_from_text(width_text);
    assert(width > 0);
    return .{ .generation = generation, .width = width };
}

/// Raw per-lane bandwidth of a PCIe generation: transfer rate, encoding and
/// eight bits per byte. Gen1/2 use 8b/10b, Gen3 and later 128b/130b.
pub fn lane_gib_per_second(generation: u32) LinkError!f64 {
    assert(generation <= 5);
    assert(@sizeOf(f64) == 8);

    return switch (generation) {
        1 => 0.233, // 2.5 GT/s, 8b/10b
        2 => 0.466, // 5 GT/s, 8b/10b
        3 => 0.917, // 8 GT/s, 128b/130b
        4 => 1.834, // 16 GT/s, 128b/130b
        5 => 3.667, // 32 GT/s, 128b/130b
        else => LinkError.link_state_unavailable,
    };
}

/// Raw bandwidth of a link in GiB/s, from its generation and lane count.
pub fn link_gib_per_second(state: LinkState) LinkError!f64 {
    assert(state.width > 0);
    assert(state.width <= 32);

    const lane = try lane_gib_per_second(state.generation);
    const lanes: f64 = @floatFromInt(state.width);
    assert(lane > 0);
    return lane * lanes;
}

/// Reads one attribute file and copies its trimmed text into `buffer`, which
/// must be at least `link_text_bytes_max` bytes.
fn read_attribute(
    pci_bus_id: []const u8,
    attribute_name: []const u8,
    buffer: []u8,
) LinkError![]const u8 {
    assert(attribute_name.len > 0);
    assert(buffer.len >= link_text_bytes_max);

    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(
        path_buffer[0..],
        "/sys/bus/pci/devices/{s}/{s}",
        .{ pci_bus_id, attribute_name },
    ) catch return LinkError.link_state_unavailable;
    const file = std.fs.openFileAbsolute(path, .{}) catch {
        return LinkError.link_state_unavailable;
    };
    defer file.close();

    var contents: [link_text_bytes_max]u8 = undefined;
    const length = file.readAll(contents[0..]) catch {
        return LinkError.link_state_unavailable;
    };
    const text = std.mem.trim(u8, contents[0..length], " \t\r\n");
    if (text.len == 0) {
        return LinkError.link_state_unavailable;
    } else {
        assert(text.len <= link_text_bytes_max);
    }
    @memcpy(buffer[0..text.len], text);
    return buffer[0..text.len];
}

/// Maps "16.0 GT/s PCIe" onto generation 4.
fn generation_from_text(text: []const u8) LinkError!u32 {
    assert(text.len > 0);
    assert(link_speeds.len > 0);

    var iterator = std.mem.tokenizeAny(u8, text, " \t");
    const number = iterator.next() orelse {
        return LinkError.link_state_unavailable;
    };
    const rate = std.fmt.parseFloat(f64, number) catch {
        return LinkError.link_state_unavailable;
    };

    for (link_speeds) |speed| {
        const matches = std.math.approxEqAbs(
            f64,
            rate,
            speed.gigatransfers_per_second,
            0.001,
        );
        if (matches) {
            assert(speed.generation >= 1);
            return speed.generation;
        } else {
            // Keep looking: the kernel may report a generation we do not know.
        }
    }
    return LinkError.link_state_unavailable;
}

/// Parses "4" or "4\n" into a lane count.
fn width_from_text(text: []const u8) LinkError!u32 {
    assert(text.len > 0);
    assert(text.len <= link_text_bytes_max);

    const width = std.fmt.parseInt(u32, text, 10) catch {
        return LinkError.link_state_unavailable;
    };
    if (width == 0) {
        return LinkError.link_state_unavailable;
    } else {
        assert(width > 0);
    }
    return width;
}

test "a Gen4 x4 link is 7.34 GiB/s raw" {
    const state = LinkState{ .generation = 4, .width = 4 };
    const bandwidth = try link_gib_per_second(state);
    try std.testing.expectApproxEqAbs(@as(f64, 7.336), bandwidth, 0.01);
}

test "link_gib_per_second scales with width" {
    const narrow = try link_gib_per_second(.{ .generation = 4, .width = 4 });
    const wide = try link_gib_per_second(.{ .generation = 4, .width = 16 });
    try std.testing.expectApproxEqAbs(wide, narrow * 4.0, 0.01);
}

test "an unknown generation is an error, not a zero" {
    try std.testing.expectError(
        LinkError.link_state_unavailable,
        lane_gib_per_second(0),
    );
}

test "generation_from_text reads the kernel's link speed lines" {
    try std.testing.expectEqual(
        @as(u32, 4),
        try generation_from_text("16.0 GT/s PCIe"),
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        try generation_from_text("2.5 GT/s PCIe"),
    );
    try std.testing.expectError(
        LinkError.link_state_unavailable,
        generation_from_text("nonsense"),
    );
}

test "width_from_text reads the kernel's lane count" {
    try std.testing.expectEqual(@as(u32, 4), try width_from_text("4"));
    try std.testing.expectError(
        LinkError.link_state_unavailable,
        width_from_text("0"),
    );
}

test "this host's first GPU reports a link state" {
    // The first GPU's bus id is fixed for this machine's board layout.
    const state = link_state("0000:05:00.0") catch |err| switch (err) {
        // A machine without this board layout simply has no such device.
        error.link_state_unavailable => return error.SkipZigTest,
    };
    try std.testing.expect(state.width > 0);
    try std.testing.expect(state.generation >= 1);
    try std.testing.expect(state.generation <= 5);
}
