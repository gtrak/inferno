//! Safetensors reader: header parsing plus a read-only mapping of the weights.
//!
//! A safetensors file is an 8 byte little-endian header length, the JSON header
//! (`name -> {dtype, shape, data_offsets}` plus an optional `__metadata__`),
//! then the tensor bytes. The bytes are never copied: every tensor becomes
//! `{file_index, file_offset, byte_len}` plus a view into the mapping, which is
//! what makes the 132 GB checkpoint cost (almost) no host memory.
//!
//! The header is untrusted input, so every offset is checked twice: it must
//! land after the header, and the ranges must tile the data section exactly.
//! A header that overlaps itself or runs past the end of the file is a hard
//! error, not a clamp.

const std = @import("std");

const assert = std.debug.assert;

/// Deepest tensor rank in the checkpoint. The video patch embedding of the
/// vision tower is rank 5; the deepest text tensor is the rank 3 `conv1d`.
pub const rank_max: u32 = 5;

/// The largest real header is 5.3 MiB; anything near this bound is corrupt.
pub const header_bytes_max: u64 = 1 << 26;

/// Bytes of the leading little-endian header length.
pub const length_prefix_bytes: u64 = 8;

/// Key of the optional free-form metadata object.
pub const metadata_key = "__metadata__";

/// Element formats that actually occur; anything else is rejected.
pub const Dtype = enum {
    bf16,
    f32,
    f8_e4m3,
    u8,
    i64,

    /// Bytes per element. The 4-bit expert weights are stored as `u8` pairs,
    /// never as nibbles, so this is also the packed byte size.
    pub fn elementBytes(self: Dtype) u32 {
        assert(@sizeOf(u32) == 4);
        const bytes: u32 = switch (self) {
            .bf16 => 2,
            .f32 => 4,
            .f8_e4m3 => 1,
            .u8 => 1,
            .i64 => 8,
        };
        assert(bytes == 1 or bytes == 2 or bytes == 4 or bytes == 8);
        return bytes;
    }

    /// Maps the checkpoint's dtype tag onto `Dtype`; unknown tags are fatal.
    pub fn fromName(name: []const u8) !Dtype {
        assert(name.len > 0);
        assert(name.len <= 16);

        const Known = struct { tag: []const u8, dtype: Dtype };
        const known = [_]Known{
            .{ .tag = "BF16", .dtype = .bf16 },
            .{ .tag = "F32", .dtype = .f32 },
            .{ .tag = "F8_E4M3", .dtype = .f8_e4m3 },
            .{ .tag = "U8", .dtype = .u8 },
            .{ .tag = "I64", .dtype = .i64 },
        };
        for (known) |entry| {
            if (std.mem.eql(u8, name, entry.tag)) {
                return entry.dtype;
            } else {
                continue;
            }
        }
        std.debug.print("safetensors: unsupported dtype {s}\n", .{name});
        return error.unsupported_dtype;
    }
};

/// Tensor extents, at most `rank_max` of them. A scalar has rank zero.
pub const Dims = struct {
    values: [rank_max]u32 = [_]u32{0} ** rank_max,
    rank: u32 = 0,

    /// Product of the extents; a rank zero tensor holds exactly one element.
    pub fn elementCount(self: Dims) u64 {
        assert(self.rank <= rank_max);
        assert(@sizeOf(u64) == 8);

        var total: u64 = 1;
        for (self.values[0..self.rank]) |extent| {
            assert(extent > 0);
            total *= extent;
        }
        assert(total > 0);
        return total;
    }

    /// True when `self` names exactly `extents`, used to validate roles.
    pub fn equals(self: Dims, extents: []const u32) bool {
        assert(self.rank <= rank_max);
        assert(extents.len <= rank_max);

        if (self.rank != extents.len) {
            return false;
        } else {
            for (self.values[0..self.rank], extents) |extent, expected| {
                if (extent != expected) {
                    return false;
                } else {
                    continue;
                }
            }
            return true;
        }
    }
};

/// One tensor as described by the header, with no bytes attached.
pub const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    dims: Dims,
    /// Absolute byte offset of the first element inside its file.
    file_offset: u64,
    byte_len: u64,
    /// Index into the caller's file list, not a device number.
    file_index: u32,

    pub fn elementCount(self: *const Tensor) u64 {
        assert(self.byte_len > 0);
        assert(self.dims.rank <= rank_max);

        const elements = self.dims.elementCount();
        assert(elements * self.dtype.elementBytes() == self.byte_len);
        return elements;
    }
};

/// Parsed header: the tensors it describes and where its data section ends.
pub const Header = struct {
    tensors: []Tensor,
    has_metadata: bool,
    /// One past the last tensor byte, according to the header.
    data_end: u64,

    /// False while the file is still being written: a partially downloaded
    /// shard has a complete header and an incomplete data section, and that is
    /// the only situation where `data_end` may exceed the mapping.
    pub fn dataComplete(self: *const Header, file: *const File) bool {
        assert(self.data_end >= file.dataStart());
        assert(self.tensors.len > 0);

        return self.data_end <= file.bytes.len;
    }
};

/// A mapped safetensors file. `map` owns the mapping; `bytes` views it.
pub const File = struct {
    path: []const u8,
    bytes: []const u8,
    map: ?[]align(std.heap.page_size_min) u8,
    /// Length of the JSON header, excluding the 8 byte length prefix.
    header_len: u64,

    /// Wraps bytes that are already in memory (golden fixtures and tests).
    pub fn initFromBytes(path: []const u8, bytes: []const u8) !File {
        assert(path.len > 0);
        assert(bytes.len > length_prefix_bytes);

        const file = File{
            .path = path,
            .bytes = bytes,
            .map = null,
            .header_len = try readHeaderLength(bytes),
        };
        assert(file.dataStart() <= bytes.len);
        return file;
    }

    /// Maps `path` read-only. A shard that is still downloading maps short and
    /// is reported through `Header.dataComplete`.
    pub fn open(path: []const u8) !File {
        assert(path.len > 0);
        assert(length_prefix_bytes == 8);

        const handle = try std.fs.cwd().openFile(path, .{});
        defer handle.close();
        const size = (try handle.stat()).size;
        if (size <= length_prefix_bytes) {
            return error.file_too_short;
        } else {
            assert(size > length_prefix_bytes);
        }

        const map = try std.posix.mmap(
            null,
            @intCast(size),
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            handle.handle,
            0,
        );
        errdefer std.posix.munmap(map);

        const file = File{
            .path = path,
            .bytes = map,
            .map = map,
            .header_len = try readHeaderLength(map),
        };
        assert(file.map != null);
        return file;
    }

    pub fn close(self: *File) void {
        assert(self.bytes.len > length_prefix_bytes);
        assert(self.map == null or self.map.?.len == self.bytes.len);

        if (self.map) |map| {
            std.posix.munmap(map);
        } else {
            // Bytes borrowed from the caller stay the caller's to free.
        }
        self.map = null;
        self.bytes = "";
    }

    /// Offset of the first tensor byte: the length prefix plus the header.
    pub fn dataStart(self: *const File) u64 {
        assert(self.header_len > 0);
        assert(self.header_len <= header_bytes_max);

        const start = length_prefix_bytes + self.header_len;
        assert(start <= self.bytes.len);
        return start;
    }

    pub fn headerBytes(self: *const File) []const u8 {
        assert(self.header_len > 0);
        assert(self.bytes.len >= length_prefix_bytes + self.header_len);

        return self.bytes[length_prefix_bytes..][0..@intCast(self.header_len)];
    }

    /// View of one tensor's bytes; the mapping is read-only by construction.
    pub fn tensorBytes(self: *const File, tensor: *const Tensor) []const u8 {
        assert(tensor.byte_len > 0);
        assert(tensor.file_offset >= self.dataStart());

        const end = tensor.file_offset + tensor.byte_len;
        assert(end <= self.bytes.len);
        return self.bytes[@intCast(tensor.file_offset)..@intCast(end)];
    }
};

/// Reads the leading little-endian u64 header length.
fn readHeaderLength(bytes: []const u8) !u64 {
    assert(bytes.len > length_prefix_bytes);
    assert(length_prefix_bytes == 8);

    const length = std.mem.readInt(u64, bytes[0..length_prefix_bytes], .little);
    if (length == 0) {
        return error.empty_header;
    } else {
        assert(length > 0);
    }
    if (length > header_bytes_max) {
        return error.header_too_long;
    } else {
        assert(length <= header_bytes_max);
    }
    if (length_prefix_bytes + length > bytes.len) {
        return error.truncated_header;
    } else {
        assert(length_prefix_bytes + length <= bytes.len);
    }
    return length;
}

/// Parses `file`'s header. `allocator` must outlive the returned tensors; it
/// is an arena in practice, because the header text is copied per string.
pub fn parseHeader(
    allocator: std.mem.Allocator,
    file: *const File,
    file_index: u32,
) !Header {
    assert(file.header_len > 0);
    assert(file.bytes.len > length_prefix_bytes);

    const document = std.json.parseFromSliceLeaky(
        std.json.Value,
        allocator,
        file.headerBytes(),
        .{ .allocate = .alloc_always },
    ) catch |err| {
        std.debug.print("safetensors: {s}: header is not JSON: {s}\n", .{
            file.path,
            @errorName(err),
        });
        return error.malformed_header;
    };
    if (document != .object) {
        return error.malformed_header;
    } else {
        assert(document == .object);
    }

    var tensors = std.ArrayList(Tensor).init(allocator);
    var has_metadata = false;
    var iterator = document.object.iterator();
    while (iterator.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.eql(u8, name, metadata_key)) {
            has_metadata = try isMetadataObject(entry.value_ptr.*);
        } else {
            try tensors.append(try parseTensor(
                name,
                entry.value_ptr.*,
                file_index,
                file.dataStart(),
            ));
        }
    }
    assert(tensors.items.len + @intFromBool(has_metadata) == document.object.count());
    if (tensors.items.len == 0) {
        return error.empty_header;
    } else {
        assert(tensors.items.len > 0);
    }

    const slice = try tensors.toOwnedSlice();
    const data_end = try validatePlacement(allocator, slice, file);
    const header = Header{
        .tensors = slice,
        .has_metadata = has_metadata,
        .data_end = data_end,
    };
    assert(header.tensors.len > 0);
    return header;
}

/// Metadata only has to be an object of strings; its contents are unused.
fn isMetadataObject(value: std.json.Value) !bool {
    assert(metadata_key.len > 0);
    assert(value != .null);

    if (value != .object) {
        return error.malformed_header;
    } else {
        assert(value == .object);
    }
    var iterator = value.object.iterator();
    while (iterator.next()) |entry| {
        if (entry.value_ptr.* != .string) {
            return error.malformed_header;
        } else {
            continue;
        }
    }
    return true;
}

/// Reads one `{dtype, shape, data_offsets}` object. The header's offsets are
/// relative to the first tensor byte, so `data_start` turns them into file
/// offsets.
fn parseTensor(
    name: []const u8,
    value: std.json.Value,
    file_index: u32,
    data_start: u64,
) !Tensor {
    assert(name.len > 0);
    assert(file_index < std.math.maxInt(u32));

    if (value != .object) {
        std.debug.print("safetensors: {s}: tensor entry is not an object\n", .{name});
        return error.malformed_header;
    } else {
        assert(value == .object);
    }

    const dtype_field = value.object.get("dtype") orelse
        return error.malformed_header;
    const shape_field = value.object.get("shape") orelse
        return error.malformed_header;
    const offsets_field = value.object.get("data_offsets") orelse
        return error.malformed_header;
    if (dtype_field != .string) {
        return error.malformed_header;
    } else {
        assert(dtype_field == .string);
    }

    const dtype = try Dtype.fromName(dtype_field.string);
    const dims = try parseDims(name, shape_field);
    const offsets = try parseOffsets(shape_field, offsets_field);
    const tensor = Tensor{
        .name = name,
        .dtype = dtype,
        .dims = dims,
        .file_offset = data_start + offsets.start,
        .byte_len = offsets.end - offsets.start,
        .file_index = file_index,
    };
    if (tensor.byte_len != dims.elementCount() * dtype.elementBytes()) {
        std.debug.print("safetensors: {s}: {d} bytes for {d} {s} elements\n", .{
            name,
            tensor.byte_len,
            dims.elementCount(),
            @tagName(dtype),
        });
        return error.element_count_mismatch;
    } else {
        assert(tensor.byte_len > 0);
    }
    assert(tensor.file_index == file_index);
    assert(tensor.file_offset >= data_start);
    return tensor;
}

/// Reads the `shape` array into fixed extents; rank zero is a scalar.
fn parseDims(name: []const u8, shape_field: std.json.Value) !Dims {
    assert(name.len > 0);
    assert(shape_field != .null);

    if (shape_field != .array) {
        return error.malformed_header;
    } else {
        assert(shape_field == .array);
    }
    const items = shape_field.array.items;
    if (items.len > rank_max) {
        std.debug.print("safetensors: {s}: rank {d} is too deep\n", .{
            name,
            items.len,
        });
        return error.rank_too_deep;
    } else {
        assert(items.len <= rank_max);
    }

    var dims = Dims{ .rank = @intCast(items.len) };
    for (items, 0..) |item, position| {
        if (item != .integer) {
            return error.malformed_header;
        } else {
            assert(item == .integer);
        }
        if (item.integer <= 0) {
            return error.malformed_header;
        } else {
            assert(item.integer > 0);
        }
        if (item.integer > std.math.maxInt(u32)) {
            return error.malformed_header;
        } else {
            dims.values[position] = @intCast(item.integer);
        }
    }
    assert(dims.rank == items.len);
    return dims;
}

/// Byte range of a tensor inside its file, as `[start, end)`.
const Range = struct { start: u64, end: u64 };

/// Reads `data_offsets`, which must be two ascending non-negative integers.
fn parseOffsets(shape_field: std.json.Value, offsets_field: std.json.Value) !Range {
    assert(offsets_field != .null);
    assert(shape_field != .null);

    if (offsets_field != .array) {
        return error.malformed_header;
    } else {
        assert(offsets_field == .array);
    }
    const items = offsets_field.array.items;
    if (items.len != 2) {
        return error.malformed_header;
    } else {
        assert(items.len == 2);
    }
    if (items[0] != .integer) {
        return error.malformed_header;
    } else {
        assert(items[0] == .integer);
    }
    if (items[1] != .integer) {
        return error.malformed_header;
    } else {
        assert(items[1] == .integer);
    }
    if (items[0].integer < 0) {
        return error.malformed_header;
    } else {
        assert(items[0].integer >= 0);
    }
    if (items[1].integer <= items[0].integer) {
        return error.malformed_header;
    } else {
        assert(items[1].integer > items[0].integer);
    }
    const range = Range{
        .start = @intCast(items[0].integer),
        .end = @intCast(items[1].integer),
    };
    assert(range.end > range.start);
    return range;
}

/// Tensor order in the data section must be a permutation of the tensors.
fn lessByOffset(tensors: []const Tensor, lhs: u32, rhs: u32) bool {
    assert(lhs < tensors.len);
    assert(rhs < tensors.len);

    return tensors[lhs].file_offset < tensors[rhs].file_offset;
}

/// Proves the data section is tiled exactly once, and returns its end.
///
/// The data section starts right after the header, so the first tensor must
/// start there and every tensor must begin where the previous one ended. The
/// resulting end is compared with the file size by the caller: shorter than the
/// file means trailing garbage, longer means a shard that is still downloading.
fn validatePlacement(
    allocator: std.mem.Allocator,
    tensors: []const Tensor,
    file: *const File,
) !u64 {
    assert(tensors.len > 0);
    assert(file.header_len > 0);

    const order = try allocator.alloc(u32, tensors.len);
    for (order, 0..) |*position, index| {
        position.* = @intCast(index);
    }
    std.mem.sort(u32, order, tensors, lessByOffset);

    var cursor = file.dataStart();
    for (order) |index| {
        const tensor = tensors[index];
        if (tensor.file_offset != cursor) {
            std.debug.print("safetensors: {s}: {s} starts at {d}, expected {d}\n", .{
                file.path,
                tensor.name,
                tensor.file_offset,
                cursor,
            });
            return error.overlapping_offsets;
        } else {
            assert(tensor.file_offset == cursor);
        }
        cursor += tensor.byte_len;
        assert(cursor > tensor.file_offset);
    }
    if (cursor < file.bytes.len) {
        std.debug.print("safetensors: {s}: {d} trailing bytes\n", .{
            file.path,
            file.bytes.len - cursor,
        });
        return error.trailing_bytes;
    } else {
        assert(cursor >= file.bytes.len);
    }
    assert(cursor >= file.dataStart());
    return cursor;
}

/// One tensor of a synthetic test file.
const TestSpec = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const u32,
    byte_len: u64,
};

/// Bytes of the synthetic payload, so views can be compared against a pattern.
fn payloadByte(tensor_index: u32, position: u64) u8 {
    assert(tensor_index < 16);
    assert(position < 1024);

    const value: u8 = @truncate(0x40 +% tensor_index *% 7 +% position);
    assert(value >= 0x40);
    return value;
}

/// Builds a whole safetensors image around `header_text` and a zero payload.
fn composeImage(
    allocator: std.mem.Allocator,
    header_text: []const u8,
    payload_bytes: u64,
) ![]u8 {
    assert(header_text.len > 0);
    assert(payload_bytes > 0);

    const total = length_prefix_bytes + header_text.len + payload_bytes;
    const image = try allocator.alloc(u8, @intCast(total));
    std.mem.writeInt(u64, image[0..length_prefix_bytes], header_text.len, .little);
    @memcpy(image[length_prefix_bytes..][0..header_text.len], header_text);
    @memset(image[@intCast(length_prefix_bytes + header_text.len)..][0..], 0);
    assert(image.len == total);
    return image;
}

/// Builds a file whose tensors carry a recognizable per tensor payload.
fn composeFile(
    allocator: std.mem.Allocator,
    specs: []const TestSpec,
    metadata_json: []const u8,
) ![]u8 {
    assert(specs.len > 0);
    assert(metadata_json.len <= 128);

    var header = std.ArrayList(u8).init(allocator);
    try header.append('{');
    var payload_bytes: u64 = 0;
    for (specs, 0..) |spec, index| {
        if (index > 0) {
            try header.append(',');
        } else {
            assert(index == 0);
        }
        try header.writer().print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{
            spec.name,
            spec.dtype,
        });
        for (spec.shape, 0..) |extent, position| {
            if (position > 0) {
                try header.append(',');
            } else {
                assert(position == 0);
            }
            try header.writer().print("{d}", .{extent});
        }
        try header.writer().print("],\"data_offsets\":[{d},{d}]}}", .{
            payload_bytes,
            payload_bytes + spec.byte_len,
        });
        payload_bytes += spec.byte_len;
    }
    if (metadata_json.len > 0) {
        try header.append(',');
        try header.writer().print("\"{s}\":", .{metadata_key});
        try header.appendSlice(metadata_json);
    }
    try header.append('}');

    const image = try composeImage(allocator, header.items, payload_bytes);
    const data_start = length_prefix_bytes + header.items.len;
    var cursor = data_start;
    for (specs, 0..) |spec, index| {
        for (0..@intCast(spec.byte_len)) |position| {
            image[@intCast(cursor + position)] = payloadByte(@intCast(index), position);
        }
        cursor += spec.byte_len;
    }
    assert(cursor == image.len);
    return image;
}

test "a two tensor file maps both tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{
        .{ .name = "weight", .dtype = "BF16", .shape = &.{ 2, 3 }, .byte_len = 12 },
        .{ .name = "scale_2", .dtype = "F32", .shape = &.{}, .byte_len = 4 },
    };
    const image = try composeFile(allocator, specs[0..], "");
    var file = try File.initFromBytes("synthetic.safetensors", image);
    defer file.close();
    const header = try parseHeader(allocator, &file, 3);

    try std.testing.expectEqual(@as(usize, 2), header.tensors.len);
    try std.testing.expect(!header.has_metadata);
    try std.testing.expectEqualStrings("weight", header.tensors[0].name);
    try std.testing.expectEqual(Dtype.bf16, header.tensors[0].dtype);
    try std.testing.expect(header.tensors[0].dims.equals(&.{ 2, 3 }));
    try std.testing.expectEqual(@as(u64, 6), header.tensors[0].elementCount());
    try std.testing.expectEqual(@as(u32, 3), header.tensors[0].file_index);
    try std.testing.expect(header.tensors[1].dims.equals(&.{}));
    try std.testing.expectEqual(@as(u64, 1), header.tensors[1].elementCount());
    try std.testing.expectEqual(@as(u64, image.len), header.data_end);
    try std.testing.expect(header.dataComplete(&file));
    try std.testing.expectEqualSlices(
        u8,
        image[@intCast(header.tensors[0].file_offset)..][0..12],
        file.tensorBytes(&header.tensors[0]),
    );
    try std.testing.expectEqual(@as(u8, 0x40), file.tensorBytes(&header.tensors[0])[0]);
    try std.testing.expectEqual(@as(u8, 0x47), file.tensorBytes(&header.tensors[1])[0]);
}

test "a metadata object is not a tensor" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{
        .{ .name = "weight", .dtype = "F32", .shape = &.{2}, .byte_len = 8 },
    };
    const image = try composeFile(allocator, specs[0..], "{\"format\":\"pt\"}");
    var file = try File.initFromBytes("synthetic.safetensors", image);
    defer file.close();
    const header = try parseHeader(allocator, &file, 0);

    try std.testing.expect(header.has_metadata);
    try std.testing.expectEqual(@as(usize, 1), header.tensors.len);
}

test "an unsupported dtype is a hard error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{
        .{ .name = "weight", .dtype = "F16", .shape = &.{2}, .byte_len = 4 },
    };
    const image = try composeFile(allocator, specs[0..], "");
    var file = try File.initFromBytes("synthetic.safetensors", image);
    defer file.close();

    try std.testing.expectError(error.unsupported_dtype, parseHeader(allocator, &file, 0));
}

test "a shape that contradicts the byte length is rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{
        .{ .name = "weight", .dtype = "BF16", .shape = &.{ 2, 3 }, .byte_len = 11 },
    };
    const image = try composeFile(allocator, specs[0..], "");
    var file = try File.initFromBytes("synthetic.safetensors", image);
    defer file.close();

    try std.testing.expectError(
        error.element_count_mismatch,
        parseHeader(allocator, &file, 0),
    );
}

test "overlapping data offsets are rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const header_text =
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}," ++
        "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}";
    const image = try composeImage(allocator, header_text, 16);
    var file = try File.initFromBytes("synthetic.safetensors", image);
    defer file.close();

    try std.testing.expectError(
        error.overlapping_offsets,
        parseHeader(allocator, &file, 0),
    );
}

test "a shard that is still downloading keeps a complete header" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{
        .{ .name = "a", .dtype = "U8", .shape = &.{16}, .byte_len = 16 },
        .{ .name = "b", .dtype = "U8", .shape = &.{16}, .byte_len = 16 },
    };
    const image = try composeFile(allocator, specs[0..], "");
    // The last four bytes of `b` have not arrived yet.
    var file = try File.initFromBytes("partial.safetensors", image[0 .. image.len - 4]);
    defer file.close();
    const header = try parseHeader(allocator, &file, 0);

    try std.testing.expectEqual(@as(usize, 2), header.tensors.len);
    try std.testing.expectEqual(@as(u64, image.len), header.data_end);
    try std.testing.expect(!header.dataComplete(&file));
}

test "trailing bytes after the last tensor are rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{
        .{ .name = "a", .dtype = "U8", .shape = &.{4}, .byte_len = 4 },
    };
    const composed = try composeFile(allocator, specs[0..], "");
    const image = try allocator.dupe(u8, composed);
    const longer = try allocator.alloc(u8, image.len + 1);
    @memcpy(longer[0..image.len], image);
    longer[image.len] = 0;
    var file = try File.initFromBytes("long.safetensors", longer);
    defer file.close();

    try std.testing.expectError(error.trailing_bytes, parseHeader(allocator, &file, 0));
}

test "a header length past the end of the file is rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const image = try allocator.alloc(u8, 16);
    std.mem.writeInt(u64, image[0..8], 4096, .little);
    @memset(image[8..], 0);

    try std.testing.expectError(error.truncated_header, File.initFromBytes("short", image));
}

test "a rank six shape is rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const specs = [_]TestSpec{.{
        .name = "deep",
        .dtype = "U8",
        .shape = &.{ 1, 1, 1, 1, 1, 4 },
        .byte_len = 4,
    }};
    const image = try composeFile(allocator, specs[0..], "");
    var file = try File.initFromBytes("synthetic.safetensors", image);
    defer file.close();

    try std.testing.expectError(error.rank_too_deep, parseHeader(allocator, &file, 0));
}
