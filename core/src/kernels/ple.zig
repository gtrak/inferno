//! Per-Layer Embedding: the n-gram id derivation, the 16-row table gather, and
//! the launches that turn the gathered rows into the block output.
//!
//! `src/kernels/ple.cu` holds the device code; this file owns the host half:
//!
//! * the **ids** (`IdStream`) — splitmix64 multipliers, the nth-prime head
//!   vocabularies and offsets, and the EOS-aware shift of the reference's
//!   `_shift_right_ignore_eos`. This is exact integer arithmetic: the ids index
//!   a 320M-row table, so a one-off id reads a different row, and the tests pin
//!   every id to `src/ref/ple.zig` for the golden input, for an input with an
//!   EOS in the middle, and for the decode steps after it.
//! * the **gather** (`Table` + `GatherPool`) — 16 scattered rows of 160 B per
//!   token out of the 51.2 GB table. A serial gather is a per-token stall on
//!   NVMe (measured below), so the rows of one token are read concurrently by a
//!   pool of workers and land in one pinned staging buffer, which is then one
//!   H2D copy.
//! * the **block** (`Block`) — cast/scale, the two projections, the signed-sqrt
//!   gate, `norm_conv`, the dilated depthwise conv with its ring-buffer state,
//!   and the injection into the hyper-connection residual.
//!
//! Tolerances: the whole block is f32 in the reference and in the HF golden
//! trace, so the gate is the issue's `max|err| / max(1e-6, max|golden|) <= 2e-5`
//! scaled error, per stage. That is the f32 output format's own precision, not a
//! blanket `1e-6`; if the engine stores the block output in bf16 on the way into
//! the residual, the tolerance at that point becomes 0.5 bf16 ulp instead, and
//! nothing in this file rounds to bf16.

const std = @import("std");

const cuda_context = @import("../hw/cuda_context.zig");
const cuda_driver = @import("../hw/cuda_driver.zig");
const generated_cubins = @import("generated_cubins");
const kernel = @import("../hw/kernel.zig");
const quant = @import("../weights/quant.zig");
const reference_attention = @import("../ref/attention.zig");
const reference_fixtures = @import("../ref/fixtures.zig");
const reference_layer = @import("../ref/layer.zig");
const reference_model = @import("../ref/model.zig");
const reference_ple = @import("../ref/ple.zig");
const ops = @import("../ref/ops.zig");
const safetensors = @import("../weights/safetensors.zig");

const assert = std.debug.assert;

/// Cubin compiled from `src/kernels/ple.cu`.
const cubin_name = "ple";

/// Shards the checkpoint may split the n-gram table into.
pub const shards_max: u32 = 128;

/// n-gram heads one PLE layer may derive: `(ngram_size - 1) * heads_per_ngram`.
pub const heads_max: u32 = 32;

/// Largest n-gram order the derivation supports.
pub const ngram_max: u32 = 8;

/// Largest `ple_conv_kernel_size` the derivation supports.
pub const conv_kernel_max: u32 = 8;

/// Largest conv state length the device ring update keeps in registers; it
/// mirrors `ple_state_max` in the device file.
pub const state_len_max: u32 = 64;

/// Tokens one `Block` can be sized for.
pub const tokens_max: u32 = 4096;

/// Threads per block the kernels launch; mirrors `ple_threads` in the device
/// file.
const conv_threads: u32 = 256;

/// Bytes one device allocation rounds up to (`cuMemAlloc` granularity).
const allocation_granularity: u64 = 4096;

/// The splitmix64 constants of the reference.
const splitmix_gamma: u64 = 0x9E3779B97F4A7C15;
const splitmix_m1: u64 = 0xBF58476D1CE4E5B9;
const splitmix_m2: u64 = 0x94D049BB133111EB;
const prime_seed_stride: u64 = 10007;

// ---------------------------------------------------------------------------
// n-gram ids.
// ---------------------------------------------------------------------------

/// The checkpoint's n-gram parameters (`config.json`, `text_config`).
pub const HashSpec = struct {
    /// `ngram_size`: the longest n-gram order hashed.
    ngram_size: u32,
    /// `heads_per_ngram`: heads per order (order 2 and up).
    heads_per_ngram: u32,
    /// `eos_token_id`: the boundary the history restarts at.
    eos_token_id: i64,
    /// `seed`, mixed into the layer multipliers.
    seed: u64,
    /// `vocab_size`: the unigram vocabulary bounds the multiplier magnitude.
    vocab_size: u64,
    /// `ngram_vocab_size_base`: the first head's vocabulary starts after this.
    ngram_vocab_size_base: u64,
    /// `make_ngram_vocab_size_divisible_by`: the table rows are padded to it.
    make_ngram_vocab_size_divisible_by: u32,
    /// `split_ngram_parts`: how many shards the table file is split into.
    split_ngram_parts: u32,
    /// Index of this layer inside `ple_layer_ids`, which offsets the heads.
    ple_layer_index: u32,

    /// `(ngram_size - 1) * heads_per_ngram`.
    pub fn headCount(self: HashSpec) u32 {
        assert(self.ngram_size >= 2);
        assert(self.heads_per_ngram >= 1);

        return (self.ngram_size - 1) * self.heads_per_ngram;
    }

    /// Rejects parameters the derivation cannot honour.
    pub fn validate(self: HashSpec) !void {
        assert(self.ngram_size >= 2);
        assert(self.make_ngram_vocab_size_divisible_by > 0);

        if (self.ngram_size > ngram_max) return error.NgramSizeTooLarge;
        if (self.headCount() > heads_max) return error.TooManyNgramHeads;
        if (self.ngram_vocab_size_base < 3) return error.BadNgramVocabBase;
        if (self.split_ngram_parts == 0) return error.NoTableShards;
        if (self.split_ngram_parts > shards_max) return error.TooManyTableShards;
    }
};

/// `nvidia/Qwen3.8-Flash-Next-NVFP4`, text layer 1 (`ple_layer_ids = [2]`).
///
/// The derived head table must cover exactly the 128 x 2,500,012 rows of the
/// 51.2 GB `model-fp8-mtp-ple.safetensors` table; the drift guard below pins
/// that.
pub const real_spec: HashSpec = .{
    .ngram_size = 3,
    .heads_per_ngram = 8,
    .eos_token_id = 248044,
    .seed = 1234,
    .vocab_size = 248320,
    .ngram_vocab_size_base = 20_000_000,
    .make_ngram_vocab_size_divisible_by = 128,
    .split_ngram_parts = 128,
    .ple_layer_index = 0,
};

/// One PLE layer's derived head vocabulary: where each head's ids land in the
/// flat table, and the multipliers the hash of each order uses.
pub const HeadTable = struct {
    /// `[head_count]`: the vocabulary size of each head.
    sizes: [heads_max]i64 = [_]i64{0} ** heads_max,
    /// `[head_count]`: the first table row of each head.
    offsets: [heads_max]i64 = [_]i64{0} ** heads_max,
    /// `[ngram_size]`: one odd multiplier per order.
    multipliers: [ngram_max]i64 = [_]i64{0} ** ngram_max,
    head_count: u32 = 0,
    heads_per_ngram: u32 = 0,
    ngram_size: u32 = 0,
    /// Rows the heads address, i.e. the sum of the head sizes.
    total_vocab: u64 = 0,
    /// `total_vocab` padded up to `make_ngram_vocab_size_divisible_by`.
    padded_rows: u64 = 0,
    /// Rows per shard: `padded_rows / split_ngram_parts`.
    rows_per_shard: u64 = 0,
};

/// `_splitmix64` from the reference, on wrapping u64.
pub fn splitmix64(value: u64) u64 {
    assert(@sizeOf(u64) == 8);
    assert(splitmix_gamma > 0);

    var mixed = value +% splitmix_gamma;
    mixed = (mixed ^ (mixed >> 30)) *% splitmix_m1;
    mixed = (mixed ^ (mixed >> 27)) *% splitmix_m2;
    return mixed ^ (mixed >> 31);
}

/// `_build_layer_multipliers`: one odd multiplier per n-gram order.
pub fn deriveMultipliers(spec: *const HashSpec, out: []i64) void {
    assert(spec.ngram_size >= 2);
    assert(out.len >= spec.ngram_size);

    const max_long: u64 = (1 << 63) - 1;
    const vocabulary = @max(spec.vocab_size, 1);
    const half_bound = @max(max_long / vocabulary / 2, 1);
    const base_seed = spec.seed +% prime_seed_stride *% spec.ple_layer_index;
    for (0..spec.ngram_size) |order| {
        const value = base_seed +% splitmix_gamma *% (order + 1);
        out[order] = @intCast(2 * (splitmix64(value) % half_bound) + 1);
    }
    assert(half_bound >= 1);
}

/// `_is_prime`, by trial division up to the integer square root.
pub fn isPrime(value: u64) bool {
    assert(value <= 1 << 40);
    assert(@sizeOf(u64) == 8);

    if (value < 2) {
        return false;
    } else {
        assert(value >= 2);
    }
    if (value % 2 == 0) {
        return value == 2;
    } else {
        assert(value % 2 != 0);
    }
    var divisor: u64 = 3;
    while (divisor * divisor <= value) : (divisor += 2) {
        if (value % divisor == 0) {
            return false;
        } else {
            continue;
        }
    }
    return true;
}

/// `_find_nth_prime_after`: the `count`-th prime strictly after `start`.
pub fn findNthPrimeAfter(start: u64, count: u32) u64 {
    assert(count >= 1);
    assert(start <= 1 << 40);

    var prime = start;
    for (0..count) |_| {
        prime += 1;
        while (!isPrime(prime)) {
            prime += 1;
        }
    }
    assert(isPrime(prime));
    return prime;
}

/// Derives the head vocabulary, the offsets and the multipliers of one layer.
pub fn deriveHeadTable(spec: *const HashSpec) !HeadTable {
    try spec.validate();
    assert(spec.headCount() <= heads_max);

    var table = HeadTable{
        .head_count = spec.headCount(),
        .heads_per_ngram = spec.heads_per_ngram,
        .ngram_size = spec.ngram_size,
    };
    var total: u64 = 0;
    for (0..table.head_count) |head| {
        const global: u64 = spec.ple_layer_index * spec.headCount() + head;
        const size = findNthPrimeAfter(spec.ngram_vocab_size_base - 1, @intCast(global + 1));
        table.sizes[head] = @intCast(size);
        table.offsets[head] = @intCast(total);
        total += size;
    }
    const divisor: u64 = spec.make_ngram_vocab_size_divisible_by;
    table.total_vocab = total;
    table.padded_rows = ((total + divisor - 1) / divisor) * divisor;
    assert(table.padded_rows % divisor == 0);
    if (table.padded_rows % spec.split_ngram_parts != 0) {
        return error.TableRowsNotDivisibleByShards;
    }
    table.rows_per_shard = table.padded_rows / spec.split_ngram_parts;
    deriveMultipliers(spec, table.multipliers[0..]);
    assert(table.rows_per_shard > 0);
    return table;
}

/// The EOS-aware rolling state that derives one id vector per token.
///
/// It reproduces `_shift_right_ignore_eos` in O(1) per token instead of
/// building the shifted arrays: at absolute position `p` the value of shift `s`
/// is the token at `p - s` when `p` sits at least `s` positions into the
/// current segment, and EOS otherwise. Positions before the first appended
/// token are EOS, which is what the reference's EOS-filled context is.
pub const IdStream = struct {
    /// The last `ngram_size - 1` tokens, position `p` at `recent[p % span]`.
    recent: [ngram_max]i64 = [_]i64{0} ** ngram_max,
    /// Tokens appended so far, i.e. the absolute position of the next one.
    count: u64 = 0,
    /// Absolute position of the open segment's first token.
    segment_start: i64 = 0,
    /// `ngram_size - 1`; the shifting never looks further back.
    span: u32 = 0,
    eos_token_id: i64 = 0,

    /// An empty stream: every shift reads the virtual EOS prefix.
    pub fn init(spec: *const HashSpec) IdStream {
        assert(spec.ngram_size >= 2);
        assert(spec.ngram_size <= ngram_max);

        return .{
            .span = spec.ngram_size - 1,
            .eos_token_id = spec.eos_token_id,
        };
    }

    /// Forgets every token; the EOS prefix is virtual, so nothing is filled.
    pub fn reset(self: *IdStream) void {
        assert(self.span >= 1);
        assert(self.span < ngram_max);

        self.count = 0;
        self.segment_start = 0;
    }

    /// The token at absolute position `position`, or EOS outside the stream.
    fn valueAt(self: *const IdStream, position: i64) i64 {
        assert(position >= 0);
        assert(self.span >= 1);

        const index: u64 = @intCast(position);
        if (index >= self.count) {
            return self.eos_token_id;
        } else {
            assert(index < self.count);
        }
        // The slot was overwritten if a later token of the same residue class
        // has already been appended; callers only reach back `span` positions.
        assert(index + self.span >= self.count);
        return self.recent[@intCast(index % self.span)];
    }

    /// Appends `token` and writes the `head_count` ids of its position.
    pub fn append(
        self: *IdStream,
        table: *const HeadTable,
        token: i64,
        ids: []i64,
    ) void {
        assert(ids.len == table.head_count);
        assert(self.span == table.ngram_size - 1);

        const position: i64 = @intCast(self.count);
        const in_segment = position - self.segment_start;
        var shifted: [ngram_max]i64 = undefined;
        shifted[0] = token;
        for (1..table.ngram_size) |order| {
            const distance: i64 = @intCast(order);
            shifted[order] = if (in_segment >= distance)
                self.valueAt(position - distance)
            else
                self.eos_token_id;
        }
        var head: u32 = 0;
        var order: u32 = 2;
        while (order <= table.ngram_size) : (order += 1) {
            var mixed: i64 = shifted[0] *% table.multipliers[0];
            for (1..order) |shift| {
                mixed ^= shifted[shift] *% table.multipliers[shift];
            }
            for (0..table.heads_per_ngram) |_| {
                const modulus = table.sizes[head];
                assert(modulus > 0);
                ids[head] = @mod(mixed, modulus) + table.offsets[head];
                head += 1;
            }
        }
        self.recent[@intCast(self.count % self.span)] = token;
        if (token == self.eos_token_id) {
            self.segment_start = position + 1;
        } else {
            assert(token != self.eos_token_id);
        }
        self.count += 1;
        assert(head == table.head_count);
    }

    /// Appends every token of `tokens_in`, writing `tokens_in.len` id vectors.
    pub fn appendAll(
        self: *IdStream,
        table: *const HeadTable,
        tokens_in: []const i64,
        ids: []i64,
    ) void {
        assert(tokens_in.len > 0);
        assert(ids.len == tokens_in.len * table.head_count);

        for (tokens_in, 0..) |token, index| {
            self.append(
                table,
                token,
                ids[index * table.head_count ..][0..table.head_count],
            );
        }
        assert(self.count >= tokens_in.len);
    }
};

// ---------------------------------------------------------------------------
// The n-gram table: one mmap for whole-row reads, one fd for the concurrent
// gather, one place that maps an id onto a file offset.
// ---------------------------------------------------------------------------

/// Element formats the n-gram table is stored in: the checkpoint's F8_E4M3 rows
/// and the golden fixture's f32 rows.
pub const RowFormat = enum {
    f32,
    e4m3,

    /// Bytes per table element.
    pub fn elementBytes(self: RowFormat) u32 {
        assert(@sizeOf(f32) == 4);
        assert(@sizeOf(u8) == 1);

        return switch (self) {
            .f32 => 4,
            .e4m3 => 1,
        };
    }

    /// The device kernel's element-format code; it mirrors `ple_format_*`.
    pub fn deviceCode(self: RowFormat) u32 {
        assert(@intFromEnum(self) <= 1);
        assert(@sizeOf(u32) == 4);

        return switch (self) {
            .f32 => 0,
            .e4m3 => 1,
        };
    }
};

/// Where one table id lives in the file.
pub const RowLocation = struct {
    shard: u32,
    row: u32,
    /// Absolute byte offset of the row inside the table file.
    offset: u64,
};

/// The n-gram table of one PLE layer, mapped from disk.
///
/// The 51.2 GB table does not fit in VRAM (and barely in RAM), so it stays in
/// the file: whole rows are read through the mapping, and the concurrent gather
/// reads them with `pread` on the same file descriptor. Both routes return the
/// same bytes from the same offset, which is what the test asserts.
pub const Table = struct {
    handle: std.fs.File,
    mapping: []align(std.heap.page_size_min) u8,
    shard_offset: [shards_max]u64 = [_]u64{0} ** shards_max,
    shard_rows: [shards_max]u32 = [_]u32{0} ** shards_max,
    shard_count: u32,
    head_dim: u32,
    format: RowFormat,
    /// Rows per shard, taken from shard 0: `padded_rows / split_ngram_parts`.
    rows_per_shard: u64,

    /// Maps `path` and indexes the `prefix{N}.weight` shards it holds.
    pub fn open(
        path: []const u8,
        prefix: []const u8,
        split_parts: u32,
        head_dim: u32,
        format: RowFormat,
    ) !Table {
        if (split_parts == 0 or split_parts > shards_max) {
            return error.TooManyTableShards;
        }
        if (head_dim == 0 or prefix.len == 0) {
            return error.BadTableShape;
        }

        const handle = try std.fs.cwd().openFile(path, .{});
        errdefer handle.close();
        const size = (try handle.stat()).size;
        if (size <= safetensors.length_prefix_bytes) {
            return error.file_too_short;
        } else {
            assert(size > safetensors.length_prefix_bytes);
        }
        const mapping = try std.posix.mmap(
            null,
            @intCast(size),
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            handle.handle,
            0,
        );
        errdefer std.posix.munmap(mapping);

        var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena_state.deinit();
        var file = try safetensors.File.initFromBytes(path, mapping);
        const header = try safetensors.parseHeader(
            arena_state.allocator(),
            &file,
            0,
        );
        if (!header.dataComplete(&file)) {
            return error.table_incomplete;
        }

        var table = Table{
            .handle = handle,
            .mapping = mapping,
            .shard_count = split_parts,
            .head_dim = head_dim,
            .format = format,
            .rows_per_shard = 0,
        };
        for (0..split_parts) |shard| {
            try table.indexShard(&header, prefix, @intCast(shard));
        }
        table.rows_per_shard = table.shard_rows[0];
        assert(table.rows_per_shard > 0);
        // The id mapping assumes shard 0's row count; a wider shard would mean a
        // different split than the config declares.
        for (table.shard_rows[0..split_parts]) |rows| {
            if (rows > table.rows_per_shard) return error.TableShardTooWide;
        }
        return table;
    }

    /// Fills one shard's row count and file offset from the parsed header.
    fn indexShard(
        self: *Table,
        header: *const safetensors.Header,
        prefix: []const u8,
        shard: u32,
    ) !void {
        assert(shard < self.shard_count);
        assert(prefix.len > 0);

        var name_buffer: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(
            &name_buffer,
            "{s}{d}.weight",
            .{ prefix, shard },
        );
        var found: ?safetensors.Tensor = null;
        for (header.tensors) |tensor| {
            if (std.mem.eql(u8, tensor.name, name)) {
                found = tensor;
                break;
            } else {
                continue;
            }
        }
        const tensor = found orelse return error.MissingTableShard;
        const expected: safetensors.Dtype = switch (self.format) {
            .f32 => .f32,
            .e4m3 => .f8_e4m3,
        };
        if (tensor.dtype != expected) {
            return error.TableShardDtype;
        }
        const extents = [_]u32{ tensor.dims.values[0], self.head_dim };
        if (!tensor.dims.equals(&extents)) {
            return error.TableShardShape;
        }
        const rows = tensor.dims.values[0];
        if (rows > self.rows_per_shard and self.rows_per_shard != 0) {
            return error.TableShardTooWide;
        }
        self.shard_offset[shard] = tensor.file_offset;
        self.shard_rows[shard] = rows;
        assert(self.shard_offset[shard] >= safetensors.length_prefix_bytes);
    }

    /// Unmaps the table and closes its descriptor.
    pub fn close(self: *Table) void {
        assert(self.shard_count > 0);
        assert(self.mapping.len > 0);

        std.posix.munmap(self.mapping);
        self.handle.close();
        self.mapping = self.mapping[0..0];
    }

    /// Maps `id` onto its shard, row and file offset.
    pub fn location(self: *const Table, id: i64) RowLocation {
        assert(id >= 0);
        assert(self.rows_per_shard > 0);

        const index: u64 = @intCast(id);
        const shard: u32 = @intCast(index / self.rows_per_shard);
        const row_index: u32 = @intCast(index % self.rows_per_shard);
        // An id is derived in range by construction, so an id past the table is
        // a programming error, not a data condition.
        assert(shard < self.shard_count);
        assert(row_index < self.shard_rows[shard]);
        return .{
            .shard = shard,
            .row = row_index,
            .offset = self.shard_offset[shard] +
                @as(u64, row_index) * self.rowBytes(),
        };
    }

    /// Bytes one table row holds.
    pub fn rowBytes(self: *const Table) u64 {
        assert(self.head_dim > 0);
        assert(self.format.elementBytes() > 0);

        return @as(u64, self.head_dim) * self.format.elementBytes();
    }

    /// One row's bytes from the file, the single read both the serial and the
    /// pooled gather use.
    fn readRow(self: *const Table, at: RowLocation, out: []u8) !void {
        assert(out.len == self.rowBytes());
        assert(at.offset + self.rowBytes() <= self.mapping.len);

        const read = try std.posix.pread(self.handle.handle, out, at.offset);
        if (read != out.len) return error.ShortTableRead;
    }

    /// One row through the mapping; the bytes are the file's, never copied.
    pub fn row(self: *const Table, id: i64) []const u8 {
        assert(self.mapping.len > 0);
        assert(id >= 0);

        const at = self.location(id);
        const end = at.offset + self.rowBytes();
        assert(end <= self.mapping.len);
        return self.mapping[@intCast(at.offset)..@intCast(end)];
    }

    /// Reads every id's row into `out` one `pread` at a time.
    ///
    /// This is the reference route: it is also what a gather without a worker
    /// pool costs, which the latency test measures as the serial baseline.
    pub fn gather(self: *const Table, ids: []const i64, out: []u8) !void {
        assert(ids.len > 0);
        assert(out.len == ids.len * self.rowBytes());

        const width: usize = @intCast(self.rowBytes());
        for (ids, 0..) |id, index| {
            const at = self.location(id);
            try self.readRow(at, out[index * width ..][0..width]);
        }
        assert(out.len == ids.len * width);
    }
};

/// Workers one gather pool may run; 16 covers one token's 16 rows with no
/// queueing, which is what makes the gather one NVMe latency instead of 16.
pub const gather_workers_max: u32 = 16;

/// Reads one token's table rows concurrently on a fixed set of threads.
///
/// The threads live for the whole run (no thread is spawned per token), each
/// grabs rows from a shared cursor so a slow row does not idle the others, and
/// every row is a `pread` of its own file offset, so the reads overlap in the
/// block layer instead of serialising into 16 page faults.
pub const GatherPool = struct {
    workers: [gather_workers_max]std.Thread = undefined,
    worker_count: u32,
    table: *const Table,
    mutex: std.Thread.Mutex = .{},
    start: std.Thread.Condition = .{},
    done: std.Thread.Condition = .{},
    epoch: u64 = 0,
    finished: u32 = 0,
    /// Set once at deinit; the workers then leave their loop.
    stopped: bool = false,
    cursor: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    failures: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    ids: []const i64 = &.{},
    out: []u8 = &.{},

    /// Starts `worker_count` workers for `table`; both must outlive the pool.
    ///
    /// The workers keep a pointer to the pool, so it is heap allocated and the
    /// caller only ever holds the pointer.
    pub fn create(
        allocator: std.mem.Allocator,
        table: *const Table,
        worker_count: u32,
    ) !*GatherPool {
        assert(table.head_dim > 0);
        if (worker_count == 0 or worker_count > gather_workers_max) {
            return error.BadWorkerCount;
        }

        const pool = try allocator.create(GatherPool);
        errdefer allocator.destroy(pool);
        pool.* = .{ .worker_count = 0, .table = table };
        while (pool.worker_count < worker_count) {
            pool.workers[pool.worker_count] = try std.Thread.spawn(
                .{},
                worker,
                .{pool},
            );
            pool.worker_count += 1;
        }
        assert(pool.worker_count == worker_count);
        return pool;
    }

    /// Stops the workers and releases the pool's own allocation.
    pub fn destroy(self: *GatherPool, allocator: std.mem.Allocator) void {
        assert(self.worker_count > 0);
        assert(self.stopped == false);

        const count = self.worker_count;
        self.mutex.lock();
        self.stopped = true;
        self.start.broadcast();
        self.mutex.unlock();
        for (self.workers[0..count]) |handle| {
            handle.join();
        }
        allocator.destroy(self);
    }

    /// Reads every id's row into `out`, one token's worth in one round trip.
    pub fn gather(self: *GatherPool, ids: []const i64, out: []u8) !void {
        assert(ids.len > 0);
        assert(out.len == ids.len * self.table.rowBytes());
        assert(ids.len <= std.math.maxInt(u32));

        self.mutex.lock();
        defer self.mutex.unlock();
        self.ids = ids;
        self.out = out;
        self.cursor.store(0, .monotonic);
        self.failures.store(0, .monotonic);
        self.finished = 0;
        self.epoch +%= 1;
        self.start.broadcast();
        while (self.finished < self.worker_count) {
            self.done.wait(&self.mutex);
        }
        if (self.failures.load(.monotonic) != 0) {
            return error.GatherFailed;
        }
        assert(self.finished == self.worker_count);
    }

    /// One worker's share of the current batch: rows until the cursor is past
    /// the end.
    fn runShare(self: *GatherPool) void {
        assert(self.worker_count > 0);
        assert(self.table.head_dim > 0);

        const width: usize = @intCast(self.table.rowBytes());
        const count: u32 = @intCast(self.ids.len);
        while (true) {
            const index = self.cursor.fetchAdd(1, .monotonic);
            if (index >= count) {
                return;
            } else {
                assert(index < count);
            }
            const at = self.table.location(self.ids[index]);
            const slice = self.out[@as(usize, index) * width ..][0..width];
            self.table.readRow(at, slice) catch {
                _ = self.failures.fetchAdd(1, .monotonic);
            };
        }
    }
};

/// One worker thread: wait for a batch, run its share, report it finished.
fn worker(pool: *GatherPool) void {
    assert(pool.worker_count > 0);
    assert(pool.table.head_dim > 0);

    var seen: u64 = 0;
    while (true) {
        pool.mutex.lock();
        while (pool.epoch == seen and !pool.stopped) {
            pool.start.wait(&pool.mutex);
        }
        if (pool.stopped) {
            pool.mutex.unlock();
            return;
        }
        seen = pool.epoch;
        pool.mutex.unlock();

        pool.runShare();

        pool.mutex.lock();
        pool.finished += 1;
        if (pool.finished == pool.worker_count) {
            pool.done.signal();
        }
        pool.mutex.unlock();
    }
}

// ---------------------------------------------------------------------------
// The device side: the shapes, the six entry points, and the block that runs
// them in order.
// ---------------------------------------------------------------------------

/// One PLE layer's shapes. The kernels take every extent as an argument, so the
/// tiny golden fixture (embed 32, hidden 128, 4 heads of 8) and the real
/// checkpoint (embed 2560, hidden 2560, 16 heads of 160) share the same code.
pub const Shapes = struct {
    /// `ple_embed_dim`: the gathered row count times the row width.
    ple_embed_dim: u32,
    /// `hidden_size`: the `value_proj` width and the RMSNorm group size.
    hidden_size: u32,
    /// `hc_count`: residual streams the gate has one weight per.
    hc_count: u32,
    /// `(ngram_size - 1) * heads_per_ngram`.
    ngram_heads: u32,
    /// `ple_embed_dim / ngram_heads`: columns one n-gram table row holds.
    head_dim: u32,
    /// `ple_conv_kernel_size`.
    conv_kernel: u32,
    /// The conv dilation, which the checkpoint sets to `ngram_size`.
    conv_dilation: u32,
    /// `rms_norm_eps`.
    eps: f32,

    /// `hc_count * hidden_size`: the width of one token's PLE row.
    pub fn hyperSize(self: Shapes) u32 {
        assert(self.hc_count > 0);
        assert(self.hidden_size > 0);

        return self.hc_count * self.hidden_size;
    }

    /// `(conv_kernel - 1) * conv_dilation`: the conv ring buffer's length.
    pub fn stateLen(self: Shapes) u32 {
        assert(self.conv_kernel >= 2);
        assert(self.conv_dilation >= 1);

        return (self.conv_kernel - 1) * self.conv_dilation;
    }

    /// Rejects shapes the kernels cannot launch.
    pub fn validate(self: Shapes) !void {
        assert(self.eps > 0.0);
        assert(self.eps <= 1.0);

        if (self.ple_embed_dim == 0 or self.hidden_size == 0) {
            return error.BadPleShape;
        }
        if (self.hc_count == 0 or self.ngram_heads == 0) return error.BadPleShape;
        if (self.head_dim * self.ngram_heads != self.ple_embed_dim) {
            return error.BadPleShape;
        }
        if (self.conv_kernel < 2 or self.conv_kernel > conv_kernel_max) {
            return error.BadConvKernel;
        }
        if (self.conv_dilation == 0) return error.BadConvDilation;
        if (self.stateLen() > state_len_max) return error.ConvStateTooLong;
        assert(self.stateLen() >= 1);
    }
};

/// `nvidia/Qwen3.8-Flash-Next-NVFP4`, text layer 1.
pub const real_shapes: Shapes = .{
    .ple_embed_dim = 2560,
    .hidden_size = 2560,
    .hc_count = 4,
    .ngram_heads = 16,
    .head_dim = 160,
    .conv_kernel = 4,
    .conv_dilation = 3,
    .eps = 1e-6,
};

/// Every entry point of the `ple` cubin.
pub const Kernels = struct {
    module: kernel.Module,
    cast_rows: kernel.Function,
    project: kernel.Function,
    gate: kernel.Function,
    norm_conv: kernel.Function,
    conv_apply: kernel.Function,
    conv_state: kernel.Function,

    /// Loads the cubin and resolves every entry point; the context must be
    /// current.
    pub fn init() !Kernels {
        assert(cubin_name.len > 0);
        assert(generated_cubins.cubins.len > 0);

        const image = try cubinImage(cubin_name);
        const module = try kernel.Module.load(image);
        errdefer module.unload();
        return .{
            .module = module,
            .cast_rows = try module.function("ple_cast_rows"),
            .project = try module.function("ple_project"),
            .gate = try module.function("ple_gate"),
            .norm_conv = try module.function("ple_norm_conv"),
            .conv_apply = try module.function("ple_conv_apply"),
            .conv_state = try module.function("ple_conv_state"),
        };
    }

    /// Unloads the module; the owning context must be current.
    pub fn deinit(self: *Kernels) void {
        assert(self.module.handle != null);
        assert(@intFromPtr(self.module.handle) != 0);

        self.module.unload();
    }
};

/// The image of the cubin named `name` inside the static library.
fn cubinImage(name: []const u8) ![]const u8 {
    assert(name.len > 0);
    assert(generated_cubins.cubins.len > 0);

    for (generated_cubins.cubins) |cubin| {
        const text = std.mem.span(cubin.name);
        if (std.mem.eql(u8, text, name)) {
            assert(cubin.length > 0);
            return cubin.bytes[0..cubin.length];
        } else {
            continue;
        }
    }
    return error.MissingCubin;
}

/// Arguments of `ple_cast_rows`.
pub const CastArgs = struct {
    /// Raw staged table rows, `count * head_dim` elements packed.
    rows: cuda_driver.DevicePtr,
    /// f32 `[count, head_dim]` out.
    out: cuda_driver.DevicePtr,
    count: u32,
    head_dim: u32,
    format: RowFormat,
    /// The per-tensor `weight_scale` of the table; 1.0 for an f32 fixture.
    scale: f32,
};

/// Arguments of `ple_project`.
pub const ProjectArgs = struct {
    /// f32 `[rows, in_features]`.
    x: cuda_driver.DevicePtr,
    /// f32 `[out_features, in_features]`, the `nn.Linear` layout.
    weight: cuda_driver.DevicePtr,
    /// f32 `[rows, out_features]` out.
    out: cuda_driver.DevicePtr,
    rows: u32,
    in_features: u32,
    out_features: u32,
};

/// Arguments of `ple_gate`.
pub const GateArgs = struct {
    /// f32 `[tokens, hyper]`: the projected key.
    key: cuda_driver.DevicePtr,
    /// f32 `[tokens, hyper]`: the residual entering the block.
    hidden: cuda_driver.DevicePtr,
    /// f32 `[hyper]` each.
    norm_key: cuda_driver.DevicePtr,
    norm_query: cuda_driver.DevicePtr,
    /// f32 `[tokens, hyper]` out.
    key_normed: cuda_driver.DevicePtr,
    query_normed: cuda_driver.DevicePtr,
    gated_value: cuda_driver.DevicePtr,
    /// f32 `[tokens, hc_count]` out: the signed-sqrt gate itself.
    gates: cuda_driver.DevicePtr,
    /// f32 `[tokens, hidden]`: the projected value, shared by every stream.
    value: cuda_driver.DevicePtr,
    tokens: u32,
    shapes: Shapes,
};

/// Arguments of `ple_norm_conv`.
pub const NormConvArgs = struct {
    /// f32 `[tokens, hyper]` in and out.
    gated_value: cuda_driver.DevicePtr,
    /// f32 `[hyper]` norm weight.
    weight: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,
    tokens: u32,
    shapes: Shapes,
};

/// Arguments of `ple_conv_apply`.
pub const ConvArgs = struct {
    /// f32 `[tokens, hyper]`.
    gated_value: cuda_driver.DevicePtr,
    /// f32 `[tokens, hyper]`: `norm_conv` of `gated_value`.
    normed: cuda_driver.DevicePtr,
    /// f32 `[hyper, conv_kernel]` depthwise taps.
    taps: cuda_driver.DevicePtr,
    /// f32 `[hyper, state_len]`: the ring buffer before this step.
    state: cuda_driver.DevicePtr,
    /// f32 `[tokens, hyper]` out: `gated_value + silu(conv)`.
    out: cuda_driver.DevicePtr,
    /// f32 `[tokens, hyper]`: the residual, accumulated in place.
    hyper: cuda_driver.DevicePtr,
    tokens: u32,
    shapes: Shapes,
};

/// Arguments of `ple_conv_state`.
pub const StateArgs = struct {
    /// f32 `[tokens, hyper]`: this step's `norm_conv` output.
    normed: cuda_driver.DevicePtr,
    /// f32 `[hyper, state_len]`, shifted in place.
    state: cuda_driver.DevicePtr,
    tokens: u32,
    shapes: Shapes,
};

/// The `ple` kernels plus the stream they run on.
pub const Ple = struct {
    kernels: Kernels,
    stream: cuda_context.Stream,

    /// Loads the cubin and creates the queue; the context must be current.
    pub fn init() !Ple {
        assert(cubin_name.len > 0);
        assert(generated_cubins.cubins.len > 0);

        const kernels = try Kernels.init();
        errdefer {
            var owned = kernels;
            owned.deinit();
        }
        const stream = try cuda_context.Stream.create();
        assert(@intFromPtr(stream.handle) != 0);
        return .{ .kernels = kernels, .stream = stream };
    }

    /// Releases the queue and the module.
    pub fn deinit(self: *Ple) void {
        assert(@intFromPtr(self.stream.handle) != 0);
        assert(@intFromPtr(self.kernels.module.handle) != 0);

        self.stream.destroy();
        self.kernels.deinit();
    }

    /// Decodes the staged rows into f32 embeddings and applies the table scale.
    pub fn castRows(self: *const Ple, args: CastArgs) !void {
        assert(args.count > 0);
        assert(args.head_dim > 0);

        var rows = args.rows;
        var out = args.out;
        var count: u32 = args.count;
        var head_dim: u32 = args.head_dim;
        var format: u32 = args.format.deviceCode();
        var scale: f32 = args.scale;
        const total: u64 = @as(u64, args.count) * args.head_dim;
        const blocks: u32 = @intCast(@min(
            std.math.divCeil(u64, total, conv_threads) catch unreachable,
            4096,
        ));
        const arguments = [_]?*anyopaque{
            @ptrCast(&rows),     @ptrCast(&out),    @ptrCast(&count),
            @ptrCast(&head_dim), @ptrCast(&format), @ptrCast(&scale),
        };
        try self.kernels.cast_rows.launch(
            .{ blocks, 1, 1 },
            .{ conv_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// `out = x @ weight^T` in f32, one warp per output column.
    pub fn project(self: *const Ple, args: ProjectArgs) !void {
        assert(args.rows > 0);
        assert(args.out_features > 0);

        var x = args.x;
        var weight = args.weight;
        var out = args.out;
        var rows: u32 = args.rows;
        var in_features: u32 = args.in_features;
        var out_features: u32 = args.out_features;
        // Eight warps per block, i.e. eight output columns.
        const columns: u32 = @intCast(std.math.divCeil(
            u64,
            args.out_features,
            8,
        ) catch unreachable);
        const arguments = [_]?*anyopaque{
            @ptrCast(&x),    @ptrCast(&weight),      @ptrCast(&out),
            @ptrCast(&rows), @ptrCast(&in_features), @ptrCast(&out_features),
        };
        try self.kernels.project.launch(
            .{ columns, args.rows, 1 },
            .{ conv_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// The two group norms, the signed-sqrt gate and the gated value.
    pub fn gate(self: *const Ple, args: GateArgs) !void {
        assert(args.tokens > 0);
        assert(args.shapes.hc_count > 0);
        try args.shapes.validate();

        var key = args.key;
        var hidden = args.hidden;
        var norm_key = args.norm_key;
        var norm_query = args.norm_query;
        var key_normed = args.key_normed;
        var query_normed = args.query_normed;
        var gates = args.gates;
        var value = args.value;
        var gated_value = args.gated_value;
        var hidden_size: u32 = args.shapes.hidden_size;
        var streams: u32 = args.shapes.hc_count;
        var eps: f32 = args.shapes.eps;
        const tokens: u32 = args.tokens;
        const arguments = [_]?*anyopaque{
            @ptrCast(&key),         @ptrCast(&hidden),     @ptrCast(&norm_key),
            @ptrCast(&norm_query),  @ptrCast(&key_normed), @ptrCast(&query_normed),
            @ptrCast(&gates),       @ptrCast(&value),      @ptrCast(&gated_value),
            @ptrCast(&hidden_size), @ptrCast(&streams),    @ptrCast(&eps),
        };
        try self.kernels.gate.launch(
            .{ tokens, 1, 1 },
            .{ conv_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// `norm_conv` over the flattened gated value, one group per stream.
    pub fn normConv(self: *const Ple, args: NormConvArgs) !void {
        assert(args.tokens > 0);
        assert(args.shapes.hc_count > 0);

        var gated_value = args.gated_value;
        var weight = args.weight;
        var out = args.out;
        var hidden_size: u32 = args.shapes.hidden_size;
        var streams: u32 = args.shapes.hc_count;
        var eps: f32 = args.shapes.eps;
        const arguments = [_]?*anyopaque{
            @ptrCast(&gated_value), @ptrCast(&weight),  @ptrCast(&out),
            @ptrCast(&hidden_size), @ptrCast(&streams), @ptrCast(&eps),
        };
        try self.kernels.norm_conv.launch(
            .{ args.tokens, 1, 1 },
            .{ conv_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// The dilated depthwise conv, `silu`, and the injection into `hyper`.
    pub fn applyConv(self: *const Ple, args: ConvArgs) !void {
        assert(args.tokens > 0);
        assert(args.shapes.conv_kernel > 1);
        try args.shapes.validate();

        const channels = args.shapes.hyperSize();
        var gated_value = args.gated_value;
        var normed = args.normed;
        var taps = args.taps;
        var state = args.state;
        var out = args.out;
        var hyper = args.hyper;
        var tokens: u32 = args.tokens;
        var channels_arg: u32 = channels;
        var kernel_size: u32 = args.shapes.conv_kernel;
        var dilation: u32 = args.shapes.conv_dilation;
        var state_len: u32 = args.shapes.stateLen();
        const blocks: u32 = @intCast(std.math.divCeil(
            u64,
            channels,
            conv_threads,
        ) catch unreachable);
        const arguments = [_]?*anyopaque{
            @ptrCast(&gated_value), @ptrCast(&normed),       @ptrCast(&taps),
            @ptrCast(&state),       @ptrCast(&out),          @ptrCast(&hyper),
            @ptrCast(&tokens),      @ptrCast(&channels_arg), @ptrCast(&kernel_size),
            @ptrCast(&dilation),    @ptrCast(&state_len),
        };
        try self.kernels.conv_apply.launch(
            .{ blocks, args.tokens, 1 },
            .{ conv_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Shifts the conv ring buffer to the last `state_len` positions.
    pub fn saveConvState(self: *const Ple, args: StateArgs) !void {
        assert(args.tokens > 0);
        assert(args.shapes.stateLen() > 0);
        try args.shapes.validate();

        const channels = args.shapes.hyperSize();
        var normed = args.normed;
        var state = args.state;
        var tokens: u32 = args.tokens;
        var channels_arg: u32 = channels;
        var state_len: u32 = args.shapes.stateLen();
        const blocks: u32 = @intCast(std.math.divCeil(
            u64,
            channels,
            conv_threads,
        ) catch unreachable);
        const arguments = [_]?*anyopaque{
            @ptrCast(&normed),       @ptrCast(&state),     @ptrCast(&tokens),
            @ptrCast(&channels_arg), @ptrCast(&state_len),
        };
        try self.kernels.conv_state.launch(
            .{ blocks, 1, 1 },
            .{ conv_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }
};

/// One PLE layer's device weights.
pub const Weights = struct {
    /// f32 `[hyper, ple_embed_dim]`, `key_proj.weight`.
    key_proj: cuda_driver.DevicePtr,
    /// f32 `[hidden_size, ple_embed_dim]`, `value_proj.weight`.
    value_proj: cuda_driver.DevicePtr,
    /// f32 `[hyper]` each: `norm_key`, `norm_query`, `norm_conv`.
    norm_key: cuda_driver.DevicePtr,
    norm_query: cuda_driver.DevicePtr,
    norm_conv: cuda_driver.DevicePtr,
    /// f32 `[hyper, conv_kernel]`, `conv1d.weight`.
    taps: cuda_driver.DevicePtr,
    /// The table's element format and per-tensor scale.
    table_format: RowFormat,
    table_scale: f32,
};

/// The device buffers one `Block` runs on.
pub const Buffers = struct {
    /// Raw staged table rows: `capacity * ngram_heads * head_dim` elements.
    rows: cuda_context.DeviceBuffer,
    /// f32 `[capacity, ple_embed_dim]`.
    embeddings: cuda_context.DeviceBuffer,
    /// f32 `[capacity, hyper]`.
    key: cuda_context.DeviceBuffer,
    /// f32 `[capacity, hidden_size]`.
    value: cuda_context.DeviceBuffer,
    key_normed: cuda_context.DeviceBuffer,
    query_normed: cuda_context.DeviceBuffer,
    gated_value: cuda_context.DeviceBuffer,
    gated_value_normed: cuda_context.DeviceBuffer,
    /// f32 `[capacity, hc_count]`.
    gates: cuda_context.DeviceBuffer,
    /// f32 `[capacity, hyper]`: the block output, before the residual add.
    out: cuda_context.DeviceBuffer,
    /// f32 `[hyper, state_len]`: the conv ring buffer.
    conv_state: cuda_context.DeviceBuffer,
};

/// Rounds `bytes` up to what `cuMemAlloc` and the pinned allocator accept.
fn roundToAllocation(bytes: u64) u64 {
    assert(bytes > 0);
    assert(allocation_granularity > 0);

    const rounded = (bytes + allocation_granularity - 1) /
        allocation_granularity * allocation_granularity;
    assert(rounded >= bytes);
    return rounded;
}

/// Allocates one device buffer of at least `bytes`.
fn allocateBuffer(bytes: u64) !cuda_context.DeviceBuffer {
    assert(bytes > 0);
    assert(allocation_granularity == 4096);

    const buffer = try cuda_context.DeviceBuffer.allocate(roundToAllocation(bytes));
    assert(buffer.pointer != 0);
    return buffer;
}

/// The whole PLE block of one layer: gather, cast, project, gate, conv, inject.
///
/// One instance owns the device buffers and the one pinned staging window, so a
/// token batch costs exactly one pinned buffer and one H2D copy of
/// `16 rows * 160 B` (2.5 KB) regardless of how the rows are spread over the
/// table.
pub const Block = struct {
    ple: *const Ple,
    table: *const Table,
    pool: *GatherPool,
    staging: cuda_context.PinnedHostBuffer,
    buffers: Buffers,
    weights: Weights,
    shapes: Shapes,
    capacity_tokens: u32,

    /// Allocates the buffers for `capacity_tokens` tokens per call.
    pub fn init(
        ple: *const Ple,
        table: *const Table,
        pool: *GatherPool,
        weights: Weights,
        shapes: Shapes,
        capacity_tokens: u32,
    ) !Block {
        assert(pool.table == table);
        assert(shapes.head_dim > 0);
        try shapes.validate();
        if (capacity_tokens == 0 or capacity_tokens > tokens_max) {
            return error.BadTokenCapacity;
        }
        if (table.head_dim != shapes.head_dim) return error.TableHeadDimMismatch;
        if (table.format != weights.table_format) return error.TableFormatMismatch;

        const tokens: u64 = capacity_tokens;
        const hyper: u64 = shapes.hyperSize();
        const hidden: u64 = shapes.hidden_size;
        const rows_bytes = @as(u64, capacity_tokens) * shapes.ngram_heads *
            table.rowBytes();
        // Zeroed first so a partial allocation is freed by the errdefer below.
        var buffers = std.mem.zeroes(Buffers);
        errdefer freeBuffers(&buffers);
        buffers.rows = try allocateBuffer(rows_bytes);
        buffers.embeddings = try allocateBuffer(tokens * shapes.ple_embed_dim * 4);
        buffers.key = try allocateBuffer(tokens * hyper * 4);
        buffers.value = try allocateBuffer(tokens * hidden * 4);
        buffers.key_normed = try allocateBuffer(tokens * hyper * 4);
        buffers.query_normed = try allocateBuffer(tokens * hyper * 4);
        buffers.gated_value = try allocateBuffer(tokens * hyper * 4);
        buffers.gated_value_normed = try allocateBuffer(tokens * hyper * 4);
        buffers.gates = try allocateBuffer(tokens * shapes.hc_count * 4);
        buffers.out = try allocateBuffer(tokens * hyper * 4);
        buffers.conv_state = try allocateBuffer(hyper * shapes.stateLen() * 4);
        const staging = try cuda_context.PinnedHostBuffer.allocate(
            roundToAllocation(rows_bytes),
        );
        assert(staging.bytes >= rows_bytes);
        return .{
            .ple = ple,
            .table = table,
            .pool = pool,
            .staging = staging,
            .buffers = buffers,
            .weights = weights,
            .shapes = shapes,
            .capacity_tokens = capacity_tokens,
        };
    }

    /// Frees the device buffers and the pinned window.
    pub fn deinit(self: *Block) void {
        assert(self.capacity_tokens > 0);
        assert(self.staging.bytes > 0);

        self.staging.free();
        freeBuffers(&self.buffers);
        self.capacity_tokens = 0;
    }

    /// Forgets the conv state (the reference's zeroed state after a reset).
    ///
    /// The conv state can be wider than the staging window the rows are
    /// gathered through, so this copies it in chunks. The destination has to
    /// advance with those chunks: a copy that re-used offset 0 for every one of
    /// them left the tail of the ring holding the previous sequence's newest
    /// column, which the dilated conv then convolves into the first rows of the
    /// next sequence.
    pub fn reset(self: *Block) !void {
        assert(self.shapes.stateLen() > 0);
        assert(self.buffers.conv_state.bytes > 0);

        const total = self.buffers.conv_state.bytes;
        const window = self.staging.bytes;
        assert(window > 0);
        @memset(self.staging.slice(), 0);
        var written: u64 = 0;
        while (written < total) {
            const size: u64 = @min(window, total - written);
            const destination = cuda_context.DeviceBuffer{
                .pointer = self.buffers.conv_state.pointer + written,
                .bytes = size,
            };
            try self.ple.stream.copy_h_to_d(
                &destination,
                &self.staging,
                size,
            );
            written += size;
        }
        try self.ple.stream.synchronize();
        assert(written == total);
    }

    /// Runs one token batch: `ids` is `[tokens, ngram_heads]` table ids.
    ///
    /// `hidden` is the residual entering the block (`[tokens, hyper]`, f32) and
    /// `hyper` is that same residual, which the block output is added into.
    pub fn run(
        self: *Block,
        ids: []const i64,
        tokens: u32,
        hidden: cuda_driver.DevicePtr,
        hyper: cuda_driver.DevicePtr,
    ) !void {
        if (tokens == 0 or tokens > self.capacity_tokens) {
            return error.BadTokenCount;
        }
        assert(ids.len == tokens * self.shapes.ngram_heads);
        assert(hidden != 0 and hyper != 0);

        try self.uploadRows(ids);
        try self.launch(ids.len, tokens, hidden, hyper);
        assert(ids.len > 0);
    }

    /// Gathers the token's rows and copies them to the device as one transfer.
    fn uploadRows(self: *Block, ids: []const i64) !void {
        assert(ids.len > 0);
        assert(ids.len <= self.capacity_tokens * self.shapes.ngram_heads);

        const row_bytes: usize = @intCast(self.table.rowBytes());
        const staged_bytes: u64 = @as(u64, ids.len) * row_bytes;
        assert(staged_bytes <= self.staging.bytes);
        try self.pool.gather(
            ids,
            self.staging.slice()[0..@intCast(staged_bytes)],
        );
        try self.ple.stream.copy_h_to_d(
            &self.buffers.rows,
            &self.staging,
            staged_bytes,
        );
        // The staged bytes are pinned host memory the next gather overwrites, so
        // the copy has to land before this call returns; the kernels below then
        // queue behind it and the host is free to gather the next token.
        try self.ple.stream.synchronize();
        assert(staged_bytes > 0);
    }

    /// Casts, projects, gates and convolves; `hyper` accumulates the block output.
    fn launch(
        self: *Block,
        rows: usize,
        tokens: u32,
        hidden: cuda_driver.DevicePtr,
        hyper: cuda_driver.DevicePtr,
    ) !void {
        assert(tokens > 0);
        assert(rows == tokens * self.shapes.ngram_heads);

        try self.ple.castRows(.{
            .rows = self.buffers.rows.pointer,
            .out = self.buffers.embeddings.pointer,
            .count = @intCast(rows),
            .head_dim = self.shapes.head_dim,
            .format = self.weights.table_format,
            .scale = self.weights.table_scale,
        });
        try self.ple.project(.{
            .x = self.buffers.embeddings.pointer,
            .weight = self.weights.key_proj,
            .out = self.buffers.key.pointer,
            .rows = tokens,
            .in_features = self.shapes.ple_embed_dim,
            .out_features = self.shapes.hyperSize(),
        });
        try self.ple.project(.{
            .x = self.buffers.embeddings.pointer,
            .weight = self.weights.value_proj,
            .out = self.buffers.value.pointer,
            .rows = tokens,
            .in_features = self.shapes.ple_embed_dim,
            .out_features = self.shapes.hidden_size,
        });
        try self.ple.gate(.{
            .key = self.buffers.key.pointer,
            .hidden = hidden,
            .norm_key = self.weights.norm_key,
            .norm_query = self.weights.norm_query,
            .key_normed = self.buffers.key_normed.pointer,
            .query_normed = self.buffers.query_normed.pointer,
            .gated_value = self.buffers.gated_value.pointer,
            .gates = self.buffers.gates.pointer,
            .value = self.buffers.value.pointer,
            .tokens = tokens,
            .shapes = self.shapes,
        });
        try self.convolve(tokens, hyper);
        assert(hyper != 0);
    }

    /// `norm_conv`, the dilated conv, `silu`, the residual add, and the state.
    fn convolve(
        self: *Block,
        tokens: u32,
        hyper: cuda_driver.DevicePtr,
    ) !void {
        assert(tokens > 0);
        assert(hyper != 0);

        try self.ple.normConv(.{
            .gated_value = self.buffers.gated_value.pointer,
            .weight = self.weights.norm_conv,
            .out = self.buffers.gated_value_normed.pointer,
            .tokens = tokens,
            .shapes = self.shapes,
        });
        try self.ple.applyConv(.{
            .gated_value = self.buffers.gated_value.pointer,
            .normed = self.buffers.gated_value_normed.pointer,
            .taps = self.weights.taps,
            .state = self.buffers.conv_state.pointer,
            .out = self.buffers.out.pointer,
            .hyper = hyper,
            .tokens = tokens,
            .shapes = self.shapes,
        });
        try self.ple.saveConvState(.{
            .normed = self.buffers.gated_value_normed.pointer,
            .state = self.buffers.conv_state.pointer,
            .tokens = tokens,
            .shapes = self.shapes,
        });
        assert(tokens > 0);
    }
};

/// Frees every buffer of `buffers`, skipping the ones a failed `init` never
/// allocated (they are all-zero then).
fn freeBuffers(buffers: *const Buffers) void {
    assert(@sizeOf(Buffers) > 0);
    assert(@sizeOf(cuda_context.DeviceBuffer) > 0);

    inline for (std.meta.fields(Buffers)) |field| {
        const buffer = @field(buffers, field.name);
        if (buffer.pointer != 0) {
            buffer.free();
        }
    }
}

// ---------------------------------------------------------------------------
// The harness: one context, one queue, the cubin, and buffer copies.
// ---------------------------------------------------------------------------

/// Directory of the golden fixtures, injected by `core/build.zig`.
const fixtures_root: []const u8 = @import("fixtures_dir").path;

/// Environment variable that overrides the checkpoint directory.
const checkpoint_variable = "INFERNO_MODEL_DIR";

/// Checkpoint directory when the environment does not name one.
const checkpoint_default = "/home/gary/models/qwen38-flash-next-nvfp4";

/// The file the real n-gram table lives in (53.7 GB, table plus the MTP head).
const real_table_file = "model-fp8-mtp-ple.safetensors";

/// Tensor names of the table shards, without the index and `.weight`; the tiny
/// checkpoint uses the same names.
const table_prefix =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

/// Largest scaled error a stage may show (issue I-011). The block output is f32
/// in the reference and in the golden trace, so this is the f32 format's own
/// precision at the tensor's magnitude, not a blanket `1e-6`; a bf16 copy of a
/// stage would be gated at 0.5 ulp instead.
const scaled_error_max: f64 = 2e-5;

/// Floor of the error scale, so near-zero tensors are compared absolutely.
const error_scale_min: f64 = 1e-6;

/// The golden prefill ids of the trace (the tiny vocabulary needs no tokenizer).
const golden_prefill_ids = [12]i64{ 13, 4, 199, 42, 7, 88, 211, 30, 96, 155, 61, 240 };

/// The id the reference feeds for its single decode step.
const golden_decode_id = [1]i64{169};

/// A GPU to run on: one context, one queue, one loaded cubin.
const Harness = struct {
    allocator: std.mem.Allocator,
    devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined,
    context: cuda_context.Context = undefined,
    ple: Ple = undefined,
    allocations: std.ArrayList(cuda_context.DeviceBuffer),

    /// Opens device 0 and loads the kernels. A host without the driver skips the
    /// test instead of failing it.
    fn open(allocator: std.mem.Allocator) !Harness {
        assert(cuda_driver.device_count_max > 0);
        assert(allocator.vtable != undefined);

        var harness = Harness{
            .allocator = allocator,
            .allocations = std.ArrayList(cuda_context.DeviceBuffer).init(allocator),
        };
        const count = cuda_driver.query_devices(harness.devices[0..]) catch |err| switch (err) {
            error.driver_library_missing => return error.SkipZigTest,
            else => return err,
        };
        assert(count > 0);
        harness.context = try cuda_context.Context.retain(&harness.devices[0]);
        errdefer harness.context.release();
        try harness.context.set_current();
        harness.ple = try Ple.init();
        assert(@intFromPtr(harness.ple.stream.handle) != 0);
        return harness;
    }

    fn close(self: *Harness) void {
        assert(self.allocations.items.len > 0);
        assert(@intFromPtr(self.ple.stream.handle) != 0);

        for (self.allocations.items) |buffer| {
            buffer.free();
        }
        self.allocations.deinit();
        self.ple.deinit();
        self.context.release();
    }

    /// Copies `bytes` to a fresh device buffer and tracks it for cleanup.
    fn upload(self: *Harness, bytes: []const u8) !cuda_driver.DevicePtr {
        assert(bytes.len > 0);
        assert(@as(u64, @intCast(bytes.len)) <= allocation_granularity * (1 << 20));

        const buffer = try allocateBuffer(@intCast(bytes.len));
        errdefer buffer.free();
        const window = try cuda_context.PinnedHostBuffer.allocate(buffer.bytes);
        defer window.free();
        @memset(window.slice(), 0);
        @memcpy(window.slice()[0..bytes.len], bytes);
        try self.ple.stream.copy_h_to_d(&buffer, &window, buffer.bytes);
        try self.ple.stream.synchronize();
        try self.allocations.append(buffer);
        assert(buffer.pointer != 0);
        return buffer.pointer;
    }

    /// Copies a device buffer back into `out`.
    fn download(self: *Harness, pointer: cuda_driver.DevicePtr, out: []u8) !void {
        assert(out.len > 0);
        assert(pointer != 0);

        const window = try cuda_context.PinnedHostBuffer.allocate(
            roundToAllocation(@intCast(out.len)),
        );
        defer window.free();
        const source = cuda_context.DeviceBuffer{
            .pointer = pointer,
            .bytes = window.bytes,
        };
        try self.ple.stream.copy_d_to_h(&window, &source, window.bytes);
        try self.ple.stream.synchronize();
        @memcpy(out, window.slice()[0..out.len]);
    }

    fn uploadF32(self: *Harness, values: []const f32) !cuda_driver.DevicePtr {
        assert(values.len > 0);
        assert(@sizeOf(f32) == 4);

        return self.upload(std.mem.sliceAsBytes(values));
    }

    fn downloadF32(
        self: *Harness,
        pointer: cuda_driver.DevicePtr,
        out: []f32,
    ) !void {
        assert(out.len > 0);
        assert(pointer != 0);

        try self.download(pointer, std.mem.sliceAsBytes(out));
    }

    /// An uninitialised device buffer for scratch or kernel output.
    fn allocate(self: *Harness, bytes: u64) !cuda_driver.DevicePtr {
        assert(bytes > 0);
        assert(bytes <= allocation_granularity * (1 << 20));

        const buffer = try allocateBuffer(bytes);
        try self.allocations.append(buffer);
        assert(buffer.pointer != 0);
        return buffer.pointer;
    }
};

// ---------------------------------------------------------------------------
// Fixture configuration and comparison helpers.
// ---------------------------------------------------------------------------

/// The shapes one `RefConfig` runs at.
fn shapesFromConfig(config: *const reference_fixtures.RefConfig) Shapes {
    assert(config.ple_embed_dim > 0);
    assert(config.hc_count > 0);

    return .{
        .ple_embed_dim = config.ple_embed_dim,
        .hidden_size = config.hidden_size,
        .hc_count = config.hc_count,
        .ngram_heads = config.ngram_heads(),
        .head_dim = config.ngram_head_dim(),
        .conv_kernel = config.ple_conv_kernel_size,
        .conv_dilation = config.ngram_size,
        .eps = config.rms_norm_eps,
    };
}

/// The n-gram parameters one `RefConfig` runs at, for layer `layer_index`.
fn specFromConfig(
    config: *const reference_fixtures.RefConfig,
    layer_index: u32,
) HashSpec {
    assert(layer_index < config.num_hidden_layers);
    assert(config.ngram_size >= 2);

    return .{
        .ngram_size = config.ngram_size,
        .heads_per_ngram = config.heads_per_ngram,
        .eos_token_id = config.eos_token_id,
        .seed = config.seed,
        .vocab_size = config.vocab_size,
        .ngram_vocab_size_base = config.ngram_vocab_size_base,
        .make_ngram_vocab_size_divisible_by = config.make_ngram_vocab_size_divisible_by,
        .split_ngram_parts = config.split_ngram_parts,
        .ple_layer_index = config.ple_layer_index(layer_index) orelse
            unreachable,
    };
}

/// Path of one golden fixture file.
fn fixturePath(allocator: std.mem.Allocator, comptime name: []const u8) ![]u8 {
    assert(fixtures_root.len > 0);
    assert(name.len > 0);

    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ fixtures_root, name });
}

/// Path of the real n-gram table, or null when the checkpoint is not on disk.
fn realTablePath(allocator: std.mem.Allocator) !?[]u8 {
    assert(real_table_file.len > 0);
    assert(checkpoint_variable.len > 0);

    const directory = std.posix.getenv(checkpoint_variable) orelse
        checkpoint_default;
    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ directory, real_table_file },
    );
    errdefer allocator.free(path);
    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            allocator.free(path);
            return null;
        },
        else => return err,
    };
    file.close();
    return path;
}

/// How far `actual` is from `expected`: the largest absolute error, the
/// magnitude it is measured against, and their ratio.
const Deviation = struct {
    absolute: f64,
    magnitude: f64,
    ratio: f64,
};

/// Measures one comparison; the absolute pair travels with the ratio because a
/// small reference magnitude inflates the ratio under cancellation.
fn measureDeviation(actual: []const f32, expected: []const f32) Deviation {
    assert(actual.len == expected.len);
    assert(actual.len > 0);

    var absolute: f64 = 0.0;
    var magnitude: f64 = 0.0;
    for (actual, expected) |got, want| {
        absolute = @max(absolute, @abs(@as(f64, got) - @as(f64, want)));
        magnitude = @max(magnitude, @abs(@as(f64, want)));
    }
    const ratio = absolute / @max(error_scale_min, magnitude);
    assert(ratio >= 0.0);
    return .{ .absolute = absolute, .magnitude = magnitude, .ratio = ratio };
}

/// Prints one stage's error and fails it past `scaled_error_max`.
fn printDeviation(
    label: []const u8,
    actual: []const f32,
    expected: []const f32,
) Deviation {
    assert(label.len > 0);
    assert(actual.len > 0);

    const deviation = measureDeviation(actual, expected);
    std.debug.print(
        "{s:<34} max_abs_err {e:.3} max|ref| {e:.3} scaled {e:.3}\n",
        .{ label, deviation.absolute, deviation.magnitude, deviation.ratio },
    );
    assert(deviation.ratio >= 0.0);
    return deviation;
}

/// Compares one f32 stage against its oracle and the golden trace.
fn expectStage(
    label: []const u8,
    actual: []const f32,
    oracle: ?[]const f32,
    golden: ?[]const f32,
) !void {
    assert(label.len > 0);
    assert(actual.len > 0);

    if (oracle) |reference| {
        try std.testing.expectEqual(actual.len, reference.len);
        const deviation = printDeviation(label, actual, reference);
        try std.testing.expect(deviation.ratio <= scaled_error_max);
    } else {
        assert(oracle == null);
    }
    if (golden) |reference| {
        try std.testing.expectEqual(actual.len, reference.len);
        var buffer: [128]u8 = undefined;
        const text = try std.fmt.bufPrint(&buffer, "{s} (golden)", .{label});
        const deviation = printDeviation(text, actual, reference);
        try std.testing.expect(deviation.ratio <= scaled_error_max);
    } else {
        assert(golden == null);
    }
}

/// Mean and population standard deviation of a slice, for the non-vacuity
/// report.
fn meanAndDeviation(values: []const f32) [2]f64 {
    assert(values.len > 0);
    assert(values.len <= 1 << 26);

    var total: f64 = 0.0;
    for (values) |value| {
        total += value;
    }
    const mean = total / @as(f64, @floatFromInt(values.len));
    var squares: f64 = 0.0;
    for (values) |value| {
        const difference = @as(f64, value) - mean;
        squares += difference * difference;
    }
    const deviation = @sqrt(squares / @as(f64, @floatFromInt(values.len)));
    assert(deviation >= 0.0);
    return .{ mean, deviation };
}

// ---------------------------------------------------------------------------
// Tests. Every one of them gates against `src/ref/ple.zig` or the HF golden
// trace, never against another GPU path.
// ---------------------------------------------------------------------------

test "the head vocabulary, the offsets and the multipliers match the checkpoint" {
    const allocator = std.testing.allocator;
    const config = try reference_fixtures.load_config(allocator);
    var tensors = try reference_fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);

    const spec = specFromConfig(&config, 1);
    const table = try deriveHeadTable(&spec);
    try std.testing.expectEqual(@as(u32, 4), table.head_count);
    try std.testing.expectEqual(@as(u32, 3), table.ngram_size);

    const prefix = "model.language_model.layers.1.ple.ple_embedding.";
    const sizes = try tensors.tensor_i64(prefix ++ "ngram_heads_vocab_sizes");
    const offsets = try tensors.tensor_i64(prefix ++ "ngram_heads_offsets");
    const stored = try tensors.tensor_i64(prefix ++ "layer_multipliers");
    try std.testing.expectEqual(@as(usize, table.head_count), sizes.len);
    try std.testing.expectEqual(stored.len, @as(usize, spec.ngram_size));
    for (0..table.head_count) |head| {
        try std.testing.expectEqual(sizes[head], table.sizes[head]);
        try std.testing.expectEqual(offsets[head], table.offsets[head]);
    }
    for (0..spec.ngram_size) |order| {
        try std.testing.expectEqual(stored[order], table.multipliers[order]);
    }

    // The oracle derives the same numbers from the same config.
    var oracle_multipliers: [ngram_max]i64 = undefined;
    reference_ple.build_layer_multipliers(
        spec.vocab_size,
        spec.ngram_size,
        spec.ple_layer_index,
        spec.seed,
        oracle_multipliers[0..],
    );
    for (0..spec.ngram_size) |order| {
        try std.testing.expectEqual(
            oracle_multipliers[order],
            table.multipliers[order],
        );
    }
    var total: u64 = 0;
    for (0..table.head_count) |head| {
        const size = reference_ple.find_nth_prime_after(
            spec.ngram_vocab_size_base - 1,
            @intCast(head + 1),
        );
        try std.testing.expectEqual(@as(i64, @intCast(size)), table.sizes[head]);
        total += size;
    }
    try std.testing.expectEqual(total, table.total_vocab);
    // 1009 + 1013 + 1019 + 1021 = 4062 rows, padded to 4064 (a multiple of 8)
    // and split into two shards of 2032 rows.
    try std.testing.expectEqual(@as(u64, 4062), table.total_vocab);
    try std.testing.expectEqual(@as(u64, 4064), table.padded_rows);
    try std.testing.expectEqual(@as(u64, 2032), table.rows_per_shard);
}

test "the real head table covers the 51.2 GB n-gram table exactly" {
    const table = try deriveHeadTable(&real_spec);
    try std.testing.expectEqual(@as(u32, 16), table.head_count);
    try std.testing.expectEqual(@as(u32, 8), table.heads_per_ngram);
    // The checkpoint's table is 128 shards of 2,500,012 rows of 160 F8_E4M3
    // bytes: 51,200,245,760 Table bytes, and every derived id lands inside it.
    try std.testing.expectEqual(@as(u64, 2_500_012), table.rows_per_shard);
    try std.testing.expectEqual(@as(u64, 320_001_536), table.padded_rows);
    try std.testing.expectEqual(
        @as(u64, 51_200_245_760),
        table.padded_rows * 160,
    );
    try std.testing.expect(table.total_vocab <= table.padded_rows);
    try std.testing.expect(table.total_vocab > table.padded_rows - 128);

    for (0..table.head_count) |head| {
        const size = reference_ple.find_nth_prime_after(
            real_spec.ngram_vocab_size_base - 1,
            @intCast(head + 1),
        );
        try std.testing.expectEqual(@as(i64, @intCast(size)), table.sizes[head]);
    }
    var oracle_multipliers: [ngram_max]i64 = undefined;
    reference_ple.build_layer_multipliers(
        real_spec.vocab_size,
        real_spec.ngram_size,
        real_spec.ple_layer_index,
        real_spec.seed,
        oracle_multipliers[0..],
    );
    for (0..real_spec.ngram_size) |order| {
        try std.testing.expectEqual(
            oracle_multipliers[order],
            table.multipliers[order],
        );
    }
    // The real shapes, pinned: 4 streams of 2560, 16 heads of 160.
    try std.testing.expectEqual(@as(u32, 10240), real_shapes.hyperSize());
    try std.testing.expectEqual(@as(u32, 9), real_shapes.stateLen());
}

/// The reference PLE of the tiny checkpoint, driven one section at a time so the
/// intermediate the block needs (the residual entering layer 1) is available.
const Oracle = struct {
    allocator: std.mem.Allocator,
    checkpoint: reference_fixtures.Tensors,
    trace: reference_fixtures.Tensors,
    model: reference_model.Model,
    scratch: reference_model.Scratch,
    ple_scratch: reference_ple.Scratch,
    hyper_in: []f32,
    output: []f32,
    /// Tokens already through layer 0, i.e. the next absolute position.
    past: u32,

    fn open(allocator: std.mem.Allocator) !Oracle {
        assert(allocator.vtable != undefined);
        assert(fixtures_root.len > 0);

        var checkpoint = try reference_fixtures.open_checkpoint(allocator);
        errdefer checkpoint.close(allocator);
        var trace = try reference_fixtures.open_trace(allocator);
        errdefer trace.close(allocator);

        var oracle = Oracle{
            .allocator = allocator,
            .checkpoint = checkpoint,
            .trace = trace,
            .model = undefined,
            .scratch = undefined,
            .ple_scratch = undefined,
            .hyper_in = &.{},
            .output = &.{},
            .past = 0,
        };
        try oracle.model.init(allocator, &oracle.checkpoint);
        errdefer oracle.model.deinit(allocator);
        oracle.scratch = try reference_model.Scratch.init(
            allocator,
            &oracle.model.config,
        );
        errdefer oracle.scratch.deinit(allocator);
        const tokens: usize = 32;
        oracle.ple_scratch = try reference_ple.Scratch.init(
            allocator,
            &oracle.model.config,
        );
        oracle.hyper_in = try allocator.alloc(
            f32,
            tokens * oracle.model.config.hyper_size(),
        );
        oracle.output = try allocator.alloc(
            f32,
            tokens * oracle.model.config.hyper_size(),
        );
        assert(oracle.hyper_in.len > 0);
        return oracle;
    }

    fn close(self: *Oracle) void {
        assert(self.output.len > 0);
        assert(self.model.config.num_hidden_layers > 0);

        self.allocator.free(self.output);
        self.allocator.free(self.hyper_in);
        self.ple_scratch.deinit(self.allocator);
        self.scratch.deinit(self.allocator);
        self.model.deinit(self.allocator);
        self.trace.close(self.allocator);
        self.checkpoint.close(self.allocator);
    }

    /// Runs the golden trace's layer 0 and returns the residual entering layer 1.
    fn advance(self: *Oracle, ids: []const i64, tokens: u32) ![]const f32 {
        assert(tokens > 0);
        assert(ids.len == tokens);

        const config = &self.model.config;
        const hidden: usize = config.hidden_size;
        const hyper: usize = config.hyper_size();
        for (0..tokens) |t| {
            self.scratch.positions[t] = self.past + @as(u32, @intCast(t));
        }
        ops.embed_gather(
            self.model.weights.embed_tokens,
            config.hidden_size,
            ids,
            self.scratch.embed[0 .. tokens * hidden],
        );
        for (0..tokens) |t| {
            for (0..config.hc_count) |c| {
                for (0..hidden) |h| {
                    self.scratch.hyper[t * hyper + c * hidden + h] =
                        self.scratch.embed[t * hidden + h];
                }
            }
        }
        const table = reference_attention.Rope{
            .cos = self.scratch.cos,
            .sin = self.scratch.sin,
        };
        const recurrent = tokens == 1 and self.past > 0;
        reference_layer.forward(
            self.scratch.hyper[0 .. tokens * hyper],
            tokens,
            self.scratch.positions[0..tokens],
            ids,
            &self.model.weights.layers[0],
            &self.model.cache.layers[0],
            &self.scratch.layer,
            table,
            config,
            recurrent,
            null,
        );
        @memcpy(
            self.hyper_in[0 .. tokens * hyper],
            self.scratch.hyper[0 .. tokens * hyper],
        );
        self.past += tokens;
        return self.hyper_in[0 .. tokens * hyper];
    }

    /// The reference PLE block output for the residual `advance` produced.
    fn pleForward(self: *Oracle, ids: []const i64, tokens: u32) ![]const f32 {
        assert(tokens > 0);
        assert(ids.len == tokens);

        const config = &self.model.config;
        const hyper: usize = config.hyper_size();
        const weights = if (self.model.weights.layers[1].ple) |*value|
            value
        else
            return error.NoPleLayer;
        const state = if (self.model.cache.layers[1].ple) |*value|
            value
        else
            return error.NoPleLayer;
        reference_ple.forward(
            self.hyper_in[0 .. tokens * hyper],
            ids,
            tokens,
            self.output[0 .. tokens * hyper],
            weights,
            state,
            &self.ple_scratch,
            config,
        );
        return self.output[0 .. tokens * hyper];
    }

    /// The reference PLE intermediates of the last forward.
    fn stage(self: *const Oracle, comptime key: []const u8, tokens: u32) []const f32 {
        assert(tokens > 0);
        assert(key.len > 0);

        const config = &self.model.config;
        const width: usize = if (comptime std.mem.eql(u8, key, "embeddings"))
            config.ple_embed_dim
        else if (comptime std.mem.eql(u8, key, "value"))
            config.hidden_size
        else if (comptime std.mem.eql(u8, key, "gate"))
            config.hc_count
        else
            config.hyper_size();
        const length = tokens * width;
        if (comptime std.mem.eql(u8, key, "out")) {
            // The block output is the reference's return value, not scratch.
            return self.output[0 .. tokens * config.hyper_size()];
        } else if (comptime std.mem.eql(u8, key, "embeddings")) {
            return self.ple_scratch.embeddings[0..length];
        } else if (comptime std.mem.eql(u8, key, "key_normed")) {
            return self.ple_scratch.key_normed[0..length];
        } else if (comptime std.mem.eql(u8, key, "query_normed")) {
            return self.ple_scratch.query_normed[0..length];
        } else if (comptime std.mem.eql(u8, key, "value")) {
            return self.ple_scratch.value[0..length];
        } else if (comptime std.mem.eql(u8, key, "gate")) {
            return self.ple_scratch.gate[0..length];
        } else if (comptime std.mem.eql(u8, key, "gated_value")) {
            return self.ple_scratch.gated_value[0..length];
        } else if (comptime std.mem.eql(u8, key, "gated_value_normed")) {
            return self.ple_scratch.gated_value_normed[0..length];
        } else {
            return self.ple_scratch.conv_out[0..length];
        }
    }

    /// A golden f32 tensor of the trace.
    fn golden(self: *const Oracle, name: []const u8) ![]const f32 {
        assert(name.len > 0);
        assert(self.trace.entry_count > 0);

        const tensor = try self.trace.tensor(name);
        return tensor.values;
    }
};

test "the n-gram ids match the fp32 oracle, including across an EOS" {
    const allocator = std.testing.allocator;
    var oracle = try Oracle.open(allocator);
    defer oracle.close();

    const config = &oracle.model.config;
    const spec = specFromConfig(config, 1);
    const table = try deriveHeadTable(&spec);
    const heads: usize = table.head_count;
    try std.testing.expectEqual(@as(usize, 4), heads);

    const cases = [_][]const i64{
        &golden_prefill_ids,
        // The trap: an EOS in the middle of the batch, then decodes across it,
        // including two EOS in a row.
        &[_]i64{ 4, 199, 255, 42, 7, 88 },
        &[_]i64{ 7, 88, 255, 9 },
    };
    const decode_steps = [_][]const i64{
        &golden_decode_id,
        &[_]i64{ 12, 255 },
        &[_]i64{255},
    };
    var ids: [16 * heads_max]i64 = undefined;
    var stream = IdStream.init(&spec);
    for (cases, 0..) |case, case_index| {
        try std.testing.expectEqual(@as(usize, table.head_count), heads);
        // Each case is its own sequence: both sides start from an empty stream,
        // whose first two positions read the virtual EOS prefix.
        stream.reset();
        try std.testing.expectEqual(@as(u64, 0), stream.count);
        resetOraclePle(&oracle);
        stream.appendAll(&table, case, ids[0 .. case.len * heads]);
        try checkIds(&oracle, case, ids[0 .. case.len * heads]);

        for (decode_steps[case_index]) |token| {
            const single = [_]i64{token};
            stream.append(&table, token, ids[0..heads]);
            try checkIds(&oracle, &single, ids[0..heads]);
        }
    }

    // Non-vacuity: two different tokens must not produce the same id vector.
    stream.reset();
    stream.appendAll(&table, &golden_prefill_ids, ids[0 .. 12 * heads]);
    std.debug.print(
        "n-gram ids of token 0: {any}\n",
        .{ids[0..heads]},
    );
    std.debug.print(
        "n-gram ids of token 5: {any}\n",
        .{ids[5 * heads ..][0..heads]},
    );
    try std.testing.expect(
        !std.mem.eql(i64, ids[0..heads], ids[5 * heads ..][0..heads]),
    );
}

/// Runs the reference PLE for `tokens_in` and compares every id against `ids`.
fn checkIds(
    oracle: *Oracle,
    tokens_in: []const i64,
    ids: []const i64,
) !void {
    assert(tokens_in.len > 0);
    assert(ids.len == tokens_in.len * oracle.model.config.ngram_heads());

    const config = &oracle.model.config;
    const tokens: u32 = @intCast(tokens_in.len);
    const hyper: usize = config.hyper_size();
    // `forward` only needs a residual to read; zeros are enough because the ids
    // are derived before anything else happens.
    @memset(oracle.hyper_in[0 .. tokens * hyper], 0.0);
    const weights = if (oracle.model.weights.layers[1].ple) |*value|
        value
    else
        return error.NoPleLayer;
    const state = if (oracle.model.cache.layers[1].ple) |*value|
        value
    else
        return error.NoPleLayer;
    reference_ple.forward(
        oracle.hyper_in[0 .. tokens * hyper],
        tokens_in,
        tokens,
        oracle.output[0 .. tokens * hyper],
        weights,
        state,
        &oracle.ple_scratch,
        config,
    );
    const reference_ids = oracle.ple_scratch.ngram_ids[0 .. tokens_in.len *
        config.ngram_heads()];
    try std.testing.expectEqualSlices(i64, reference_ids, ids);
}

/// Restores the reference PLE state to the start of a sequence.
fn resetOraclePle(oracle: *Oracle) void {
    assert(oracle.model.config.ngram_size >= 2);
    assert(oracle.model.config.num_hidden_layers > 1);

    if (oracle.model.cache.layers[1].ple) |*state| {
        state.reset(&oracle.model.config);
    } else {
        assert(oracle.model.cache.layers[1].ple == null);
    }
}

test "the gather fetches the bytes at the id's file offset" {
    const allocator = std.testing.allocator;
    var tensors = try reference_fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);
    const config = try reference_fixtures.load_config(allocator);
    const spec = specFromConfig(&config, 1);
    const table_ids = try deriveHeadTable(&spec);
    const shapes = shapesFromConfig(&config);
    const path = try fixturePath(allocator, "checkpoint.safetensors");
    defer allocator.free(path);

    var table = try Table.open(
        path,
        table_prefix,
        config.split_ngram_parts,
        shapes.head_dim,
        .f32,
    );
    defer table.close();
    try std.testing.expectEqual(table_ids.rows_per_shard, table.rows_per_shard);
    try std.testing.expectEqual(@as(u32, 2), table.shard_count);

    const heads: usize = table_ids.head_count;
    const tokens_in = [4]i64{ 13, 4, 199, 42 };
    var stream = IdStream.init(&spec);
    var ids: [4 * heads_max]i64 = undefined;
    stream.appendAll(&table_ids, &tokens_in, ids[0 .. tokens_in.len * heads]);

    const width: u32 = @intCast(table.rowBytes());
    const rows = ids[0 .. tokens_in.len * heads];
    const staged = try allocator.alloc(u8, rows.len * width);
    defer allocator.free(staged);
    const pool = try GatherPool.create(allocator, &table, gather_workers_max);
    defer pool.destroy(allocator);
    try pool.gather(rows, staged);

    var read_buffer: [256]u8 = undefined;
    var name_buffer: [256]u8 = undefined;
    try std.testing.expect(width <= read_buffer.len);
    for (rows, 0..) |id, index| {
        // Route A: the mapping.
        const mapped = table.row(id);
        const at = table.location(id);
        try std.testing.expectEqual(width, mapped.len);
        // Route B: a read of the same file offset through the descriptor.
        const read = try std.posix.pread(
            table.handle.handle,
            read_buffer[0..width],
            at.offset,
        );
        try std.testing.expectEqual(width, read);
        try std.testing.expectEqualSlices(u8, mapped, read_buffer[0..width]);
        // Route C: the safetensors view of the shard the reference reads.
        const shard_name = try std.fmt.bufPrint(
            &name_buffer,
            "{s}{d}.weight",
            .{ table_prefix, at.shard },
        );
        const shard = try tensors.tensor(shard_name);
        const shard_bytes = std.mem.sliceAsBytes(shard.values);
        try std.testing.expectEqualSlices(
            u8,
            mapped,
            shard_bytes[@as(usize, at.row) * width ..][0..width],
        );
        // And the worker pool copied that same row.
        try std.testing.expectEqualSlices(
            u8,
            mapped,
            staged[index * width ..][0..width],
        );
    }
    // Non-vacuity: consecutive tokens read different rows of the table.
    try std.testing.expect(
        !std.mem.eql(u8, staged[0..width], staged[width..][0..width]),
    );
    std.debug.print(
        "tiny table: {d} shards, {d} rows/shard, {d} B/row, f32\n",
        .{ table.shard_count, table.rows_per_shard, width },
    );
}

test "the real table's rows are identical through the mapping and pread" {
    const allocator = std.testing.allocator;
    const maybe_path = try realTablePath(allocator);
    if (maybe_path == null) {
        std.debug.print(
            "SKIP: no real n-gram table (set {s} to the checkpoint)\n",
            .{checkpoint_variable},
        );
        return error.SkipZigTest;
    }
    const path = maybe_path.?;
    defer allocator.free(path);

    const table_ids = try deriveHeadTable(&real_spec);
    var table = try Table.open(
        path,
        table_prefix,
        real_spec.split_ngram_parts,
        real_shapes.head_dim,
        .e4m3,
    );
    defer table.close();
    try std.testing.expectEqual(
        table_ids.rows_per_shard,
        table.rows_per_shard,
    );
    try std.testing.expectEqual(@as(u32, 128), table.shard_count);
    try std.testing.expectEqual(@as(u64, 160), table.rowBytes());

    const heads: usize = table_ids.head_count;
    const tokens_in = [4]i64{ 0, 1, 2, 3 };
    var stream = IdStream.init(&real_spec);
    var ids: [4 * heads_max]i64 = undefined;
    stream.appendAll(&table_ids, &tokens_in, ids[0 .. tokens_in.len * heads]);

    const pool = try GatherPool.create(allocator, &table, gather_workers_max);
    defer pool.destroy(allocator);
    var staged: [4 * heads_max * 160]u8 = undefined;
    const rows = ids[0 .. tokens_in.len * heads];
    try pool.gather(rows, staged[0 .. rows.len * 160]);

    var read_buffer: [160]u8 = undefined;
    var distinct: u32 = 0;
    for (rows, 0..) |id, index| {
        const mapped = table.row(id);
        const at = table.location(id);
        const read = try std.posix.pread(
            table.handle.handle,
            &read_buffer,
            at.offset,
        );
        try std.testing.expectEqual(@as(usize, 160), read);
        try std.testing.expectEqualSlices(u8, mapped, &read_buffer);
        try std.testing.expectEqualSlices(
            u8,
            mapped,
            staged[index * 160 ..][0..160],
        );
        if (index > 0) {
            const previous = staged[(index - 1) * 160 ..][0..160];
            if (!std.mem.eql(u8, mapped, previous)) distinct += 1;
        }
    }
    try std.testing.expect(distinct > 0);
    std.debug.print(
        "real table: {d} shards, {d} rows/shard, {d:.1} GB on disk, " ++
            "{d} B/row, f8_e4m3\n",
        .{
            table.shard_count,
            table.rows_per_shard,
            @as(f64, @floatFromInt(table.mapping.len)) / 1e9,
            table.rowBytes(),
        },
    );
}

test "the e4m3 cast decodes every code like the quant decoder" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open(allocator);
    defer harness.close();

    const count: u32 = 32;
    const head_dim: u32 = 8;
    const total: usize = count * head_dim;
    try std.testing.expectEqual(@as(usize, 256), total);
    var codes: [256]u8 = undefined;
    for (&codes, 0..) |*code, index| {
        code.* = @intCast(index);
    }
    const scale: f32 = 41.999;
    const rows = try harness.upload(&codes);
    const decoded_pointer = try harness.allocate(total * @sizeOf(f32));
    var decoded: [256]f32 = undefined;
    try harness.ple.castRows(.{
        .rows = rows,
        .out = decoded_pointer,
        .count = count,
        .head_dim = head_dim,
        .format = .e4m3,
        .scale = scale,
    });
    try harness.ple.stream.synchronize();
    try harness.downloadF32(decoded_pointer, &decoded);

    // The Zig decoder is the oracle: every code must decode to the same f32, and
    // the scale multiply must be the same one instruction.
    var original: [256]f32 = undefined;
    var largest: f64 = 0.0;
    var nan_codes: u32 = 0;
    for (&original, 0..) |*want, index| {
        const code: u8 = @intCast(index);
        want.* = quant.f8_e4m3_to_f32(@enumFromInt(code));
        const expected = want.* * scale;
        if (std.math.isNan(expected)) {
            // Only codes 0x7f and 0xff are NaN in E4M3 (FN).
            try std.testing.expect(std.math.isNan(decoded[index]));
            nan_codes += 1;
        } else {
            try std.testing.expectEqual(expected, decoded[index]);
            const error_value = @abs(
                @as(f64, decoded[index]) - @as(f64, expected),
            );
            largest = @max(largest, error_value);
        }
    }
    try std.testing.expectEqual(@as(u32, 2), nan_codes);
    try std.testing.expectEqual(@as(f64, 0.0), largest);

    // The f32 route is the identity on the fixture's rows.
    const f32_rows = try harness.uploadF32(&original);
    const identity_pointer = try harness.allocate(total * @sizeOf(f32));
    try harness.ple.castRows(.{
        .rows = f32_rows,
        .out = identity_pointer,
        .count = count,
        .head_dim = head_dim,
        .format = .f32,
        .scale = 1.0,
    });
    try harness.ple.stream.synchronize();
    var identity: [256]f32 = undefined;
    try harness.downloadF32(identity_pointer, &identity);
    for (identity, original) |got, want| {
        if (std.math.isNan(want)) {
            try std.testing.expect(std.math.isNan(got));
        } else {
            try std.testing.expectEqual(want, got);
        }
    }
    std.debug.print(
        "e4m3 cast: 256 codes decoded exactly, scale {d}\n",
        .{scale},
    );
}

/// Downloads every stage buffer of `block` and compares it against the oracle's
/// intermediates of the same forward and against the golden trace section.
fn checkStages(
    allocator: std.mem.Allocator,
    harness: *Harness,
    oracle: *Oracle,
    block: *const Block,
    tokens: u32,
    section: []const u8,
) !void {
    assert(tokens > 0);
    assert(section.len > 0);

    const config = &oracle.model.config;
    const hyper: usize = config.hyper_size();
    const downloaded = try allocator.alloc(f32, tokens * hyper);
    defer allocator.free(downloaded);
    inline for (.{
        .{ "embeddings", block.buffers.embeddings.pointer },
        .{ "key_normed", block.buffers.key_normed.pointer },
        .{ "query_normed", block.buffers.query_normed.pointer },
        .{ "value", block.buffers.value.pointer },
        .{ "gate", block.buffers.gates.pointer },
        .{ "gated_value", block.buffers.gated_value.pointer },
        .{ "gated_value_normed", block.buffers.gated_value_normed.pointer },
        .{ "out", block.buffers.out.pointer },
    }) |entry| {
        const name = entry[0];
        const expected = oracle.stage(name, tokens);
        try std.testing.expect(expected.len <= downloaded.len);
        try harness.downloadF32(entry[1], downloaded[0..expected.len]);
        var label_buffer: [128]u8 = undefined;
        const label = try std.fmt.bufPrint(
            &label_buffer,
            "{s}/ple/{s}",
            .{ section, name },
        );
        try expectStage(
            label,
            downloaded[0..expected.len],
            expected,
            oracle.golden(label) catch null,
        );
    }
    assert(hyper > 0);
}

/// Reconstructs the conv output (`out - gated_value`) and compares it with the
/// golden section's channel-major `conv_out`.
fn checkConvOutput(
    allocator: std.mem.Allocator,
    harness: *Harness,
    oracle: *Oracle,
    block: *const Block,
    tokens: u32,
    section: []const u8,
) !void {
    assert(tokens > 0);
    assert(section.len > 0);

    const config = &oracle.model.config;
    const hyper: usize = config.hyper_size();
    const out_values = try allocator.alloc(f32, tokens * hyper);
    defer allocator.free(out_values);
    const reconstructed = try allocator.alloc(f32, tokens * hyper);
    defer allocator.free(reconstructed);
    try harness.downloadF32(block.buffers.out.pointer, out_values);
    const gated = oracle.stage("gated_value", tokens);
    for (0..tokens) |token| {
        for (0..hyper) |channel| {
            reconstructed[channel * tokens + token] =
                out_values[token * hyper + channel] -
                gated[token * hyper + channel];
        }
    }
    var label_buffer: [128]u8 = undefined;
    const label = try std.fmt.bufPrint(
        &label_buffer,
        "{s}/ple/conv_out",
        .{section},
    );
    const golden = try oracle.golden(label);
    const deviation = printDeviation(label, reconstructed, golden);
    const stats = meanAndDeviation(reconstructed);
    std.debug.print(
        "{s} mean {e:.4} std {e:.4}\n",
        .{ label, stats[0], stats[1] },
    );
    try std.testing.expect(deviation.ratio <= 2e-5);
}

/// Checks that the block's output was added into the residual in place.
fn checkResidual(
    allocator: std.mem.Allocator,
    harness: *Harness,
    block: *const Block,
    residual: cuda_driver.DevicePtr,
    input: []const f32,
    tokens: u32,
    hyper: usize,
) !void {
    assert(tokens > 0);
    assert(input.len == tokens * hyper);

    const values = try allocator.alloc(f32, tokens * hyper);
    defer allocator.free(values);
    const added = try allocator.alloc(f32, tokens * hyper);
    defer allocator.free(added);
    try harness.downloadF32(residual, values);
    try harness.downloadF32(block.buffers.out.pointer, added);
    for (values, input, added) |got, before, out| {
        try std.testing.expectEqual(before + out, got);
    }
    assert(values.len == input.len);
}

test "the PLE block matches the HF golden layer1/ple/out" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open(allocator);
    defer harness.close();
    var oracle = try Oracle.open(allocator);
    defer oracle.close();

    const config = &oracle.model.config;
    const spec = specFromConfig(config, 1);
    const shapes = shapesFromConfig(config);
    const table_ids = try deriveHeadTable(&spec);
    const hyper: usize = config.hyper_size();
    const heads: usize = table_ids.head_count;
    try std.testing.expectEqual(@as(usize, 256), hyper);

    const path = try fixturePath(allocator, "checkpoint.safetensors");
    defer allocator.free(path);
    var table = try Table.open(
        path,
        table_prefix,
        config.split_ngram_parts,
        shapes.head_dim,
        .f32,
    );
    defer table.close();
    try std.testing.expectEqual(table_ids.rows_per_shard, table.rows_per_shard);

    const pool = try GatherPool.create(allocator, &table, gather_workers_max);
    defer pool.destroy(allocator);
    const reference_weights = if (oracle.model.weights.layers[1].ple) |*value|
        value
    else
        return error.NoPleLayer;
    const weights = Weights{
        .key_proj = try harness.uploadF32(reference_weights.key_proj),
        .value_proj = try harness.uploadF32(reference_weights.value_proj),
        .norm_key = try harness.uploadF32(reference_weights.norm_key),
        .norm_query = try harness.uploadF32(reference_weights.norm_query),
        .norm_conv = try harness.uploadF32(reference_weights.norm_conv),
        .taps = try harness.uploadF32(reference_weights.conv),
        .table_format = .f32,
        .table_scale = 1.0,
    };
    var block = try Block.init(&harness.ple, &table, pool, weights, shapes, 32);
    defer block.deinit();
    try block.reset();

    const ids = try allocator.alloc(i64, 32 * heads);
    defer allocator.free(ids);
    var stream = IdStream.init(&spec);

    // Prefill: the golden trace's twelve tokens.
    const tokens: u32 = @intCast(golden_prefill_ids.len);
    stream.appendAll(&table_ids, &golden_prefill_ids, ids[0 .. tokens * heads]);
    const residual = try oracle.advance(&golden_prefill_ids, tokens);
    const expected = try oracle.pleForward(&golden_prefill_ids, tokens);
    const hidden_device = try harness.uploadF32(residual);
    const residual_device = try harness.uploadF32(residual);
    try block.run(ids[0 .. tokens * heads], tokens, hidden_device, residual_device);
    try checkStages(allocator, &harness, &oracle, &block, tokens, "layer1");
    try checkConvOutput(allocator, &harness, &oracle, &block, tokens, "layer1");
    try checkResidual(
        allocator,
        &harness,
        &block,
        residual_device,
        residual,
        tokens,
        hyper,
    );
    {
        var downloaded: [12 * 256]f32 = undefined;
        try std.testing.expectEqual(tokens * hyper, downloaded.len);
        try harness.downloadF32(block.buffers.out.pointer, &downloaded);
        const deviation = printDeviation(
            "layer1/ple/out vs oracle",
            &downloaded,
            expected,
        );
        try std.testing.expect(deviation.ratio <= scaled_error_max);
        try expectStage(
            "layer1/ple/out",
            &downloaded,
            null,
            try oracle.golden("layer1/ple/out"),
        );
        const stats = meanAndDeviation(&downloaded);
        std.debug.print(
            "layer1/ple/out mean {e:.4} std {e:.4}\n",
            .{ stats[0], stats[1] },
        );
    }

    // Decode: one token, through the conv ring buffer the prefill left behind.
    stream.append(&table_ids, golden_decode_id[0], ids[0..heads]);
    const residual_decode = try oracle.advance(&golden_decode_id, 1);
    const expected_decode = try oracle.pleForward(&golden_decode_id, 1);
    const hidden_decode = try harness.uploadF32(residual_decode);
    const residual_decode_device = try harness.uploadF32(residual_decode);
    try block.run(ids[0..heads], 1, hidden_decode, residual_decode_device);
    try checkStages(
        allocator,
        &harness,
        &oracle,
        &block,
        1,
        "decode/layer1",
    );
    try checkConvOutput(allocator, &harness, &oracle, &block, 1, "decode/layer1");
    {
        var downloaded: [256]f32 = undefined;
        try harness.downloadF32(block.buffers.out.pointer, &downloaded);
        const deviation = printDeviation(
            "decode/layer1/ple/out vs oracle",
            &downloaded,
            expected_decode,
        );
        try std.testing.expect(deviation.ratio <= scaled_error_max);
        try expectStage(
            "decode/layer1/ple/out",
            &downloaded,
            null,
            try oracle.golden("decode/layer1/ple/out"),
        );
    }

    // Non-vacuity: the ids of two tokens differ, and the gate covers both signs.
    try std.testing.expect(!std.mem.eql(
        i64,
        ids[0..heads],
        ids[heads..][0..heads],
    ));
    var gate_values: [32 * 2]f32 = undefined;
    try harness.downloadF32(
        block.buffers.gates.pointer,
        gate_values[0 .. 13 * shapes.hc_count],
    );
    var lowest: f32 = gate_values[0];
    var highest: f32 = gate_values[0];
    for (gate_values[0 .. 13 * shapes.hc_count]) |value| {
        lowest = @min(lowest, value);
        highest = @max(highest, value);
    }
    std.debug.print(
        "gate range [{e:.4}, {e:.4}] over {d} values (negative present: {})\n",
        .{ lowest, highest, 13 * shapes.hc_count, lowest < 0.0 },
    );
    try std.testing.expect(lowest < 0.0);
    try std.testing.expect(highest > 0.0);
}

// ---------------------------------------------------------------------------
// The gather on the real 51.2 GB table: correctness of the concurrent route and
// the per-token latency the issue asks for.
// ---------------------------------------------------------------------------

/// Tokens the latency loop measures; 256 tokens is 4096 rows.
const latency_tokens: u32 = 256;

/// Ids each derived candidate stream holds, as a multiple of the tokens timed:
/// a cold pass consumes 16 rows per token, so it needs more candidates than
/// tokens while the page cache is filling up.
const candidate_tokens: u32 = 4 * latency_tokens;

/// Single rows the two residency probes time.
const row_probe_samples: usize = 512;

/// Table pages the residency snapshot samples.
const residency_samples: u32 = 512;

/// Widest table row a probe reads (the real table's row is 160 bytes).
const row_probe_bytes_max: u32 = 1024;

/// p50, p99 and the maximum of `samples`, in the samples' own order.
fn percentiles(samples: []u64, sorted: []u64) [3]u64 {
    assert(samples.len > 0);
    assert(sorted.len == samples.len);

    @memcpy(sorted, samples);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
    const last = sorted.len - 1;
    const p50 = sorted[sorted.len / 2];
    const p99 = sorted[@min(last, (sorted.len * 99) / 100)];
    assert(p50 <= sorted[last]);
    return .{ p50, p99, sorted[last] };
}

/// Drops the pages holding `ids`' rows from the page cache.
///
/// `POSIX_FADV_DONTNEED` is what makes a cold measurement possible at all on a
/// machine whose 60 GB of RAM can hold the whole 51.2 GB table: without it a
/// "latency" is a memory copy. Returns how many of the calls failed.
fn evictRows(table: *const Table, ids: []const i64) u32 {
    assert(ids.len > 0);
    assert(table.rowBytes() > 0);

    const page: u64 = std.heap.page_size_min;
    var failures: u32 = 0;
    for (ids) |id| {
        const at = table.location(id);
        const start = at.offset / page * page;
        const end = at.offset + table.rowBytes();
        const result = std.os.linux.fadvise(
            table.handle.handle,
            @intCast(start),
            @intCast(end - start),
            std.os.linux.POSIX_FADV.DONTNEED,
        );
        if (std.os.linux.E.init(result) != .SUCCESS) failures += 1;
    }
    assert(failures <= ids.len);
    return failures;
}

/// Times the pooled and the serial gather on the same token, over two disjoint
/// halves of an evicted row batch.
///
/// The halves must be disjoint: whichever gather ran second would otherwise find
/// the rows the first one read already in the page cache. Which half goes to
/// which pool alternates per token, so a positional bias in what the page cache
/// still holds cannot favour either side.
fn timeGatherPair(
    pool: *GatherPool,
    serial: *GatherPool,
    ids: []const i64,
    heads: u32,
    staged: []u8,
    pool_samples: []u64,
    serial_samples: []u64,
    evict: bool,
) !u32 {
    assert(ids.len > 0);
    assert(pool_samples.len <= ids.len / (2 * heads));
    assert(pool_samples.len == serial_samples.len);

    const row_bytes: usize = @intCast(pool.table.rowBytes());
    const token_bytes = row_bytes * heads;
    assert(staged.len == 2 * token_bytes);
    var failures: u32 = 0;
    var timer = try std.time.Timer.start();
    for (pool_samples, 0..) |*pool_sample, token| {
        const rows = ids[token * 2 * heads ..][0..(2 * heads)];
        if (evict) failures += evictRows(pool.table, rows);
        const first = if (token % 2 == 0) pool else serial;
        const second = if (token % 2 == 0) serial else pool;
        timer.reset();
        try first.gather(rows[0..heads], staged[0..token_bytes]);
        const first_sample = timer.read();
        timer.reset();
        try second.gather(rows[heads..], staged[token_bytes..][0..token_bytes]);
        const second_sample = timer.read();
        if (token % 2 == 0) {
            pool_sample.* = first_sample;
            serial_samples[token] = second_sample;
        } else {
            pool_sample.* = second_sample;
            serial_samples[token] = first_sample;
        }
        assert(pool_sample.* > 0);
    }
    return failures;
}

/// Times the same rows twice: once right after their pages were evicted, and once
/// again immediately after, when they are back in the page cache.
///
/// The pair is the control for the whole measurement: if the evicted read is not
/// slower than the second one, the eviction did nothing and the "device" column
/// was really a memory copy.
fn timeRowPair(
    table: *const Table,
    ids: []const i64,
    evicted: []u64,
    cached: []u64,
) !u32 {
    assert(ids.len >= evicted.len);
    assert(evicted.len == cached.len);

    const width: usize = @intCast(table.rowBytes());
    var row: [row_probe_bytes_max]u8 = undefined;
    assert(width <= row.len);
    var failures: u32 = 0;
    var timer = try std.time.Timer.start();
    for (evicted, 0..) |*sample, index| {
        const at = table.location(ids[index]);
        failures += evictRows(table, ids[index..][0..1]);
        timer.reset();
        try table.readRow(at, row[0..width]);
        sample.* = timer.read();
        timer.reset();
        try table.readRow(at, row[0..width]);
        cached[index] = timer.read();
        assert(sample.* > 0);
        assert(cached[index] > 0);
    }
    return failures;
}

/// Prints one latency pass: p50/p99/max per token and the read rate.
fn reportLatency(
    label: []const u8,
    stats: [3]u64,
    heads: u32,
    row_bytes: u32,
) void {
    assert(label.len > 0);
    assert(stats[0] > 0);

    const requested = @as(f64, @floatFromInt(heads)) *
        @as(f64, @floatFromInt(row_bytes));
    const pages = @as(f64, @floatFromInt(heads)) *
        @as(f64, @floatFromInt(std.heap.page_size_min));
    const p50: f64 = @as(f64, @floatFromInt(stats[0])) / 1e9;
    std.debug.print(
        "  {s:<30} p50 {d:>6} us  p99 {d:>6} us  max {d:>7} us  " ++
            "{d:>7.1} MB/s requested  {d:>8.1} MB/s at page granularity\n",
        .{
            label,
            stats[0] / 1000,
            stats[1] / 1000,
            stats[2] / 1000,
            requested / p50 / 1e6,
            pages / p50 / 1e6,
        },
    );
}

/// Derives `latency_tokens` tokens' ids from `seed`, exactly as the engine does.
fn deriveLatencyIds(
    allocator: std.mem.Allocator,
    table_ids: *const HeadTable,
    seed: u64,
) ![]i64 {
    assert(allocator.vtable != undefined);
    assert(table_ids.head_count > 0);

    const heads: usize = table_ids.head_count;
    const ids = try allocator.alloc(i64, candidate_tokens * heads);
    errdefer allocator.free(ids);
    const tokens_in = try allocator.alloc(i64, candidate_tokens);
    defer allocator.free(tokens_in);
    var random = std.Random.DefaultPrng.init(seed);
    for (tokens_in) |*token| {
        token.* = random.random().uintLessThan(u32, 248320);
    }
    var stream = IdStream.init(&real_spec);
    stream.appendAll(table_ids, tokens_in, ids);
    assert(stream.count == candidate_tokens);
    return ids;
}

test "the gather latency per token on the real 51.2 GB table" {
    const allocator = std.testing.allocator;
    const maybe_path = try realTablePath(allocator);
    if (maybe_path == null) {
        std.debug.print(
            "SKIP: no real n-gram table (set {s} to the checkpoint)\n",
            .{checkpoint_variable},
        );
        return error.SkipZigTest;
    }
    const path = maybe_path.?;
    defer allocator.free(path);

    const table_ids = try deriveHeadTable(&real_spec);
    const heads: u32 = table_ids.head_count;
    var table = try Table.open(
        path,
        table_prefix,
        real_spec.split_ngram_parts,
        real_shapes.head_dim,
        .e4m3,
    );
    defer table.close();

    // The gather is host I/O, so this test needs no GPU.
    const row_bytes: u32 = @intCast(table.rowBytes());
    const token_bytes: usize = row_bytes * heads;
    const staged = try allocator.alignedAlloc(
        u8,
        std.heap.page_size_min,
        2 * token_bytes,
    );
    defer allocator.free(staged);
    const pool = try GatherPool.create(allocator, &table, gather_workers_max);
    defer pool.destroy(allocator);
    const serial = try GatherPool.create(allocator, &table, 1);
    defer serial.destroy(allocator);

    // One candidate stream per residency treatment: a pass reads its rows, so a
    // second pass over the same ids would measure the page cache the first
    // pass filled.
    const evicted_ids = try deriveLatencyIds(allocator, &table_ids, 0x0110_0001);
    defer allocator.free(evicted_ids);
    const cached_ids = try deriveLatencyIds(allocator, &table_ids, 0x0110_0002);
    defer allocator.free(cached_ids);
    const probe_ids = try deriveLatencyIds(allocator, &table_ids, 0x0110_0003);
    defer allocator.free(probe_ids);

    const evicted_pool = try allocator.alloc(u64, latency_tokens);
    defer allocator.free(evicted_pool);
    const evicted_serial = try allocator.alloc(u64, latency_tokens);
    defer allocator.free(evicted_serial);
    const cached_pool = try allocator.alloc(u64, latency_tokens);
    defer allocator.free(cached_pool);
    const cached_serial = try allocator.alloc(u64, latency_tokens);
    defer allocator.free(cached_serial);
    const evicted_row = try allocator.alloc(u64, row_probe_samples);
    defer allocator.free(evicted_row);
    const cached_row = try allocator.alloc(u64, row_probe_samples);
    defer allocator.free(cached_row);
    const sorted = try allocator.alloc(u64, latency_tokens);
    defer allocator.free(sorted);
    const row_sorted = try allocator.alloc(u64, row_probe_samples);
    defer allocator.free(row_sorted);

    // The evicted passes run first, so nothing this test did has warmed the rows.
    const row_failures = try timeRowPair(
        &table,
        probe_ids,
        evicted_row,
        cached_row,
    );
    const evicted_failures = try timeGatherPair(
        pool,
        serial,
        evicted_ids,
        heads,
        staged,
        evicted_pool,
        evicted_serial,
        true,
    );
    // The last pair runs without eviction, to show what the gather costs when the
    // page cache already holds the rows.
    _ = try timeGatherPair(
        pool,
        serial,
        cached_ids,
        heads,
        staged,
        cached_pool,
        cached_serial,
        false,
    );

    {
        const at = table.location(probe_ids[0]);
        const page: u64 = std.heap.page_size_min;
        const start = at.offset / page * page;
        var vector: [1]u8 = .{0};
        const address = table.mapping.ptr + @as(usize, @intCast(start));
        _ = std.os.linux.mincore(address, @intCast(page), &vector);
        const before = vector[0] & 1;
        _ = std.os.linux.fadvise(table.handle.handle, @intCast(start), @intCast(page), std.os.linux.POSIX_FADV.DONTNEED);
        vector[0] = 0;
        _ = std.os.linux.mincore(address, @intCast(page), &vector);
        const after = vector[0] & 1;
        var timer = try std.time.Timer.start();
        var probe_buffer: [160]u8 = undefined;
        _ = try std.posix.pread(table.handle.handle, &probe_buffer, at.offset);
        const read_ns = timer.read();
        std.debug.print("DBG resident before {d} after {d}, read {d} us\n", .{ before, after, read_ns / 1000 });
        std.debug.print("DBG evicted samples us:", .{});
        for (evicted_row[0..24]) |value| std.debug.print(" {d}", .{value / 1000});
        std.debug.print("\n", .{});
    }
    const evicted_row_stats = percentiles(evicted_row, row_sorted);
    const cached_row_stats = percentiles(cached_row, row_sorted);
    const evicted_pool_stats = percentiles(evicted_pool, sorted);
    const evicted_serial_stats = percentiles(evicted_serial, sorted);
    const cached_pool_stats = percentiles(cached_pool, sorted);
    const cached_serial_stats = percentiles(cached_serial, sorted);

    std.debug.print(
        "n-gram gather on the real table ({d:.1} GB, {d} shards x {d} rows, " ++
            "{d} B/row): {d} rows = {d} B requested per token\n",
        .{
            @as(f64, @floatFromInt(table.mapping.len)) / 1e9,
            table.shard_count,
            table.rows_per_shard,
            row_bytes,
            heads,
            token_bytes,
        },
    );
    std.debug.print(
        "  one row, page evicted:  p50 {d} us p99 {d} us  ({d} of {d} " ++
            "evictions failed)\n",
        .{
            evicted_row_stats[0] / 1000,
            evicted_row_stats[1] / 1000,
            row_failures,
            row_probe_samples,
        },
    );
    std.debug.print(
        "  the same rows, warm:    p50 {d} us p99 {d} us  = the control for " ++
            "whether the eviction reached the device\n",
        .{ cached_row_stats[0] / 1000, cached_row_stats[1] / 1000 },
    );
    reportLatency(
        "16 evicted rows, 16 workers",
        evicted_pool_stats,
        heads,
        row_bytes,
    );
    reportLatency(
        "16 evicted rows, 1 worker",
        evicted_serial_stats,
        heads,
        row_bytes,
    );
    reportLatency(
        "16 rows, no eviction, 16 w",
        cached_pool_stats,
        heads,
        row_bytes,
    );
    reportLatency(
        "16 rows, no eviction, 1 w",
        cached_serial_stats,
        heads,
        row_bytes,
    );
    std.debug.print(
        "  concurrency win, evicted: {d:.2}x at p50, {d:.2}x at p99; cached: " ++
            "{d:.2}x at p50. The evicted and cached rows are the same token " ++
            "batch split in half ({d} of {d} evictions failed), so the pair " ++
            "says which regime the machine was in\n",
        .{
            @as(f64, @floatFromInt(evicted_serial_stats[0])) /
                @as(f64, @floatFromInt(evicted_pool_stats[0])),
            @as(f64, @floatFromInt(evicted_serial_stats[1])) /
                @as(f64, @floatFromInt(evicted_pool_stats[1])),
            @as(f64, @floatFromInt(cached_serial_stats[0])) /
                @as(f64, @floatFromInt(cached_pool_stats[0])),
            evicted_failures,
            2 * latency_tokens * heads,
        },
    );

    // The timings are reported, not gated: the page cache and the NVMe device are
    // shared with every other process on this machine, which is why the control
    // pass is printed next to them. What is gated is that the concurrent route
    // read the bytes the table holds.
    const verify = try allocator.alloc(u8, token_bytes);
    defer allocator.free(verify);
    const verified = evicted_ids[evicted_ids.len - heads ..][0..heads];
    try pool.gather(verified, verify);
    var read_buffer: [160]u8 = undefined;
    for (verified, 0..) |id, index| {
        const at = table.location(id);
        const read = try std.posix.pread(
            table.handle.handle,
            &read_buffer,
            at.offset,
        );
        try std.testing.expectEqual(@as(usize, 160), read);
        try std.testing.expectEqualSlices(u8, table.row(id), &read_buffer);
        try std.testing.expectEqualSlices(
            u8,
            &read_buffer,
            verify[index * 160 ..][0..160],
        );
    }
    try std.testing.expect(evicted_pool_stats[1] >= evicted_pool_stats[0]);
}
