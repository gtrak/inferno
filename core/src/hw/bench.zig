//! Hardware probe: measures the ceilings of this machine.
//!
//! Every number here replaces a spec-sheet guess in THEORY.md. Each timing is
//! one warm-up run followed by a median of `sample_count` runs (200 for
//! latencies), because the first run carries clock ramp-up and page faults.
//!
//! Allocation rule: tiger-style forbids dynamic allocation after init, and this
//! module takes exactly one documented exception - the measurement buffers.
//! Device memory comes from `cuMemAlloc`, pinned host memory from
//! `cuMemHostAlloc`, and both are freed before the probe exits;
//! `verify_no_leak` asserts that `cuMemGetInfo` returns to its starting value
//! afterwards.

const std = @import("std");

/// Re-exported so the report renderers can name driver types (`Device`,
/// `DriverVersion`) through this one module import.
pub const cuda_driver = @import("cuda_driver.zig");
const cuda_context = @import("cuda_context.zig");
const gemm = @import("gemm.zig");
/// Public so the probe binary can name the peak record and its matrix size.
pub const gemm_peak = @import("gemm_peak.zig");
const pcie_link = @import("pcie_link.zig");

const assert = std.debug.assert;

/// A host with more GPUs than this reports an error instead of truncating.
pub const device_count_max: u32 = 8;

/// Median over six samples, per the measurement skill.
pub const sample_count: u32 = 6;

/// Latency tails are noisy; the skill asks for 200 samples there.
pub const latency_sample_count: u32 = 200;

pub const kib: u64 = 1024;
pub const mib: u64 = 1024 * kib;
pub const gib: u64 = 1024 * mib;

/// Sliced host<->device sweep: 64 MiB, 256 MiB and 1 GiB.
pub const transfer_64_mib_bytes: u64 = 64 * mib;
pub const transfer_256_mib_bytes: u64 = 256 * mib;
pub const transfer_1_gib_bytes: u64 = gib;
pub const transfer_size_count: u32 = 3;

/// The three sweep sizes in ascending order.
pub const transfer_sizes_bytes = [transfer_size_count]u64{
    transfer_64_mib_bytes,
    transfer_256_mib_bytes,
    transfer_1_gib_bytes,
};

/// Intra-device copy: 1 GiB, which must beat the PCIe link by a wide margin.
pub const intra_device_copy_bytes: u64 = transfer_1_gib_bytes;

/// Peer copy for bandwidth: 256 MiB.
pub const p2p_bandwidth_bytes: u64 = transfer_256_mib_bytes;

/// Peer copy for latency: one page, the smallest realistic message. Measured
/// as a round trip (A -> B -> A), which is what a per-token exchange costs.
pub const p2p_latency_bytes: u64 = 4 * kib;

/// Per-token PCIe traffic is tens of bytes, so measure a 64 B round trip.
pub const host_device_latency_bytes: u64 = 64;

/// One page: the smallest allocation the driver accepts for these buffers.
pub const page_bytes: u64 = 4 * kib;

/// Host `memcpy`: 2 GiB, large enough to stream past every cache level.
pub const host_copy_bytes: u64 = 2 * gib;

/// Aggregate H2D: 256 MiB per GPU, all GPUs in flight at once.
pub const aggregate_bytes_per_device: u64 = transfer_256_mib_bytes;

/// Driver bookkeeping may keep a little memory; 32 MiB is the tolerance.
pub const leak_slack_bytes: u64 = 32 * mib;

/// `memcpy` is called through libc so the compiler cannot fold the loop.
extern "c" fn memcpy(
    destination: *anyopaque,
    source: *const anyopaque,
    bytes: usize,
) *anyopaque;

pub const ProbeError = cuda_driver.DriverError ||
    gemm_peak.Error ||
    error{
        too_many_devices,
        out_of_memory,
        impossible_timing,
        timer_unsupported,
        thread_count_unavailable,
        vram_leak,
    };

/// Direction of a staged copy, i.e. which side holds the pinned buffer.
const StagedDirection = enum { host_to_device, device_to_host };

/// Whether a matrix entry is a device reaching itself or a peer.
pub const P2pKind = enum { self, peer };

/// One cell of the 4x4 P2P matrix.
pub const P2pEntry = struct {
    from: u32,
    to: u32,
    kind: P2pKind,
    can_access: bool,
    enabled: bool,
    /// Bytes copied once, over the P2P link.
    bw_gib_s: f64,
    /// Median of `latency_sample_count` 4 KiB round trips, A -> B -> A.
    latency_us: f64,
};

/// Everything the probe measured; the JSON document is a rendering of this.
pub const Report = struct {
    device_count: u32,
    devices: [device_count_max]cuda_driver.Device,
    cuda_api_version: u32,
    driver_version: cuda_driver.DriverVersion,
    h2d_bw_gib_s: [device_count_max]f64,
    d2h_bw_gib_s: [device_count_max]f64,
    h2d_bw_by_size_gib_s: [transfer_size_count][device_count_max]f64,
    d2h_bw_by_size_gib_s: [transfer_size_count][device_count_max]f64,
    d2d_copied_gib_s: [device_count_max]f64,
    d2d_hbm_gib_s: [device_count_max]f64,
    pcie_link_state: [device_count_max]pcie_link.LinkState,
    link_gib_s_theoretical: [device_count_max]f64,
    /// H2D bandwidth as a percentage of the raw link rate, so the number can
    /// be audited against the link instead of believed.
    h2d_achieved_percent: [device_count_max]f64,
    p2p: [device_count_max][device_count_max]P2pEntry,
    p2p_failure_count: u32,
    aggregate_h2d_gib_s: f64,
    host_memcpy_bw_gib_s: f64,
    cpu_count: u32,
    host_device_latency_us: [device_count_max]f64,
    vram_leak_bytes: [device_count_max]u64,
    /// Tensor throughput of the four prefill GEMM kernels at M=N=K=4096.
    /// THEORY.md §3's prefill ceiling is stated against these.
    gemm_peak_tflops: gemm_peak.Peaks,
    /// Worst run-to-run spread over every measured number, as a percentage, or
    /// null while the probe has not been run enough times to know it. An
    /// unmeasured stability is never written as a stable machine.
    stability_percent_max: ?f64 = null,
};

/// The probe's live state: measured data plus the contexts it must release.
pub const Probe = struct {
    report: Report,
    contexts: [device_count_max]cuda_context.Context,
    baseline_free_bytes: [device_count_max]u64,
    context_count: u32,

    /// Enumerates the devices, retains one context each, records the free
    /// memory baseline and enables peer access between every ordered pair.
    pub fn init(probe: *Probe) ProbeError!void {
        assert(device_count_max > 0);
        assert(sample_count >= 6);

        // Set before anything can fail, so `release_contexts` is always safe.
        probe.report.device_count = 0;
        probe.context_count = 0;
        init_steps(probe) catch |err| {
            probe.release_contexts();
            return err;
        };
    }

    /// Releases every retained context; call `verify_no_leak` before this.
    pub fn deinit(probe: *Probe) void {
        assert(probe.context_count <= device_count_max);
        assert(device_count_max > 0);

        probe.release_contexts();
        assert(probe.context_count == 0);
    }

    /// Runs every measurement in order. GPU work is serialized: one
    /// measurement at a time, so the numbers are not each other's noise.
    pub fn measure_all(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(probe.context_count == probe.report.device_count);

        try probe.measure_device_bandwidth();
        try probe.measure_p2p();
        try probe.measure_aggregate_h2d();
        try probe.measure_host();
        try probe.measure_host_device_latency();
    }

    /// Measures the prefill GEMM peaks on GPU0 from an embedded cubin image.
    ///
    /// The probe binary owns the cubin table (it is compiled from the same
    /// `src/kernels` directory), so the caller passes the bytes in rather than
    /// this module knowing how the build embeds them.
    pub fn measure_gemm_peaks(probe: *Probe, cubin: []const u8) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(cubin.len > 0);

        try probe.contexts[0].set_current();
        const stream = try cuda_context.Stream.create();
        defer stream.destroy();
        const kernels = try gemm.Kernels.init(cubin);
        defer kernels.deinit();

        probe.report.gemm_peak_tflops = try gemm_peak.measure(
            std.heap.page_allocator,
            &kernels,
            &stream,
        );
        assert(probe.report.gemm_peak_tflops.nvfp4_mma > 0);
    }

    /// Asserts that the device allocations were all returned to the driver.
    pub fn verify_no_leak(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(probe.context_count == probe.report.device_count);

        var leaked_any = false;
        var index: u32 = 0;
        while (index < probe.report.device_count) : (index += 1) {
            try probe.contexts[index].set_current();
            const free_now = try probe.contexts[index].free_bytes();
            const baseline = probe.baseline_free_bytes[index];
            const leaked = if (baseline > free_now) baseline - free_now else 0;
            probe.report.vram_leak_bytes[index] = leaked;
            if (leaked >= leak_slack_bytes) {
                std.log.err(
                    "GPU{d} leaked {d} bytes of VRAM",
                    .{ index, leaked },
                );
                leaked_any = true;
            } else {
                assert(probe.report.vram_leak_bytes[index] < leak_slack_bytes);
            }
        }
        if (leaked_any) {
            return ProbeError.vram_leak;
        } else {
            assert(probe.context_count == probe.report.device_count);
        }
    }

    fn init_steps(probe: *Probe) ProbeError!void {
        assert(probe.context_count == 0);
        assert(probe.report.device_count == 0);

        var devices: [cuda_driver.device_count_max]cuda_driver.Device =
            undefined;
        const device_count = try cuda_driver.query_devices(devices[0..]);
        assert(device_count > 0);
        if (device_count > device_count_max) {
            return ProbeError.too_many_devices;
        }
        probe.report.device_count = device_count;
        probe.report.p2p_failure_count = 0;
        probe.report.stability_percent_max = null;
        for (0..device_count) |index| {
            probe.report.devices[index] = devices[index];
        }

        probe.report.gemm_peak_tflops = .{
            .bf16 = 0,
            .fp8 = 0,
            .nvfp4_mma = 0,
            .nvfp4_dequant_bf16 = 0,
        };
        probe.report.cuda_api_version = try cuda_driver.cuda_api_version();
        probe.report.driver_version = cuda_driver.DriverVersion.unknown();
        cuda_driver.display_driver_version(
            &probe.report.driver_version,
        ) catch |err| switch (err) {
            // procfs is the only source of the display version; without it the
            // document says "unknown" rather than failing the probe.
            error.version_unavailable => {
                probe.report.driver_version =
                    cuda_driver.DriverVersion.unknown();
            },
        };

        try probe.retain_contexts();
        try probe.enable_peer_access();
        assert(probe.report.device_count <= device_count_max);
    }

    fn retain_contexts(probe: *Probe) ProbeError!void {
        assert(probe.context_count <= probe.report.device_count);
        assert(probe.report.device_count > 0);

        var index: u32 = 0;
        while (index < probe.report.device_count) : (index += 1) {
            probe.contexts[index] = try cuda_context.Context.retain(
                &probe.report.devices[index],
            );
            probe.context_count = index + 1;
            try probe.contexts[index].set_current();
            probe.baseline_free_bytes[index] =
                try probe.contexts[index].free_bytes();
        }
        assert(probe.context_count == probe.report.device_count);
    }

    fn release_contexts(probe: *Probe) void {
        assert(probe.context_count <= device_count_max);
        assert(probe.context_count <= probe.report.device_count);

        var index = probe.context_count;
        while (index > 0) {
            index -= 1;
            probe.contexts[index].release();
        }
        probe.context_count = 0;
        assert(probe.context_count == 0);
    }

    fn enable_peer_access(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(probe.context_count == probe.report.device_count);

        var from: u32 = 0;
        while (from < probe.report.device_count) : (from += 1) {
            var to: u32 = 0;
            while (to < probe.report.device_count) : (to += 1) {
                try probe.enable_peer_pair(from, to);
            }
        }
    }

    /// Records one matrix cell, enabling peer access when the driver allows it.
    fn enable_peer_pair(probe: *Probe, from: u32, to: u32) ProbeError!void {
        assert(from < probe.report.device_count);
        assert(to < probe.report.device_count);

        var entry: P2pEntry = .{
            .from = from,
            .to = to,
            .kind = .self,
            .can_access = true,
            .enabled = true,
            .bw_gib_s = 0,
            .latency_us = 0,
        };
        if (from == to) {
            // A device reaching itself needs no peer access; this cell is the
            // local reference the peer numbers are read against.
            probe.report.p2p[from][to] = entry;
            return;
        } else {
            entry.kind = .peer;
        }

        entry.can_access = try cuda_driver.can_access_peer(
            probe.report.devices[from].handle,
            probe.report.devices[to].handle,
        );
        var enabled = false;
        if (entry.can_access) {
            try probe.contexts[from].set_current();
            enabled = probe.enable_peer_or_log(from, to);
        } else {
            std.log.err(
                "P2P unavailable from GPU{d} to GPU{d}",
                .{ from, to },
            );
        }
        entry.enabled = enabled;
        if (enabled) {
            assert(probe.report.p2p_failure_count <= 64);
        } else {
            entry.can_access = false;
            probe.report.p2p_failure_count += 1;
        }
        probe.report.p2p[from][to] = entry;
    }

    fn enable_peer_or_log(probe: *Probe, from: u32, to: u32) bool {
        assert(from < probe.report.device_count);
        assert(to < probe.report.device_count);

        probe.contexts[from].enable_peer_access(
            &probe.contexts[to],
        ) catch |err| {
            std.log.err(
                "cuCtxEnablePeerAccess GPU{d}->GPU{d}: {s}",
                .{ from, to, @errorName(err) },
            );
            return false;
        };
        return true;
    }

    fn measure_device_bandwidth(probe: *Probe) ProbeError!void {
        assert(probe.context_count == probe.report.device_count);
        assert(probe.report.device_count > 0);

        var index: u32 = 0;
        while (index < probe.report.device_count) : (index += 1) {
            try probe.measure_device_bandwidth_of(index);
        }
    }

    /// Staged H2D/D2H at three sizes plus the intra-device copy, one GPU.
    fn measure_device_bandwidth_of(
        probe: *Probe,
        index: u32,
    ) ProbeError!void {
        assert(index < probe.report.device_count);
        assert(transfer_sizes_bytes[transfer_size_count - 1] == gib);

        try probe.contexts[index].set_current();
        const staged = try cuda_context.PinnedHostBuffer.allocate(gib);
        defer staged.free();
        const source = try cuda_context.DeviceBuffer.allocate(gib);
        defer probe.free_device_buffer(index, &source);
        const destination = try cuda_context.DeviceBuffer.allocate(gib);
        defer probe.free_device_buffer(index, &destination);
        const stream = try cuda_context.Stream.create();
        defer stream.destroy();

        var size_index: u32 = 0;
        while (size_index < transfer_size_count) : (size_index += 1) {
            const bytes = transfer_sizes_bytes[size_index];
            probe.report.h2d_bw_by_size_gib_s[size_index][index] =
                try measure_staged_bandwidth(
                    &stream,
                    .host_to_device,
                    &source,
                    &staged,
                    bytes,
                );
            probe.report.d2h_bw_by_size_gib_s[size_index][index] =
                try measure_staged_bandwidth(
                    &stream,
                    .device_to_host,
                    &source,
                    &staged,
                    bytes,
                );
        }

        const largest = transfer_size_count - 1;
        probe.report.h2d_bw_gib_s[index] =
            probe.report.h2d_bw_by_size_gib_s[largest][index];
        probe.report.d2h_bw_gib_s[index] =
            probe.report.d2h_bw_by_size_gib_s[largest][index];

        // One copy, two bases: the bytes copied once, and the HBM traffic the
        // copy causes (read + write). Both are reported; neither is derived
        // from the other's name.
        const copy_nanoseconds = try measure_copy_nanoseconds(
            &stream,
            &destination,
            &source,
            intra_device_copy_bytes,
        );
        probe.report.d2d_copied_gib_s[index] = gib_per_second(
            intra_device_copy_bytes,
            copy_nanoseconds,
        );
        probe.report.d2d_hbm_gib_s[index] = gib_per_second(
            intra_device_copy_bytes * 2,
            copy_nanoseconds,
        );

        try probe.measure_link_state(index, &stream, &staged, &source);
        assert(probe.report.d2d_copied_gib_s[index] > 0);
    }

    /// Reads the negotiated PCIe link state while a 1 GiB copy keeps the link
    /// at full speed; an idle link reports Gen1 and would make the achieved
    /// percentage nonsense.
    fn measure_link_state(
        probe: *Probe,
        index: u32,
        stream: *const cuda_context.Stream,
        staged: *const cuda_context.PinnedHostBuffer,
        device: *const cuda_context.DeviceBuffer,
    ) ProbeError!void {
        assert(index < probe.report.device_count);
        assert(staged.bytes >= gib);

        try stream.copy_h_to_d(device, staged, gib);
        const bus_id = probe.report.devices[index].pci_bus_id_slice();
        const state = pcie_link.link_state(bus_id) catch |err| blk: {
            std.log.warn(
                "GPU{d} PCIe link state unreadable: {s}",
                .{ index, @errorName(err) },
            );
            break :blk pcie_link.LinkState{ .generation = 0, .width = 0 };
        };
        try stream.synchronize();

        probe.report.pcie_link_state[index] = state;
        if (state.width > 0) {
            probe.report.link_gib_s_theoretical[index] =
                pcie_link.link_gib_per_second(state) catch 0;
        } else {
            // No link state means no ratio; reporting a made-up one would be
            // worse than reporting none.
            probe.report.link_gib_s_theoretical[index] = 0;
        }
        probe.report.h2d_achieved_percent[index] = achieved_percent(
            probe.report.h2d_bw_gib_s[index],
            probe.report.link_gib_s_theoretical[index],
        );
        assert(probe.report.link_gib_s_theoretical[index] >= 0);
    }

    fn measure_p2p(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(probe.context_count == probe.report.device_count);

        var from: u32 = 0;
        while (from < probe.report.device_count) : (from += 1) {
            var to: u32 = 0;
            while (to < probe.report.device_count) : (to += 1) {
                if (probe.report.p2p[from][to].enabled) {
                    try probe.measure_p2p_pair(from, to);
                } else {
                    // Unavailable pairs keep their zeroes; the probe exits
                    // non-zero for them.
                    assert(probe.report.p2p[from][to].can_access == false);
                }
            }
        }
    }

    /// 256 MiB bandwidth and 4 KiB latency for one ordered pair.
    fn measure_p2p_pair(probe: *Probe, from: u32, to: u32) ProbeError!void {
        assert(from < probe.report.device_count);
        assert(to < probe.report.device_count);
        assert(probe.report.p2p[from][to].enabled);

        try probe.contexts[from].set_current();
        const source = try cuda_context.DeviceBuffer.allocate(
            p2p_bandwidth_bytes,
        );
        defer probe.free_device_buffer(from, &source);
        const stream = try cuda_context.Stream.create();
        // The peer buffer is freed with the peer context current, so the
        // stream must be destroyed with its own context current afterwards.
        defer probe.destroy_stream(from, &stream);

        try probe.contexts[to].set_current();
        const destination = try cuda_context.DeviceBuffer.allocate(
            p2p_bandwidth_bytes,
        );
        defer probe.free_device_buffer(to, &destination);

        try probe.contexts[from].set_current();
        const bandwidth_nanoseconds = try measure_copy_nanoseconds(
            &stream,
            &destination,
            &source,
            p2p_bandwidth_bytes,
        );
        probe.report.p2p[from][to].bw_gib_s = gib_per_second(
            p2p_bandwidth_bytes,
            bandwidth_nanoseconds,
        );
        probe.report.p2p[from][to].latency_us =
            try measure_copy_round_trip_latency(
                &stream,
                &destination,
                &source,
                p2p_latency_bytes,
            );
        assert(probe.report.p2p[from][to].latency_us > 0);
    }

    fn measure_aggregate_h2d(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(aggregate_bytes_per_device == transfer_256_mib_bytes);

        var setup: AggregateSetup = .{};
        setup.init(probe) catch |err| {
            setup.deinit(probe);
            return err;
        };
        defer setup.deinit(probe);
        try setup.run(probe);
    }

    fn measure_host(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(host_copy_bytes == 2 * gib);

        const cpu_count = std.Thread.getCpuCount() catch
            return ProbeError.thread_count_unavailable;
        if (cpu_count == 0) {
            return ProbeError.impossible_timing;
        } else {
            probe.report.cpu_count = @intCast(cpu_count);
        }
        probe.report.host_memcpy_bw_gib_s = try measure_memcpy_bandwidth();
    }

    fn measure_host_device_latency(probe: *Probe) ProbeError!void {
        assert(probe.report.device_count > 0);
        assert(host_device_latency_bytes > 0);

        var index: u32 = 0;
        while (index < probe.report.device_count) : (index += 1) {
            try probe.measure_host_device_latency_of(index);
        }
    }

    /// Round trip of one 64 B message: H2D, wait, D2H, wait, host timing.
    fn measure_host_device_latency_of(
        probe: *Probe,
        index: u32,
    ) ProbeError!void {
        assert(index < probe.report.device_count);
        assert(host_device_latency_bytes <= transfer_64_mib_bytes);

        try probe.contexts[index].set_current();
        const staged = try cuda_context.PinnedHostBuffer.allocate(page_bytes);
        defer staged.free();
        const device = try cuda_context.DeviceBuffer.allocate(page_bytes);
        defer probe.free_device_buffer(index, &device);
        const stream = try cuda_context.Stream.create();
        defer stream.destroy();

        try round_trip_once(&stream, &staged, &device);
        var samples: [latency_sample_count]u64 = undefined;
        for (&samples) |*sample| {
            var timer = std.time.Timer.start() catch
                return ProbeError.timer_unsupported;
            try round_trip_once(&stream, &staged, &device);
            sample.* = timer.read();
        }
        const nanoseconds = median(samples[0..]);
        probe.report.host_device_latency_us[index] =
            nanoseconds_to_microseconds(nanoseconds);
    }

    /// Destroys a stream with its owning context current.
    fn destroy_stream(
        probe: *Probe,
        device_index: u32,
        stream: *const cuda_context.Stream,
    ) void {
        assert(device_index < probe.report.device_count);
        assert(@intFromPtr(stream.handle) != 0);

        probe.contexts[device_index].set_current() catch |err| {
            std.log.err(
                "cannot destroy a stream on GPU{d}: {s}",
                .{ device_index, @errorName(err) },
            );
            return;
        };
        stream.destroy();
    }

    /// Frees a device buffer with its owning context current.
    fn free_device_buffer(
        probe: *Probe,
        device_index: u32,
        buffer: *const cuda_context.DeviceBuffer,
    ) void {
        assert(device_index < probe.report.device_count);
        assert(buffer.bytes > 0);

        probe.contexts[device_index].set_current() catch |err| {
            std.log.err(
                "cannot free a buffer on GPU{d}: {s}",
                .{ device_index, @errorName(err) },
            );
            return;
        };
        buffer.free();
    }

    /// Samples the clock around one round trip of `host_device_latency_bytes`.
    fn round_trip_once(
        stream: *const cuda_context.Stream,
        staged: *const cuda_context.PinnedHostBuffer,
        device: *const cuda_context.DeviceBuffer,
    ) ProbeError!void {
        assert(staged.bytes >= host_device_latency_bytes);
        assert(device.bytes >= host_device_latency_bytes);

        try stream.copy_h_to_d(device, staged, host_device_latency_bytes);
        try stream.synchronize();
        try stream.copy_d_to_h(staged, device, host_device_latency_bytes);
        try stream.synchronize();
        assert(@intFromPtr(staged.pointer) != 0);
    }
};

/// Buffers and streams for the aggregate test: every GPU copies at once.
const AggregateSetup = struct {
    staged: [device_count_max]cuda_context.PinnedHostBuffer = undefined,
    device_buffers: [device_count_max]cuda_context.DeviceBuffer = undefined,
    streams: [device_count_max]cuda_context.Stream = undefined,
    ready_count: u32 = 0,

    fn init(setup: *AggregateSetup, probe: *Probe) ProbeError!void {
        assert(setup.ready_count == 0);
        assert(probe.report.device_count <= device_count_max);

        setup.ready_count = 0;
        var index: u32 = 0;
        while (index < probe.report.device_count) : (index += 1) {
            try probe.contexts[index].set_current();
            setup.staged[index] = try cuda_context.PinnedHostBuffer.allocate(
                aggregate_bytes_per_device,
            );
            setup.device_buffers[index] =
                cuda_context.DeviceBuffer.allocate(
                    aggregate_bytes_per_device,
                ) catch |err| {
                    setup.staged[index].free();
                    return err;
                };
            setup.streams[index] = cuda_context.Stream.create() catch |err| {
                setup.device_buffers[index].free();
                setup.staged[index].free();
                return err;
            };
            setup.ready_count = index + 1;
        }
        assert(setup.ready_count == probe.report.device_count);
    }

    /// Frees everything `init` took, newest device first.
    fn deinit(setup: *AggregateSetup, probe: *Probe) void {
        assert(setup.ready_count <= device_count_max);
        assert(probe.report.device_count <= device_count_max);

        var index = setup.ready_count;
        while (index > 0) {
            index -= 1;
            probe.free_device_buffer(
                index,
                &setup.device_buffers[index],
            );
            probe.contexts[index].set_current() catch |err| {
                std.log.err(
                    "cannot release GPU{d} resources: {s}",
                    .{ index, @errorName(err) },
                );
            };
            setup.streams[index].destroy();
            setup.staged[index].free();
        }
        setup.ready_count = 0;
        assert(setup.ready_count == 0);
    }

    /// Launches every copy, then waits: the transfers have to overlap.
    fn transfer_all(setup: *AggregateSetup, probe: *Probe) ProbeError!void {
        assert(setup.ready_count > 0);
        assert(setup.ready_count == probe.report.device_count);

        var index: u32 = 0;
        while (index < setup.ready_count) : (index += 1) {
            try probe.contexts[index].set_current();
            try setup.streams[index].copy_h_to_d(
                &setup.device_buffers[index],
                &setup.staged[index],
                aggregate_bytes_per_device,
            );
        }
        index = 0;
        while (index < setup.ready_count) : (index += 1) {
            try probe.contexts[index].set_current();
            try setup.streams[index].synchronize();
        }
    }

    fn run(setup: *AggregateSetup, probe: *Probe) ProbeError!void {
        assert(setup.ready_count > 0);
        assert(aggregate_bytes_per_device > 0);

        setup.transfer_all(probe) catch |err| return err;
        var samples: [sample_count]u64 = undefined;
        for (&samples) |*sample| {
            var timer = std.time.Timer.start() catch
                return ProbeError.timer_unsupported;
            try setup.transfer_all(probe);
            sample.* = timer.read();
        }
        const nanoseconds = median(samples[0..]);
        const device_count: u64 = probe.report.device_count;
        const total_bytes = aggregate_bytes_per_device * device_count;
        if (nanoseconds > 0) {
            probe.report.aggregate_h2d_gib_s =
                gib_per_second(total_bytes, nanoseconds);
        } else {
            return ProbeError.impossible_timing;
        }
        assert(probe.report.aggregate_h2d_gib_s > 0);
    }
};

/// Times one staged copy of `bytes`, warm-up plus a median of six runs.
fn measure_staged_bandwidth(
    stream: *const cuda_context.Stream,
    direction: StagedDirection,
    device: *const cuda_context.DeviceBuffer,
    staged: *const cuda_context.PinnedHostBuffer,
    bytes: u64,
) ProbeError!f64 {
    assert(bytes > 0);
    assert(bytes <= device.bytes);
    assert(bytes <= staged.bytes);

    try staged_transfer_once(stream, direction, device, staged, bytes);
    try stream.synchronize();

    var samples: [sample_count]u64 = undefined;
    for (&samples) |*sample| {
        var timer = std.time.Timer.start() catch
            return ProbeError.timer_unsupported;
        try staged_transfer_once(stream, direction, device, staged, bytes);
        try stream.synchronize();
        sample.* = timer.read();
    }
    const nanoseconds = median(samples[0..]);
    if (nanoseconds > 0) {
        return gib_per_second(bytes, nanoseconds);
    } else {
        return ProbeError.impossible_timing;
    }
}

/// Times one device-to-device copy of `bytes`, warm-up plus median of six.
/// The basis is applied by the caller, which keeps each reported number's
/// meaning visible where it is written.
fn measure_copy_nanoseconds(
    stream: *const cuda_context.Stream,
    destination: *const cuda_context.DeviceBuffer,
    source: *const cuda_context.DeviceBuffer,
    bytes: u64,
) ProbeError!u64 {
    assert(bytes > 0);
    assert(bytes <= destination.bytes);
    assert(bytes <= source.bytes);

    try stream.copy_d_to_d(destination, source, bytes);
    try stream.synchronize();

    var samples: [sample_count]u64 = undefined;
    for (&samples) |*sample| {
        var timer = std.time.Timer.start() catch
            return ProbeError.timer_unsupported;
        try stream.copy_d_to_d(destination, source, bytes);
        try stream.synchronize();
        sample.* = timer.read();
    }
    const nanoseconds = median(samples[0..]);
    if (nanoseconds == 0) {
        return ProbeError.impossible_timing;
    } else {
        assert(nanoseconds > 0);
    }
    return nanoseconds;
}

/// Device-side latency of a 4 KiB round trip (A -> B -> A), median of
/// `latency_sample_count` event pairs.
///
/// A round trip is what a per-token exchange actually costs: the requester pays
/// for both hops. The event timer resolves about 0.5 us, so the median matters.
fn measure_copy_round_trip_latency(
    stream: *const cuda_context.Stream,
    destination: *const cuda_context.DeviceBuffer,
    source: *const cuda_context.DeviceBuffer,
    bytes: u64,
) ProbeError!f64 {
    assert(bytes > 0);
    assert(bytes <= destination.bytes);
    assert(bytes <= source.bytes);

    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();

    try stream.copy_d_to_d(destination, source, bytes);
    try stream.copy_d_to_d(source, destination, bytes);
    try stream.synchronize();

    var samples: [latency_sample_count]u64 = undefined;
    for (&samples) |*sample| {
        try start.record(stream);
        try stream.copy_d_to_d(destination, source, bytes);
        try stream.copy_d_to_d(source, destination, bytes);
        try end.record(stream);
        try end.synchronize();
        const elapsed = try cuda_context.Event.elapsed_nanoseconds(
            &start,
            &end,
        );
        if (elapsed == 0) {
            return ProbeError.impossible_timing;
        } else {
            sample.* = elapsed;
        }
    }
    return nanoseconds_to_microseconds(median(samples[0..]));
}

/// Host memory bandwidth: `memcpy` over a 2 GiB buffer, median of six.
fn measure_memcpy_bandwidth() ProbeError!f64 {
    assert(host_copy_bytes == 2 * gib);
    assert(sample_count >= 6);

    // The documented allocation exception: the two host buffers this
    // measurement needs. Both are freed before the probe exits.
    const allocator = std.heap.page_allocator;
    const source: []u8 = allocator.alloc(u8, @intCast(host_copy_bytes)) catch
        return ProbeError.out_of_memory;
    defer allocator.free(source);
    const destination: []u8 = allocator.alloc(
        u8,
        @intCast(host_copy_bytes),
    ) catch return ProbeError.out_of_memory;
    defer allocator.free(destination);

    // Touch both buffers so the timing does not measure page faults.
    @memset(source, 0xa5);
    @memset(destination, 0x00);

    _ = memcpy(
        @ptrCast(&destination[0]),
        @ptrCast(&source[0]),
        @intCast(host_copy_bytes),
    );
    var samples: [sample_count]u64 = undefined;
    for (&samples) |*sample| {
        var timer = std.time.Timer.start() catch
            return ProbeError.timer_unsupported;
        _ = memcpy(
            @ptrCast(&destination[0]),
            @ptrCast(&source[0]),
            @intCast(host_copy_bytes),
        );
        sample.* = timer.read();
    }
    const nanoseconds = median(samples[0..]);
    // A copy reads one buffer and writes the other, so both move.
    const moved_bytes = host_copy_bytes * 2;
    if (nanoseconds > 0) {
        return gib_per_second(moved_bytes, nanoseconds);
    } else {
        return ProbeError.impossible_timing;
    }
}

fn staged_transfer_once(
    stream: *const cuda_context.Stream,
    direction: StagedDirection,
    device: *const cuda_context.DeviceBuffer,
    staged: *const cuda_context.PinnedHostBuffer,
    bytes: u64,
) ProbeError!void {
    assert(bytes > 0);
    assert(bytes <= device.bytes);
    assert(bytes <= staged.bytes);

    switch (direction) {
        .host_to_device => try stream.copy_h_to_d(device, staged, bytes),
        .device_to_host => try stream.copy_d_to_h(staged, device, bytes),
    }
    assert(@intFromPtr(staged.pointer) != 0);
}

/// Median of a sample array; the caller's buffer is scratch and gets sorted.
fn median(samples: []u64) u64 {
    assert(samples.len > 0);
    assert(samples.len <= latency_sample_count);

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const middle = @divFloor(samples.len, 2);
    assert(middle < samples.len);
    return samples[middle];
}

fn gib_per_second(bytes: u64, nanoseconds: u64) f64 {
    assert(bytes > 0);
    assert(nanoseconds > 0);

    const bytes_float: f64 = @floatFromInt(bytes);
    const seconds: f64 = @as(f64, @floatFromInt(nanoseconds)) / 1e9;
    const gibibytes: f64 = bytes_float / @as(f64, @floatFromInt(gib));
    assert(seconds > 0);
    return gibibytes / seconds;
}

/// `measured` as a percentage of `theoretical`; zero when the link state is
/// unknown, which is honest about not knowing rather than inventing a ratio.
fn achieved_percent(measured: f64, theoretical: f64) f64 {
    assert(std.math.isFinite(measured));
    assert(theoretical >= 0);

    if (theoretical > 0) {
        return measured / theoretical * 100.0;
    } else {
        return 0;
    }
}

fn nanoseconds_to_microseconds(nanoseconds: u64) f64 {
    assert(nanoseconds > 0);
    assert(nanoseconds <= std.math.maxInt(u64) / 2);

    return @as(f64, @floatFromInt(nanoseconds)) / 1000.0;
}

comptime {
    assert(transfer_sizes_bytes[0] == transfer_64_mib_bytes);
    assert(transfer_sizes_bytes[1] == transfer_256_mib_bytes);
    assert(transfer_sizes_bytes[2] == transfer_1_gib_bytes);
    assert(transfer_sizes_bytes[0] < transfer_sizes_bytes[1]);
    assert(transfer_sizes_bytes[1] < transfer_sizes_bytes[2]);
    assert(device_count_max >= 4);
    assert(sample_count >= 6);
}

test "median picks the middle sample of an odd count" {
    var samples = [_]u64{ 5, 1, 3 };
    try std.testing.expectEqual(@as(u64, 3), median(samples[0..]));

    var even = [_]u64{ 4, 1, 3, 2 };
    try std.testing.expectEqual(@as(u64, 3), median(even[0..]));
}

test "gib_per_second converts bytes and nanoseconds" {
    // 1 GiB in 1 s is 1 GiB/s; 2 GiB in 0.5 s is 4 GiB/s.
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.0),
        gib_per_second(gib, std.time.ns_per_s),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 4.0),
        gib_per_second(2 * gib, std.time.ns_per_s / 2),
        1e-6,
    );
}

test "achieved_percent is a ratio against the raw link rate" {
    try std.testing.expectApproxEqAbs(
        @as(f64, 50.0),
        achieved_percent(3.67, 7.34),
        1e-9,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 0.0),
        achieved_percent(3.67, 0),
        1e-9,
    );
}

test "nanoseconds_to_microseconds converts one microsecond" {
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.0),
        nanoseconds_to_microseconds(1000),
        1e-9,
    );
}

test "the transfer sizes ascend and end at 1 GiB" {
    try std.testing.expect(
        @as(usize, transfer_size_count) == transfer_sizes_bytes.len,
    );
    try std.testing.expectEqual(
        gib,
        transfer_sizes_bytes[transfer_size_count - 1],
    );
}
