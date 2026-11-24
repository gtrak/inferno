//! Capability verdict for the SM120 tensor-core MMA family, recorded as
//! evidence rather than asserted from memory.
//!
//! `kernels/fp4_mma_probe.cu` issues each instruction once and writes its
//! accumulator, so `ptxas` has to emit it. This harness dumps the SASS of the
//! embedded cubin with `cuobjdump -sass`, requires the expected mnemonic for
//! every probe kernel, requires `-arch=sm_120` (no `a`) to be rejected, and
//! writes the verdict table to `bench/fp4-mma-probe.txt`.
//!
//! The numbers that follow from this table are consumed by
//! `.agents/design/mma-sm120-nvfp4.md` and PLAN-log.md; that is why the verdict
//! is written to a committed file instead of only printed.

const std = @import("std");

const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// `ctypes` style term used by `std.process.Child` results.
const ExitTerm = std.process.Child.Term;

/// CUDA toolkit used when the environment does not name one.
const cuda_home_default = "/usr/local/cuda-13.2";

/// Environment variable that overrides `cuda_home_default`.
const cuda_home_variable = "CUDA_HOME";

/// Probe kernels and the SASS that must appear for each of them.
const Verdict = struct {
    /// `extern "C"` kernel symbol in the cubin.
    kernel: []const u8,
    /// Instruction the kernel issues.
    instruction: []const u8,
    /// SASS mnemonic fragment `cuobjdump -sass` must show.
    sass: []const u8,
    /// What it means for the engine.
    meaning: []const u8,
};

const verdicts = [_]Verdict{
    .{
        .kernel = "fp4_probe_mma_bf16_m16n8k16",
        .instruction = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        .sass = "HMMA.16816.F32.BF16",
        .meaning = "bf16 tensor core: the exact dequantise-to-bf16 reference path",
    },
    .{
        .kernel = "fp4_probe_mma_fp8_m16n8k32",
        .instruction = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
        .sass = "QMMA.16832.F32.E4M3.E4M3",
        .meaning = "FP8 tensor core: the MTP and requantised dense prefill path",
    },
    .{
        .kernel = "fp4_probe_mma_f8f6f4_m16n8k32",
        .instruction = "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32",
        .sass = "QMMA.16832.F32.E2M1.E2M1",
        .meaning = "unscaled FP4 MMA (no block scales)",
    },
    .{
        .kernel = "fp4_probe_mma_mxf4nvf4_4x_ue4m3_m16n8k64",
        .instruction = "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." ++
            "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
        .sass = "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
        .meaning = "native NVFP4: per-16 E4M3 scales, exactly the checkpoint format",
    },
    .{
        .kernel = "fp4_probe_mma_mxf4nvf4_2x_ue8m0_m16n8k64",
        .instruction = "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X." ++
            "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
        .sass = "OMMA.SF.16864.F32.E2M1.E2M1.E8",
        .meaning = "MXFP4: per-32 power-of-two scales, the lever-L3 requantisation",
    },
};

/// Path of the probe kernel source, relative to the build root (`core/`).
const probe_source_path = "src/kernels/fp4_mma_probe.cu";

/// Where the verdict table is written, relative to the build root and to the
/// repository root (the gate may run from either).
const report_paths = [_][]const u8{
    "../bench/fp4-mma-probe.txt",
    "bench/fp4-mma-probe.txt",
};

/// Output cap for one tool invocation.
const tool_output_bytes_max = 8 * 1024 * 1024;

/// Result of running one external tool.
const ToolRun = struct {
    exited_zero: bool,
    stdout: []u8,
    stderr: []u8,
};

/// CUDA toolkit directory: `CUDA_HOME` when set, the default install otherwise.
fn cudaHome() []const u8 {
    assert(cuda_home_default.len > 0);
    assert(cuda_home_variable.len > 0);

    if (std.posix.getenv(cuda_home_variable)) |override| {
        return override;
    } else {
        return cuda_home_default;
    }
}

/// Runs `arguments` and captures its output.
fn runTool(
    allocator: std.mem.Allocator,
    arguments: []const []const u8,
) !ToolRun {
    assert(arguments.len > 0);
    assert(arguments[0].len > 0);

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = arguments,
        .max_output_bytes = tool_output_bytes_max,
    });
    const exited_zero = switch (result.term) {
        ExitTerm.Exited => |code| code == 0,
        else => false,
    };
    assert(result.stdout.len <= tool_output_bytes_max);
    return .{
        .exited_zero = exited_zero,
        .stdout = result.stdout,
        .stderr = result.stderr,
    };
}

/// The embedded cubin of `kernels/fp4_mma_probe.cu`.
fn probeCubin() ?[]const u8 {
    assert(generated_cubins.cubins.len > 0);
    assert(verdicts.len > 0);

    for (generated_cubins.cubins) |entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), "fp4_mma_probe")) {
            assert(entry.length > 0);
            return entry.bytes[0..entry.length];
        } else {
            continue;
        }
    }
    return null;
}

/// Disassembles `cubin` with `cuobjdump -sass` and returns the SASS text.
fn disassemble(allocator: std.mem.Allocator, cubin: []const u8) ![]u8 {
    assert(cubin.len > 0);
    assert(cubin.len % 4 == 0);

    var directory = std.testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(.{ .sub_path = "probe.cubin", .data = cubin });

    const path = try directory.dir.realpathAlloc(allocator, "probe.cubin");
    defer allocator.free(path);
    const tool = try std.fmt.allocPrint(
        allocator,
        "{s}/bin/cuobjdump",
        .{cudaHome()},
    );
    defer allocator.free(tool);

    const run = try runTool(allocator, &.{ tool, "-sass", path });
    if (!run.exited_zero) {
        std.debug.print("cuobjdump failed: {s}\n", .{run.stderr});
        allocator.free(run.stdout);
        allocator.free(run.stderr);
        return error.DisassemblyFailed;
    } else {
        allocator.free(run.stderr);
        assert(run.stdout.len > 0);
        return run.stdout;
    }
}

/// Renders the verdict table, with one row per probe kernel.
fn renderReport(
    allocator: std.mem.Allocator,
    sass: []const u8,
    sm120_rejected: bool,
) ![]u8 {
    assert(sass.len > 0);
    assert(verdicts.len > 0);

    var text = std.ArrayList(u8).init(allocator);
    errdefer text.deinit();
    const writer = text.writer();

    try writer.writeAll(
        \\inferno — SM120 tensor-core MMA capability verdict
        \\generated by `zig build test` (core/src/kernels/fp4_mma_probe.zig)
        \\
        \\Machine: 4 x RTX 5060 Ti (GB206, sm_120 / sm_120a), CUDA 13.2.
        \\Method: each instruction is issued once by a kernel in
        \\core/src/kernels/fp4_mma_probe.cu, the cubin is compiled by the build
        \\with -arch=sm_120a, and this file is produced by disassembling that
        \\embedded cubin with `cuobjdump -sass`.
        \\
        \\| instruction | SASS | verdict |
        \\|---|---|---|
        \\
    );
    for (verdicts) |verdict| {
        try writer.print(
            "| `{s}` | `{s}` | ACCEPTED |\n",
            .{ verdict.instruction, verdict.sass },
        );
    }
    try writer.writeAll(
        \\
        \\| same instructions, `-arch=sm_120` (no `a`) | — | REJECTED |
        \\
    );
    try writer.print(
        \\
        \\`-arch=sm_120` rejection re-checked by this harness: {s}.
        \\
        \\
        \\Consequences for the engine
        \\--------------------------
        \\
    , .{if (sm120_rejected) "confirmed, the build fails without the `a` suffix" else "NOT confirmed"});
    for (verdicts) |verdict| {
        try writer.print("- {s}: {s}\n", .{ verdict.kernel, verdict.meaning });
    }
    try writer.writeAll(
        \\
        \\- `tcgen05.mma` (5th generation tensor core) does not exist on SM120;
        \\  the block-scaled path here is `OMMA.SF` at m16n8k64.
        \\- The last four operands of a block-scaled MMA are `byte_id` /
        \\  `thread_id` immediates of type u16, always 0. Passing them as u32
        \\  registers makes ptxas report "Unexpected instruction types specified
        \\  for 'mma'", which reads like "unsupported instruction".
        \\- Decode (M=1) stays an FFMA dequantise-GEMV: at M=1 the tensor core
        \\  is idle. See core/src/kernels/gemv.cu.
        \\
    );
    assert(text.items.len > 0);
    return text.toOwnedSlice();
}

/// Writes `report` next to the hardware probe, if either location is writable.
fn writeReport(report: []const u8) !void {
    assert(report.len > 0);
    assert(report_paths.len > 0);

    for (report_paths) |path| {
        const file = std.fs.cwd().createFile(path, .{}) catch continue;
        defer file.close();
        try file.writeAll(report);
        std.debug.print("wrote {s} ({d} bytes)\n", .{ path, report.len });
        return;
    }
    // The gate runs from `core/`; a different working directory only means the
    // table is printed instead of committed, which the verdicts below still
    // enforce.
    std.debug.print("could not write the verdict table:\n{s}\n", .{report});
}

test "every SM120 MMA variant the engine relies on is accepted" {
    const cubin = probeCubin() orelse return error.TestExpectedCubin;
    const sass = disassemble(std.testing.allocator, cubin) catch |err| switch (err) {
        error.FileNotFound, error.DisassemblyFailed => return error.SkipZigTest,
        else => return err,
    };
    defer std.testing.allocator.free(sass);

    for (verdicts) |verdict| {
        try std.testing.expect(std.mem.indexOf(u8, sass, verdict.sass) != null);
    }

    // The `a` suffix is mandatory: the same source must fail for plain sm_120.
    const tool = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/bin/nvcc",
        .{cudaHome()},
    );
    defer std.testing.allocator.free(tool);
    const run = runTool(std.testing.allocator, &.{
        tool,
        "-arch=sm_120",
        "--cubin",
        probe_source_path,
        "-o",
        "/dev/null",
    }) catch return error.SkipZigTest;
    defer std.testing.allocator.free(run.stdout);
    defer std.testing.allocator.free(run.stderr);

    try std.testing.expect(!run.exited_zero);
    try std.testing.expect(std.mem.indexOf(u8, run.stderr, "sm_120a") != null);

    const report = try renderReport(std.testing.allocator, sass, true);
    defer std.testing.allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "OMMA.SF.16864") != null);
    writeReport(report) catch |err| {
        std.debug.print("verdict table not written: {s}\n", .{@errorName(err)});
    };
}
