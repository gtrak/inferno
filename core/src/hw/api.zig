//! The public hardware boundary: the one `hw/` file the rest of the engine
//! may import.
//!
//! `tools/invariants.py` (G1) forbids every other directory from importing
//! `hw/cuda_driver.zig` or `hw/cuda_context.zig`. The point is not ceremony:
//! those two files expose the raw driver entry-point table (`functions`,
//! `check_result`) and the ABI types behind it. Keeping them private to `hw/`
//! means engine code cannot reach a raw `cuMemAlloc`/`cuLaunchKernel` even by
//! accident, and the typed handles below are the only door.
//!
//! The re-exports are intentionally explicit, not `pub const cuda_driver =
//! @import(...)`: a whole-module alias would re-export `functions` too and the
//! boundary would again exist in name only (I-028, G1).

const raw_context = @import("cuda_context.zig");
const raw_driver = @import("cuda_driver.zig");

/// Typed driver-resource handles. Each owns exactly one driver resource.
pub const cuda_context = struct {
    pub const Context = raw_context.Context;
    pub const DeviceBuffer = raw_context.DeviceBuffer;
    pub const PinnedHostBuffer = raw_context.PinnedHostBuffer;
    pub const Stream = raw_context.Stream;
    pub const Event = raw_context.Event;
    pub const DriverError = raw_context.DriverError;
};

/// Device enumeration and version queries. The raw symbol table is absent.
pub const cuda_driver = struct {
    pub const Device = raw_driver.Device;
    pub const DevicePtr = raw_driver.DevicePtr;
    pub const DriverError = raw_driver.DriverError;
    pub const DriverVersion = raw_driver.DriverVersion;
    pub const device_count_max = raw_driver.device_count_max;
    pub const device_name_bytes_max = raw_driver.device_name_bytes_max;
    pub const driver_version_bytes_max = raw_driver.driver_version_bytes_max;
    pub const pci_bus_id_bytes_max = raw_driver.pci_bus_id_bytes_max;

    /// Enumerates the devices this process can use, into `devices`.
    pub const query_devices = raw_driver.query_devices;

    /// Whether `from` can reach `to`'s memory through peer access.
    pub const can_access_peer = raw_driver.can_access_peer;

    /// The CUDA API version the driver implements.
    pub const cuda_api_version = raw_driver.cuda_api_version;

    /// The display-driver version, read from procfs.
    pub const display_driver_version = raw_driver.display_driver_version;
};

test {
    _ = @import("cuda_context.zig");
    _ = @import("cuda_driver.zig");
    _ = @import("epochs.zig");
}
