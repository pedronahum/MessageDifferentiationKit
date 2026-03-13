import CMPIBindings
import Foundation

/// MPI 5.0 Hardware Offload Support
///
/// This module provides support for GPU-aware MPI and hardware-accelerated communication.
/// It enables direct communication from GPU memory without intermediate host copies.

// MARK: - Device Type

/// Supported device types for hardware offload
public enum MPIDeviceType: String {
    case cuda = "cuda"           // NVIDIA CUDA
    case rocm = "rocm"           // AMD ROCm
    case levelZero = "level_zero" // Intel Level Zero
    case opencl = "opencl"       // OpenCL
    case host = "host"           // CPU memory
}

// MARK: - Device Buffer

/// Descriptor for device (GPU) memory buffers
public struct MPIDeviceBuffer<T: MPIDataRepresentable> {
    /// Pointer to device memory
    public let devicePointer: UnsafeMutableRawPointer

    /// Number of elements
    public let count: Int

    /// Device type
    public let deviceType: MPIDeviceType

    /// Device ID (for multi-GPU systems)
    public let deviceID: Int32

    public init(
        devicePointer: UnsafeMutableRawPointer,
        count: Int,
        deviceType: MPIDeviceType,
        deviceID: Int32 = 0
    ) {
        self.devicePointer = devicePointer
        self.count = count
        self.deviceType = deviceType
        self.deviceID = deviceID
    }

    /// Get MPI datatype for this buffer
    public var datatype: MPIDatatype {
        return T.mpiDatatype
    }

    /// Get count as Int32
    public var count32: Int32 {
        return Int32(count)
    }
}

// MARK: - Hardware Offload Info Builder

extension MPIInfoBuilder {
    /// Enable GPU-aware MPI
    /// - Parameter deviceType: Type of GPU device
    /// - Returns: Self for chaining
    @discardableResult
    public func enableGPU(deviceType: MPIDeviceType) -> MPIInfoBuilder {
        try? info.set(key: "mpi_hw_offload", value: "true")
        try? info.set(key: "mpi_device_pointer", value: deviceType.rawValue)
        return self
    }

    /// Set GPU device ID
    /// - Parameter deviceID: Device identifier
    /// - Returns: Self for chaining
    @discardableResult
    public func setDeviceID(_ deviceID: Int32) -> MPIInfoBuilder {
        try? info.set(key: "mpi_device_id", value: "\(deviceID)")
        return self
    }

    /// Enable unified memory support
    /// - Returns: Self for chaining
    @discardableResult
    public func enableUnifiedMemory() -> MPIInfoBuilder {
        try? info.set(key: "mpi_unified_memory", value: "true")
        return self
    }

    /// Set memory pool allocation strategy
    /// - Parameter strategy: Allocation strategy hint
    /// - Returns: Self for chaining
    @discardableResult
    public func setMemoryPool(_ strategy: String) -> MPIInfoBuilder {
        try? info.set(key: "mpi_memory_pool", value: strategy)
        return self
    }
}

// MARK: - GPU-Aware Communication

extension MPICommunicator {
    /// Send data from device (GPU) memory
    /// - Parameters:
    ///   - deviceBuffer: Device buffer descriptor
    ///   - dest: Destination rank
    ///   - tag: Message tag
    ///   - info: MPI Info with hardware offload hints
    /// - Throws: MPIError if the operation fails
    public func sendFromDevice<T: MPIDataRepresentable>(
        _ deviceBuffer: MPIDeviceBuffer<T>,
        to dest: Int32,
        tag: Int32 = 0,
        info: MPIInfo? = nil
    ) throws {
        // Note: info parameter is provided for API consistency but GPU-aware MPI
        // handles device pointers automatically. The MPI implementation detects
        // device memory and uses appropriate transport.
        _ = info  // Silence unused warning

        // Perform send with device pointer
        try send(
            buffer: deviceBuffer.devicePointer,
            count: deviceBuffer.count32,
            datatype: deviceBuffer.datatype,
            dest: dest,
            tag: tag
        )
    }

    /// Receive data into device (GPU) memory
    /// - Parameters:
    ///   - deviceBuffer: Device buffer descriptor
    ///   - source: Source rank
    ///   - tag: Message tag
    ///   - info: MPI Info with hardware offload hints
    /// - Throws: MPIError if the operation fails
    public func recvIntoDevice<T: MPIDataRepresentable>(
        _ deviceBuffer: MPIDeviceBuffer<T>,
        from source: Int32,
        tag: Int32 = 0,
        info: MPIInfo? = nil
    ) throws {
        var status: MPIStatus? = MPIStatus()

        try recv(
            buffer: deviceBuffer.devicePointer,
            count: deviceBuffer.count32,
            datatype: deviceBuffer.datatype,
            source: source,
            tag: tag,
            status: &status
        )
    }

    /// Non-blocking send from device memory
    /// - Parameters:
    ///   - deviceBuffer: Device buffer descriptor
    ///   - dest: Destination rank
    ///   - tag: Message tag
    ///   - request: Request handle
    ///   - info: MPI Info with hardware offload hints
    /// - Throws: MPIError if the operation fails
    public func isendFromDevice<T: MPIDataRepresentable>(
        _ deviceBuffer: MPIDeviceBuffer<T>,
        to dest: Int32,
        tag: Int32 = 0,
        request: inout MPIRequest,
        info: MPIInfo? = nil
    ) throws {
        try isend(
            buffer: deviceBuffer.devicePointer,
            count: deviceBuffer.count32,
            datatype: deviceBuffer.datatype,
            dest: dest,
            tag: tag,
            request: &request
        )
    }

    /// Non-blocking receive into device memory
    /// - Parameters:
    ///   - deviceBuffer: Device buffer descriptor
    ///   - source: Source rank
    ///   - tag: Message tag
    ///   - request: Request handle
    ///   - info: MPI Info with hardware offload hints
    /// - Throws: MPIError if the operation fails
    public func irecvIntoDevice<T: MPIDataRepresentable>(
        _ deviceBuffer: MPIDeviceBuffer<T>,
        from source: Int32,
        tag: Int32 = 0,
        request: inout MPIRequest,
        info: MPIInfo? = nil
    ) throws {
        try irecv(
            buffer: deviceBuffer.devicePointer,
            count: deviceBuffer.count32,
            datatype: deviceBuffer.datatype,
            source: source,
            tag: tag,
            request: &request
        )
    }
}

// MARK: - GPU-Aware Collectives

extension MPICommunicator {
    /// Allreduce operation on device memory
    /// - Parameters:
    ///   - sendBuffer: Device buffer containing send data
    ///   - recvBuffer: Device buffer for receive data
    ///   - op: Reduction operation
    ///   - info: MPI Info with hardware offload hints
    /// - Throws: MPIError if the operation fails
    public func allreduceOnDevice<T: MPIDataRepresentable>(
        sendbuf: MPIDeviceBuffer<T>,
        recvbuf: MPIDeviceBuffer<T>,
        operation op: MPIOperation,
        info: MPIInfo? = nil
    ) throws {
        guard sendbuf.count == recvbuf.count else {
            throw MPIError.invalidCount
        }

        try allreduce(
            sendbuf: sendbuf.devicePointer,
            recvbuf: recvbuf.devicePointer,
            count: sendbuf.count32,
            datatype: sendbuf.datatype,
            op: op
        )
    }

    /// Broadcast from device memory
    /// - Parameters:
    ///   - deviceBuffer: Device buffer (root has data, others receive)
    ///   - root: Root rank
    ///   - info: MPI Info with hardware offload hints
    /// - Throws: MPIError if the operation fails
    public func broadcastOnDevice<T: MPIDataRepresentable>(
        _ deviceBuffer: MPIDeviceBuffer<T>,
        root: Int32,
        info: MPIInfo? = nil
    ) throws {
        try bcast(
            buffer: deviceBuffer.devicePointer,
            count: deviceBuffer.count32,
            datatype: deviceBuffer.datatype,
            root: root
        )
    }
}

// MARK: - Memory Management Helpers

/// GPU memory registration and management
public final class MPIDeviceMemoryManager {
    private var registeredBuffers: [UInt: DeviceBufferInfo] = [:]
    private let lock = NSLock()

    public init() {}

    private struct DeviceBufferInfo {
        let pointer: UnsafeMutableRawPointer
        let size: Int
        let deviceType: MPIDeviceType
        let deviceID: Int32
    }

    /// Register a device buffer for MPI communication
    /// - Parameters:
    ///   - pointer: Device memory pointer
    ///   - size: Size in bytes
    ///   - deviceType: Type of device
    ///   - deviceID: Device identifier
    /// - Returns: Handle for the registered buffer
    public func registerBuffer(
        pointer: UnsafeMutableRawPointer,
        size: Int,
        deviceType: MPIDeviceType,
        deviceID: Int32 = 0
    ) -> UInt {
        lock.lock()
        defer { lock.unlock() }

        let handle = UInt(bitPattern: pointer)
        registeredBuffers[handle] = DeviceBufferInfo(
            pointer: pointer,
            size: size,
            deviceType: deviceType,
            deviceID: deviceID
        )
        return handle
    }

    /// Unregister a device buffer
    /// - Parameter handle: Handle returned from registerBuffer
    public func unregisterBuffer(_ handle: UInt) {
        lock.lock()
        defer { lock.unlock() }
        registeredBuffers.removeValue(forKey: handle)
    }

    /// Check if a buffer is registered
    /// - Parameter pointer: Device memory pointer
    /// - Returns: True if registered
    public func isRegistered(pointer: UnsafeMutableRawPointer) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let handle = UInt(bitPattern: pointer)
        return registeredBuffers[handle] != nil
    }

    /// Get device type for a registered buffer
    /// - Parameter pointer: Device memory pointer
    /// - Returns: Device type if registered
    public func getDeviceType(pointer: UnsafeMutableRawPointer) -> MPIDeviceType? {
        lock.lock()
        defer { lock.unlock() }
        let handle = UInt(bitPattern: pointer)
        return registeredBuffers[handle]?.deviceType
    }
}

// MARK: - CUDA-Specific Helpers (placeholder)

#if canImport(CUDA)
import CUDA

extension MPIDeviceBuffer {
    /// Create device buffer from CUDA device pointer
    /// - Parameters:
    ///   - cudaPointer: CUDA device pointer
    ///   - count: Number of elements
    ///   - deviceID: CUDA device ID
    /// - Returns: MPI device buffer
    public static func fromCUDA(
        cudaPointer: CUdeviceptr,
        count: Int,
        deviceID: Int32 = 0
    ) -> MPIDeviceBuffer<T> {
        let rawPointer = UnsafeMutableRawPointer(bitPattern: UInt(cudaPointer))!
        return MPIDeviceBuffer(
            devicePointer: rawPointer,
            count: count,
            deviceType: .cuda,
            deviceID: deviceID
        )
    }
}
#endif

// MARK: - Documentation and Best Practices

/*
 Hardware Offload Best Practices:

 1. Device Memory: Use GPU-aware MPI only when both sender and receiver have
    MPI implementations that support it.

 2. Performance: Direct GPU communication avoids host-device copies, improving
    performance for large data transfers.

 3. Verification: Check MPI implementation capabilities:
    - OpenMPI: Requires --with-cuda or --with-rocm
    - MPICH: Requires CH4 device with GPU support

 4. Memory Registration: Some MPI implementations benefit from registering
    GPU memory buffers for better performance.

 5. Unified Memory: CUDA unified memory can simplify programming but may
    have performance implications.

 Example Usage:

 ```swift
 // Assuming CUDA device pointer
 let devicePtr: UnsafeMutableRawPointer = ...
 let deviceBuffer = MPIDeviceBuffer<Float>(
     devicePointer: devicePtr,
     count: 1000000,
     deviceType: .cuda,
     deviceID: 0
 )

 // GPU-aware send
 try comm.sendFromDevice(deviceBuffer, to: 1, tag: 0)

 // GPU-aware allreduce
 try comm.allreduceOnDevice(
     sendbuf: deviceBuffer,
     recvbuf: resultBuffer,
     operation: .sum
 )
 ```
 */
