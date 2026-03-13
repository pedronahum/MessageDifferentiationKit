import CMPIBindings
import Foundation

/// MPI Info object for key-value pairs and performance hints
public final class MPIInfo {
    private var info: MPI_Info
    private var isFreed: Bool = false

    /// Create a new MPI Info object
    public init() throws {
        var tempInfo: MPI_Info? = MPI_INFO_NULL_get()
        let errorCode = MPI_Info_create(&tempInfo)
        try checkMPIError(errorCode)
        guard let unwrapped = tempInfo else {
            throw MPIError.internalError
        }
        self.info = unwrapped
    }

    /// Initialize from existing MPI_Info handle
    init(handle: MPI_Info) {
        self.info = handle
    }

    /// Set a key-value pair
    public func set(key: String, value: String) throws {
        let errorCode = key.withCString { keyStr in
            value.withCString { valueStr in
                MPI_Info_set(info, keyStr, valueStr)
            }
        }
        try checkMPIError(errorCode)
    }

    /// Get value for a key
    public func get(key: String) throws -> String? {
        var valuelen: Int32 = 0
        var flag: Int32 = 0

        // First get the length
        let errorCode1 = key.withCString { keyStr in
            MPI_Info_get_valuelen(info, keyStr, &valuelen, &flag)
        }
        try checkMPIError(errorCode1)

        guard flag != 0 else {
            return nil
        }

        // Now get the actual value
        var value = [CChar](repeating: 0, count: Int(valuelen) + 1)
        let errorCode2 = key.withCString { keyStr in
            MPI_Info_get(info, keyStr, valuelen, &value, &flag)
        }
        try checkMPIError(errorCode2)

        return String(cString: value)
    }

    /// Delete a key
    public func delete(key: String) throws {
        let errorCode = key.withCString { keyStr in
            MPI_Info_delete(info, keyStr)
        }
        try checkMPIError(errorCode)
    }

    /// Get number of keys
    public var numberOfKeys: Int32 {
        var nkeys: Int32 = 0
        MPI_Info_get_nkeys(info, &nkeys)
        return nkeys
    }

    /// Get all keys
    public func getAllKeys() throws -> [String] {
        let count = numberOfKeys
        var keys: [String] = []

        for i in 0..<count {
            var key = [CChar](repeating: 0, count: Int(MPI_MAX_INFO_KEY) + 1)
            let errorCode = MPI_Info_get_nthkey(info, i, &key)
            try checkMPIError(errorCode)
            keys.append(String(cString: key))
        }

        return keys
    }

    /// Duplicate the info object
    public func duplicate() throws -> MPIInfo {
        var newInfo: MPI_Info? = MPI_INFO_NULL_get()
        let errorCode = MPI_Info_dup(info, &newInfo)
        try checkMPIError(errorCode)
        guard let unwrapped = newInfo else {
            throw MPIError.internalError
        }
        return MPIInfo(handle: unwrapped)
    }

    /// Free the info object
    public func free() throws {
        guard !isFreed else { return }
        var tempInfo: MPI_Info? = info
        let errorCode = MPI_Info_free(&tempInfo)
        try checkMPIError(errorCode)
        isFreed = true
    }

    deinit {
        if !isFreed && info != MPI_INFO_NULL_get() {
            try? free()
        }
    }

    /// Get the raw MPI_Info handle
    internal var handle: MPI_Info {
        return info
    }
}

/// MPI 5.0 Application Info Assertions - Performance hints
public struct MPIAssertions {
    private let info: MPIInfo

    public init() throws {
        self.info = try MPIInfo()
    }

    init(info: MPIInfo) {
        self.info = info
    }

    /// Assert no locks (for RMA operations)
    public func assertNoLocks() throws {
        try info.set(key: "no_locks", value: "true")
    }

    /// Assert no concurrent puts (for RMA operations)
    public func assertNoConcurrentPuts() throws {
        try info.set(key: "no_concurrent_puts", value: "true")
    }

    /// Assert same operation for all processes (collective optimization)
    public func assertSameOp() throws {
        try info.set(key: "same_op", value: "true")
    }

    /// Assert same datatype for all processes
    public func assertSameDatatype() throws {
        try info.set(key: "same_datatype", value: "true")
    }

    /// Enable hardware offload (GPU-aware MPI)
    public func enableHardwareOffload() throws {
        try info.set(key: "mpi_hw_offload", value: "true")
    }

    /// Set device pointer type (cuda, rocm, level_zero, etc.)
    public func setDevicePointerType(_ type: String) throws {
        try info.set(key: "mpi_device_pointer", value: type)
    }

    /// Enable CUDA-aware MPI
    public func enableCUDAOffload() throws {
        try setDevicePointerType("cuda")
        try enableHardwareOffload()
    }

    /// Enable ROCm-aware MPI
    public func enableROCmOffload() throws {
        try setDevicePointerType("rocm")
        try enableHardwareOffload()
    }

    /// Set memory allocation alignment
    public func setMemoryAlignment(bytes: Int) throws {
        try info.set(key: "mpi_memory_align", value: "\(bytes)")
    }

    /// Enable large count operations
    public func enableLargeCount() throws {
        try info.set(key: "mpi_large_count", value: "true")
    }

    /// Hint for persistent collective optimization
    public func hintPersistentCollective() throws {
        try info.set(key: "mpi_persistent_coll", value: "true")
    }

    /// Hint for partitioned communication
    public func hintPartitionedComm() throws {
        try info.set(key: "mpi_partitioned", value: "true")
    }

    /// Set expected message size for optimization
    public func setExpectedMessageSize(bytes: Int) throws {
        try info.set(key: "mpi_expected_msg_size", value: "\(bytes)")
    }

    /// Custom assertion
    public func setCustomAssertion(key: String, value: String) throws {
        try info.set(key: key, value: value)
    }

    /// Get the underlying MPI Info object
    public var mpiInfo: MPIInfo {
        return info
    }
}

/// Property wrapper for performance assertions
@propertyWrapper
public struct MPIPerformanceHint {
    private let info: MPIInfo
    private let key: String

    public var wrappedValue: String {
        get {
            (try? info.get(key: key)) ?? ""
        }
        nonmutating set {
            try? info.set(key: key, value: newValue)
        }
    }

    public init(key: String) throws {
        self.info = try MPIInfo()
        self.key = key
    }
}

/// Builder pattern for MPI Info objects
public class MPIInfoBuilder {
    internal let info: MPIInfo

    public init() throws {
        self.info = try MPIInfo()
    }

    @discardableResult
    public func set(_ key: String, _ value: String) -> MPIInfoBuilder {
        try? info.set(key: key, value: value)
        return self
    }

    @discardableResult
    public func enableHardwareOffload() -> MPIInfoBuilder {
        try? info.set(key: "mpi_hw_offload", value: "true")
        return self
    }

    @discardableResult
    public func enableLargeCount() -> MPIInfoBuilder {
        try? info.set(key: "mpi_large_count", value: "true")
        return self
    }

    @discardableResult
    public func deviceType(_ type: String) -> MPIInfoBuilder {
        try? info.set(key: "mpi_device_pointer", value: type)
        return self
    }

    public func build() -> MPIInfo {
        return info
    }
}
