import CMPIBindings
import Foundation

/// MPI 5.0 Count type for large messages (64-bit)
public typealias MPICount = MPI_Count

/// MPI Datatype wrapper
public struct MPIDatatype: Equatable {
    let handle: MPI_Datatype

    // Standard MPI datatypes
    public static let int8 = MPIDatatype(handle: MPI_INT8_T_get())
    public static let int16 = MPIDatatype(handle: MPI_INT16_T_get())
    public static let int32 = MPIDatatype(handle: MPI_INT32_T_get())
    public static let int64 = MPIDatatype(handle: MPI_INT64_T_get())
    public static let uint8 = MPIDatatype(handle: MPI_UINT8_T_get())
    public static let uint16 = MPIDatatype(handle: MPI_UINT16_T_get())
    public static let uint32 = MPIDatatype(handle: MPI_UINT32_T_get())
    public static let uint64 = MPIDatatype(handle: MPI_UINT64_T_get())
    public static let float = MPIDatatype(handle: MPI_FLOAT_get())
    public static let double = MPIDatatype(handle: MPI_DOUBLE_get())
    public static let char = MPIDatatype(handle: MPI_CHAR_get())
    public static let byte = MPIDatatype(handle: MPI_BYTE_get())

    /// Get size of the datatype in bytes
    public var size: Int {
        var typeSize: Int32 = 0
        MPI_Type_size(handle, &typeSize)
        return Int(typeSize)
    }

    /// Get extent of the datatype
    public var extent: (lower: Int, upper: Int) {
        var lb = MPI_Aint()
        var extent = MPI_Aint()
        MPI_Type_get_extent(handle, &lb, &extent)
        return (Int(lb), Int(extent))
    }
}

/// Protocol for types that can be sent via MPI
public protocol MPIDataRepresentable {
    static var mpiDatatype: MPIDatatype { get }
}

extension Int8: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .int8 }
}

extension Int16: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .int16 }
}

extension Int32: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .int32 }
}

extension Int64: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .int64 }
}

extension UInt8: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .uint8 }
}

extension UInt16: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .uint16 }
}

extension UInt32: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .uint32 }
}

extension UInt64: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .uint64 }
}

extension Float: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .float }
}

extension Double: MPIDataRepresentable {
    public static var mpiDatatype: MPIDatatype { .double }
}

/// MPI Buffer descriptor - type-safe buffer handling
public struct MPIBufferDescriptor<T: MPIDataRepresentable> {
    public let pointer: UnsafeMutablePointer<T>
    public let count: Int

    public init(pointer: UnsafeMutablePointer<T>, count: Int) {
        self.pointer = pointer
        self.count = count
    }

    /// Create from array pointer
    /// - Important: Use within array.withUnsafeMutableBufferPointer { } to ensure pointer validity
    /// Example:
    /// ```
    /// var data = [1.0, 2.0, 3.0]
    /// data.withUnsafeMutableBufferPointer { buffer in
    ///     let descriptor = MPIBufferDescriptor(pointer: buffer.baseAddress!, count: buffer.count)
    ///     // Use descriptor with MPI operations
    /// }
    /// ```
    public static func from(pointer: UnsafeMutablePointer<T>, count: Int) -> MPIBufferDescriptor<T> {
        return MPIBufferDescriptor(pointer: pointer, count: count)
    }

    /// Get MPI datatype for this buffer
    public var datatype: MPIDatatype {
        return T.mpiDatatype
    }

    /// Get count as MPI_Count for large count operations
    public var largeCount: MPICount {
        return MPICount(count)
    }
}

/// MPI Large Count operations support
public struct MPILargeCount {
    /// Check if large count interface is available
    public static var isAvailable: Bool {
#if MPI_LARGE_COUNT_AVAILABLE
        return true
#else
        return false
#endif
    }

    /// Maximum count for standard MPI operations
    public static let maxStandardCount = Int(Int32.max)

    /// Check if count requires large count interface
    public static func requiresLargeCount(_ count: Int) -> Bool {
        return count > maxStandardCount
    }
}

/// MPI Operation for reduction operations
public struct MPIOperation: Equatable {
    let handle: MPI_Op

    public static func == (lhs: MPIOperation, rhs: MPIOperation) -> Bool {
        return lhs.handle == rhs.handle
    }

    // Standard MPI operations
    public static let max = MPIOperation(handle: MPI_MAX_get())
    public static let min = MPIOperation(handle: MPI_MIN_get())
    public static let sum = MPIOperation(handle: MPI_SUM_get())
    public static let prod = MPIOperation(handle: MPI_PROD_get())
    public static let land = MPIOperation(handle: MPI_LAND_get())
    public static let lor = MPIOperation(handle: MPI_LOR_get())
    public static let band = MPIOperation(handle: MPI_BAND_get())
    public static let bor = MPIOperation(handle: MPI_BOR_get())
    public static let lxor = MPIOperation(handle: MPI_LXOR_get())
    public static let bxor = MPIOperation(handle: MPI_BXOR_get())
    public static let maxloc = MPIOperation(handle: MPI_MAXLOC_get())
    public static let minloc = MPIOperation(handle: MPI_MINLOC_get())
    public static let replace = MPIOperation(handle: MPI_REPLACE_get())
}

/// MPI Request for non-blocking operations
public struct MPIRequest {
    var handle: MPI_Request

    public init() {
        self.handle = MPI_REQUEST_NULL_get()
    }

    /// Start a persistent request
    public mutating func start() throws {
        var tempHandle: MPI_Request? = handle
        let errorCode = MPI_Start(&tempHandle)
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
    }

    /// Wait for the request to complete
    public mutating func wait() throws {
        var status = MPI_Status()
        var tempHandle: MPI_Request? = handle
        let errorCode = MPI_Wait(&tempHandle, &status)
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
    }

    /// Test if the request is complete
    public mutating func test() throws -> Bool {
        var flag: Int32 = 0
        var status = MPI_Status()
        var tempHandle: MPI_Request? = handle
        let errorCode = MPI_Test(&tempHandle, &flag, &status)
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
        return flag != 0
    }

    /// Cancel the request
    public mutating func cancel() throws {
        var tempHandle: MPI_Request? = handle
        let errorCode = MPI_Cancel(&tempHandle)
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
    }
}

/// MPI Status for message information
public struct MPIStatus {
    var status: MPI_Status

    public init() {
        self.status = MPI_Status()
    }

    init(status: MPI_Status) {
        self.status = status
    }

    /// Source rank of the message
    public var source: Int32 {
        return status.MPI_SOURCE
    }

    /// Tag of the message
    public var tag: Int32 {
        return status.MPI_TAG
    }

    /// Error code
    public var error: Int32 {
        return status.MPI_ERROR
    }

    /// Get count of received elements
    public mutating func getCount(datatype: MPIDatatype) -> Int {
        var count: Int32 = 0
        MPI_Get_count(&status, datatype.handle, &count)
        return Int(count)
    }
}
