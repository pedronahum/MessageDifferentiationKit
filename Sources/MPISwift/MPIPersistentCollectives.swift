import CMPIBindings
import Foundation

/// MPI 5.0 Persistent Collective Operations
///
/// Persistent collectives allow pre-initialization of collective operations
/// for better performance when the same operation is repeated multiple times.

// MARK: - Persistent Request

/// Persistent MPI Request for reusable operations
public final class MPIPersistentRequest {
    private var request: MPI_Request
    private var isActive: Bool = false

    init(request: MPI_Request) {
        self.request = request
    }

    /// Start the persistent request
    /// - Throws: MPIError if the operation fails
    public func start() throws {
        var tempRequest: MPI_Request? = request
        let errorCode = MPI_Start(&tempRequest)
        try checkMPIError(errorCode)
        if let unwrapped = tempRequest {
            request = unwrapped
        }
        isActive = true
    }

    /// Wait for the persistent request to complete
    /// - Throws: MPIError if the operation fails
    public func wait() throws {
        guard isActive else { return }

        var status = MPI_Status()
        var tempRequest: MPI_Request? = request
        let errorCode = MPI_Wait(&tempRequest, &status)
        try checkMPIError(errorCode)
        if let unwrapped = tempRequest {
            request = unwrapped
        }
        isActive = false
    }

    /// Free the persistent request
    /// - Throws: MPIError if the operation fails
    public func free() throws {
        var tempRequest: MPI_Request? = request
        let errorCode = MPI_Request_free(&tempRequest)
        try checkMPIError(errorCode)
        if let unwrapped = tempRequest {
            request = unwrapped
        }
    }

    deinit {
        try? free()
    }
}

// MARK: - Persistent Collectives (MPI 5.0)

#if MPI5_PERSISTENT_COLLECTIVES_AVAILABLE

extension MPICommunicator {
    /// Initialize a persistent broadcast operation
    /// - Parameters:
    ///   - buffer: Buffer for data
    ///   - count: Number of elements
    ///   - datatype: MPI datatype
    ///   - root: Root process rank
    ///   - info: MPI Info for optimization hints
    /// - Returns: Persistent request that can be reused
    /// - Throws: MPIError if initialization fails
    public func bcastInit(
        buffer: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        root: Int32,
        info: MPIInfo? = nil
    ) throws -> MPIPersistentRequest {
        var request: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Bcast_init(
            buffer, count, datatype.handle, root,
            handle, infoHandle, &request
        )
        try checkMPIError(errorCode)

        guard let unwrappedRequest = request else {
            throw MPIError.invalidRequest
        }

        return MPIPersistentRequest(request: unwrappedRequest)
    }

    /// Initialize a persistent reduce operation
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - recvbuf: Receive buffer
    ///   - count: Number of elements
    ///   - datatype: MPI datatype
    ///   - op: Reduction operation
    ///   - root: Root process rank
    ///   - info: MPI Info for optimization hints
    /// - Returns: Persistent request that can be reused
    /// - Throws: MPIError if initialization fails
    public func reduceInit(
        sendbuf: UnsafeRawPointer,
        recvbuf: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        op: MPIOperation,
        root: Int32,
        info: MPIInfo? = nil
    ) throws -> MPIPersistentRequest {
        var request: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Reduce_init(
            sendbuf, recvbuf, count,
            datatype.handle, op.handle, root,
            handle, infoHandle, &request
        )
        try checkMPIError(errorCode)

        guard let unwrappedRequest = request else {
            throw MPIError.invalidRequest
        }

        return MPIPersistentRequest(request: unwrappedRequest)
    }

    /// Initialize a persistent allreduce operation
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - recvbuf: Receive buffer
    ///   - count: Number of elements
    ///   - datatype: MPI datatype
    ///   - op: Reduction operation
    ///   - info: MPI Info for optimization hints
    /// - Returns: Persistent request that can be reused
    /// - Throws: MPIError if initialization fails
    public func allreduceInit(
        sendbuf: UnsafeRawPointer,
        recvbuf: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        op: MPIOperation,
        info: MPIInfo? = nil
    ) throws -> MPIPersistentRequest {
        var request: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Allreduce_init(
            sendbuf, recvbuf, count,
            datatype.handle, op.handle,
            handle, infoHandle, &request
        )
        try checkMPIError(errorCode)

        guard let unwrappedRequest = request else {
            throw MPIError.invalidRequest
        }

        return MPIPersistentRequest(request: unwrappedRequest)
    }

    /// Initialize a persistent gather operation
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send
    ///   - sendtype: Send datatype
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive per process
    ///   - recvtype: Receive datatype
    ///   - root: Root process rank
    ///   - info: MPI Info for optimization hints
    /// - Returns: Persistent request that can be reused
    /// - Throws: MPIError if initialization fails
    public func gatherInit(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype,
        root: Int32,
        info: MPIInfo? = nil
    ) throws -> MPIPersistentRequest {
        var request: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Gather_init(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            root, handle, infoHandle, &request
        )
        try checkMPIError(errorCode)

        guard let unwrappedRequest = request else {
            throw MPIError.invalidRequest
        }

        return MPIPersistentRequest(request: unwrappedRequest)
    }

    /// Initialize a persistent allgather operation
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send
    ///   - sendtype: Send datatype
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive per process
    ///   - recvtype: Receive datatype
    ///   - info: MPI Info for optimization hints
    /// - Returns: Persistent request that can be reused
    /// - Throws: MPIError if initialization fails
    public func allgatherInit(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype,
        info: MPIInfo? = nil
    ) throws -> MPIPersistentRequest {
        var request: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Allgather_init(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            handle, infoHandle, &request
        )
        try checkMPIError(errorCode)

        guard let unwrappedRequest = request else {
            throw MPIError.invalidRequest
        }

        return MPIPersistentRequest(request: unwrappedRequest)
    }
}

#endif // MPI5_PERSISTENT_COLLECTIVES_AVAILABLE

// MARK: - Operation Cache

/// Cache for persistent collective operations
/// Provides a convenient way to manage and reuse persistent operations
public final class MPIOperationCache {
    private var operations: [String: MPIPersistentRequest] = [:]
    private let lock = NSLock()

    public init() {}

    /// Store a persistent operation with a key
    /// - Parameters:
    ///   - key: Identifier for the operation
    ///   - operation: Persistent request to cache
    public func store(key: String, operation: MPIPersistentRequest) {
        lock.lock()
        defer { lock.unlock() }
        operations[key] = operation
    }

    /// Retrieve a cached operation
    /// - Parameter key: Identifier for the operation
    /// - Returns: Cached persistent request, or nil if not found
    public func get(key: String) -> MPIPersistentRequest? {
        lock.lock()
        defer { lock.unlock() }
        return operations[key]
    }

    /// Execute a cached operation
    /// - Parameter key: Identifier for the operation
    /// - Throws: MPIError if the operation fails or is not found
    public func execute(key: String) throws {
        guard let operation = get(key: key) else {
            throw MPIError.invalidRequest
        }
        try operation.start()
        try operation.wait()
    }

    /// Remove and free a cached operation
    /// - Parameter key: Identifier for the operation
    /// - Throws: MPIError if freeing fails
    public func remove(key: String) throws {
        lock.lock()
        defer { lock.unlock() }

        if let operation = operations.removeValue(forKey: key) {
            try operation.free()
        }
    }

    /// Clear all cached operations
    /// - Throws: MPIError if freeing any operation fails
    public func clearAll() throws {
        lock.lock()
        defer { lock.unlock() }

        for operation in operations.values {
            try operation.free()
        }
        operations.removeAll()
    }

    deinit {
        try? clearAll()
    }
}
