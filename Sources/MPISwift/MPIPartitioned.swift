import CMPIBindings
import Foundation

/// MPI 5.0 Partitioned Point-to-Point Communication
///
/// Partitioned communication allows splitting large messages into smaller partitions
/// for better overlap between computation and communication. This is particularly
/// useful for large gradient tensors in distributed machine learning.

#if MPI5_PARTITIONED_AVAILABLE

// MARK: - Partitioned Send

extension MPICommunicator {
    /// Initialize a partitioned send operation
    /// - Parameters:
    ///   - buffer: Buffer containing data to send
    ///   - partitions: Number of partitions
    ///   - count: Total number of elements
    ///   - datatype: MPI datatype
    ///   - dest: Destination rank
    ///   - tag: Message tag
    ///   - info: MPI Info for hints
    ///   - request: Request handle for the operation
    /// - Throws: MPIError if initialization fails
    public func psendInit(
        buffer: UnsafeRawPointer,
        partitions: Int32,
        count: Int32,
        datatype: MPIDatatype,
        dest: Int32,
        tag: Int32,
        info: MPIInfo? = nil,
        request: inout MPIRequest
    ) throws {
        var tempRequest: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Psend_init(
            buffer, partitions, count,
            datatype.handle, dest, tag,
            handle, infoHandle, &tempRequest
        )
        try checkMPIError(errorCode)

        if let unwrapped = tempRequest {
            request.handle = unwrapped
        }
    }

    /// Initialize a partitioned receive operation
    /// - Parameters:
    ///   - buffer: Buffer to receive data
    ///   - partitions: Number of partitions
    ///   - count: Total number of elements
    ///   - datatype: MPI datatype
    ///   - source: Source rank
    ///   - tag: Message tag
    ///   - info: MPI Info for hints
    ///   - request: Request handle for the operation
    /// - Throws: MPIError if initialization fails
    public func precvInit(
        buffer: UnsafeMutableRawPointer,
        partitions: Int32,
        count: Int32,
        datatype: MPIDatatype,
        source: Int32,
        tag: Int32,
        info: MPIInfo? = nil,
        request: inout MPIRequest
    ) throws {
        var tempRequest: MPI_Request?
        let infoHandle = info?.handle ?? MPI_INFO_NULL_get()

        let errorCode = MPI_Precv_init(
            buffer, partitions, count,
            datatype.handle, source, tag,
            handle, infoHandle, &tempRequest
        )
        try checkMPIError(errorCode)

        if let unwrapped = tempRequest {
            request.handle = unwrapped
        }
    }
}

#endif // MPI5_PARTITIONED_AVAILABLE

// MARK: - Partition Management

/// Manager for partitioned communication operations
public final class MPIPartitionManager {
    private var partitions: [PartitionInfo] = []
    private let lock = NSLock()

    public init() {}

    /// Information about a partition
    public struct PartitionInfo {
        public let partitionIndex: Int32
        public let offset: Int
        public let size: Int
        public var isReady: Bool

        public init(partitionIndex: Int32, offset: Int, size: Int, isReady: Bool = false) {
            self.partitionIndex = partitionIndex
            self.offset = offset
            self.size = size
            self.isReady = isReady
        }
    }

    /// Create partitions for a buffer
    /// - Parameters:
    ///   - totalSize: Total size of the buffer
    ///   - numPartitions: Number of partitions to create
    /// - Returns: Array of partition information
    public func createPartitions(totalSize: Int, numPartitions: Int32) -> [PartitionInfo] {
        lock.lock()
        defer { lock.unlock() }

        partitions.removeAll()
        let partitionSize = totalSize / Int(numPartitions)
        let remainder = totalSize % Int(numPartitions)

        for i in 0..<Int(numPartitions) {
            let offset = i * partitionSize
            let size = partitionSize + (i < remainder ? 1 : 0)
            partitions.append(PartitionInfo(
                partitionIndex: Int32(i),
                offset: offset,
                size: size,
                isReady: false
            ))
        }

        return partitions
    }

    /// Mark a partition as ready
    /// - Parameter index: Partition index to mark ready
    public func markReady(_ index: Int32) {
        lock.lock()
        defer { lock.unlock() }

        if let idx = partitions.firstIndex(where: { $0.partitionIndex == index }) {
            partitions[idx].isReady = true
        }
    }

    /// Check if all partitions are ready
    public var allReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return partitions.allSatisfy { $0.isReady }
    }

    /// Get partition info
    /// - Parameter index: Partition index
    /// - Returns: Partition information if found
    public func getPartition(_ index: Int32) -> PartitionInfo? {
        lock.lock()
        defer { lock.unlock() }
        return partitions.first(where: { $0.partitionIndex == index })
    }

    /// Get all partitions
    public var allPartitions: [PartitionInfo] {
        lock.lock()
        defer { lock.unlock() }
        return partitions
    }
}

// MARK: - Partitioned Request Extensions

extension MPIRequest {
    #if MPI5_PARTITIONED_AVAILABLE

    /// Mark a partition as ready for sending (MPI_Pready)
    /// - Parameter partition: Partition index to mark ready
    /// - Throws: MPIError if the operation fails
    public mutating func pready(partition: Int32) throws {
        var tempHandle: MPI_Request? = handle
        let errorCode = MPI_Pready(partition, &tempHandle)
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
    }

    /// Mark a range of partitions as ready (MPI_Pready_range)
    /// - Parameters:
    ///   - startPartition: First partition index
    ///   - endPartition: Last partition index (inclusive)
    /// - Throws: MPIError if the operation fails
    public mutating func preadyRange(start: Int32, end: Int32) throws {
        var tempHandle: MPI_Request? = handle
        let errorCode = MPI_Pready_range(start, end, &tempHandle)
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
    }

    /// Mark a list of partitions as ready (MPI_Pready_list)
    /// - Parameter partitions: Array of partition indices
    /// - Throws: MPIError if the operation fails
    public mutating func preadyList(_ partitions: [Int32]) throws {
        var tempHandle: MPI_Request? = handle
        var partitionArray = partitions
        let errorCode = MPI_Pready_list(
            Int32(partitions.count),
            &partitionArray,
            &tempHandle
        )
        try checkMPIError(errorCode)
        if let unwrapped = tempHandle {
            handle = unwrapped
        }
    }

    #endif
}

// MARK: - High-Level Partitioned Operations

/// High-level wrapper for partitioned send operations
public final class PartitionedSender<T: MPIDataRepresentable> {
    private let comm: MPICommunicator
    private let dest: Int32
    private let tag: Int32
    private var request: MPIRequest
    private let manager: MPIPartitionManager
    private var buffer: [T]
    private let numPartitions: Int32

    /// Initialize a partitioned sender
    /// - Parameters:
    ///   - data: Data to send
    ///   - numPartitions: Number of partitions
    ///   - dest: Destination rank
    ///   - tag: Message tag
    ///   - comm: Communicator
    /// - Throws: MPIError if initialization fails
    public init(
        data: [T],
        numPartitions: Int32,
        dest: Int32,
        tag: Int32 = 0,
        comm: MPICommunicator
    ) throws {
        self.comm = comm
        self.dest = dest
        self.tag = tag
        self.buffer = data
        self.numPartitions = numPartitions
        self.request = MPIRequest()
        self.manager = MPIPartitionManager()

        #if MPI5_PARTITIONED_AVAILABLE
        // Initialize partitioned send
        try buffer.withUnsafeBufferPointer { bufferPtr in
            guard let baseAddress = bufferPtr.baseAddress else {
                throw MPIError.invalidBuffer
            }
            try comm.psendInit(
                buffer: baseAddress,
                partitions: numPartitions,
                count: Int32(buffer.count),
                datatype: T.mpiDatatype,
                dest: dest,
                tag: tag,
                request: &request
            )
        }

        // Create partition information
        _ = manager.createPartitions(totalSize: data.count, numPartitions: numPartitions)
        #else
        throw MPIError.otherError
        #endif
    }

    /// Start the send operation
    /// - Throws: MPIError if start fails
    public func start() throws {
        try request.start()
    }

    /// Mark a partition as ready
    /// - Parameter partition: Partition index
    /// - Throws: MPIError if marking fails
    public func markPartitionReady(_ partition: Int32) throws {
        #if MPI5_PARTITIONED_AVAILABLE
        try request.pready(partition: partition)
        manager.markReady(partition)
        #endif
    }

    /// Mark a range of partitions as ready
    /// - Parameters:
    ///   - start: First partition
    ///   - end: Last partition (inclusive)
    /// - Throws: MPIError if marking fails
    public func markRangeReady(start: Int32, end: Int32) throws {
        #if MPI5_PARTITIONED_AVAILABLE
        try request.preadyRange(start: start, end: end)
        for i in start...end {
            manager.markReady(i)
        }
        #endif
    }

    /// Wait for completion
    /// - Throws: MPIError if wait fails
    public func wait() throws {
        try request.wait()
    }

    /// Check if all partitions are ready
    public var allPartitionsReady: Bool {
        return manager.allReady
    }
}

/// High-level wrapper for partitioned receive operations
public final class PartitionedReceiver<T: MPIDataRepresentable> {
    private let comm: MPICommunicator
    private let source: Int32
    private let tag: Int32
    private var request: MPIRequest
    private let manager: MPIPartitionManager
    private var buffer: [T]
    private let numPartitions: Int32

    /// Initialize a partitioned receiver
    /// - Parameters:
    ///   - count: Total number of elements to receive
    ///   - numPartitions: Number of partitions
    ///   - source: Source rank
    ///   - tag: Message tag
    ///   - comm: Communicator
    /// - Throws: MPIError if initialization fails
    public init(
        count: Int,
        numPartitions: Int32,
        source: Int32,
        tag: Int32 = 0,
        comm: MPICommunicator
    ) throws {
        self.comm = comm
        self.source = source
        self.tag = tag
        self.numPartitions = numPartitions
        self.request = MPIRequest()
        self.manager = MPIPartitionManager()

        // Initialize buffer with default values
        self.buffer = [T](repeating: T.self == Double.self ? 0.0 as! T : T.self as! T, count: count)

        #if MPI5_PARTITIONED_AVAILABLE
        // Initialize partitioned receive
        try buffer.withUnsafeMutableBufferPointer { bufferPtr in
            guard let baseAddress = bufferPtr.baseAddress else {
                throw MPIError.invalidBuffer
            }
            try comm.precvInit(
                buffer: baseAddress,
                partitions: numPartitions,
                count: Int32(count),
                datatype: T.mpiDatatype,
                source: source,
                tag: tag,
                request: &request
            )
        }

        // Create partition information
        _ = manager.createPartitions(totalSize: count, numPartitions: numPartitions)
        #else
        throw MPIError.otherError
        #endif
    }

    /// Start the receive operation
    /// - Throws: MPIError if start fails
    public func start() throws {
        try request.start()
    }

    /// Wait for completion and return the received data
    /// - Returns: Received data
    /// - Throws: MPIError if wait fails
    public func wait() throws -> [T] {
        try request.wait()
        return buffer
    }

    /// Get partition information
    public var partitions: [MPIPartitionManager.PartitionInfo] {
        return manager.allPartitions
    }
}

// MARK: - Async/Await Support

#if compiler(>=5.5) && MPI5_PARTITIONED_AVAILABLE

extension PartitionedSender {
    /// Asynchronously send data with partitioned communication
    /// - Returns: Completion status
    @available(macOS 10.15, *)
    public func send() async throws {
        try start()

        // Mark all partitions ready asynchronously
        for i in 0..<numPartitions {
            try await Task {
                try self.markPartitionReady(i)
            }.value
        }

        // Wait for completion
        try await Task {
            try self.wait()
        }.value
    }
}

extension PartitionedReceiver {
    /// Asynchronously receive data with partitioned communication
    /// - Returns: Received data
    @available(macOS 10.15, *)
    public func receive() async throws -> [T] {
        try start()

        return try await Task {
            try self.wait()
        }.value
    }
}

#endif
