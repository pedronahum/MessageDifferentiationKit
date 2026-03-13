import CMPIBindings
import Foundation

/// Point-to-Point Communication Operations
///
/// This module provides blocking and non-blocking send/receive operations
/// for direct communication between MPI processes.

// MARK: - Blocking Point-to-Point

extension MPICommunicator {
    /// Blocking send operation
    /// - Parameters:
    ///   - buffer: Pointer to the data to send
    ///   - count: Number of elements to send
    ///   - datatype: MPI datatype of the elements
    ///   - dest: Rank of the destination process
    ///   - tag: Message tag
    /// - Throws: MPIError if the operation fails
    public func send(
        buffer: UnsafeRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        dest: Int32,
        tag: Int32
    ) throws {
        let errorCode = MPI_Send(buffer, count, datatype.handle, dest, tag, handle)
        try checkMPIError(errorCode)
    }

    /// Type-safe blocking send
    /// - Parameters:
    ///   - data: Array of data to send
    ///   - dest: Rank of the destination process
    ///   - tag: Message tag (default: 0)
    /// - Throws: MPIError if the operation fails
    public func send<T: MPIDataRepresentable>(
        _ data: [T],
        to dest: Int32,
        tag: Int32 = 0
    ) throws {
        try data.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                throw MPIError.invalidBuffer
            }
            try send(
                buffer: baseAddress,
                count: Int32(buffer.count),
                datatype: T.mpiDatatype,
                dest: dest,
                tag: tag
            )
        }
    }

    /// Blocking receive operation
    /// - Parameters:
    ///   - buffer: Pointer to the receive buffer
    ///   - count: Maximum number of elements to receive
    ///   - datatype: MPI datatype of the elements
    ///   - source: Rank of the source process (or MPI_ANY_SOURCE)
    ///   - tag: Message tag (or MPI_ANY_TAG)
    ///   - status: Optional status object to receive message information
    /// - Throws: MPIError if the operation fails
    public func recv(
        buffer: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        source: Int32,
        tag: Int32,
        status: inout MPIStatus?
    ) throws {
        var mpiStatus = MPI_Status()
        let errorCode = MPI_Recv(buffer, count, datatype.handle, source, tag, handle, &mpiStatus)
        try checkMPIError(errorCode)

        if status != nil {
            status = MPIStatus(status: mpiStatus)
        }
    }

    /// Type-safe blocking receive
    /// - Parameters:
    ///   - count: Number of elements to receive
    ///   - source: Rank of the source process
    ///   - tag: Message tag (default: 0)
    /// - Returns: Array of received data
    /// - Throws: MPIError if the operation fails
    public func recv<T: MPIDataRepresentable>(
        count: Int,
        from source: Int32,
        tag: Int32 = 0
    ) throws -> [T] {
        var data = [T](repeating: T.self == Double.self ? 0.0 as! T : T.self as! T, count: count)
        var status: MPIStatus? = MPIStatus()

        try data.withUnsafeMutableBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                throw MPIError.invalidBuffer
            }
            try recv(
                buffer: baseAddress,
                count: Int32(count),
                datatype: T.mpiDatatype,
                source: source,
                tag: tag,
                status: &status
            )
        }

        return data
    }
}

// MARK: - Non-Blocking Point-to-Point

extension MPICommunicator {
    /// Non-blocking send operation
    /// - Parameters:
    ///   - buffer: Pointer to the data to send
    ///   - count: Number of elements to send
    ///   - datatype: MPI datatype of the elements
    ///   - dest: Rank of the destination process
    ///   - tag: Message tag
    ///   - request: Request object for tracking completion
    /// - Throws: MPIError if the operation fails
    public func isend(
        buffer: UnsafeRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        dest: Int32,
        tag: Int32,
        request: inout MPIRequest
    ) throws {
        var tempRequest: MPI_Request? = request.handle
        let errorCode = MPI_Isend(buffer, count, datatype.handle, dest, tag, handle, &tempRequest)
        try checkMPIError(errorCode)
        if let unwrapped = tempRequest {
            request.handle = unwrapped
        }
    }

    /// Non-blocking receive operation
    /// - Parameters:
    ///   - buffer: Pointer to the receive buffer
    ///   - count: Maximum number of elements to receive
    ///   - datatype: MPI datatype of the elements
    ///   - source: Rank of the source process
    ///   - tag: Message tag
    ///   - request: Request object for tracking completion
    /// - Throws: MPIError if the operation fails
    public func irecv(
        buffer: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        source: Int32,
        tag: Int32,
        request: inout MPIRequest
    ) throws {
        var tempRequest: MPI_Request? = request.handle
        let errorCode = MPI_Irecv(buffer, count, datatype.handle, source, tag, handle, &tempRequest)
        try checkMPIError(errorCode)
        if let unwrapped = tempRequest {
            request.handle = unwrapped
        }
    }
}

// MARK: - Probe Operations

extension MPICommunicator {
    /// Probe for incoming messages
    /// - Parameters:
    ///   - source: Rank of the source process
    ///   - tag: Message tag
    /// - Returns: Status of the probed message
    /// - Throws: MPIError if the operation fails
    public func probe(source: Int32, tag: Int32) throws -> MPIStatus {
        var status = MPI_Status()
        let errorCode = MPI_Probe(source, tag, handle, &status)
        try checkMPIError(errorCode)
        return MPIStatus(status: status)
    }

    /// Non-blocking probe for incoming messages
    /// - Parameters:
    ///   - source: Rank of the source process
    ///   - tag: Message tag
    /// - Returns: Tuple of (message available, status if available)
    /// - Throws: MPIError if the operation fails
    public func iprobe(source: Int32, tag: Int32) throws -> (Bool, MPIStatus?) {
        var flag: Int32 = 0
        var status = MPI_Status()
        let errorCode = MPI_Iprobe(source, tag, handle, &flag, &status)
        try checkMPIError(errorCode)

        if flag != 0 {
            return (true, MPIStatus(status: status))
        } else {
            return (false, nil)
        }
    }
}

// MARK: - Synchronous and Ready Send

extension MPICommunicator {
    /// Synchronous blocking send
    /// Completes only when the receiver has started receiving the message
    /// - Parameters:
    ///   - buffer: Pointer to the data to send
    ///   - count: Number of elements to send
    ///   - datatype: MPI datatype of the elements
    ///   - dest: Rank of the destination process
    ///   - tag: Message tag
    /// - Throws: MPIError if the operation fails
    public func ssend(
        buffer: UnsafeRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        dest: Int32,
        tag: Int32
    ) throws {
        let errorCode = MPI_Ssend(buffer, count, datatype.handle, dest, tag, handle)
        try checkMPIError(errorCode)
    }

    /// Ready send - assumes receiver is already posted
    /// - Parameters:
    ///   - buffer: Pointer to the data to send
    ///   - count: Number of elements to send
    ///   - datatype: MPI datatype of the elements
    ///   - dest: Rank of the destination process
    ///   - tag: Message tag
    /// - Throws: MPIError if the operation fails
    public func rsend(
        buffer: UnsafeRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        dest: Int32,
        tag: Int32
    ) throws {
        let errorCode = MPI_Rsend(buffer, count, datatype.handle, dest, tag, handle)
        try checkMPIError(errorCode)
    }
}

// MARK: - Buffered Send

extension MPICommunicator {
    /// Buffered send operation
    /// - Parameters:
    ///   - buffer: Pointer to the data to send
    ///   - count: Number of elements to send
    ///   - datatype: MPI datatype of the elements
    ///   - dest: Rank of the destination process
    ///   - tag: Message tag
    /// - Throws: MPIError if the operation fails
    public func bsend(
        buffer: UnsafeRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        dest: Int32,
        tag: Int32
    ) throws {
        let errorCode = MPI_Bsend(buffer, count, datatype.handle, dest, tag, handle)
        try checkMPIError(errorCode)
    }
}

// MARK: - MPIStatus Updates

extension MPIStatus {
    /// Get the actual count of received elements
    public mutating func getElementCount(datatype: MPIDatatype) -> Int {
        return getCount(datatype: datatype)
    }
}
