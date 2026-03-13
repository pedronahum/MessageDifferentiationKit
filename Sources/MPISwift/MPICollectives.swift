import CMPIBindings
import Foundation

/// Collective Communication Operations
///
/// This module provides collective operations that involve all processes
/// in a communicator: broadcast, reduce, gather, scatter, and all-to-all.

// MARK: - Broadcast

extension MPICommunicator {
    /// Broadcast data from root to all processes
    /// - Parameters:
    ///   - buffer: Buffer containing data (on root) or receiving data (on others)
    ///   - count: Number of elements
    ///   - datatype: MPI datatype
    ///   - root: Rank of the root process
    /// - Throws: MPIError if the operation fails
    public func bcast(
        buffer: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        root: Int32
    ) throws {
        let errorCode = MPI_Bcast(buffer, count, datatype.handle, root, handle)
        try checkMPIError(errorCode)
    }

    /// Type-safe broadcast
    /// - Parameters:
    ///   - data: Data to broadcast (on root) or array to receive into (on others)
    ///   - root: Rank of the root process
    /// - Returns: Broadcasted data
    /// - Throws: MPIError if the operation fails
    public func broadcast<T: MPIDataRepresentable>(
        _ data: inout [T],
        root: Int32
    ) throws {
        try data.withUnsafeMutableBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                throw MPIError.invalidBuffer
            }
            try bcast(
                buffer: baseAddress,
                count: Int32(buffer.count),
                datatype: T.mpiDatatype,
                root: root
            )
        }
    }
}

// MARK: - Reduce

extension MPICommunicator {
    /// Reduce data from all processes to root
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - recvbuf: Receive buffer (significant only at root)
    ///   - count: Number of elements
    ///   - datatype: MPI datatype
    ///   - op: Reduction operation
    ///   - root: Rank of the root process
    /// - Throws: MPIError if the operation fails
    public func reduce(
        sendbuf: UnsafeRawPointer,
        recvbuf: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        op: MPIOperation,
        root: Int32
    ) throws {
        let errorCode = MPI_Reduce(sendbuf, recvbuf, count, datatype.handle, op.handle, root, handle)
        try checkMPIError(errorCode)
    }

    /// Type-safe reduce operation
    /// - Parameters:
    ///   - data: Data to reduce
    ///   - op: Reduction operation
    ///   - root: Rank of the root process
    /// - Returns: Reduced result (on root) or nil (on other processes)
    /// - Throws: MPIError if the operation fails
    public func reduce<T: MPIDataRepresentable>(
        _ data: [T],
        operation op: MPIOperation,
        root: Int32
    ) throws -> [T]? {
        var result: [T]? = nil
        let isRoot = (rank == root)

        if isRoot {
            result = [T](repeating: data[0], count: data.count)
        }

        try data.withUnsafeBufferPointer { sendBuffer in
            guard let sendBase = sendBuffer.baseAddress else {
                throw MPIError.invalidBuffer
            }

            if isRoot {
                try result!.withUnsafeMutableBufferPointer { recvBuffer in
                    guard let recvBase = recvBuffer.baseAddress else {
                        throw MPIError.invalidBuffer
                    }
                    try reduce(
                        sendbuf: sendBase,
                        recvbuf: recvBase,
                        count: Int32(sendBuffer.count),
                        datatype: T.mpiDatatype,
                        op: op,
                        root: root
                    )
                }
            } else {
                // Non-root processes use a dummy receive buffer
                var dummy: UInt8 = 0
                try reduce(
                    sendbuf: sendBase,
                    recvbuf: &dummy,
                    count: Int32(sendBuffer.count),
                    datatype: T.mpiDatatype,
                    op: op,
                    root: root
                )
            }
        }

        return result
    }
}

// MARK: - Allreduce

extension MPICommunicator {
    /// All-reduce: reduce and broadcast result to all processes
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - recvbuf: Receive buffer
    ///   - count: Number of elements
    ///   - datatype: MPI datatype
    ///   - op: Reduction operation
    /// - Throws: MPIError if the operation fails
    public func allreduce(
        sendbuf: UnsafeRawPointer,
        recvbuf: UnsafeMutableRawPointer,
        count: Int32,
        datatype: MPIDatatype,
        op: MPIOperation
    ) throws {
        let errorCode = MPI_Allreduce(sendbuf, recvbuf, count, datatype.handle, op.handle, handle)
        try checkMPIError(errorCode)
    }

    /// Type-safe allreduce operation
    /// - Parameters:
    ///   - data: Data to reduce
    ///   - op: Reduction operation
    /// - Returns: Reduced result available on all processes
    /// - Throws: MPIError if the operation fails
    public func allreduce<T: MPIDataRepresentable>(
        _ data: [T],
        operation op: MPIOperation
    ) throws -> [T] {
        var result = [T](repeating: data[0], count: data.count)

        try data.withUnsafeBufferPointer { sendBuffer in
            guard let sendBase = sendBuffer.baseAddress else {
                throw MPIError.invalidBuffer
            }

            try result.withUnsafeMutableBufferPointer { recvBuffer in
                guard let recvBase = recvBuffer.baseAddress else {
                    throw MPIError.invalidBuffer
                }

                try allreduce(
                    sendbuf: sendBase,
                    recvbuf: recvBase,
                    count: Int32(sendBuffer.count),
                    datatype: T.mpiDatatype,
                    op: op
                )
            }
        }

        return result
    }
}

// MARK: - Gather

extension MPICommunicator {
    /// Gather data from all processes to root
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send
    ///   - sendtype: Datatype of send elements
    ///   - recvbuf: Receive buffer (significant only at root)
    ///   - recvcount: Number of elements to receive from each process
    ///   - recvtype: Datatype of receive elements
    ///   - root: Rank of the root process
    /// - Throws: MPIError if the operation fails
    public func gather(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype,
        root: Int32
    ) throws {
        let errorCode = MPI_Gather(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            root, handle
        )
        try checkMPIError(errorCode)
    }

    /// Type-safe gather operation
    /// - Parameters:
    ///   - data: Data to send from this process
    ///   - root: Rank of the root process
    /// - Returns: Gathered data (on root) or nil (on other processes)
    /// - Throws: MPIError if the operation fails
    public func gather<T: MPIDataRepresentable>(
        _ data: [T],
        root: Int32
    ) throws -> [T]? {
        let isRoot = (rank == root)
        var result: [T]? = nil

        if isRoot {
            result = [T](repeating: data[0], count: data.count * Int(size))
        }

        try data.withUnsafeBufferPointer { sendBuffer in
            guard let sendBase = sendBuffer.baseAddress else {
                throw MPIError.invalidBuffer
            }

            if isRoot {
                try result!.withUnsafeMutableBufferPointer { recvBuffer in
                    guard let recvBase = recvBuffer.baseAddress else {
                        throw MPIError.invalidBuffer
                    }
                    try gather(
                        sendbuf: sendBase,
                        sendcount: Int32(sendBuffer.count),
                        sendtype: T.mpiDatatype,
                        recvbuf: recvBase,
                        recvcount: Int32(sendBuffer.count),
                        recvtype: T.mpiDatatype,
                        root: root
                    )
                }
            } else {
                var dummy: UInt8 = 0
                try gather(
                    sendbuf: sendBase,
                    sendcount: Int32(sendBuffer.count),
                    sendtype: T.mpiDatatype,
                    recvbuf: &dummy,
                    recvcount: Int32(sendBuffer.count),
                    recvtype: T.mpiDatatype,
                    root: root
                )
            }
        }

        return result
    }
}

// MARK: - Scatter

extension MPICommunicator {
    /// Scatter data from root to all processes
    /// - Parameters:
    ///   - sendbuf: Send buffer (significant only at root)
    ///   - sendcount: Number of elements to send to each process
    ///   - sendtype: Datatype of send elements
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive
    ///   - recvtype: Datatype of receive elements
    ///   - root: Rank of the root process
    /// - Throws: MPIError if the operation fails
    public func scatter(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype,
        root: Int32
    ) throws {
        let errorCode = MPI_Scatter(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            root, handle
        )
        try checkMPIError(errorCode)
    }
}

// MARK: - Allgather

extension MPICommunicator {
    /// All-gather: gather data from all processes to all processes
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send
    ///   - sendtype: Datatype of send elements
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive from each process
    ///   - recvtype: Datatype of receive elements
    /// - Throws: MPIError if the operation fails
    public func allgather(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype
    ) throws {
        let errorCode = MPI_Allgather(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            handle
        )
        try checkMPIError(errorCode)
    }

    /// Type-safe allgather operation
    /// - Parameter data: Data to send from this process
    /// - Returns: Gathered data from all processes
    /// - Throws: MPIError if the operation fails
    public func allgather<T: MPIDataRepresentable>(_ data: [T]) throws -> [T] {
        var result = [T](repeating: data[0], count: data.count * Int(size))

        try data.withUnsafeBufferPointer { sendBuffer in
            guard let sendBase = sendBuffer.baseAddress else {
                throw MPIError.invalidBuffer
            }

            try result.withUnsafeMutableBufferPointer { recvBuffer in
                guard let recvBase = recvBuffer.baseAddress else {
                    throw MPIError.invalidBuffer
                }

                try allgather(
                    sendbuf: sendBase,
                    sendcount: Int32(sendBuffer.count),
                    sendtype: T.mpiDatatype,
                    recvbuf: recvBase,
                    recvcount: Int32(sendBuffer.count),
                    recvtype: T.mpiDatatype
                )
            }
        }

        return result
    }
}

// MARK: - Barrier

extension MPICommunicator {
    /// Barrier synchronization
    /// Blocks until all processes in the communicator have called it
    /// - Throws: MPIError if the operation fails
    public func barrier() throws {
        let errorCode = MPI_Barrier(handle)
        try checkMPIError(errorCode)
    }
}
