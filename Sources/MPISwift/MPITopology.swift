import CMPIBindings
import Foundation

/// MPI Topology and Neighborhood Communication
///
/// This module provides support for topology-aware communication patterns,
/// including Cartesian grids, graph topologies, and neighborhood collectives.

// MARK: - Cartesian Topology

/// Cartesian grid topology
public final class MPICartesianTopology {
    public let comm: MPICommunicator
    public let dimensions: [Int32]
    public let periods: [Bool]
    public let reorder: Bool

    /// Create a Cartesian topology communicator
    /// - Parameters:
    ///   - comm: Input communicator
    ///   - dims: Array of dimension sizes
    ///   - periods: Periodicity flags for each dimension
    ///   - reorder: Allow rank reordering for optimization
    /// - Throws: MPIError if creation fails
    public init(
        comm: MPICommunicator,
        dims: [Int32],
        periods: [Bool],
        reorder: Bool = true
    ) throws {
        guard dims.count == periods.count else {
            throw MPIError.invalidArgument
        }

        var newComm = MPI_COMM_NULL_get()
        var dimsArray = dims
        var periodsArray = periods.map { $0 ? Int32(1) : Int32(0) }

        let errorCode = MPI_Cart_create(
            comm.handle,
            Int32(dims.count),
            &dimsArray,
            &periodsArray,
            reorder ? 1 : 0,
            &newComm
        )
        try checkMPIError(errorCode)

        guard let unwrappedComm = newComm else {
            throw MPIError.invalidCommunicator
        }

        self.comm = MPICommunicator(handle: unwrappedComm)
        self.dimensions = dims
        self.periods = periods
        self.reorder = reorder
    }

    /// Get Cartesian coordinates for a rank
    /// - Parameter rank: Process rank
    /// - Returns: Coordinates in the Cartesian grid
    /// - Throws: MPIError if the operation fails
    public func getCoordinates(rank: Int32) throws -> [Int32] {
        var coords = [Int32](repeating: 0, count: dimensions.count)
        let errorCode = MPI_Cart_coords(
            comm.handle,
            rank,
            Int32(dimensions.count),
            &coords
        )
        try checkMPIError(errorCode)
        return coords
    }

    /// Get rank for given Cartesian coordinates
    /// - Parameter coords: Coordinates in the grid
    /// - Returns: Process rank
    /// - Throws: MPIError if the operation fails
    public func getRank(coords: [Int32]) throws -> Int32 {
        guard coords.count == dimensions.count else {
            throw MPIError.invalidArgument
        }

        var rank: Int32 = 0
        var coordsArray = coords
        let errorCode = MPI_Cart_rank(
            comm.handle,
            &coordsArray,
            &rank
        )
        try checkMPIError(errorCode)
        return rank
    }

    /// Get ranks of neighboring processes along a dimension
    /// - Parameter dimension: Dimension index
    /// - Returns: Tuple of (predecessor rank, successor rank)
    /// - Throws: MPIError if the operation fails
    public func getNeighbors(dimension: Int32) throws -> (predecessor: Int32, successor: Int32) {
        var pred: Int32 = 0
        var succ: Int32 = 0

        let errorCode = MPI_Cart_shift(
            comm.handle,
            dimension,
            1, // displacement
            &pred,
            &succ
        )
        try checkMPIError(errorCode)
        return (pred, succ)
    }

    /// Get all neighbors in the Cartesian grid
    /// - Returns: Array of neighbor ranks for each dimension (predecessor, successor)
    /// - Throws: MPIError if the operation fails
    public func getAllNeighbors() throws -> [(predecessor: Int32, successor: Int32)] {
        var neighbors: [(Int32, Int32)] = []
        for dim in 0..<Int32(dimensions.count) {
            let (pred, succ) = try getNeighbors(dimension: dim)
            neighbors.append((pred, succ))
        }
        return neighbors
    }
}

// MARK: - Graph Topology

/// Graph topology
public final class MPIGraphTopology {
    public let comm: MPICommunicator
    public let index: [Int32]
    public let edges: [Int32]

    /// Create a graph topology communicator
    /// - Parameters:
    ///   - comm: Input communicator
    ///   - index: Index array (cumulative neighbor counts)
    ///   - edges: Edge array (neighbor ranks)
    ///   - reorder: Allow rank reordering
    /// - Throws: MPIError if creation fails
    public init(
        comm: MPICommunicator,
        index: [Int32],
        edges: [Int32],
        reorder: Bool = true
    ) throws {
        var newComm = MPI_COMM_NULL_get()
        var indexArray = index
        var edgesArray = edges

        let errorCode = MPI_Graph_create(
            comm.handle,
            Int32(index.count),
            &indexArray,
            &edgesArray,
            reorder ? 1 : 0,
            &newComm
        )
        try checkMPIError(errorCode)

        guard let unwrappedComm = newComm else {
            throw MPIError.invalidCommunicator
        }

        self.comm = MPICommunicator(handle: unwrappedComm)
        self.index = index
        self.edges = edges
    }

    /// Get neighbors of a given rank
    /// - Parameter rank: Process rank
    /// - Returns: Array of neighbor ranks
    /// - Throws: MPIError if the operation fails
    public func getNeighbors(rank: Int32) throws -> [Int32] {
        var numNeighbors: Int32 = 0

        // Get number of neighbors
        var errorCode = MPI_Graph_neighbors_count(
            comm.handle,
            rank,
            &numNeighbors
        )
        try checkMPIError(errorCode)

        // Get the neighbors
        var neighbors = [Int32](repeating: 0, count: Int(numNeighbors))
        errorCode = MPI_Graph_neighbors(
            comm.handle,
            rank,
            numNeighbors,
            &neighbors
        )
        try checkMPIError(errorCode)

        return neighbors
    }
}

// MARK: - Neighborhood Collectives

extension MPICommunicator {
    /// Neighbor allgather - gather from all neighbors
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send
    ///   - sendtype: Send datatype
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive from each neighbor
    ///   - recvtype: Receive datatype
    /// - Throws: MPIError if the operation fails
    public func neighborAllgather(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype
    ) throws {
        let errorCode = MPI_Neighbor_allgather(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            handle
        )
        try checkMPIError(errorCode)
    }

    /// Neighbor alltoall - all-to-all among neighbors
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send to each neighbor
    ///   - sendtype: Send datatype
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive from each neighbor
    ///   - recvtype: Receive datatype
    /// - Throws: MPIError if the operation fails
    public func neighborAlltoall(
        sendbuf: UnsafeRawPointer,
        sendcount: Int32,
        sendtype: MPIDatatype,
        recvbuf: UnsafeMutableRawPointer,
        recvcount: Int32,
        recvtype: MPIDatatype
    ) throws {
        let errorCode = MPI_Neighbor_alltoall(
            sendbuf, sendcount, sendtype.handle,
            recvbuf, recvcount, recvtype.handle,
            handle
        )
        try checkMPIError(errorCode)
    }
}

// MARK: - Persistent Neighborhood Collectives (MPI 5.0)

#if MPI5_PERSISTENT_COLLECTIVES_AVAILABLE

extension MPICommunicator {
    /// Initialize persistent neighbor allgather
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send
    ///   - sendtype: Send datatype
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive from each neighbor
    ///   - recvtype: Receive datatype
    ///   - info: MPI Info for hints
    /// - Returns: Persistent request
    /// - Throws: MPIError if initialization fails
    public func neighborAllgatherInit(
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

        let errorCode = MPI_Neighbor_allgather_init(
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

    /// Initialize persistent neighbor alltoall
    /// - Parameters:
    ///   - sendbuf: Send buffer
    ///   - sendcount: Number of elements to send to each neighbor
    ///   - sendtype: Send datatype
    ///   - recvbuf: Receive buffer
    ///   - recvcount: Number of elements to receive from each neighbor
    ///   - recvtype: Receive datatype
    ///   - info: MPI Info for hints
    /// - Returns: Persistent request
    /// - Throws: MPIError if initialization fails
    public func neighborAlltoallInit(
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

        let errorCode = MPI_Neighbor_alltoall_init(
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

// MARK: - Topology Helpers

/// Helper class for common topology patterns
public struct MPITopologyHelpers {
    /// Create a 2D Cartesian grid
    /// - Parameters:
    ///   - comm: Input communicator
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    ///   - periodicX: Periodic in X dimension
    ///   - periodicY: Periodic in Y dimension
    /// - Returns: Cartesian topology
    /// - Throws: MPIError if creation fails
    public static func create2DGrid(
        comm: MPICommunicator,
        rows: Int32,
        cols: Int32,
        periodicX: Bool = false,
        periodicY: Bool = false
    ) throws -> MPICartesianTopology {
        return try MPICartesianTopology(
            comm: comm,
            dims: [rows, cols],
            periods: [periodicX, periodicY]
        )
    }

    /// Create a 3D Cartesian grid
    /// - Parameters:
    ///   - comm: Input communicator
    ///   - x: Size in X dimension
    ///   - y: Size in Y dimension
    ///   - z: Size in Z dimension
    ///   - periodicX: Periodic in X
    ///   - periodicY: Periodic in Y
    ///   - periodicZ: Periodic in Z
    /// - Returns: Cartesian topology
    /// - Throws: MPIError if creation fails
    public static func create3DGrid(
        comm: MPICommunicator,
        x: Int32,
        y: Int32,
        z: Int32,
        periodicX: Bool = false,
        periodicY: Bool = false,
        periodicZ: Bool = false
    ) throws -> MPICartesianTopology {
        return try MPICartesianTopology(
            comm: comm,
            dims: [x, y, z],
            periods: [periodicX, periodicY, periodicZ]
        )
    }

    /// Create a ring topology (1D periodic grid)
    /// - Parameters:
    ///   - comm: Input communicator
    ///   - size: Number of processes in ring
    /// - Returns: Cartesian topology representing a ring
    /// - Throws: MPIError if creation fails
    public static func createRing(
        comm: MPICommunicator,
        size: Int32
    ) throws -> MPICartesianTopology {
        return try MPICartesianTopology(
            comm: comm,
            dims: [size],
            periods: [true]
        )
    }
}

// MARK: - Documentation

/*
 Topology Communication Patterns:

 1. Cartesian Topologies:
    - Ideal for regular grid-based computations (stencil operations, image processing)
    - Supports periodic boundaries for wraparound communication
    - Automatic neighbor discovery

 2. Graph Topologies:
    - Flexible for irregular communication patterns
    - Useful for mesh-based computations, molecular dynamics
    - Custom neighbor relationships

 3. Neighborhood Collectives:
    - Optimized for communication with topological neighbors
    - Better performance than manual point-to-point
    - MPI 5.0 adds persistent versions for repeated patterns

 Example Usage:

 ```swift
 // Create a 2D Cartesian grid (4x4)
 let cart = try MPITopologyHelpers.create2DGrid(
     comm: MPI5.Communicator.world,
     rows: 4,
     cols: 4,
     periodicX: true,
     periodicY: true
 )

 // Get my coordinates
 let myCoords = try cart.getCoordinates(rank: cart.comm.rank)
 print("My coordinates: \(myCoords)")

 // Find neighbors in each dimension
 let neighbors = try cart.getAllNeighbors()
 print("Neighbors: \(neighbors)")

 // Communicate with neighbors
 try cart.comm.neighborAllgather(...)
 ```
 */
