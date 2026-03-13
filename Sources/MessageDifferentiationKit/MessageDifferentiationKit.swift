/// MessageDifferentiationKit - Automatic Differentiation for MPI 5.0 Parallel Programs
///
/// This library provides comprehensive Swift bindings for MPI 5.0 with automatic differentiation support,
/// enabling gradient-based optimization in distributed computing environments.
///
/// ## Features
/// - **MPI 5.0 Sessions**: Modern resource isolation and error handling
/// - **Large Count Support**: 64-bit counts for handling large data (>2GB)
/// - **Persistent Collectives**: Pre-initialized collective operations for better performance
/// - **Partitioned Communication**: Split large messages for improved overlap
/// - **Hardware Offload**: GPU-aware communication for accelerated computing
/// - **Automatic Differentiation**: Full AD support for distributed operations
///
/// ## Example Usage
/// ```swift
/// import MessageDifferentiationKit
///
/// // Initialize MPI 5.0 session
/// let session = try MPI5.Session(threadLevel: .multiple)
/// let world = try session.createCommunicator(processSet: .world)
///
/// // Define differentiable distributed computation
/// @differentiable
/// func distributedComputation(_ input: MPIFloat) -> MPIFloat {
///     var result = input
///     result = MPI5.broadcast(result, root: 0)
///     result = MPI5.allreduce(result, operation: .sum)
///     return result
/// }
///
/// // Gradients automatically flow through MPI operations
/// let gradient = gradient(of: distributedComputation)
/// ```

import MPISwift
import _Differentiation

/// Main namespace for MPI 5.0 operations
public enum MPI5 {
    /// Session-based MPI initialization
    public typealias Session = MPISession

    /// Communicator type
    public typealias Communicator = MPICommunicator

    /// Process set type
    public typealias ProcessSet = MPIProcessSet

    /// Info object for performance hints
    public typealias Info = MPIInfo

    /// Assertions for optimization
    public typealias Assertions = MPIAssertions

    /// Datatype for MPI operations
    public typealias Datatype = MPIDatatype

    /// Operation for reductions
    public typealias Operation = MPIOperation

    /// Request for non-blocking operations
    public typealias Request = MPIRequest

    /// MPI error type
    public typealias Error = MPIError
}

/// Float type for MPI operations (alias for clarity)
public typealias MPIFloat = Double
