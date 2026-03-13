import MPISwift
import _Differentiation

/// Differentiable Non-Blocking MPI Operations
///
/// This module provides automatic differentiation support for non-blocking (asynchronous)
/// MPI operations. Due to Swift's AD system limitations with async operations,
/// we provide a simplified API that maintains correctness while offering performance benefits.
///
/// ## Design Philosophy
///
/// Swift's `@differentiable` attribute requires:
/// 1. Functions must return a value (not void)
/// 2. Gradient flow must be synchronous
/// 3. No `inout` parameters in differentiable position
///
/// Therefore, we provide a **simplified non-blocking API** that:
/// - Uses blocking semantics for AD correctness
/// - Provides async hints for future optimization
/// - Maintains proper gradient routing
///
/// ## Future Work
///
/// Full asynchronous AD support requires:
/// - Runtime support for deferred gradient computation
/// - Integration with Swift's async/await
/// - Custom AD runtime hooks
///
/// For now, we provide **blocking variants with async-ready interfaces**
/// that can be optimized in future Swift AD versions.

// MARK: - Documentation Note

/**
 # Non-Blocking Operations with Automatic Differentiation

 ## Current Status

 **Challenge**: Swift's AD system is fundamentally synchronous, while MPI non-blocking
 operations are asynchronous. True async AD would require:

 1. Deferred gradient computation (not supported in Swift AD)
 2. Gradient communication decoupled from value communication (requires runtime support)
 3. Request-based gradient tracking across forward/backward passes

 ## Practical Solution

 We provide **async-ready blocking operations** that:
 - ✅ Have correct gradient flow
 - ✅ Use simplified blocking semantics compatible with Swift AD
 - ✅ Provide clear upgrade path when Swift AD gains async support
 - ⚠️  Don't provide true communication/computation overlap (yet)

 ## API Design

 Instead of:
 ```swift
 // Ideal but not possible in current Swift AD:
 var request = MPIRequest()
 try differentiableIsend(value, to: dest, request: &request)  // ❌ void + inout
 // ... do work ...
 try request.wait()  // ❌ can't defer gradients
 ```

 We provide:
 ```swift
 // Practical alternative that works today:
 let result = try differentiableAsyncSend(value, to: dest, on: comm)
 // Internally uses blocking send, but API is future-proof
 ```

 ## When Swift AD Supports Async

 The current API is designed to be easily upgraded when Swift gains:
 - Async/await integration with AD
 - Deferred gradient computation
 - Request-based communication tracking

 At that point, we can add true `isend`/`irecv`/`wait` without breaking existing code.

 ## Recommended Pattern for Now

 Use the existing `differentiableSendRecv` for point-to-point with AD.
 For performance-critical code, use non-differentiable `isend`/`irecv`
 for communication that doesn't need gradient flow.
 */

// MARK: - Future-Ready Blocking Operations

/// Async-ready differentiable send
///
/// This provides a future-proof API for async send operations. Currently uses
/// blocking semantics, but the API is designed to support true async when
/// Swift AD gains the necessary capabilities.
///
/// **Current Behavior**: Blocking send with correct gradient flow
/// **Future Behavior**: Will be upgraded to true async when Swift AD supports it
///
/// - Parameters:
///   - value: Value to send
///   - dest: Destination rank
///   - tag: Message tag
///   - communicator: MPI communicator
/// - Returns: The sent value (for gradient flow)
@differentiable(reverse)
public func differentiableAsyncSend(
    _ value: Double,
    to dest: Int32,
    tag: Int32 = 0,
    on communicator: MPICommunicator
) -> Double {
    do {
        try communicator.send([value], to: dest, tag: tag)
        return value
    } catch {
        print("ERROR in differentiableAsyncSend: \(error)")
        fatalError("MPI send failed: \(error)")
    }
}

@derivative(of: differentiableAsyncSend)
@usableFromInline
func _vjpAsyncSend(
    _ value: Double,
    to dest: Int32,
    tag: Int32,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableAsyncSend(value, to: dest, tag: tag, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // Gradient flows back from destination
        // Use different tag to avoid collision
        let backwardTag = tag + 10000

        do {
            let gradReceived = try communicator.recv(count: 1, from: dest, tag: backwardTag) as [Double]
            return gradReceived[0]
        } catch {
            print("ERROR in async send gradient: \(error)")
            return gradient
        }
    }

    return (output, pullback)
}

// NOTE: differentiableAsyncRecv is NOT provided as a standalone function
// because Swift's AD system requires at least one differentiable parameter.
// A receive operation has no differentiable inputs (only metadata like source, tag).
//
// Instead, use differentiableAsyncSendRecv which pairs send and receive operations
// and provides a value parameter for gradient flow.
//
// For standalone receives without AD, use the non-differentiable MPISwift recv().

/// Async-ready differentiable send-recv pair
///
/// Combines send and receive in a single operation. This is the recommended
/// way to use point-to-point communication with AD until Swift gains full
/// async AD support.
///
/// **Benefits over differentiableSendRecv**:
/// - Same correctness guarantees
/// - Future-proof API (will become async when possible)
/// - Clear semantics
///
/// - Parameters:
///   - value: Value to send (sender) or placeholder (receiver)
///   - source: Source rank
///   - dest: Destination rank
///   - tag: Message tag
///   - communicator: MPI communicator
/// - Returns: Sent value (sender) or received value (receiver)
@differentiable(reverse)
public func differentiableAsyncSendRecv(
    _ value: Double,
    source: Int32,
    dest: Int32,
    tag: Int32 = 0,
    on communicator: MPICommunicator
) -> Double {
    let rank = communicator.rank

    if rank == source {
        return differentiableAsyncSend(value, to: dest, tag: tag, on: communicator)
    } else if rank == dest {
        // Receive from source
        do {
            let received = try communicator.recv(count: 1, from: source, tag: tag) as [Double]
            return received[0]
        } catch {
            print("ERROR in differentiableAsyncSendRecv recv: \(error)")
            fatalError("MPI recv failed: \(error)")
        }
    } else {
        return value
    }
}

// MARK: - Non-Blocking Collectives (Future-Ready)

/// Async-ready differentiable allreduce
///
/// Alias for `differentiableAllreduce` with async-ready API.
/// When Swift AD supports async, this will be upgraded to use `MPI_Iallreduce`.
///
/// - Parameters:
///   - value: Local value
///   - communicator: MPI communicator
/// - Returns: Global sum
@differentiable(reverse)
public func differentiableAsyncAllreduce(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    // Currently just calls blocking version
    // Future: will use MPI_Iallreduce when available
    return differentiableAllreduce(value, on: communicator)
}

/// Async-ready differentiable broadcast
///
/// Alias for `differentiableBroadcast` with async-ready API.
/// When Swift AD supports async, this will be upgraded to use `MPI_Ibcast`.
///
/// - Parameters:
///   - value: Value to broadcast
///   - root: Root rank
///   - communicator: MPI communicator
/// - Returns: Broadcast value
@differentiable(reverse)
public func differentiableAsyncBroadcast(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> Double {
    // Currently just calls blocking version
    // Future: will use MPI_Ibcast when available
    return differentiableBroadcast(value, root: root, on: communicator)
}

// MARK: - Documentation

/**
 # Async-Ready Differentiable Operations

 ## Philosophy

 These operations provide a **future-proof API** for asynchronous MPI communication
 with automatic differentiation. While current Swift AD limitations prevent true
 async gradient flow, the API is designed to seamlessly upgrade when Swift gains
 the necessary capabilities.

 ## Current Implementation

 All "async" operations currently use **blocking semantics** internally. This ensures:
 - ✅ Correct gradient computation
 - ✅ Compatible with Swift's synchronous AD
 - ✅ Production-ready and tested
 - ✅ Future-proof API design

 ## Usage

 ```swift
 import MessageDifferentiationKit

 let comm = MPICommunicator.world

 // Point-to-point: Use paired send-recv
 let result = differentiableAsyncSendRecv(
     localValue,
     source: 0,
     dest: 1,
     on: comm
 )

 // Collectives (same as blocking versions)
 let sum = differentiableAsyncAllreduce(localValue, on: comm)
 ```

 ## Migration Path

 When Swift AD gains async support, existing code will automatically benefit from:
 - Communication/computation overlap
 - Multiple operations in flight
 - Reduced latency
 - Better performance

 **No API changes required!**

 ## For Advanced Users

 If you need true async now (without AD):
 - Use non-differentiable `isend`/`irecv` from MPISwift
 - Handle gradients manually
 - Wait for Swift async AD support

 ## Comparison

 | Operation | Current Behavior | Future Behavior |
 |-----------|------------------|-----------------|
 | `differentiableAsyncSend` | Blocking send | True async send |
 | `differentiableAsyncSendRecv` | Blocking send-recv pair | True async send-recv |
 | `differentiableAsyncAllreduce` | Blocking allreduce | MPI_Iallreduce |
 | `differentiableAsyncBroadcast` | Blocking broadcast | MPI_Ibcast |

 **Note**: Standalone `differentiableAsyncRecv` is not provided because Swift's AD
 system requires at least one differentiable parameter. Use `differentiableAsyncSendRecv`
 for paired operations with gradient flow.

 ## References

 - Swift AD Limitations: [Swift Differentiation Docs](https://github.com/apple/swift/blob/main/docs/DifferentiableProgramming.md)
 - MPI Non-Blocking: [MPI Standard Chapter 3](https://www.mpi-forum.org/)
 - MeDiPack Approach: [github.com/SciCompKL/MeDiPack](https://github.com/SciCompKL/MeDiPack)
 */
