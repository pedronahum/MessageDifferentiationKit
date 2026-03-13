import MPISwift
import _Differentiation

/// Differentiable MPI Point-to-Point Operations
///
/// This module provides automatic differentiation support for MPI point-to-point
/// communication operations (send and recv).
///
/// ## Gradient Routing for Point-to-Point
///
/// Point-to-point operations create a dependency between sender and receiver:
/// - **Send-Recv pair**: Forms a communication channel
/// - **Forward**: Data flows from sender to receiver
/// - **Backward**: Gradient flows from receiver back to sender
///
/// This requires careful coordination to ensure gradients flow in reverse direction.

// MARK: - Differentiable Send-Recv Pair

/// Differentiable send-recv pair operation
///
/// This represents a matched send-recv pair where one process sends a value
/// and another receives it. The gradient automatically flows in reverse.
///
/// **Important**: This function must be called on both sender and receiver
/// with matching parameters to maintain gradient flow.
///
/// - Parameters:
///   - value: Value to send (on sender) or initial value (on receiver)
///   - source: Rank of the sending process
///   - dest: Rank of the receiving process
///   - tag: Message tag for matching (default: 0)
///   - communicator: MPI communicator
/// - Returns: Sent value (on sender) or received value (on receiver)
@differentiable(reverse)
public func differentiableSendRecv(
    _ value: Double,
    source: Int32,
    dest: Int32,
    tag: Int32 = 0,
    on communicator: MPICommunicator
) -> Double {
    let rank = communicator.rank

    do {
        if rank == source {
            // Send value to destination
            try communicator.send([value], to: dest, tag: tag)
            return value  // Sender returns its own value
        } else if rank == dest {
            // Receive value from source
            let received = try communicator.recv(count: 1, from: source, tag: tag) as [Double]
            return received[0]
        } else {
            // Not involved in this communication
            return value
        }
    } catch {
        return value
    }
}

@derivative(of: differentiableSendRecv)
@usableFromInline
func _vjpSendRecv(
    _ value: Double,
    source: Int32,
    dest: Int32,
    tag: Int32,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableSendRecv(value, source: source, dest: dest, tag: tag, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // In backward pass, gradient flows in reverse direction:
        // - Receiver sends gradient back to sender
        // - Sender receives gradient from receiver
        let rank = communicator.rank

        do {
            if rank == dest {
                // Receiver sends gradient back to sender
                // Use a different tag to avoid collision with forward pass
                let backwardTag = tag + 10000
                try communicator.send([gradient], to: source, tag: backwardTag)
                return 0.0  // Receiver doesn't get gradient for input value
            } else if rank == source {
                // Sender receives gradient from receiver
                let backwardTag = tag + 10000
                let gradReceived = try communicator.recv(count: 1, from: dest, tag: backwardTag) as [Double]
                return gradReceived[0]
            } else {
                // Not involved in this communication
                return 0.0
            }
        } catch {
            return gradient
        }
    }

    return (output, pullback)
}

// MARK: - Future: Scatter-like Pattern
//
// Note: Implementing differentiable scatter/gather requires careful handling
// of array gradients with Swift's DifferentiableView. This is planned for
// future work when tensor operations are added.

// MARK: - Documentation

/**
 # Differentiable Point-to-Point Communication

 Point-to-point operations create directed communication channels between processes.
 Automatic differentiation requires reversing these channels in the backward pass.

 ## Usage Example

 ```swift
 import MessageDifferentiationKit

 let comm = MPI5.Communicator.world
 let value = Double(comm.rank)

 // Define a computation with point-to-point communication
 func computation(_ x: Double) -> Double {
     // Process 0 sends to process 1
     return differentiableSendRecv(
         x,
         source: 0,
         dest: 1,
         on: comm
     )
 }

 // Compute gradients
 if comm.rank == 0 {
     let grad = gradient(at: 5.0, of: computation)
     print("Sender gradient: \\(grad)")
 } else if comm.rank == 1 {
     let grad = gradient(at: 0.0, of: computation)
     print("Receiver gradient: \\(grad)")
 }
 ```

 ## Gradient Flow

 ### Send-Recv Pair
 - **Forward**: `sender → receiver` (data flows)
 - **Backward**: `receiver → sender` (gradient flows)
 - **Key**: Gradient goes to sender, not receiver's input

 ## Important Notes

 1. **Matching Calls**: All processes involved must call differentiable operations
    with matching parameters (source, dest, tag)

 2. **Tag Management**: Backward pass uses different tags to avoid collisions
    with forward pass messages

 3. **Deadlock Prevention**: Ensure proper ordering of send/recv to avoid deadlocks
    in both forward and backward passes

 4. **Zero Gradients**: Processes not involved in communication receive zero gradients

 ## Performance Considerations

 - Each differentiable send-recv adds one reverse send-recv in backward pass
 - Careful tag management prevents message conflicts
 - Consider using non-blocking operations for better overlap (future work)

 ## Limitations

 - Currently supports only blocking send-recv operations
 - Arrays require element-wise handling (future: vectorized operations)
 - No support for wildcard receives (MPI_ANY_SOURCE) due to gradient ambiguity

 ## Future Enhancements

 - [ ] Non-blocking isend/irecv with differentiability
 - [ ] Persistent send-recv with gradients
 - [ ] Neighbor collectives for topology-aware gradients
 - [ ] Automatic deadlock detection in gradient flow
 */
