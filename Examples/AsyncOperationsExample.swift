/// Async Differentiable Operations Example
///
/// Demonstrates the use of async-ready differentiable MPI operations.
/// These operations currently use blocking semantics but are designed
/// to support true async when Swift AD gains the necessary capabilities.
///
/// Run with:
///   swift build --target AsyncOperationsExample
///   mpirun -np 4 swift run AsyncOperationsExample

import CMPIBindings
import MPISwift
import MessageDifferentiationKit
import _Differentiation
import Foundation

@main
struct AsyncOperationsExample {
    static func main() {
        MPI_Init(nil, nil)

        let comm = MPICommunicator.world
        let rank = comm.rank
        let size = comm.size

        guard size >= 2 else {
            if rank == 0 {
                print("This example requires at least 2 processes")
                print("Run with: mpirun -np 4 swift run AsyncOperationsExample")
            }
            MPI_Finalize()
            return
        }

        print("Process \(rank) of \(size): Async Operations Example")

        // Example 1: Async Send-Recv
        asyncSendRecvExample(comm: comm)

        // Example 2: Async Collectives
        asyncCollectivesExample(comm: comm)

        MPI_Finalize()
    }
}

/// Example 1: Basic async send-recv pattern
func asyncSendRecvExample(comm: MPICommunicator) {
    let rank = comm.rank

    print("\n=== Example 1: Async Send-Recv ===")

    let localValue = Double(rank + 1) * 10.0

    // Process 0 sends to process 1
    let result = differentiableAsyncSendRecv(
        localValue,
        source: 0,
        dest: 1,
        on: comm
    )

    if rank == 0 {
        print("Process 0: Sent \(localValue), result=\(result)")
        print("  (In backward pass, gradients flow from rank 1 back to rank 0)")
    } else if rank == 1 {
        print("Process 1: Received \(result)")
        print("  (In backward pass, gradients are sent back to rank 0)")
    }
}

/// Example 2: Async collective operations
func asyncCollectivesExample(comm: MPICommunicator) {
    let rank = comm.rank

    print("\n=== Example 2: Async Collectives ===")

    let localValue = Double(rank + 1)

    // Async allreduce
    let sum = differentiableAsyncAllreduce(localValue, on: comm)
    print("Process \(rank): Allreduce - Input=\(localValue), Sum=\(sum)")

    // Async broadcast
    let broadcastValue = differentiableAsyncBroadcast(
        Double(rank * 100),
        root: 0,
        on: comm
    )

    print("Process \(rank): Broadcast - Value=\(broadcastValue)")
}
