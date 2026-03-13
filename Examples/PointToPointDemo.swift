/// MPI Point-to-Point Communication Demo
///
/// Demonstrates send/recv operations
/// Run with:
///   swift build --target PointToPointDemo
///   mpirun -np 2 swift run PointToPointDemo

import CMPIBindings
import MPISwift
import MessageDifferentiationKit
import Foundation

func demonstratePointToPoint(comm: MPICommunicator) {
    let rank = comm.rank
    let size = comm.size

    guard size >= 2 else {
        print("This demo requires at least 2 processes")
        return
    }

    print("Process \(rank) of \(size) starting point-to-point demonstrations")

    // Simple Send/Recv
    print("\n=== Blocking Send/Recv Demo ===")
    if rank == 0 {
        let sendData: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0]
        print("Rank 0 sending: \(sendData)")
        do {
            try comm.send(sendData, to: 1, tag: 0)
            print("Rank 0 sent data")
        } catch {
            print("Rank 0 send error: \(error)")
        }
    } else if rank == 1 {
        print("Rank 1 receiving...")
        do {
            let recvData: [Double] = try comm.recv(count: 5, from: 0, tag: 0)
            print("Rank 1 received: \(recvData)")
        } catch {
            print("Rank 1 recv error: \(error)")
        }
    }

    print("\nRank \(rank) finished point-to-point demonstrations")
}

@main
struct PointToPointDemo {
    static func main() {
        MPI_Init(nil, nil)

        let world = MPICommunicator.world
        demonstratePointToPoint(comm: world)

        MPI_Finalize()
    }
}
