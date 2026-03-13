/// MPI Collective Operations Demo
///
/// Demonstrates broadcast, reduce, and allreduce operations
/// Run with:
///   swift build --target CollectivesDemo
///   mpirun -np 4 swift run CollectivesDemo

import CMPIBindings
import MPISwift
import MessageDifferentiationKit
import Foundation

func demonstrateCollectives(comm: MPICommunicator) {
    let rank = comm.rank
    let size = comm.size

    print("Process \(rank) of \(size) starting collective demonstrations")

    // 1. Allreduce Demo
    print("\n=== Allreduce Demo ===")
    let contribution: [Double] = [Double(rank * 10)]
    print("Rank \(rank) contributing: \(contribution)")

    do {
        let allreduceResult = try comm.allreduce(contribution, operation: .sum)
        print("Rank \(rank) received global sum: \(allreduceResult)")
        // Expected: sum of (0*10 + 1*10 + 2*10 + 3*10) = [60.0] for 4 processes
    } catch {
        print("Rank \(rank) allreduce error: \(error)")
    }

    // 2. Broadcast Demo
    print("\n=== Broadcast Demo ===")
    var broadcastData: [Double] = rank == 0 ? [1.0, 2.0, 3.0, 4.0] : [0.0, 0.0, 0.0, 0.0]
    print("Rank \(rank) before broadcast: \(broadcastData)")

    do {
        try comm.broadcast(&broadcastData, root: 0)
        print("Rank \(rank) after broadcast: \(broadcastData)")
    } catch {
        print("Rank \(rank) broadcast error: \(error)")
    }

    // 3. Reduce Demo
    print("\n=== Reduce Demo ===")
    let localData: [Double] = [Double(rank + 1), Double(rank + 2)]
    print("Rank \(rank) contributing: \(localData)")

    do {
        if let result = try comm.reduce(localData, operation: .sum, root: 0) {
            print("Rank \(rank) (root) received sum: \(result)")
        } else {
            print("Rank \(rank) (non-root) sent data")
        }
    } catch {
        print("Rank \(rank) reduce error: \(error)")
    }

    print("\nRank \(rank) finished collective demonstrations")
}

@main
struct CollectivesDemo {
    static func main() {
        MPI_Init(nil, nil)

        let world = MPICommunicator.world
        demonstrateCollectives(comm: world)

        MPI_Finalize()
    }
}
