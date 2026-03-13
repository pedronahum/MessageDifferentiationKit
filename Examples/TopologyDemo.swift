/// MPI Topology Communication Demo
///
/// Demonstrates basic topology concepts
/// Run with:
///   swift build --target TopologyDemo
///   mpirun -np 4 swift run TopologyDemo

import CMPIBindings
import MPISwift
import MessageDifferentiationKit
import Foundation

func demonstrateTopology(comm: MPICommunicator) {
    let rank = comm.rank
    let size = comm.size

    guard size >= 4 else {
        print("This demo requires at least 4 processes")
        return
    }

    print("Process \(rank) of \(size) demonstrating topology concepts")

    // Simple ring topology simulation
    print("\n=== Ring Topology Simulation ===")
    let next = (rank + 1) % size
    let prev = (rank - 1 + size) % size

    print("Rank \(rank): prev=\(prev), next=\(next)")
    print("  In a ring, rank \(rank) would:")
    print("  - Receive data from rank \(prev)")
    print("  - Process or transform the data")
    print("  - Send data to rank \(next)")

    // 2D grid topology
    print("\n=== 2D Grid Topology (2x2) ===")
    let rows = 2
    let cols = 2
    let row = Int(rank) / cols
    let col = Int(rank) % cols

    print("Rank \(rank): position=(\(row), \(col))")
    print("  In a 2D grid, rank \(rank) would communicate with:")
    if row > 0 { print("  - North: rank \((row-1)*cols + col)") }
    if row < rows-1 { print("  - South: rank \((row+1)*cols + col)") }
    if col > 0 { print("  - West: rank \(row*cols + (col-1))") }
    if col < cols-1 { print("  - East: rank \(row*cols + (col+1))") }

    print("\nRank \(rank) finished topology demonstration")
}

@main
struct TopologyDemo {
    static func main() {
        MPI_Init(nil, nil)

        let world = MPICommunicator.world
        demonstrateTopology(comm: world)

        MPI_Finalize()
    }
}
