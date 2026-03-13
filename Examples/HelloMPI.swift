/// Hello World with MPI
///
/// This example demonstrates basic MPI usage in Swift.
/// Compile and run with:
///   swift build --target HelloMPI
///   mpirun -np 4 swift run HelloMPI

import CMPIBindings
import MPISwift
import MessageDifferentiationKit
import Foundation

func runExample() {
    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Hello from process \(rank) of \(size)")
    print("Process \(rank) running on host: \(ProcessInfo.processInfo.hostName)")
    print("Process \(rank) finished successfully")
}

@main
struct HelloMPI {
    static func main() {
        MPI_Init(nil, nil)
        runExample()
        MPI_Finalize()
    }
}
