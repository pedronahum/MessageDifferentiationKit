# MessageDifferentiationKit

**MPI (Message Passing Interface) for Swift** with automatic differentiation support for distributed machine learning and scientific computing.

## Overview

MessageDifferentiationKit provides **comprehensive Swift bindings for MPI** with a focus on enabling gradient-based distributed computing. The library includes:

1. **MPISwift** - Complete Swift bindings for MPI (collectives, point-to-point, topology, persistent operations)
2. **Differentiable MPI Operations** - 13 operations with Swift's `@differentiable` attribute
3. **DLPack Integration** - Zero-copy tensor communication for ML frameworks

```
┌──────────────────────────────────────────────────────────┐
│ MPI Operations:         ~50+ operations fully supported  │
│ Differentiable:         13 operations                    │
│ DLPack Integration:     3 operations (zero-copy)         │
│ Examples:               8 executable examples            │
│ Tests:                  86 tests (100% passing)          │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install MPI
brew install open-mpi  # macOS
# or
sudo apt-get install libopenmpi-dev  # Linux
```

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/pedronahum/MessageDifferentiationKit", branch: "main")
]
```

### Using MPI in Swift (MPISwift)

```swift
import MPISwift
import CMPIBindings

// Initialize MPI
MPI_Init(nil, nil)
let comm = MPICommunicator.world

// Any MPI operation - full standard support
let data = [1.0, 2.0, 3.0, 4.0]
let result = try comm.allreduce(data, op: .sum)
print("Sum across all processes: \(result)")

// Non-blocking operations
var request: MPI_Request? = nil
try comm.isend([42.0], to: 1, tag: 0, request: &request)

// Topology operations
let dims = [2, 2]
let periods = [false, false]
let cartComm = try comm.createCart(dims: dims, periods: periods, reorder: false)

MPI_Finalize()
```

### Differentiable Operations (Swift AD)

```swift
import MessageDifferentiationKit
import MPISwift
import _Differentiation

let comm = MPICommunicator.world

// Differentiable allreduce with automatic gradients
let localGradient = 2.5
let globalGradient = differentiableMean(localGradient, on: comm)

// Automatic gradient computation
let grad = gradient(at: localGradient) { g in
    differentiableMean(g, on: comm)
}
```

### DLPack Tensor Operations (Framework Integration)

```swift
import MessageDifferentiationKit
import CDLPack

let comm = MPICommunicator.world

// Create DLPack tensor (zero-copy)
var data = [1.0, 2.0, 3.0, 4.0]
let tensor = createDLPackTensor(&data)

// Zero-copy MPI allreduce
let result = allreduceDLPack(tensor, on: comm)
```

## Which API Should I Use?

| Goal | Import | Example |
|------|--------|---------|
| General MPI communication, HPC, distributed apps | `MPISwift` | Data processing, simulations, parallel algorithms |
| Distributed ML training with automatic gradients | `MessageDifferentiationKit` | Gradient aggregation, distributed loss, optimization |
| ML framework interop with zero-copy tensors | `MessageDifferentiationKit` + `CDLPack` | Multi-GPU training with PyTorch/JAX/TensorFlow |

## Operations Reference

### MPISwift - Complete MPI Bindings

| Category | Operations | Status |
|----------|------------|--------|
| **Collective** | `allreduce`, `broadcast`, `reduce`, `gather`, `scatter`, `allgather`, `alltoall`, `reduce_scatter`, `scan`, `exscan`, `barrier` | Full support |
| **Point-to-Point** | `send`, `recv`, `sendrecv`, `ssend`, `bsend`, `rsend` | Full support |
| **Non-Blocking** | `isend`, `irecv`, `iallreduce`, `ibcast`, `igather`, etc. | Full support |
| **Persistent** | `send_init`, `recv_init`, `start`, `startall` | Full support (MPI 4.0) |
| **Partitioned** | Partitioned communication patterns | Full support (MPI 4.0) |
| **Topology** | Cartesian grids, graph topologies, neighbor collectives | Full support |
| **Communicators** | `comm_split`, `comm_dup`, `comm_create`, `comm_free` | Full support |
| **Datatypes** | All MPI datatypes via Swift protocols | Full support |
| **GPU-Aware** | CUDA/ROCm device detection and communication | Full support |

### Differentiable Operations (13 total)

These operations support `@differentiable(reverse)` and automatically compute gradients in the backward pass:

| Operation | Function | Gradient Behavior |
|-----------|----------|-------------------|
| Allreduce (Sum) | `differentiableAllreduce` | Each process gets full gradient |
| Broadcast | `differentiableBroadcast` | Gradients reduce to root |
| Reduce (Sum) | `differentiableReduce` | Root's gradient broadcasts to all |
| Mean | `differentiableMean` | Gradient scaled by 1/N |
| Scatter | `differentiableScatter` | Gradients gather to root |
| Reduce-Scatter | `differentiableReduceScatter` | Gradient scaled by 1/N |
| Inclusive Scan | `differentiableScan` | Suffix sum of gradients |
| Exclusive Scan | `differentiableExscan` | Suffix sum (shifted) |
| Send-Recv Pair | `differentiableSendRecv` | Gradient flows sender <- receiver |
| Async Send | `differentiableAsyncSend` | Async-ready API (blocking impl.) |
| Async Send-Recv | `differentiableAsyncSendRecv` | Async-ready paired communication |
| Async Allreduce | `differentiableAsyncAllreduce` | Async-ready allreduce |
| Async Broadcast | `differentiableAsyncBroadcast` | Async-ready broadcast |

> **Note on Async Operations**: The async operations use blocking semantics currently due to Swift AD limitations, but provide a future-proof API that will support true async when Swift AD gains the necessary capabilities.

### DLPack Tensor Operations (3 total)

Zero-copy tensor operations for ML framework integration. Differentiation is handled by the framework's autograd system, not by MPI.

| Operation | Function | Zero-Copy | GPU Support |
|-----------|----------|-----------|-------------|
| Allreduce | `allreduceDLPack` | Yes | Yes |
| Broadcast | `broadcastDLPack` | Yes | Yes |
| Reduce | `reduceDLPack` | Yes | Yes |

## Use Cases

### Data Parallel Training

```swift
// Each process computes local gradients
let localGrad = computeGradient(localData)

// Aggregate gradients (average across all processes)
let globalGrad = differentiableMean(localGrad, on: comm)

// All processes update with same gradient
parameter -= learningRate * globalGrad
```

### Distributed Loss Computation

```swift
let (loss, grad) = valueWithGradient(at: prediction) { pred in
    distributedMSELoss(localPrediction: pred, localTarget: target, on: comm)
}
```

### Mixing Differentiable and Standard Operations

```swift
// Differentiable operations in training loop
let globalLoss = differentiableMean(localLoss, on: comm)
let grad = gradient(at: parameters, of: { p in globalLoss })

// Standard MPI for checkpointing
if epoch % 10 == 0 {
    try comm.gather(modelWeights, root: 0)
    if comm.rank == 0 { saveCheckpoint(modelWeights) }
}
```

## Examples

| Example | Description |
|---------|-------------|
| [HelloMPI](Examples/HelloMPI.swift) | Simple MPI hello world |
| [CollectivesDemo](Examples/CollectivesDemo.swift) | Collective operations demo |
| [PointToPointDemo](Examples/PointToPointDemo.swift) | Point-to-point communication |
| [TopologyDemo](Examples/TopologyDemo.swift) | MPI topology features |
| [ScalarOperationsExample](Examples/ScalarOperationsExample.swift) | All scalar differentiable operations |
| [DLPackExample](Examples/DLPackExample.swift) | Zero-copy tensor operations |
| [AsyncOperationsExample](Examples/AsyncOperationsExample.swift) | Async-ready differentiable operations |
| [DistributedGradientDescentExample](Examples/DistributedGradientDescentExample.swift) | Complete distributed training |

### Running Examples

```bash
# Build all
swift build

# Single process
swift run HelloMPI

# Multiple processes (recommended)
mpirun -np 4 swift run HelloMPI
mpirun -np 4 swift run ScalarOperationsExample
mpirun -np 4 swift run DistributedGradientDescentExample
```

See [Examples/README.md](Examples/README.md) for detailed usage patterns.

## Testing

```bash
swift test
# Test run with 86 tests in 0 suites passed after 0.004 seconds.
```

## Requirements

- **Swift**: 5.9 or later
- **MPI**: OpenMPI or MPICH
- **Platform**: macOS 15.0+ or Linux

## Project Structure

```
MessageDifferentiationKit/
├── Sources/
│   ├── CDLPack/                      # DLPack C headers
│   ├── CMPIBindings/                 # MPI C bindings
│   ├── MPISwift/                     # Complete MPI Swift bindings
│   └── MessageDifferentiationKit/    # Differentiable operations + DLPack
├── Tests/                            # 86 tests
├── Examples/                         # 8 executable examples
└── Package.swift
```

## Comparison with MeDiPack

[MeDiPack](https://github.com/SciCompKL/MeDiPack) is a mature C++ library providing AD support for MPI operations (~347/421 MPI functions).

| Aspect | MessageDifferentiationKit | MeDiPack |
|--------|---------------------------|----------|
| **Language** | Swift | C++ |
| **AD approach** | Native `@differentiable` | External (CoDiPack) |
| **Differentiable ops** | 13 (focused) | ~347 (comprehensive) |
| **Non-blocking AD** | Async-ready API (blocking impl.) | True async supported |
| **Maturity** | Early (v0.1) | Mature (years in production) |

**MessageDifferentiationKit advantages**: type-safe Swift APIs, native AD with no external dependencies, composable with Swift's `Differentiable` protocol, built for the Swift ML ecosystem.

## Future Work

- [ ] True async differentiation (requires Swift language support)
- [ ] Array-based gather/scatter/allgather/alltoall with gradients
- [ ] Custom reduction operators with AD
- [ ] GPU-aware differentiable operations
- [ ] DLPack tensor differentiation via framework integration

## Resources

- [MPI Standard](https://www.mpi-forum.org/)
- [Swift Automatic Differentiation](https://github.com/apple/swift/blob/main/docs/DifferentiableProgramming.md)
- [DLPack Specification](https://github.com/dmlc/dlpack)
- [OpenMPI](https://www.open-mpi.org/)

## License

MIT License - see [LICENSE](LICENSE).

## Citation

```bibtex
@software{messagedifferentiationkit2026,
  title={MessageDifferentiationKit: Automatic Differentiation for MPI},
  author={Pedro Nahum},
  year={2026},
  url={https://github.com/pedronahum/MessageDifferentiationKit}
}
```

## Acknowledgments

- The MPI community for the MPI standard
- Apple's Swift team for differentiable programming support
- The DLPack project for the tensor interchange standard
