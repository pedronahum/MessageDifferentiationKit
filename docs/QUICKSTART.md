# Quick Start Guide

Get up and running with MessageDifferentiationKit in 5 minutes!

## Prerequisites

- Swift 5.9 or later
- MPI (OpenMPI or MPICH)
- macOS 14.0+ or Linux

## Installation

### 1. Install MPI

**macOS:**
```bash
brew install open-mpi
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install libopenmpi-dev
```

**Verify Installation:**
```bash
mpirun --version
```

### 2. Create a Swift Package

```bash
mkdir MyMPIProject
cd MyMPIProject
swift package init --type executable
```

### 3. Add MessageDifferentiationKit

Edit `Package.swift`:

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MyMPIProject",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/pedronahum/MessageDifferentiationKit", branch: "main")
    ],
    targets: [
        .executableTarget(
            name: "MyMPIProject",
            dependencies: ["MessageDifferentiationKit"],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable")
            ]
        )
    ]
)
```

## Example 1: Basic Scalar Operations

Create `Sources/MyMPIProject/main.swift`:

```swift
import MessageDifferentiationKit
import MPISwift
import _Differentiation

// Get the world communicator
let comm = MPICommunicator.world
let rank = comm.rank
let size = comm.size

print("Hello from rank \(rank) of \(size)")

// Each process has a local value
let localValue = Double(rank + 1)

// Sum across all processes
let globalSum = differentiableAllreduce(localValue, on: comm)
print("Rank \(rank): Global sum = \(globalSum)")

// Compute gradients automatically
let grad = gradient(at: localValue) { value in
    differentiableAllreduce(value, on: comm)
}
print("Rank \(rank): Gradient = \(grad)")
```

### Build and Run

```bash
swift build
.build/debug/MyMPIProject
```

**Expected Output:**
```
Hello from rank 0 of 1
Rank 0: Global sum = 1.0
Rank 0: Gradient = 1.0
```

## Example 2: Distributed Training Pattern

```swift
import MessageDifferentiationKit
import MPISwift
import _Differentiation

let comm = MPICommunicator.world
let rank = comm.rank

// Simulate local data
let localData = Double(rank) * 10.0
let target = 50.0

// Training step with automatic gradients
func trainingStep(parameter: Double) -> Double {
    // Local forward pass
    let prediction = parameter * localData

    // Distributed loss (MSE)
    let localLoss = (prediction - target) * (prediction - target)
    let globalLoss = differentiableMean(localLoss, on: comm)

    return globalLoss
}

// Initial parameter
var parameter = 0.5

// Compute loss and gradient
let (loss, grad) = valueWithGradient(at: parameter, of: trainingStep)

print("Rank \(rank): Loss = \(loss), Gradient = \(grad)")

// Update parameter
let learningRate = 0.01
parameter -= learningRate * grad

print("Rank \(rank): Updated parameter = \(parameter)")
```

## Example 3: DLPack Tensor Operations

```swift
import MessageDifferentiationKit
import MPISwift
import CDLPack

let comm = MPICommunicator.world
let rank = comm.rank

// Create a tensor with local values
var data = Array(repeating: Double(rank + 1), count: 4)
var shape: [Int64] = [4]

print("Rank \(rank): Before allreduce: \(data)")

// Create DLPack tensor and perform zero-copy allreduce
let ndim = Int32(shape.count)
let _ = data.withUnsafeMutableBytes { dataPtr in
    shape.withUnsafeMutableBufferPointer { shapePtr in
        let dtype = DLDataType(code: UInt8(kDLFloat.rawValue), bits: 64, lanes: 1)
        let device = DLDevice(device_type: kDLCPU, device_id: 0)

        var dlTensor = DLTensor(
            data: dataPtr.baseAddress!,
            device: device,
            ndim: ndim,
            dtype: dtype,
            shape: shapePtr.baseAddress!,
            strides: nil,
            byte_offset: 0
        )

        return withUnsafeMutablePointer(to: &dlTensor) { ptr in
            let tensor = DLPackTensor(ptr)
            return allreduceDLPack(tensor, on: comm)
        }
    }
}

print("Rank \(rank): After allreduce: \(data)")
```

## Available Operations

### Scalar Differentiable Operations (10 total)

All support Swift's Automatic Differentiation:

```swift
// Collective operations
differentiableAllreduce(value, on: comm)
differentiableBroadcast(value, root: 0, on: comm)
differentiableReduce(value, root: 0, on: comm)
differentiableMean(value, on: comm)

// Array operations
differentiableAllgather(values, on: comm)

// Scan operations
differentiableScan(value, on: comm)
differentiableExscan(value, on: comm)

// Scatter operations
differentiableScatter(values, root: 0, on: comm)
differentiableReduceScatter(values, on: comm)

// Point-to-point
differentiableSendRecv(value, toRank: 1, fromRank: 0, on: comm)
```

### DLPack Tensor Operations (4 total)

Zero-copy operations for ML frameworks:

```swift
allreduceDLPack(tensor, on: comm)
broadcastDLPack(tensor, root: 0, on: comm)
reduceDLPack(tensor, root: 0, on: comm)
allgatherDLPack(input, into: output, on: comm)
```

## Running Tests

```bash
# Run all tests
swift test
```

## Common Patterns

### Pattern 1: Data Parallel Training

```swift
// Each process has local data and computes local gradients
let localGrad = computeGradient(localData)

// Aggregate gradients across all processes
let globalGrad = differentiableMean(localGrad, on: comm)

// All processes update with the same gradient
parameter -= learningRate * globalGrad
```

### Pattern 2: Distributed Loss

```swift
// Each process computes local loss
let localLoss = computeLoss(localPrediction, localTarget)

// Average loss across all processes
let globalLoss = differentiableMean(localLoss, on: comm)
```

### Pattern 3: Parameter Server

```swift
// Root broadcasts parameters to all workers
let params = differentiableBroadcast(localParams, root: 0, on: comm)

// Workers compute local gradients
let localGrad = computeGradient(params, localData)

// Reduce gradients back to root
let globalGrad = differentiableReduce(localGrad, root: 0, on: comm)
```

## Gradient Behavior

Each operation has well-defined gradient behavior:

| Operation | Forward | Backward (Gradient) |
|-----------|---------|---------------------|
| `allreduce` | Sum all inputs | Gradient = 1.0 for all |
| `broadcast` | Root → All | Gradient flows to root only |
| `reduce` | All → Root | Gradient broadcasts from root |
| `mean` | Average all inputs | Gradient / size |
| `scan` | Prefix sum | Accumulates from later ranks |
| `allgather` | Concatenate arrays | Each rank gets its gradient slice |

## Troubleshooting

### "Cannot find MPI headers"

Make sure MPI is installed and pkg-config can find it:
```bash
pkg-config --cflags ompi
```

### "Experimental feature 'Differentiable' must be enabled"

Add to your target in `Package.swift`:
```swift
swiftSettings: [
    .enableExperimentalFeature("Differentiable")
]
```

### "Missing import _Differentiation"

Add the import:
```swift
import _Differentiation
```

### MPI Initialization

The `MPICommunicator.world` accessor automatically handles MPI initialization and finalization. No manual `MPI_Init` or `MPI_Finalize` calls are needed.

## Next Steps

1. **Explore Examples**: Check out the [Examples/](../Examples/) directory for comprehensive examples
2. **Study the API**: Review the source code in [Sources/MessageDifferentiationKit/](../Sources/MessageDifferentiationKit/)
3. **Run Tests**: Examine [Tests/](../Tests/) to understand expected behavior

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pedronahum/MessageDifferentiationKit/issues)
- **Examples**: [Examples/](../Examples/) directory with working code
