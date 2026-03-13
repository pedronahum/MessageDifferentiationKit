# MessageDifferentiationKit Examples

This directory contains comprehensive examples demonstrating all features of MessageDifferentiationKit.

## Overview

MessageDifferentiationKit provides two complementary APIs:

1. **Scalar Differentiable Operations** (Phase 1-3): Swift Automatic Differentiation with MPI
2. **DLPack Tensor Operations** (Phase 4): Zero-copy tensor communication for ML frameworks

## Examples

### 1. Scalar Operations Example

**File**: [ScalarOperationsExample.swift](ScalarOperationsExample.swift)

Demonstrates scalar differentiable MPI operations with Swift's AD:

- **Example 1**: Basic Allreduce with Gradients
- **Example 2**: Broadcast with Gradients
- **Example 3**: Distributed Mean
- **Example 4**: Scan (Prefix Sum)
- **Example 5**: Distributed Training Simulation
- **Example 6**: Composition of Operations
- **Example 7**: Reduce-Scatter Pattern
- **Example 8**: Distributed MSE Loss

**Key Features**:
- ✅ Automatic gradient computation
- ✅ Correct gradient routing through MPI operations
- ✅ Type-safe Swift AD integration
- ✅ Distributed training patterns

**Running**:
```bash
# Build the example
swift build

# Note: These examples demonstrate the API usage patterns.
# Actual MPI execution requires proper MPI initialization,
# which is typically handled by your application's setup.

# The examples can be studied to understand:
# - How to structure differentiable MPI operations
# - Gradient computation patterns
# - Distributed training workflows
```

### 2. DLPack Tensor Operations Example

**File**: [DLPackExample.swift](DLPackExample.swift)

Demonstrates zero-copy tensor communication via DLPack:

- **Example 1**: DLPack Tensor Creation
- **Example 2**: DLPack Allreduce
- **Example 3**: DLPack Broadcast
- **Example 4**: DLPack Reduce
- **Example 5**: Device Detection (CPU/GPU)
- **Example 6**: Data Type Support
- **Example 7**: Tensor Validation

**Key Features**:
- ✅ Zero-copy memory sharing
- ✅ GPU-aware MPI support
- ✅ Framework-agnostic tensor operations
- ✅ Device and type detection

**Running**:
```bash
# Build the example
swift build

# Note: These examples demonstrate zero-copy tensor patterns.
# They serve as reference implementations showing:
# - DLPack tensor creation and manipulation
# - Device detection patterns
# - MPI operation integration
```

## Building All Examples

Add examples to your `Package.swift`:

```swift
products: [
    .executable(name: "ScalarOperationsExample", targets: ["ScalarOperationsExample"]),
    .executable(name: "DLPackExample", targets: ["DLPackExample"]),
],

targets: [
    .executableTarget(
        name: "ScalarOperationsExample",
        dependencies: ["MessageDifferentiationKit"],
        path: "Examples",
        sources: ["ScalarOperationsExample.swift"],
        swiftSettings: [
            .enableExperimentalFeature("Differentiable"),
        ]
    ),
    .executableTarget(
        name: "DLPackExample",
        dependencies: ["MessageDifferentiationKit"],
        path: "Examples",
        sources: ["DLPackExample.swift"],
        swiftSettings: [
            .enableExperimentalFeature("Differentiable"),
        ]
    ),
]
```

## Understanding the Examples

### Scalar Operations: When to Use

Use scalar differentiable operations when:
- Working purely in Swift
- Need automatic differentiation in Swift
- Have scalar or small data per process
- Want Swift's type safety and AD guarantees

**Example Use Case**: Distributed optimization where each process has a single parameter value.

```swift
// Each process has a parameter
var parameter = 0.5

// Compute gradient using Swift AD
let grad = gradient(at: parameter) { p in
    let loss = (p - target) * (p - target)
    return loss
}

// Aggregate gradients across processes
let avgGrad = differentiableMean(grad, on: comm)

// Update parameter
parameter -= learningRate * avgGrad
```

### DLPack Operations: When to Use

Use DLPack tensor operations when:
- Integrating with PyTorch/TensorFlow/JAX
- Working with large tensors
- Need GPU-aware MPI
- Framework handles automatic differentiation

**Example Use Case**: Distributed deep learning with PyTorch.

```python
# PyTorch (via future Python bindings)
import torch
import mpi_swift

tensor = torch.randn(1024, 512, device='cuda', requires_grad=True)

# Zero-copy MPI operation
result = mpi_swift.allreduce(tensor)

# PyTorch autograd handles gradients automatically
loss = result.sum()
loss.backward()
```

## Operation Reference

### Scalar Differentiable Operations

| Operation | Description | Gradient Behavior |
|-----------|-------------|-------------------|
| `differentiableAllreduce` | Sum across all processes | Gradient = 1.0 for all |
| `differentiableBroadcast` | Root sends to all | Gradient flows to root only |
| `differentiableReduce` | All send to root | Gradient broadcasts from root |
| `differentiableMean` | Average across processes | Gradient divided by size |
| `differentiableScan` | Prefix sum | Gradient accumulates from later ranks |
| `differentiableExscan` | Exclusive prefix sum | Similar to scan |
| `differentiableScatter` | Distribute from root | Gradient gathers to root |
| `differentiableReduceScatter` | Reduce and distribute | Combined reduce + scatter |
| `differentiableSendRecv` | Point-to-point | Gradient flows backward |

### DLPack Tensor Operations

| Operation | Description | Differentiable? |
|-----------|-------------|-----------------|
| `allreduceDLPack` | Sum tensors across all processes | No (framework handles) |
| `broadcastDLPack` | Root broadcasts tensor to all | No (framework handles) |
| `reduceDLPack` | All reduce tensors to root | No (framework handles) |

**Note**: DLPack operations are NOT differentiable in Swift. Gradients are handled by the ML framework (PyTorch/TensorFlow/JAX) that owns the tensors.

## Gradient Flow Patterns

### Pattern 1: Data Parallel Training

```swift
// Each process computes local gradients
let localGrad = computeGradient(localData)

// Aggregate gradients (average)
let globalGrad = differentiableMean(localGrad, on: comm)

// All processes update with same gradient
parameter -= learningRate * globalGrad
```

**Gradient Flow**:
```
Process 0: grad₀ ─┐
Process 1: grad₁ ─┼─> Allreduce -> (grad₀+grad₁+grad₂+grad₃)/4 -> All processes
Process 2: grad₂ ─┤
Process 3: grad₃ ─┘
```

### Pattern 2: Parameter Server

```swift
// Root has parameters
let params = differentiableBroadcast(localParams, root: 0, on: comm)

// Workers compute gradients
let grads = differentiableReduce(localGrads, root: 0, on: comm)
```

**Gradient Flow**:
```
Forward:  Root -> Broadcast -> All workers
Backward: All workers -> Reduce -> Root
```

### Pattern 3: Pipeline Parallel

```swift
// Pass forward
let output = differentiableSendRecv(
    input, toRank: nextRank, fromRank: prevRank, on: comm
)
```

**Gradient Flow**:
```
Forward:  Rank i -> Rank i+1 -> Rank i+2
Backward: Rank i+2 -> Rank i+1 -> Rank i
```

## Performance Tips

### Scalar Operations

1. **Minimize Communication**: Aggregate gradients once per iteration
2. **Use Mean Instead of Sum**: `differentiableMean` is more numerically stable
3. **Batch Operations**: Process multiple values together when possible

### DLPack Operations

1. **Zero-Copy**: Always use DLPack for large tensors (no data copying)
2. **GPU-Aware MPI**: Enable GPUDirect RDMA for ~100 GB/s transfers
3. **Contiguous Tensors**: Ensure tensors are contiguous for best performance
4. **Asynchronous Operations**: Use non-blocking MPI for overlapping computation

## Troubleshooting

### Common Issues

**Issue**: "No differentiation parameters could be inferred"
```swift
// ❌ Wrong: Trying to differentiate DLPack operations
let grad = gradient(at: tensor) { t in
    allreduceDLPack(t, on: comm)
}

// ✅ Correct: Use scalar operations for Swift AD
let grad = gradient(at: value) { v in
    differentiableAllreduce(v, on: comm)
}
```

**Issue**: "Pointer lifetime warnings"
```swift
// ❌ Wrong: Temporary pointer
var data = [1.0, 2.0, 3.0]
let tensor = createTensor(&data)  // Pointer invalidated!

// ✅ Correct: Extend lifetime
data.withUnsafeMutableBytes { ptr in
    let tensor = createTensor(ptr)
    // Use tensor here
}
```

**Issue**: "MPI not initialized"
```swift
// ✅ MPICommunicator.world handles initialization automatically
let comm = MPICommunicator.world
```

## Testing

Both examples include self-tests that verify correctness:

```bash
# Run tests
swift test

# Expected: All 77 tests pass
# - 56 original tests (Phases 1-3)
# - 21 new DLPack tests (Phase 4)
```

## Further Reading

- **Main Documentation**: [../README.md](../README.md)
- **Phase 3 Complete**: [../PHASE3_COMPLETE.md](../PHASE3_COMPLETE.md)
- **Phase 4 DLPack**: [../PHASE4_DLPACK_COMPLETE.md](../PHASE4_DLPACK_COMPLETE.md)
- **DLPack Reality Check**: [../PHASE4_DLPACK_REALITY_CHECK.md](../PHASE4_DLPACK_REALITY_CHECK.md)
- **MPI Swift Bindings**: [../Sources/MPISwift/](../Sources/MPISwift/)

## Contributing

To add new examples:

1. Create a new `.swift` file in `Examples/`
2. Add executable target to `Package.swift`
3. Document in this README
4. Add tests in `Tests/MessageDifferentiationKitTests/`

## License

Same as MessageDifferentiationKit main package.

## Questions?

Check the main documentation or file an issue on GitHub.

---

**Happy Distributed Computing with Swift! 🚀**
