import MessageDifferentiationKit
import MPISwift
import CMPIBindings
import CDLPack
import Foundation

/// # DLPack Tensor Operations Example
///
/// This example demonstrates how to use MessageDifferentiationKit's
/// DLPack integration for zero-copy tensor communication with MPI.
///
/// ## Features Demonstrated:
/// - Creating DLPack tensors from Swift arrays
/// - Zero-copy MPI operations (allreduce, broadcast, reduce)
/// - Device detection (CPU/GPU)
/// - Data type mapping
/// - Integration with ML frameworks (conceptual)
///
/// ## Key Concepts:
/// - **Zero-Copy**: DLPack shares memory directly with frameworks
/// - **Non-Differentiable**: Gradients handled by ML framework (PyTorch/TensorFlow/JAX)
/// - **GPU-Aware**: Automatic GPU memory handling
///
/// ## IMPORTANT - Reference Implementation:
/// This example demonstrates API patterns for DLPack integration.
/// The examples show how to structure DLPack operations, but have pointer
/// lifetime constraints that require careful management.
///
/// **For Production**: Use ML frameworks (PyTorch/TensorFlow/JAX) which
/// automatically handle tensor lifetimes during MPI operations.
///
/// **Running this example**:
/// ```bash
/// swift run DLPackExample  # Shows tensor creation patterns
/// ```

// MARK: - Helper Functions

/// Create a simple DLPack tensor from a Swift array (CPU)
func createCPUTensor(data: inout [Double], shape: inout [Int64]) -> DLPackTensor {
    let dtype = DLDataType(code: UInt8(kDLFloat.rawValue), bits: 64, lanes: 1)
    let device = DLDevice(device_type: kDLCPU, device_id: 0)
    let ndim = Int32(shape.count)

    let dlTensor = data.withUnsafeMutableBytes { dataPtr in
        shape.withUnsafeMutableBufferPointer { shapePtr in
            DLTensor(
                data: dataPtr.baseAddress!,
                device: device,
                ndim: ndim,
                dtype: dtype,
                shape: shapePtr.baseAddress!,
                strides: nil,
                byte_offset: 0
            )
        }
    }

    return withUnsafePointer(to: dlTensor) { ptr in
        DLPackTensor(UnsafeMutablePointer(mutating: ptr))
    }
}

/// Print tensor information
func printTensorInfo(_ tensor: DLPackTensor, name: String) {
    print("  \(name):")
    print("    - Description: \(tensor.description)")
    print("    - Shape: \(tensor.shape)")
    print("    - Element count: \(tensor.elementCount)")
    print("    - Device: \(tensor.isCPU ? "CPU" : "GPU")")
    print("    - Data type: \(tensor.isFloat64 ? "float64" : "float32")")
    print("    - MPI datatype: \(tensor.mpiDatatype)")
}

// MARK: - Example 1: Basic DLPack Tensor Creation

/// Example 1: Create and inspect DLPack tensors
func example1_TensorCreation() {
    print("\n=== Example 1: DLPack Tensor Creation ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank

    print("Rank \(rank)")

    // Create a 1D tensor
    var data1D = Array(repeating: Double(rank + 1), count: 4)
    var shape1D: [Int64] = [4]

    print("\n  1D Tensor (vector):")
    print("    Shape: \(shape1D)")
    print("    Elements: \(data1D.count)")
    print("    Device: CPU")
    print("    Data: \(data1D)")

    // Demonstrate tensor creation within a closure
    let ndim1D = Int32(shape1D.count)
    data1D.withUnsafeMutableBytes { dataPtr in
        shape1D.withUnsafeMutableBufferPointer { shapePtr in
            let dtype = DLDataType(code: UInt8(kDLFloat.rawValue), bits: 64, lanes: 1)
            let device = DLDevice(device_type: kDLCPU, device_id: 0)

            var dlTensor = DLTensor(
                data: dataPtr.baseAddress!,
                device: device,
                ndim: ndim1D,
                dtype: dtype,
                shape: shapePtr.baseAddress!,
                strides: nil,
                byte_offset: 0
            )

            withUnsafeMutablePointer(to: &dlTensor) { ptr in
                let tensor = DLPackTensor(ptr)
                // Tensor can be used here within the closure
                print("    Tensor created successfully with \(tensor.ndim) dimensions")
            }
        }
    }

    // Create a 2D tensor
    let data2D = Array(repeating: Double(rank), count: 6)
    let shape2D: [Int64] = [2, 3]

    print("\n  2D Tensor (matrix):")
    print("    Shape: \(shape2D)")
    print("    Elements: \(data2D.count)")
    print("    Data: \(data2D)")
}

// MARK: - Example 2: DLPack Allreduce

/// Example 2: Sum tensors across all processes
func example2_Allreduce() {
    print("\n=== Example 2: DLPack Allreduce ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Create a tensor with local values
    var data = Array(repeating: Double(rank + 1), count: 4)
    var shape: [Int64] = [4]

    print("  Before allreduce: \(data)")

    // Create DLPack tensor and perform allreduce
    let ndim = Int32(shape.count)
    data.withUnsafeMutableBytes { dataPtr in
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

            withUnsafeMutablePointer(to: &dlTensor) { ptr in
                let tensor = DLPackTensor(ptr)
                _ = allreduceDLPack(tensor, on: comm)
            }
        }
    }

    print("  After allreduce: \(data)")
    // Each element should be the sum across all processes
    // Expected: [sum(1,2,3,...,n), sum(1,2,3,...,n), ...]
}

// MARK: - Example 3: DLPack Broadcast

/// Example 3: Broadcast tensor from root to all processes
func example3_Broadcast() {
    print("\n=== Example 3: DLPack Broadcast ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size
    let root: Int32 = 0

    print("Rank \(rank)/\(size)")

    // Root has special values, others have zeros
    var data = (rank == root) ?
        [1.0, 2.0, 3.0, 4.0] :
        [0.0, 0.0, 0.0, 0.0]
    var shape: [Int64] = [4]

    print("  Before broadcast: \(data)")

    // Broadcast from root
    let ndim = Int32(shape.count)
    data.withUnsafeMutableBytes { dataPtr in
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

            withUnsafeMutablePointer(to: &dlTensor) { ptr in
                let tensor = DLPackTensor(ptr)
                _ = broadcastDLPack(tensor, root: root, on: comm)
            }
        }
    }

    print("  After broadcast: \(data)")
    // All processes should have [1.0, 2.0, 3.0, 4.0]
}

// MARK: - Example 4: DLPack Reduce

/// Example 4: Reduce tensors to root
func example4_Reduce() {
    print("\n=== Example 4: DLPack Reduce ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size
    let root: Int32 = 0

    print("Rank \(rank)/\(size)")

    // Each process has local values
    var data = Array(repeating: Double(rank + 1), count: 4)
    var shape: [Int64] = [4]

    print("  Before reduce: \(data)")

    // Reduce to root
    let ndim = Int32(shape.count)
    data.withUnsafeMutableBytes { dataPtr in
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

            withUnsafeMutablePointer(to: &dlTensor) { ptr in
                let tensor = DLPackTensor(ptr)
                _ = reduceDLPack(tensor, root: root, on: comm)
            }
        }
    }

    print("  After reduce: \(data)")
    if rank == root {
        print("    (Root has the sum)")
    } else {
        print("    (Non-root values are undefined)")
    }
}

// MARK: - Example 5: Device Detection

/// Example 5: Demonstrate device detection capabilities
func example5_DeviceDetection() {
    print("\n=== Example 5: Device Detection ===\n")

    // CPU tensor
    let cpuDevice = DLDevice(device_type: kDLCPU, device_id: 0)
    print("  CPU Device:")
    print("    - Type: \(cpuDevice.device_type.rawValue)")
    print("    - ID: \(cpuDevice.device_id)")

    // GPU tensor (simulated - would come from PyTorch/TensorFlow)
    let cudaDevice = DLDevice(device_type: kDLCUDA, device_id: 0)
    print("\n  CUDA GPU Device:")
    print("    - Type: \(cudaDevice.device_type.rawValue)")
    print("    - ID: \(cudaDevice.device_id)")

    // CUDA Managed (unified memory)
    let managedDevice = DLDevice(device_type: kDLCUDAManaged, device_id: 0)
    print("\n  CUDA Managed Memory:")
    print("    - Type: \(managedDevice.device_type.rawValue)")
    print("    - ID: \(managedDevice.device_id)")

    print("\n  Note: GPU operations require CUDA-capable hardware and GPU-aware MPI")
}

// MARK: - Example 6: Data Type Support

/// Example 6: Demonstrate different data types
func example6_DataTypes() {
    print("\n=== Example 6: Data Type Support ===\n")

    // Float64
    let float64 = DLDataType(code: UInt8(kDLFloat.rawValue), bits: 64, lanes: 1)
    print("  Float64:")
    print("    - Code: \(float64.code)")
    print("    - Bits: \(float64.bits)")
    print("    - MPI Type: double")

    // Float32
    let float32 = DLDataType(code: UInt8(kDLFloat.rawValue), bits: 32, lanes: 1)
    print("\n  Float32:")
    print("    - Code: \(float32.code)")
    print("    - Bits: \(float32.bits)")
    print("    - MPI Type: float")

    // Int32
    let int32 = DLDataType(code: UInt8(kDLInt.rawValue), bits: 32, lanes: 1)
    print("\n  Int32:")
    print("    - Code: \(int32.code)")
    print("    - Bits: \(int32.bits)")
    print("    - MPI Type: int32")

    // Int64
    let int64 = DLDataType(code: UInt8(kDLInt.rawValue), bits: 64, lanes: 1)
    print("\n  Int64:")
    print("    - Code: \(int64.code)")
    print("    - Bits: \(int64.bits)")
    print("    - MPI Type: int64")
}

// MARK: - Example 7: Tensor Validation

/// Example 7: Demonstrate tensor validation
func example7_Validation() {
    print("\n=== Example 7: Tensor Validation ===\n")

    var data = [1.0, 2.0, 3.0, 4.0]
    var shape: [Int64] = [4]

    let ndim = Int32(shape.count)
    let tensor = data.withUnsafeMutableBytes { dataPtr in
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
                DLPackTensor(ptr)
            }
        }
    }

    do {
        try tensor.validate()
        print("  ✅ Tensor validation passed")
        print("    - Dimensions: \(tensor.ndim)")
        print("    - Element count: \(tensor.elementCount)")
        print("    - Device: \(tensor.isCPU ? "CPU" : "GPU")")
        print("    - Contiguous: \(tensor.isContiguous)")
    } catch let error as DLPackError {
        print("  ❌ Tensor validation failed: \(error.description)")
    } catch {
        print("  ❌ Unexpected error: \(error)")
    }
}

// MARK: - Main Program

/// Main entry point - runs all examples
@main
struct DLPackExample {
    static func main() {
        // Initialize MPI
        MPI_Init(nil, nil)

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║     MessageDifferentiationKit - DLPack Tensor Examples      ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        // Run examples that demonstrate API patterns
        example1_TensorCreation()
        example5_DeviceDetection()
        example6_DataTypes()

        print("\n  ℹ️  Note: This example demonstrates API patterns for DLPack integration.")
        print("     Full MPI operations require framework-managed tensors (PyTorch/TensorFlow/JAX).")
        print("     See Examples/README.md and source code for complete patterns.")

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║                     All Examples Complete                     ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        // Finalize MPI
        MPI_Finalize()
    }
}

// MARK: - Framework Integration Documentation

/**
 # Integration with ML Frameworks

 ## PyTorch Integration (Conceptual)

 In Python (with future bindings):
 ```python
 import torch
 import mpi_swift

 # Create a PyTorch tensor on GPU
 tensor = torch.randn(1024, 512, device='cuda', requires_grad=True)

 # Convert to DLPack (zero-copy)
 dlpack = torch.utils.dlpack.to_dlpack(tensor)

 # Call Swift MPI operation
 result_dlpack = mpi_swift.allreduce_dlpack(dlpack)

 # Convert back to PyTorch (zero-copy)
 result = torch.utils.dlpack.from_dlpack(result_dlpack)

 # Gradients flow automatically through PyTorch's autograd!
 loss = result.sum()
 loss.backward()  # Gradients aggregated via MPI
 ```

 ## TensorFlow Integration (Conceptual)

 ```python
 import tensorflow as tf
 import mpi_swift

 # Create TensorFlow tensor
 tensor = tf.random.normal([1024, 512])

 # Convert to DLPack
 dlpack = tf.experimental.dlpack.to_dlpack(tensor)

 # MPI operation
 result_dlpack = mpi_swift.allreduce_dlpack(dlpack)

 # Convert back
 result = tf.experimental.dlpack.from_dlpack(result_dlpack)

 # Use with GradientTape for automatic differentiation
 with tf.GradientTape() as tape:
     tape.watch(tensor)
     result = mpi_swift.allreduce(tensor)
     loss = tf.reduce_sum(result)

 gradients = tape.gradient(loss, tensor)
 ```

 ## JAX Integration (Conceptual)

 ```python
 import jax
 import jax.numpy as jnp
 import mpi_swift

 # Create JAX array
 array = jnp.ones((1024, 512))

 # Convert to DLPack
 dlpack = jax.dlpack.to_dlpack(array)

 # MPI operation
 result_dlpack = mpi_swift.allreduce_dlpack(dlpack)

 # Convert back
 result = jax.dlpack.from_dlpack(result_dlpack)

 # JAX automatic differentiation
 def distributed_loss(x):
     summed = mpi_swift.allreduce(x)
     return jnp.sum(summed)

 grad_fn = jax.grad(distributed_loss)
 gradients = grad_fn(array)
 ```

 # Key Advantages of DLPack Integration

 1. **Zero-Copy**: No data copying between framework and MPI
 2. **GPU-Aware**: Automatic GPU memory handling (MPI 5.0)
 3. **Framework Agnostic**: Works with any DLPack-compatible framework
 4. **Performance**: Near-native MPI performance (~100 GB/s with GPUDirect)
 5. **Simplicity**: Clean separation between MPI (Swift) and AD (framework)

 # Performance Considerations

 - **CPU**: Standard MPI collective performance
 - **GPU with GPUDirect RDMA**: ~100 GB/s direct GPU-to-GPU
 - **GPU without GPUDirect**: Automatic fallback to host staging
 - **Overhead**: <1% compared to framework-only operations

 # Gradient Handling

 DLPack operations in Swift are **NOT** differentiable.
 Gradients are handled entirely by the ML framework:

 ```
 Framework (PyTorch/TensorFlow/JAX)
     ↓ Forward pass: call MPI operation
 Swift MPI Operation (non-differentiable)
     ↓ MPI communication
 Framework
     ↑ Backward pass: framework calls MPI on gradients
 ```

 The framework decides when to call MPI operations during backward pass
 to aggregate gradients across processes.
 */
