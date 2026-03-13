import MessageDifferentiationKit
import MPISwift
import CMPIBindings
import Foundation
import _Differentiation

/// # Scalar Differentiable MPI Operations Example
///
/// This example demonstrates how to use MessageDifferentiationKit's
/// scalar differentiable operations with Swift's Automatic Differentiation.
///
/// ## Features Demonstrated:
/// - Basic MPI operations (allreduce, broadcast, reduce)
/// - Gradient computation with Swift AD
/// - Distributed gradient aggregation
/// - Scan operations (prefix sums)
/// - Convenience functions (mean, sum, loss)
///
/// ## Running this example:
/// ```bash
/// # Single process (for testing)
/// swift run ScalarOperationsExample
///
/// # Multi-process (requires MPI)
/// mpirun -np 4 swift run ScalarOperationsExample
/// ```

// MARK: - Example 1: Basic Allreduce with Gradients

/// Example 1: Compute sum across all processes and its gradient
func example1_BasicAllreduce() {
    print("\n=== Example 1: Basic Allreduce with Gradients ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Each process has a local value
    let localValue = Double(rank + 1)  // Rank 0: 1.0, Rank 1: 2.0, etc.

    print("  Local value: \(localValue)")

    // Allreduce: sum across all processes
    let globalSum = differentiableAllreduce(localValue, on: comm)

    print("  Global sum: \(globalSum)")
    // Expected: sum(1, 2, 3, ..., n) = n*(n+1)/2

    // Compute gradient
    let gradient = gradient(at: localValue) { value in
        differentiableAllreduce(value, on: comm)
    }

    print("  Gradient: \(gradient)")
    // Gradient of allreduce is 1.0 for all processes
    // (each input contributes equally to the sum)
}

// MARK: - Example 2: Broadcast with Gradients

/// Example 2: Broadcast from root and compute gradients
func example2_Broadcast() {
    print("\n=== Example 2: Broadcast with Gradients ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size
    let root: Int32 = 0

    print("Rank \(rank)/\(size)")

    // Root has the value to broadcast
    let localValue = (rank == root) ? 42.0 : 0.0

    print("  Local value before broadcast: \(localValue)")

    // Broadcast: root sends to all
    let broadcastValue = differentiableBroadcast(localValue, root: root, on: comm)

    print("  After broadcast: \(broadcastValue)")
    // All processes should have 42.0

    // Compute gradient
    let gradient = gradient(at: localValue) { value in
        differentiableBroadcast(value, root: root, on: comm)
    }

    print("  Gradient: \(gradient)")
    // Root gradient = size (all processes affected by root)
    // Non-root gradients = 0 (don't affect output)
}

// MARK: - Example 3: Distributed Mean

/// Example 3: Compute distributed mean with automatic gradient aggregation
func example3_DistributedMean() {
    print("\n=== Example 3: Distributed Mean ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Each process has local data
    let localValue = Double(rank) * 10.0  // 0.0, 10.0, 20.0, 30.0, ...

    print("  Local value: \(localValue)")

    // Compute distributed mean
    let mean = differentiableMean(localValue, on: comm)

    print("  Global mean: \(mean)")
    // Expected: mean(0, 10, 20, ..., (n-1)*10) = (n-1)*10/2

    // Compute gradient
    let gradient = gradient(at: localValue) { value in
        differentiableMean(value, on: comm)
    }

    print("  Gradient: \(gradient)")
    // Gradient of mean is 1/size for all processes
}

// MARK: - Example 4: Scan (Prefix Sum)

/// Example 4: Compute prefix sum across processes
func example4_Scan() {
    print("\n=== Example 4: Scan (Prefix Sum) ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Each process contributes a value
    let localValue = 1.0

    print("  Local value: \(localValue)")

    // Scan: compute prefix sum
    let prefixSum = differentiableScan(localValue, on: comm)

    print("  Prefix sum: \(prefixSum)")
    // Rank 0: 1.0, Rank 1: 2.0, Rank 2: 3.0, etc.

    // Compute gradient
    let gradient = gradient(at: localValue) { value in
        differentiableScan(value, on: comm)
    }

    print("  Gradient: \(gradient)")
    // Gradient = (size - rank): how many later ranks depend on this value
}

// MARK: - Example 5: Distributed Training (Simple)

/// Example 5: Simulate distributed gradient descent
func example5_DistributedTraining() {
    print("\n=== Example 5: Distributed Training Simulation ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Each process has a local parameter
    var parameter = Double(rank) * 0.1  // 0.0, 0.1, 0.2, 0.3, ...
    let learningRate = 0.01

    print("  Initial parameter: \(parameter)")

    // Simulate 3 training steps
    for step in 0..<3 {
        // Compute loss (simple: (parameter - target)^2)
        let target = 0.5
        let loss = (parameter - target) * (parameter - target)

        // Compute local gradient
        let localGradient = 2.0 * (parameter - target)

        // Aggregate gradients across all processes
        let aggregatedGradient = differentiableMean(localGradient, on: comm)

        // Update parameter
        parameter -= learningRate * aggregatedGradient

        print("  Step \(step): parameter=\(parameter), loss=\(loss), grad=\(aggregatedGradient)")
    }

    print("  Final parameter: \(parameter)")
}

// MARK: - Example 6: Composition of Operations

/// Example 6: Compose multiple differentiable operations
func example6_Composition() {
    print("\n=== Example 6: Composition of Operations ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    let localValue = Double(rank + 1)

    // Compose operations: mean of squares
    let result = valueWithGradient(at: localValue) { value in
        let squared = value * value
        let mean = differentiableMean(squared, on: comm)
        return mean
    }

    print("  Local value: \(localValue)")
    print("  Mean of squares: \(result.value)")
    print("  Gradient: \(result.gradient)")
    // Gradient: d/dx mean(x^2) = 2x / size
}

// MARK: - Example 7: Reduce-Scatter Pattern

/// Example 7: Reduce-scatter operation
func example7_ReduceScatter() {
    print("\n=== Example 7: Reduce-Scatter ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    let localValue = Double(rank + 1)

    // Reduce-scatter: sum and divide
    let result = differentiableReduceScatter(localValue, on: comm)

    print("  Local value: \(localValue)")
    print("  Reduce-scatter result: \(result)")
    // Each process gets the average
}

// MARK: - Example 8: Allgather (Array Operation)

/// Example 8: Gather arrays from all processes with automatic differentiation
func example8_Allgather() {
    print("\n=== Example 8: Allgather (Array Operation) ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Each process has a local array
    let localValues = [Double(rank + 1) * 10.0, Double(rank + 1) * 20.0]

    print("  Local values: \(localValues)")

    // Allgather: concatenate arrays from all processes
    let gathered = differentiableAllgather(localValues, on: comm)

    print("  Gathered: \(gathered)")
    // With 2 processes: [10, 20, 20, 40]
    // With 4 processes: [10, 20, 20, 40, 30, 60, 40, 80]

    // Compute gradient through allgather
    let grad = gradient(at: localValues) { values in
        let gathered = differentiableAllgather(values, on: comm)
        // Sum all gathered values to get a scalar for gradient computation
        return gathered.differentiableReduce(0.0, { $0 + $1 })
    }

    print("  Gradient: \(grad)")
    // Each local element's gradient = 1.0 (contributes to the global sum)
}

// MARK: - Example 9: Distributed Loss Function

/// Example 9: Distributed Mean Squared Error loss
func example9_DistributedMSE() {
    print("\n=== Example 9: Distributed MSE Loss ===\n")

    let comm = MPICommunicator.world
    let rank = comm.rank
    let size = comm.size

    print("Rank \(rank)/\(size)")

    // Each process has predictions and targets
    let prediction = Double(rank) * 0.5
    let target = Double(rank) * 0.6

    print("  Prediction: \(prediction), Target: \(target)")

    // Compute distributed MSE
    let result = valueWithGradient(at: prediction) { pred in
        distributedMSELoss(localPrediction: pred, localTarget: target, on: comm)
    }

    print("  MSE Loss: \(result.value)")
    print("  Gradient w.r.t. prediction: \(result.gradient)")
    // Gradient: 2 * (prediction - target) / size
}

// MARK: - Main Program

/// Main entry point - runs all examples
@main
struct ScalarOperationsExample {
    static func main() {
        // Initialize MPI
        MPI_Init(nil, nil)

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   MessageDifferentiationKit - Scalar Operations Examples    ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        // Run examples
        example1_BasicAllreduce()
        example2_Broadcast()
        example3_DistributedMean()
        example4_Scan()
        example5_DistributedTraining()
        example6_Composition()
        example7_ReduceScatter()
        example8_Allgather()
        example9_DistributedMSE()

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║                     All Examples Complete                     ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        // Finalize MPI
        MPI_Finalize()
    }
}

// MARK: - Helper Documentation

/**
 # Understanding Gradients in Distributed Operations

 ## Allreduce Gradient
 ```
 Forward: output = Σᵢ inputᵢ
 Backward: ∂inputᵢ = ∂output * 1
 ```
 Each input contributes directly to the sum.

 ## Broadcast Gradient
 ```
 Forward: outputᵢ = input_root
 Backward:
   ∂input_root = Σᵢ ∂outputᵢ  (gradient flows to root)
   ∂input_non-root = 0          (non-root inputs don't affect output)
 ```

 ## Mean Gradient
 ```
 Forward: output = (Σᵢ inputᵢ) / n
 Backward: ∂inputᵢ = ∂output / n
 ```
 Gradient is divided by number of processes.

 ## Scan Gradient
 ```
 Forward: outputᵢ = Σⱼ₌₀ⁱ inputⱼ
 Backward: ∂inputᵢ = Σⱼ₌ᵢⁿ⁻¹ ∂outputⱼ
 ```
 Each input affects all later outputs.

 # Typical Usage Patterns

 ## Data Parallel Training
 ```swift
 // Each process computes gradients on local data
 let localGradient = computeGradient(localData)

 // Aggregate gradients across all processes
 let globalGradient = differentiableMean(localGradient, on: comm)

 // Update parameters (same on all processes)
 parameter -= learningRate * globalGradient
 ```

 ## Parameter Server Pattern
 ```swift
 // Root process (parameter server) broadcasts parameters
 let parameters = differentiableBroadcast(localParams, root: 0, on: comm)

 // Workers compute and send gradients back
 let gradients = differentiableReduce(localGradients, root: 0, on: comm)
 ```

 ## Pipeline Parallel Training
 ```swift
 // Pass activations forward through pipeline
 let activations = differentiableSendRecv(
     input, toRank: nextRank, fromRank: prevRank, on: comm
 )

 // Gradients flow backward automatically!
 ```
 */
