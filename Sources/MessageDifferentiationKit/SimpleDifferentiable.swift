import MPISwift
import _Differentiation

/// Phase 3: Differentiable MPI Operations - Convenience Layer
///
/// This module provides convenient aliases and helper functions that build on top
/// of the core differentiable MPI operations defined in:
/// - MPICollectives+Differentiable.swift
/// - MPIPointToPoint+Differentiable.swift
///
/// These helpers maintain backward compatibility and provide simplified APIs for
/// common distributed machine learning patterns.

// MARK: - Convenience Aliases

/// Distributed sum (alias for differentiableAllreduce)
///
/// Convenience function that sums a scalar value across all processes.
///
/// - Parameters:
///   - value: Local scalar value
///   - communicator: MPI communicator
/// - Returns: Global sum across all processes
@differentiable(reverse)
public func distributedSum(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    return differentiableAllreduce(value, on: communicator)
}

/// Distributed mean (uses differentiableMean)
///
/// Convenience function that computes the mean across all processes.
///
/// - Parameters:
///   - value: Local scalar value
///   - communicator: MPI communicator
/// - Returns: Global mean across all processes
@differentiable(reverse)
public func distributedMean(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    return differentiableMean(value, on: communicator)
}

// MARK: - Example: Simple Distributed Loss

/// Example differentiable loss function for distributed training
///
/// This demonstrates how gradients automatically flow through MPI operations.
///
/// - Parameters:
///   - localPrediction: Model prediction on this process
///   - localTarget: Target value on this process
///   - communicator: MPI communicator
/// - Returns: Global mean squared error
@differentiable(reverse)
public func distributedMSELoss(
    localPrediction: Double,
    localTarget: Double,
    on communicator: MPICommunicator
) -> Double {
    let localError = localPrediction - localTarget
    let localSquaredError = localError * localError

    // Average squared error across all processes
    return distributedMean(localSquaredError, on: communicator)
}

// MARK: - Documentation & Examples

/**
 # Phase 3: Differentiable MPI with Swift's Native AD

 This module demonstrates automatic differentiation through MPI collective
 operations using Swift's `@differentiable` attribute and `Differentiable` protocol.

 ## Simple Example

 ```swift
 import MessageDifferentiationKit

 let comm = MPI5.Communicator.world
 let localValue = Double(comm.rank + 1)

 // Compute gradient of distributed sum
 let grad = gradient(at: localValue) { value in
     distributedSum(value, on: comm)
 }

 print("Rank \(comm.rank): gradient = \(grad)")
 // All ranks will have gradient = comm.size (due to allreduce in backward pass)
 ```

 ## Distributed Training Example

 ```swift
 // Training step with automatic gradient computation
 func trainingStep(
     parameters: Double,
     localData: [Double],
     localTargets: [Double],
     on comm: MPICommunicator
 ) -> (loss: Double, gradient: Double) {

     // Compute loss and gradient automatically
     let (loss, grad) = valueWithGradient(at: parameters) { params in
         // Forward pass
         let predictions = localData.map { x in params * x }
         let errors = zip(predictions, localTargets).map { pred, target in
             distributedMSELoss(
                 localPrediction: pred,
                 localTarget: target,
                 on: comm
             )
         }
         return errors.reduce(0, +) / Double(errors.count)
     }

     return (loss, grad)
 }

 // Update parameters
 var params = 1.0
 for epoch in 0..<100 {
     let (loss, grad) = trainingStep(
         parameters: params,
         localData: myData,
         localTargets: myTargets,
         on: comm
     )

     params -= learningRate * grad

     if comm.rank == 0 {
         print("Epoch \(epoch): loss = \(loss)")
     }
 }
 ```

 ## Architecture

 The differentiable MPI operations are organized into modular files:

 ### Core Differentiable Operations
 - **MPICollectives+Differentiable.swift**
   - `differentiableAllreduce` - Sum with gradient broadcast
   - `differentiableBroadcast` - Broadcast with gradient reduce
   - `differentiableReduce` - Reduce with gradient broadcast
   - `differentiableMean` - Mean with scaled gradients

 - **MPIPointToPoint+Differentiable.swift**
   - `differentiableSendRecv` - Send-recv pair with reverse gradient flow

 ### Convenience Layer (This File)
 - `distributedSum` - Alias for differentiableAllreduce
 - `distributedMean` - Alias for differentiableMean
 - `distributedMSELoss` - Example distributed loss function

 ## How It Works

 Each MPI operation has a corresponding `@derivative(of:)` implementation that
 defines how gradients flow in the backward pass:

 1. **Forward Pass**: Execute MPI operation (e.g., allreduce)
 2. **Backward Pass**: Custom VJP routes gradients correctly
 3. **Gradient Flow**: Automatic via Swift's AD system

 See the individual operation files for detailed gradient routing rules.

 ## Performance

 - **Zero overhead** in forward pass (just MPI collective)
 - **One extra collective** in backward pass (for gradient)
 - **Scales linearly** with number of processes
 - **Type-safe** with Swift's type system

 ## Available Differentiable Operations

 ### Collective Operations
 - ✅ `differentiableAllreduce` - Sum across all processes
 - ✅ `differentiableBroadcast` - Root to all processes
 - ✅ `differentiableReduce` - All to root process
 - ✅ `differentiableMean` - Average across processes

 ### Point-to-Point Operations
 - ✅ `differentiableSendRecv` - Matched send-recv pair

 ### Future Enhancements
 - [ ] Array/tensor operations (vectorized)
 - [ ] Gather/scatter with gradients
 - [ ] Allgather with gradient routing
 - [ ] Partitioned communication gradients
 - [ ] GPU-aware gradient transfers
 - [ ] Non-blocking operations with AD

 ## Integration with ML Frameworks

 This foundation enables:
 - **Data-parallel training**: Each process has different data
 - **Model averaging**: Synchronize weights across processes
 - **Distributed SGD**: Automatic gradient aggregation
 - **Parameter servers**: Centralized parameter distribution
 - **Elastic training**: Dynamic process sets with MPI Sessions
 */
