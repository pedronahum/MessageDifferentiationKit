import MPISwift
import _Differentiation

/// Differentiable MPI Collective Operations
///
/// This module provides automatic differentiation support for MPI collective operations
/// including allreduce, broadcast, reduce, gather, and allgather.
///
/// ## Gradient Routing Rules
///
/// | Operation | Forward | Backward (Gradient) |
/// |-----------|---------|---------------------|
/// | `allreduce(sum)` | `y = Σ xᵢ` | `∂xᵢ = ∂y` (broadcast gradient) |
/// | `broadcast` | `yᵢ = x_root` | `∂x_root = Σ ∂yᵢ` (reduce gradients to root) |
/// | `reduce(sum)` | `y_root = Σ xᵢ` | `∂xᵢ = ∂y_root` (broadcast from root) |
/// | `allgather` | `y = concat(x₀, x₁, ..., xₙ)` | `∂xᵢ = y_grad[i*k..(i+1)*k]` (slice gradients) |

// MARK: - Differentiable Allreduce

/// Differentiable allreduce with SUM operation
///
/// Sums values across all processes and returns the result to all processes.
/// In the backward pass, gradients are broadcast to all processes (since all
/// processes receive the same output, they all get the same gradient).
///
/// - Parameters:
///   - value: Local scalar value to reduce
///   - communicator: MPI communicator
/// - Returns: Global sum across all processes
@differentiable(reverse)
public func differentiableAllreduce(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    do {
        let result = try communicator.allreduce([value], operation: .sum)
        return result[0]
    } catch {
        // CRITICAL ERROR: MPI operation failed!
        print("ERROR in differentiableAllreduce: \(error)")
        fatalError("MPI allreduce failed: \(error)")
    }
}

@derivative(of: differentiableAllreduce)
@usableFromInline
func _vjpAllreduce(
    _ value: Double,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableAllreduce(value, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // For allreduce with SUM: each input contributes to the sum
        // In backward pass, each process should receive the full gradient
        // since ∂(Σxᵢ)/∂xⱼ = 1 for all j
        // The gradient is already the same on all processes (from forward pass output)
        return gradient
    }

    return (output, pullback)
}

// MARK: - Differentiable Broadcast

/// Differentiable broadcast operation
///
/// Broadcasts a value from the root process to all other processes.
/// In the backward pass, gradients from all processes are reduced back to root.
///
/// - Parameters:
///   - value: Value to broadcast (meaningful on root) or placeholder (on others)
///   - root: Rank of the root process
///   - communicator: MPI communicator
/// - Returns: Broadcasted value on all processes
@differentiable(reverse)
public func differentiableBroadcast(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> Double {
    do {
        var buffer = [value]
        try communicator.broadcast(&buffer, root: root)
        return buffer[0]
    } catch {
        print("ERROR in differentiableBroadcast: \(error)")
        fatalError("MPI broadcast failed: \(error)")
    }
}

@derivative(of: differentiableBroadcast)
@usableFromInline
func _vjpBroadcast(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableBroadcast(value, root: root, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // For broadcast: y_i = x_root for all i
        // Therefore: ∂L/∂x_root = Σᵢ ∂L/∂y_i
        // We need to sum all gradients and send to root
        do {
            let result = try communicator.reduce([gradient], operation: .sum, root: root)
            // Only root gets the summed gradient, others get 0
            if communicator.rank == root {
                return result?[0] ?? 0.0
            } else {
                return 0.0
            }
        } catch {
            return gradient
        }
    }

    return (output, pullback)
}

// MARK: - Differentiable Reduce

/// Differentiable reduce operation with SUM
///
/// Reduces values from all processes to the root process.
/// In the backward pass, the gradient from root is broadcast to all processes.
///
/// - Parameters:
///   - value: Local value to reduce
///   - root: Rank of the root process
///   - communicator: MPI communicator
/// - Returns: Reduced sum (on root) or 0.0 (on other processes)
@differentiable(reverse)
public func differentiableReduce(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> Double {
    do {
        let result = try communicator.reduce([value], operation: .sum, root: root)
        return result?[0] ?? 0.0
    } catch {
        print("ERROR in differentiableReduce: \(error)")
        fatalError("MPI reduce failed: \(error)")
    }
}

@derivative(of: differentiableReduce)
@usableFromInline
func _vjpReduce(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableReduce(value, root: root, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // For reduce to root: y_root = Σᵢ xᵢ
        // Therefore: ∂L/∂xᵢ = ∂L/∂y_root for all i
        // Broadcast gradient from root to all processes
        do {
            var buffer = [gradient]
            try communicator.broadcast(&buffer, root: root)
            return buffer[0]
        } catch {
            return gradient
        }
    }

    return (output, pullback)
}

// MARK: - Differentiable Mean

/// Differentiable distributed mean
///
/// Computes the mean of values across all processes using allreduce.
/// Automatically handles gradient scaling in the backward pass.
///
/// - Parameters:
///   - value: Local value
///   - communicator: MPI communicator
/// - Returns: Global mean across all processes
@differentiable(reverse)
public func differentiableMean(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    let sum = differentiableAllreduce(value, on: communicator)
    return sum / Double(communicator.size)
}

@derivative(of: differentiableMean)
@usableFromInline
func _vjpMean(
    _ value: Double,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableMean(value, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // For mean: y = (Σᵢ xᵢ) / N
        // Therefore: ∂L/∂xⱼ = ∂L/∂y * (1/N)
        // Scale the gradient by 1/N
        return gradient / Double(communicator.size)
    }

    return (output, pullback)
}

// MARK: - Differentiable Scatter (Scalar)

/// Differentiable scatter operation (scalar version)
///
/// Root process distributes different scalar values to each process.
/// Each process receives one scalar value based on its rank.
///
/// **Note**: This is a simplified scalar version. For array scatter,
/// use the underlying MPISwift scatter with manual gradient handling
/// until tensor support is added in Phase 4.
///
/// - Parameters:
///   - value: Local value (meaningful only on root process)
///   - root: Rank of the root process
///   - communicator: MPI communicator
/// - Returns: The scattered value for this process
@differentiable(reverse)
public func differentiableScatter(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> Double {
    let rank = communicator.rank

    do {
        if rank == root {
            // Root sends value to itself (no-op, just return)
            // In a real implementation, root would have an array of values
            // For scalar version, all processes get the same value
            var buffer = [value]
            try communicator.broadcast(&buffer, root: root)
            return buffer[0]
        } else {
            // Non-root receives via broadcast (simplified)
            var buffer = [0.0]
            try communicator.broadcast(&buffer, root: root)
            return buffer[0]
        }
    } catch {
        return value
    }
}

@derivative(of: differentiableScatter)
@usableFromInline
func _vjpScatter(
    _ value: Double,
    root: Int32,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableScatter(value, root: root, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // For scatter: gradients gather back to root
        // Each process sends its gradient to root
        let rank = communicator.rank

        do {
            if rank == root {
                // Root accumulates gradients from all processes
                let result = try communicator.reduce([gradient], operation: .sum, root: root)
                return result?[0] ?? gradient
            } else {
                // Non-root sends gradient to root
                _ = try communicator.reduce([gradient], operation: .sum, root: root)
                return 0.0
            }
        } catch {
            return gradient
        }
    }

    return (output, pullback)
}

// MARK: - Differentiable Reduce-Scatter

/// Differentiable reduce-scatter operation
///
/// Combines reduction and scattering: reduces values across all processes,
/// then scatters the result so each process gets a portion.
///
/// **Simplified Implementation**: Uses allreduce + local division.
/// Full reduce-scatter with arrays requires Phase 4 tensor support.
///
/// - Parameters:
///   - value: Local value to reduce
///   - communicator: MPI communicator
/// - Returns: Reduced and scattered result for this process
@differentiable(reverse)
public func differentiableReduceScatter(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    // Simplified: sum all values and divide by size
    // Each process gets equal share
    let sum = differentiableAllreduce(value, on: communicator)
    return sum / Double(communicator.size)
}

@derivative(of: differentiableReduceScatter)
@usableFromInline
func _vjpReduceScatter(
    _ value: Double,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableReduceScatter(value, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // Gradient: inverse of reduce-scatter is allgather
        // Each process's gradient contributes to all inputs
        return gradient / Double(communicator.size)
    }

    return (output, pullback)
}

// MARK: - Differentiable Allgather (Array)

/// Differentiable allgather operation for arrays
///
/// Gathers arrays from all processes into one concatenated array on every rank.
/// Each process contributes an array of the same size, and all processes receive
/// the concatenation of all contributions ordered by rank.
///
/// **Example** (3 processes, each contributing 2 elements):
/// ```
/// Rank 0: [1.0, 2.0]
/// Rank 1: [3.0, 4.0]
/// Rank 2: [5.0, 6.0]
/// Output (all ranks): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ```
///
/// **Gradient Flow**:
/// - Forward: `y = concat(x₀, x₁, ..., xₙ₋₁)` where `xᵢ` is rank i's array
/// - Backward: `∂xᵢ = y_grad[i*k ..< (i+1)*k]` (each rank gets its slice of the gradient)
///
/// - Parameters:
///   - values: Local array of values to gather
///   - communicator: MPI communicator
/// - Returns: Concatenated array from all processes (size = values.count * worldSize)
@differentiable(reverse)
public func differentiableAllgather(
    _ values: [Double],
    on communicator: MPICommunicator
) -> [Double] {
    do {
        let result = try communicator.allgather(values)
        return result
    } catch {
        print("ERROR in differentiableAllgather: \(error)")
        fatalError("MPI allgather failed: \(error)")
    }
}

@derivative(of: differentiableAllgather)
@usableFromInline
func _vjpAllgather(
    _ values: [Double],
    on communicator: MPICommunicator
) -> (value: [Double], pullback: (Array<Double>.DifferentiableView) -> Array<Double>.DifferentiableView) {
    let output = differentiableAllgather(values, on: communicator)
    let rank = communicator.rank
    let localCount = values.count

    func pullback(_ gradient: Array<Double>.DifferentiableView) -> Array<Double>.DifferentiableView {
        // Adjoint of concatenation (allgather) is slicing (scatter):
        // Each rank extracts its own slice from the full gradient
        let start = Int(rank) * localCount
        let end = start + localCount

        if gradient.base.isEmpty {
            return Array<Double>.DifferentiableView([Double](repeating: 0.0, count: localCount))
        }

        let localGradient = Array(gradient.base[start..<end])
        return Array<Double>.DifferentiableView(localGradient)
    }

    return (output, pullback)
}

// MARK: - Documentation

/**
 # Differentiable MPI Collectives

 This module provides `@differentiable` wrappers for MPI collective operations,
 enabling automatic differentiation through distributed computations.

 ## Usage Example

 ```swift
 import MessageDifferentiationKit

 let comm = MPI5.Communicator.world
 let localValue = Double(comm.rank + 1)

 // Compute gradient of distributed sum
 let grad = gradient(at: localValue) { value in
     differentiableAllreduce(value, on: comm)
 }

 // grad will be 1.0 on all processes (since ∂sum/∂xᵢ = 1)
 ```

 ## Gradient Semantics

 ### Allreduce (SUM)
 - **Forward**: `output = Σᵢ inputᵢ` (same on all processes)
 - **Backward**: `∂inputᵢ = ∂output` (gradient flows to all)

 ### Broadcast
 - **Forward**: `outputᵢ = input_root` (root's value to all)
 - **Backward**: `∂input_root = Σᵢ ∂outputᵢ` (gradients sum to root)

 ### Reduce (SUM)
 - **Forward**: `output_root = Σᵢ inputᵢ` (only root receives)
 - **Backward**: `∂inputᵢ = ∂output_root` (root's gradient to all)

 ### Mean
 - **Forward**: `output = (Σᵢ inputᵢ) / N`
 - **Backward**: `∂inputᵢ = ∂output / N` (scaled gradient)

 ### Scatter (Scalar)
 - **Forward**: `outputᵢ = input_root` (simplified: broadcast from root)
 - **Backward**: `∂input_root = Σᵢ ∂outputᵢ` (gradients gather to root)

 ### Reduce-Scatter
 - **Forward**: `output = allreduce(input) / N`
 - **Backward**: `∂input = ∂output / N` (scaled gradient)

 ### Allgather (Array)
 - **Forward**: `output = concat(input₀, input₁, ..., inputₙ₋₁)` (all processes)
 - **Backward**: `∂inputᵢ = output_grad[i*k..(i+1)*k]` (each rank slices its gradient)

 ## Performance Notes

 - Each differentiable operation adds one additional MPI collective in backward pass
 - Gradients are communicated with the same pattern as forward data
 - Zero-copy where possible, minimal allocation overhead
 - Type-safe and compiler-verified
 */
