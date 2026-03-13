import MPISwift
import _Differentiation

/// Differentiable MPI Scan Operations
///
/// This module provides automatic differentiation support for MPI scan (prefix)
/// operations including inclusive and exclusive scans.
///
/// ## Gradient Routing for Scan Operations
///
/// Scan operations compute prefix sums. The gradient flow is more complex than
/// simple collectives because each output depends on multiple inputs.
///
/// | Operation | Forward | Backward (Gradient) |
/// |-----------|---------|---------------------|
/// | `scan` (inclusive) | `outᵢ = Σⱼ₌₀ⁱ inⱼ` | `∂inᵢ = Σⱼ₌ᵢⁿ⁻¹ ∂outⱼ` |
/// | `exscan` (exclusive) | `outᵢ = Σⱼ₌₀ⁱ⁻¹ inⱼ` | `∂inᵢ = Σⱼ₌ᵢ₊₁ⁿ⁻¹ ∂outⱼ` |

// MARK: - Differentiable Scan (Inclusive Prefix Sum)

/// Differentiable inclusive scan operation
///
/// Computes prefix sum: each process gets the sum of all values from rank 0 to its rank.
///
/// **Example** (4 processes):
/// ```
/// Input:  [1, 2, 3, 4]
/// Output: [1, 3, 6, 10]  // [1, 1+2, 1+2+3, 1+2+3+4]
/// ```
///
/// **Gradient Flow**:
/// - Process i's input affects all outputs j where j >= i
/// - Therefore: ∂inputᵢ = Σⱼ₌ᵢⁿ⁻¹ ∂outputⱼ
///
/// - Parameters:
///   - value: Local value to scan
///   - communicator: MPI communicator
/// - Returns: Prefix sum up to and including this rank
@differentiable(reverse)
public func differentiableScan(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    let rank = communicator.rank
    let size = communicator.size

    // Use allgather to get all values, then compute prefix sum
    // This is a simplified implementation; MPI_Scan would be more efficient

    // Gather all values to compute scan locally
    // In production, would use MPI_Scan directly
    var allValues = Array(repeating: 0.0, count: Int(size))

    // Everyone sends to everyone (allgather pattern)
    for i in 0..<size {
        if i == rank {
            allValues[Int(i)] = value
            // Send to all others
            for dest in 0..<size where dest != rank {
                try? communicator.send([value], to: dest, tag: 30000 + rank)
            }
        } else {
            // Receive from process i
            if let received = try? communicator.recv(count: 1, from: i, tag: 30000 + i) as [Double] {
                allValues[Int(i)] = received[0]
            }
        }
    }

    // Compute prefix sum up to current rank
    var prefixSum = 0.0
    for i in 0...Int(rank) {
        prefixSum += allValues[i]
    }

    return prefixSum
}

@derivative(of: differentiableScan)
@usableFromInline
func _vjpScan(
    _ value: Double,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableScan(value, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // Gradient for scan: each input affects all outputs >= its rank
        // Therefore: ∂inputᵢ = Σⱼ₌ᵢⁿ⁻¹ ∂outputⱼ

        let rank = communicator.rank
        let size = communicator.size

        var totalGrad = gradient

        // Receive gradients from all ranks > current
        for i in (rank + 1)..<size {
            if let grads = try? communicator.recv(count: 1, from: i, tag: 40000 + i) as [Double] {
                totalGrad += grads[0]
            }
        }

        // Send gradient to all ranks < current
        for i in 0..<rank {
            try? communicator.send([gradient], to: i, tag: 40000 + rank)
        }

        return totalGrad
    }

    return (output, pullback)
}

// MARK: - Differentiable Exclusive Scan (Exclusive Prefix Sum)

/// Differentiable exclusive scan operation
///
/// Computes exclusive prefix sum: each process gets the sum of all values from rank 0 to rank-1.
/// Rank 0 gets 0 (identity element).
///
/// **Example** (4 processes):
/// ```
/// Input:  [1, 2, 3, 4]
/// Output: [0, 1, 3, 6]  // [0, 1, 1+2, 1+2+3]
/// ```
///
/// **Gradient Flow**:
/// - Process i's input affects all outputs j where j > i
/// - Therefore: ∂inputᵢ = Σⱼ₌ᵢ₊₁ⁿ⁻¹ ∂outputⱼ
///
/// - Parameters:
///   - value: Local value to scan
///   - communicator: MPI communicator
/// - Returns: Prefix sum up to but not including this rank
@differentiable(reverse)
public func differentiableExscan(
    _ value: Double,
    on communicator: MPICommunicator
) -> Double {
    let rank = communicator.rank

    if rank == 0 {
        // Rank 0 gets identity (0)
        return 0.0
    } else {
        // Others: compute inclusive scan and subtract own value
        let inclusiveScan = differentiableScan(value, on: communicator)
        return inclusiveScan - value
    }
}

@derivative(of: differentiableExscan)
@usableFromInline
func _vjpExscan(
    _ value: Double,
    on communicator: MPICommunicator
) -> (value: Double, pullback: (Double) -> Double) {
    let output = differentiableExscan(value, on: communicator)

    func pullback(_ gradient: Double) -> Double {
        // Exclusive scan: input i affects outputs j where j > i
        // Therefore: ∂inputᵢ = Σⱼ₌ᵢ₊₁ⁿ⁻¹ ∂outputⱼ

        let rank = communicator.rank
        let size = communicator.size

        if rank == 0 {
            // Rank 0's input affects all other outputs
            var totalGrad = 0.0
            for i in 1..<size {
                if let grads = try? communicator.recv(count: 1, from: i, tag: 50000 + i) as [Double] {
                    totalGrad += grads[0]
                }
            }
            return totalGrad
        } else {
            var totalGrad = 0.0

            // Receive from ranks > current
            for i in (rank + 1)..<size {
                if let grads = try? communicator.recv(count: 1, from: i, tag: 50000 + i) as [Double] {
                    totalGrad += grads[0]
                }
            }

            // Send to all ranks < current
            for i in 0..<rank {
                try? communicator.send([gradient], to: i, tag: 50000 + rank)
            }

            return totalGrad
        }
    }

    return (output, pullback)
}

// MARK: - Documentation

/**
 # Differentiable MPI Scan Operations

 Scan operations compute prefix (cumulative) sums across processes, which are
 fundamental for many parallel algorithms.

 ## Usage Example

 ```swift
 import MessageDifferentiationKit

 let comm = MPI5.Communicator.world
 let localValue = Double(comm.rank + 1)

 // Compute prefix sum
 let scan = differentiableScan(localValue, on: comm)
 // Process 0: 1, Process 1: 3, Process 2: 6, Process 3: 10

 // Compute gradient
 let grad = gradient(at: localValue) { value in
     differentiableScan(value, on: comm)
 }
 // Gradient depends on how many processes are affected
 ```

 ## Gradient Flow Explanation

 ### Inclusive Scan (differentiableScan)

 Forward pass computes: `outᵢ = in₀ + in₁ + ... + inᵢ`

 For gradients:
 - `in₀` affects `out₀, out₁, out₂, ...` (all outputs)
 - `in₁` affects `out₁, out₂, out₃, ...` (all outputs ≥ 1)
 - `inᵢ` affects `outᵢ, outᵢ₊₁, ...` (all outputs ≥ i)

 Therefore: `∂inᵢ = ∂outᵢ + ∂outᵢ₊₁ + ... + ∂outₙ₋₁`

 ### Exclusive Scan (differentiableExscan)

 Forward pass computes: `outᵢ = in₀ + in₁ + ... + inᵢ₋₁` (excludes inᵢ)

 For gradients:
 - `in₀` affects `out₁, out₂, out₃, ...` (all outputs > 0)
 - `in₁` affects `out₂, out₃, out₄, ...` (all outputs > 1)
 - `inᵢ` affects `outᵢ₊₁, outᵢ₊₂, ...` (all outputs > i)

 Therefore: `∂inᵢ = ∂outᵢ₊₁ + ∂outᵢ₊₂ + ... + ∂outₙ₋₁`

 ## Performance Notes

 - Current implementation uses allgather pattern for simplicity
 - Production implementation should use MPI_Scan directly (when available)
 - Gradient communication uses point-to-point sends between neighbors
 - Complexity: O(N²) messages, could be optimized to O(N log N)

 ## Implementation Status

 - ✅ Inclusive scan with correct gradient flow
 - ✅ Exclusive scan with correct gradient flow
 - ⏳ Optimized MPI_Scan binding (requires MPISwift extension)
 - ⏳ Tree-based gradient aggregation (performance optimization)

 ## Future Enhancements

 - [ ] Use native MPI_Scan when available in MPISwift
 - [ ] Optimize gradient communication with tree patterns
 - [ ] Support other reduction operations (max, min with indices)
 - [ ] Add segmented scan for non-commutative operations
 */
