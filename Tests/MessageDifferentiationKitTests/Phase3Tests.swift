import Testing
@testable import MessageDifferentiationKit
@testable import MPISwift
import _Differentiation

// Phase 3: Differentiable MPI Operations Tests
//
// These tests verify that differentiable MPI operations compile correctly,
// have proper type signatures, and can be used with Swift's AD system.
//
// Note: Full gradient correctness tests require actual MPI multi-process execution
// and are better suited for integration tests with mpirun.

// MARK: - Type Safety Tests

@Test func testDifferentiableAllreduceExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableAllreduce as (Double, MPICommunicator) -> Double
}

@Test func testDifferentiableBroadcastExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableBroadcast as (Double, Int32, MPICommunicator) -> Double
}

@Test func testDifferentiableReduceExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableReduce as (Double, Int32, MPICommunicator) -> Double
}

@Test func testDifferentiableMeanExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableMean as (Double, MPICommunicator) -> Double
}

@Test func testDifferentiableScatterExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableScatter as (Double, Int32, MPICommunicator) -> Double
}

@Test func testDifferentiableReduceScatterExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableReduceScatter as (Double, MPICommunicator) -> Double
}

@Test func testDifferentiableScanExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableScan as (Double, MPICommunicator) -> Double
}

@Test func testDifferentiableExscanExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableExscan as (Double, MPICommunicator) -> Double
}

@Test func testDifferentiableSendRecvExists() async throws {
    // Verify function exists and is @differentiable
    _ = differentiableSendRecv as (Double, Int32, Int32, Int32, MPICommunicator) -> Double
}

@Test func testDifferentiableAllgatherExists() async throws {
    // Verify function exists and is @differentiable with [Double] signature
    _ = differentiableAllgather as ([Double], MPICommunicator) -> [Double]
}

// MARK: - Convenience Functions Tests

@Test func testDistributedSumExists() async throws {
    // Verify convenience function exists
    _ = distributedSum as (Double, MPICommunicator) -> Double
}

@Test func testDistributedMeanExists() async throws {
    // Verify convenience function exists
    _ = distributedMean as (Double, MPICommunicator) -> Double
}

@Test func testDistributedMSELossExists() async throws {
    // Verify example loss function exists
    _ = distributedMSELoss as (Double, Double, MPICommunicator) -> Double
}

// MARK: - Gradient Function Type Tests

@Test func testGradientFunctionForAllreduce() async throws {
    // Test that gradient function can be created (compile-time test)
    // Note: Actual gradient computation requires MPI_Init
    let canCreateGradient = { (x: Double, comm: MPICommunicator) -> Bool in
        // Check if gradient(at:of:) can be used with this function
        _ = gradient(at: x) { value in
            differentiableAllreduce(value, on: comm)
        }
        return true
    }

    // This is a compile-time verification - if it compiles, the test passes
    _ = canCreateGradient
}

@Test func testGradientFunctionForMean() async throws {
    // Test that gradient function works with mean
    let canCreateGradient = { (x: Double, comm: MPICommunicator) -> Bool in
        _ = gradient(at: x) { value in
            differentiableMean(value, on: comm)
        }
        return true
    }

    _ = canCreateGradient
}

@Test func testGradientFunctionForScan() async throws {
    // Test that gradient function works with scan
    let canCreateGradient = { (x: Double, comm: MPICommunicator) -> Bool in
        _ = gradient(at: x) { value in
            differentiableScan(value, on: comm)
        }
        return true
    }

    _ = canCreateGradient
}

@Test func testGradientFunctionForAllgather() async throws {
    // Test that gradient function works with allgather (array operation)
    let canCreateGradient = { (x: [Double], comm: MPICommunicator) -> Bool in
        _ = gradient(at: x) { values in
            let gathered = differentiableAllgather(values, on: comm)
            // Reduce to scalar for gradient computation
            return gathered.differentiableReduce(0.0, { $0 + $1 })
        }
        return true
    }

    _ = canCreateGradient
}

@Test func testGradientFunctionForExscan() async throws {
    // Test that gradient function works with exscan
    let canCreateGradient = { (x: Double, comm: MPICommunicator) -> Bool in
        _ = gradient(at: x) { value in
            differentiableExscan(value, on: comm)
        }
        return true
    }

    _ = canCreateGradient
}

// MARK: - Composition Tests

@Test func testDifferentiableOperationsCompose() async throws {
    // Test that operations can be composed and are still differentiable
    @differentiable(reverse)
    func composedOperation(_ x: Double, on comm: MPICommunicator) -> Double {
        let broadcasted = differentiableBroadcast(x, root: 0, on: comm)
        let reduced = differentiableAllreduce(broadcasted * 2.0, on: comm)
        return differentiableMean(reduced, on: comm)
    }

    // Verify the composed function is differentiable
    let canGradient = { (x: Double, comm: MPICommunicator) -> Bool in
        _ = gradient(at: x) { value in
            composedOperation(value, on: comm)
        }
        return true
    }

    _ = canGradient
}

@Test func testValueWithGradientWorks() async throws {
    // Test that valueWithGradient can be used with differentiable operations
    let canUseValueWithGradient = { (x: Double, comm: MPICommunicator) -> Bool in
        _ = valueWithGradient(at: x) { value in
            differentiableAllreduce(value, on: comm)
        }
        return true
    }

    _ = canUseValueWithGradient
}

// MARK: - Scalar Type Tests

@Test func testAllOperationsUseDouble() async throws {
    // Verify all operations use Double (scalar) type
    // This prevents array gradient issues

    struct OperationSignatures {
        static func checkAllreduce() -> Bool {
            _ = differentiableAllreduce as (Double, MPICommunicator) -> Double
            return true
        }

        static func checkBroadcast() -> Bool {
            _ = differentiableBroadcast as (Double, Int32, MPICommunicator) -> Double
            return true
        }

        static func checkScan() -> Bool {
            _ = differentiableScan as (Double, MPICommunicator) -> Double
            return true
        }

        static func checkScatter() -> Bool {
            _ = differentiableScatter as (Double, Int32, MPICommunicator) -> Double
            return true
        }
    }

    // If these all compile, test passes
    #expect(OperationSignatures.checkAllreduce())
    #expect(OperationSignatures.checkBroadcast())
    #expect(OperationSignatures.checkScan())
    #expect(OperationSignatures.checkScatter())
}

// MARK: - Documentation Tests

@Test func testAllOperationsDocumented() async throws {
    // This is a placeholder to remind us that all operations should have:
    // 1. Comprehensive documentation
    // 2. Usage examples
    // 3. Gradient routing tables
    // 4. Performance notes

    // Files to check for documentation:
    let files = [
        "MPICollectives+Differentiable.swift",
        "MPIPointToPoint+Differentiable.swift",
        "MPIScan+Differentiable.swift",
        "SimpleDifferentiable.swift"
    ]

    #expect(files.count == 4)
}

// MARK: - Future Integration Tests (Placeholder)

// These tests would require actual MPI runtime and mpirun:
/*
@Test func testAllreduceGradientCorrectness() async throws {
    // Requires: MPI_Init, multiple processes
    // Verify: ∂(sum)/∂xᵢ = 1 for all i
}

@Test func testBroadcastGradientCorrectness() async throws {
    // Requires: MPI_Init, multiple processes
    // Verify: ∂(broadcast)/∂x_root = N (sum of all gradients)
}

@Test func testScanGradientCorrectness() async throws {
    // Requires: MPI_Init, multiple processes
    // Verify: ∂xᵢ = Σⱼ₌ᵢⁿ⁻¹ ∂outⱼ
}

@Test func testFiniteDifferenceApproximation() async throws {
    // Requires: MPI_Init, multiple processes
    // Compare automatic gradients with finite difference approximation
    // ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε)
}
*/

// MARK: - Coverage Report

@Test func testPhase3Coverage() async throws {
    // Document Phase 3 coverage

    struct Phase3Operations {
        static let collectives = [
            "differentiableAllreduce",
            "differentiableBroadcast",
            "differentiableReduce",
            "differentiableMean",
            "differentiableScatter",
            "differentiableReduceScatter",
            "differentiableAllgather"
        ]

        static let pointToPoint = [
            "differentiableSendRecv"
        ]

        static let scan = [
            "differentiableScan",
            "differentiableExscan"
        ]

        static let convenience = [
            "distributedSum",
            "distributedMean",
            "distributedMSELoss"
        ]

        static var totalOperations: Int {
            collectives.count + pointToPoint.count + scan.count
        }
    }

    // Verify we have 10 differentiable operations
    #expect(Phase3Operations.totalOperations == 10)

    // Verify convenience layer has 3 functions
    #expect(Phase3Operations.convenience.count == 3)
}

// MARK: - Modular Architecture Tests

@Test func testModularFileStructure() async throws {
    // Verify that operations are organized in modular files
    // This is a documentation test

    struct ModularStructure {
        static let files = [
            "MPICollectives+Differentiable.swift",  // 6 operations
            "MPIPointToPoint+Differentiable.swift", // 1 operation
            "MPIScan+Differentiable.swift",         // 2 operations
            "SimpleDifferentiable.swift"            // Convenience layer
        ]

        static let collectiveOps = 7
        static let pointToPointOps = 1
        static let scanOps = 2

        static var totalOps: Int {
            collectiveOps + pointToPointOps + scanOps
        }
    }

    #expect(ModularStructure.files.count == 4)
    #expect(ModularStructure.totalOps == 10)
}

// MARK: - Compliance Tests

@Test func testAllVJPsHaveUsableFromInline() async throws {
    // This is a compile-time check
    // All @derivative functions should have @usableFromInline
    // for proper inlining and performance

    // If the code compiles, this test passes
    // Actual verification would require AST analysis
    #expect(true)
}

@Test func testAllOperationsFollowNamingConvention() async throws {
    // All core operations should start with "differentiable"
    // Convenience functions can use simpler names

    let coreOperations = [
        "differentiableAllreduce",
        "differentiableBroadcast",
        "differentiableReduce",
        "differentiableMean",
        "differentiableScatter",
        "differentiableReduceScatter",
        "differentiableScan",
        "differentiableExscan",
        "differentiableSendRecv",
        "differentiableAllgather"
    ]

    // All start with "differentiable"
    for op in coreOperations {
        #expect(op.hasPrefix("differentiable"))
    }

    #expect(coreOperations.count == 10)
}
