import Testing
@testable import MPISwift

/// Basic tests for MPISwift module
///
/// These tests verify that the MPISwift module is accessible
/// and basic types are available.

// MARK: - Module Accessibility

@Test func testMPISwiftModuleAccessible() async throws {
    // Verify MPISwift module can be imported
    // If this test runs, the module is accessible
    #expect(true)
}

// MARK: - Type Accessibility

@Test func testMPICommunicatorExists() async throws {
    // Verify MPICommunicator type is accessible
    _ = MPICommunicator.self
}

@Test func testMPIDatatypeExists() async throws {
    // Verify MPIDatatype type is accessible
    _ = MPIDatatype.self
}

@Test func testMPIOperationExists() async throws {
    // Verify MPIOperation type is accessible
    _ = MPIOperation.self
}

// MARK: - Constants

@Test func testMPIDatatypeConstants() async throws {
    // Verify MPIDatatype constants are accessible
    _ = MPIDatatype.double
    _ = MPIDatatype.float
    _ = MPIDatatype.int32
    _ = MPIDatatype.int64
}

@Test func testMPIOperationConstants() async throws {
    // Verify MPIOperation constants are accessible
    _ = MPIOperation.sum
    _ = MPIOperation.max
    _ = MPIOperation.min
}

// MARK: - Documentation

/**
 # MPISwift Tests

 These tests verify basic accessibility of the MPISwift module.

 ## Coverage

 - Module import verification
 - Type accessibility (MPICommunicator, MPIDatatype, MPIOperation)
 - Constant accessibility

 ## Note

 Full integration tests that require actual MPI execution are in
 the MessageDifferentiationKitTests suite, which tests the higher-level
 differentiable operations.
 */
