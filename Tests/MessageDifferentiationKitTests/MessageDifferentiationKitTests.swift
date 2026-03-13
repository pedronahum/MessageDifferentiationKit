import Testing
@testable import MessageDifferentiationKit
@testable import MPISwift

// Note: Many MPI operations require MPI_Init to be called first.
// These tests focus on API interface and non-MPI-dependent functionality.
// Full integration tests requiring MPI runtime should be in a separate test target.

@Test func testMPIDataTypeCreation() async throws {
    // Test that we can create MPI datatypes
    let doubleType = MPIDatatype.double
    let floatType = MPIDatatype.float
    let int32Type = MPIDatatype.int32
    let int64Type = MPIDatatype.int64

    // Verify they're not equal (basic sanity check)
    #expect(doubleType != floatType)
    #expect(int32Type != int64Type)
}

@Test func testMPIDataRepresentable() async throws {
    // Test that Swift types map to correct MPI datatypes
    #expect(Double.mpiDatatype == MPIDatatype.double)
    #expect(Float.mpiDatatype == MPIDatatype.float)
    #expect(Int32.mpiDatatype == MPIDatatype.int32)
    #expect(Int64.mpiDatatype == MPIDatatype.int64)
}

@Test func testMPIBufferDescriptor() async throws {
    var data: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0]

    // Create buffer descriptor within proper scope
    try data.withUnsafeMutableBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
            throw MPIError.invalidBuffer
        }
        let descriptor = MPIBufferDescriptor(pointer: baseAddress, count: buffer.count)

        #expect(descriptor.count == 5)
        #expect(descriptor.datatype == MPIDatatype.double)
        #expect(descriptor.largeCount == 5)
    }
}

@Test func testLargeCountDetection() async throws {
    let smallCount = 1000
    let largeCount = Int(Int32.max) + 1

    #expect(MPILargeCount.requiresLargeCount(smallCount) == false)
    #expect(MPILargeCount.requiresLargeCount(largeCount) == true)
    #expect(MPILargeCount.maxStandardCount == Int(Int32.max))
}

@Test func testMPIErrorHandling() async throws {
    // Test error code conversion
    let successError = MPIError(code: 0)
    #expect(successError == .success)

    let bufferError = MPIError(code: 1)
    #expect(bufferError == .invalidBuffer)

    // Test error descriptions
    #expect(MPIError.success.description.contains("successful"))
    #expect(MPIError.invalidBuffer.description.contains("buffer"))

    // Test unknown error
    let unknownError = MPIError(code: 999)
    if case .unknown(let code) = unknownError {
        #expect(code == 999)
    } else {
        Issue.record("Expected unknown error")
    }
}

@Test func testMPIOperations() async throws {
    // Test that we can access MPI operation constants
    let maxOp = MPIOperation.max
    let minOp = MPIOperation.min
    let sumOp = MPIOperation.sum

    // Basic sanity check - they should exist
    _ = maxOp
    _ = minOp
    _ = sumOp
}

@Test func testMPICommunicators() async throws {
    // Test that we can access communicator constants
    // Note: Actually using them requires MPI_Init
    let world = MPICommunicator.world
    let selfComm = MPICommunicator.self

    _ = world
    _ = selfComm
}

@Test func testLargeCountIsAvailable() async throws {
    // Test feature detection
    let available = MPILargeCount.isAvailable
    // Should be true or false, depending on MPI version
    _ = available
}
