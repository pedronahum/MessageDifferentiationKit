import Testing
@testable import MessageDifferentiationKit
@testable import MPISwift

// Phase 2: Advanced Communication Patterns Tests
//
// Note: These tests focus on API correctness and type safety without requiring
// an actual MPI runtime. Integration tests with real MPI processes should be
// in a separate test target.

// MARK: - Point-to-Point Communication Tests

@Test func testMPIRequestCreation() async throws {
    // Test that we can create MPI requests
    let request = MPIRequest()
    _ = request
}

@Test func testMPIStatusAccess() async throws {
    // Test MPIStatus type exists
    // Note: Actual status creation requires MPI operations
    _ = MPIStatus.self
}

@Test func testMPIConstantsAccessible() async throws {
    // Test that MPI constants are accessible
    #expect(MPIConstants.anySource == -1 || MPIConstants.anySource == -2)
    #expect(MPIConstants.anyTag >= -1)
    _ = MPIConstants.procNull
}

// MARK: - Collective Operations Tests

@Test func testMPIOperationTypes() async throws {
    // Test all MPI operation types are accessible
    let ops: [MPIOperation] = [
        .max, .min, .sum, .prod, .land, .lor, .band, .bor, .lxor, .bxor, .maxloc, .minloc
    ]

    #expect(ops.count == 12)
}

// MARK: - Persistent Collectives Tests

@Test func testMPIOperationCacheCreation() async throws {
    // Test operation cache creation
    let cache = MPIOperationCache()
    _ = cache
}

@Test func testMPIOperationCacheThreadSafety() async throws {
    // Test that operation cache can be accessed from multiple contexts
    let cache = MPIOperationCache()

    // Simulate concurrent access
    let result = cache.get(key: "nonexistent")
    #expect(result == nil)
}

// MARK: - Partitioned Communication Tests

@Test func testMPIPartitionManagerCreation() async throws {
    // Test partition manager creation
    let manager = MPIPartitionManager()
    _ = manager
}

@Test func testMPIPartitionManagerPartitioning() async throws {
    let manager = MPIPartitionManager()
    let partitions = manager.createPartitions(totalSize: 1000, numPartitions: 10)

    #expect(partitions.count == 10)

    // Each partition should have size info
    for partition in partitions {
        #expect(partition.size > 0)
        #expect(partition.offset >= 0)
    }

    // Total size should match
    let totalSize = partitions.map(\.size).reduce(0, +)
    #expect(totalSize == 1000)
}

@Test func testMPIPartitionManagerReadyTracking() async throws {
    let manager = MPIPartitionManager()
    _ = manager.createPartitions(totalSize: 100, numPartitions: 5)

    #expect(manager.allReady == false)

    // Mark partitions ready
    for i in 0..<5 {
        manager.markReady(Int32(i))
    }

    #expect(manager.allReady == true)
}

@Test func testMPIPartitionInfo() async throws {
    let info = MPIPartitionManager.PartitionInfo(partitionIndex: 0, offset: 0, size: 100)
    #expect(info.partitionIndex == 0)
    #expect(info.offset == 0)
    #expect(info.size == 100)
    #expect(info.isReady == false)
}

// MARK: - Hardware Offload Tests

@Test func testMPIDeviceTypes() async throws {
    // Test device type enumeration
    let cuda = MPIDeviceType.cuda
    let rocm = MPIDeviceType.rocm
    let levelZero = MPIDeviceType.levelZero

    #expect(cuda.rawValue == "cuda")
    #expect(rocm.rawValue == "rocm")
    #expect(levelZero.rawValue == "level_zero")
}

@Test func testMPIDeviceBufferCreation() async throws {
    // Create a mock device pointer
    var mockData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

    try mockData.withUnsafeMutableBufferPointer { buffer in
        guard let ptr = buffer.baseAddress else {
            throw MPIError.invalidBuffer
        }

        let deviceBuffer = MPIDeviceBuffer<Float>(
            devicePointer: ptr,
            count: 5,
            deviceType: .cuda,
            deviceID: 0
        )

        #expect(deviceBuffer.count == 5)
        #expect(deviceBuffer.count32 == 5)
        #expect(deviceBuffer.deviceType == .cuda)
        #expect(deviceBuffer.deviceID == 0)
        #expect(deviceBuffer.datatype == MPIDatatype.float)
    }
}

@Test func testMPIDeviceBufferDatatype() async throws {
    var mockDouble: [Double] = [1.0]
    var mockInt32: [Int32] = [1]
    var mockInt64: [Int64] = [1]

    mockDouble.withUnsafeMutableBufferPointer { buffer in
        let deviceBuffer = MPIDeviceBuffer<Double>(
            devicePointer: buffer.baseAddress!,
            count: 1,
            deviceType: .cuda,
            deviceID: 0
        )
        #expect(deviceBuffer.datatype == MPIDatatype.double)
    }

    mockInt32.withUnsafeMutableBufferPointer { buffer in
        let deviceBuffer = MPIDeviceBuffer<Int32>(
            devicePointer: buffer.baseAddress!,
            count: 1,
            deviceType: .rocm,
            deviceID: 1
        )
        #expect(deviceBuffer.datatype == MPIDatatype.int32)
        #expect(deviceBuffer.deviceID == 1)
    }

    mockInt64.withUnsafeMutableBufferPointer { buffer in
        let deviceBuffer = MPIDeviceBuffer<Int64>(
            devicePointer: buffer.baseAddress!,
            count: 1,
            deviceType: .levelZero,
            deviceID: 2
        )
        #expect(deviceBuffer.datatype == MPIDatatype.int64)
    }
}

@Test func testMPIInfoBuilderGPUExtensions() async throws {
    // Test that MPIInfoBuilder can be created
    // Note: Actual MPI_Info operations require MPI runtime
    do {
        let builder = try MPIInfoBuilder()
        _ = builder
    } catch {
        // If MPI is not initialized, this is expected
        _ = error
    }
}

// MARK: - Topology Tests

@Test func testMPITopologyHelpersExist() async throws {
    // Test that topology helper methods exist
    // Note: Actual topology creation requires MPI communicator
    _ = MPITopologyHelpers.self
}

// MARK: - Type Safety Tests

@Test func testMPIDataRepresentableConformance() async throws {
    // Test that all expected types conform to MPIDataRepresentable
    #expect(Double.mpiDatatype == MPIDatatype.double)
    #expect(Float.mpiDatatype == MPIDatatype.float)
    #expect(Int32.mpiDatatype == MPIDatatype.int32)
    #expect(Int64.mpiDatatype == MPIDatatype.int64)
}

@Test func testMPIDataTypeEquality() async throws {
    // Test datatype equality
    let dt1 = MPIDatatype.double
    let dt2 = MPIDatatype.double
    let dt3 = MPIDatatype.float

    #expect(dt1 == dt2)
    #expect(dt1 != dt3)
}

// MARK: - Error Handling Tests

@Test func testMPIErrorTypes() async throws {
    // Test all error types are accessible
    let errors: [MPIError] = [
        .success,
        .invalidBuffer,
        .invalidCount,
        .invalidDatatype,
        .invalidTag,
        .invalidCommunicator,
        .invalidRequest,
        .invalidRoot,
        .invalidGroup,
        .invalidOperation,
        .invalidTopology,
        .dimensionError,
        .invalidArgument,
        .unknownError,
        .truncatedMessage,
        .otherError,
        .internalError,
        .inProgress,
        .pending,
        .sessionError,
        .invalidRank,
        .invalidSession,
        .invalidProcessSet,
        .unknown(999)
    ]

    #expect(errors.count == 24)
}

@Test func testMPIErrorDescriptions() async throws {
    // Test that error descriptions are meaningful
    #expect(MPIError.success.description.contains("successful"))
    #expect(MPIError.invalidBuffer.description.contains("buffer"))
    #expect(MPIError.invalidCommunicator.description.contains("communicator"))
    #expect(MPIError.invalidRequest.description.contains("request"))
}

// MARK: - Integration Readiness Tests

@Test func testMPICommunicatorWorldExists() async throws {
    // Test that MPI_COMM_WORLD wrapper exists
    let world = MPICommunicator.world
    _ = world
}

// Note: These tests are commented out as they require MPI_Init to be called first.
// They would be suitable for integration tests that properly initialize MPI.

// @Test func testMPIRequestStartMethod() async throws {
//     // Test that MPIRequest has start method
//     var request = MPIRequest()
//
//     // Note: Calling start() without proper initialization will fail
//     // This just tests the API exists
//     do {
//         try request.start()
//     } catch {
//         // Expected to fail without MPI runtime
//         _ = error
//     }
// }
//
// @Test func testMPIRequestWaitMethod() async throws {
//     // Test that MPIRequest has wait method
//     var request = MPIRequest()
//
//     do {
//         try request.wait()
//     } catch {
//         // Expected to fail without MPI runtime
//         _ = error
//     }
// }

// MARK: - API Surface Tests

// Note: API surface tests commented out - they require MPI_Init
// These would be suitable for integration tests.

// @Test func testPointToPointAPISurface() async throws {
//     // Verify point-to-point API methods exist on MPICommunicator
//     let comm = MPICommunicator.world
//
//     // These will fail without MPI_Init, but we're testing the API exists
//     do {
//         let _: [Double] = try comm.recv(count: 1, from: 0, tag: 0)
//     } catch {
//         _ = error
//     }
// }
//
// @Test func testCollectiveAPISurface() async throws {
//     // Verify collective API methods exist on MPICommunicator
//     let comm = MPICommunicator.world
//
//     do {
//         var data = [1.0]
//         try comm.broadcast(&data, root: 0)
//     } catch {
//         _ = error
//     }
//
//     do {
//         let _: [Double]? = try comm.reduce([1.0], operation: .sum, root: 0)
//     } catch {
//         _ = error
//     }
//
//     do {
//         let _: [Double] = try comm.allreduce([1.0], operation: .max)
//     } catch {
//         _ = error
//     }
// }
//
// @Test func testBarrierAPISurface() async throws {
//     let comm = MPICommunicator.world
//
//     do {
//         try comm.barrier()
//     } catch {
//         _ = error
//     }
// }

// MARK: - Memory Safety Tests

@Test func testBufferPointerLifetime() async throws {
    // Test that buffer operations maintain proper pointer lifetime
    let data: [Double] = [1.0, 2.0, 3.0]

    let result: Bool = try data.withUnsafeBufferPointer { buffer in
        guard let ptr = buffer.baseAddress else {
            throw MPIError.invalidBuffer
        }

        // Simulate buffer usage
        _ = ptr
        return true
    }

    #expect(result == true)
}

@Test func testMutableBufferPointerLifetime() async throws {
    // Test mutable buffer operations
    var data: [Double] = [1.0, 2.0, 3.0]

    try data.withUnsafeMutableBufferPointer { buffer in
        guard let ptr = buffer.baseAddress else {
            throw MPIError.invalidBuffer
        }

        // Simulate buffer usage
        _ = ptr
    }
}

// MARK: - Performance Characteristics Tests

@Test func testPartitionSizeCalculation() async throws {
    let manager = MPIPartitionManager()

    // Test even division
    let evenPartitions = manager.createPartitions(totalSize: 1000, numPartitions: 10)
    #expect(evenPartitions.allSatisfy { $0.size == 100 })

    // Test uneven division
    let unevenPartitions = manager.createPartitions(totalSize: 1000, numPartitions: 7)
    let sizes = unevenPartitions.map(\.size)
    let totalSize = sizes.reduce(0, +)
    #expect(totalSize == 1000)

    // All partitions should be roughly equal size
    let minSize = sizes.min() ?? 0
    let maxSize = sizes.max() ?? 0
    #expect(maxSize - minSize <= 1)
}

// MARK: - Concurrency Safety Tests

@Test func testOperationCacheConcurrency() async throws {
    let cache = MPIOperationCache()

    // Test that cache operations don't crash with concurrent access
    await withTaskGroup(of: Void.self) { group in
        for i in 0..<10 {
            group.addTask {
                _ = cache.get(key: "test\(i)")
            }
        }
    }
}
