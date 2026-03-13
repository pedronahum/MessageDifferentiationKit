import CDLPack
import MPISwift

/// Bridge between DLPack tensors and MPI operations
///
/// DLPackTensor provides zero-copy integration with ML frameworks:
/// - **PyTorch**: `tensor.to_dlpack()` → MPI → `torch.from_dlpack()`
/// - **TensorFlow**: `tf.experimental.dlpack.to_dlpack()` → MPI
/// - **JAX**: `jax.dlpack.to_dlpack()` → MPI
///
/// ## Zero-Copy Operation
///
/// DLPack tensors share memory directly with ML frameworks:
/// ```
/// PyTorch Tensor Memory ←→ DLPack ←→ MPI Operation
///         (same memory, no copies!)
/// ```
///
/// ## GPU Support
///
/// Automatically detects and uses GPU memory:
/// - CPU tensors: Regular MPI operations
/// - CUDA tensors: GPU-aware MPI (MPI 5.0 hardware offload)
/// - Unified memory: Automatic handling
///
/// ## Usage Example
///
/// ```swift
/// // From Python (PyTorch):
/// // dlpack_capsule = tensor.to_dlpack()
/// // Pass dlpack_capsule to Swift
///
/// let dlTensor = DLPackTensor(dlpackPointer)
/// let result = differentiableAllreduceDLPack(dlTensor, on: comm)
///
/// // Back to Python:
/// // result_tensor = torch.from_dlpack(result)
/// ```
public struct DLPackTensor {
    /// Pointer to DLTensor structure
    let handle: UnsafeMutablePointer<DLTensor>

    /// Initialize from DLTensor pointer
    public init(_ dlTensor: UnsafeMutablePointer<DLTensor>) {
        self.handle = dlTensor
    }

    // MARK: - Tensor Properties

    /// Raw data pointer for MPI operations
    public var dataPointer: UnsafeMutableRawPointer {
        handle.pointee.data
    }

    /// Total number of elements in tensor
    public var elementCount: Int {
        var count = 1
        for i in 0..<Int(handle.pointee.ndim) {
            count *= Int(handle.pointee.shape[i])
        }
        return count
    }

    /// Size of data in bytes
    public var byteSize: Int {
        let dtype = handle.pointee.dtype
        let bitsPerElement = Int(dtype.bits) * Int(dtype.lanes)
        let bytesPerElement = (bitsPerElement + 7) / 8
        return elementCount * bytesPerElement
    }

    /// Number of dimensions
    public var ndim: Int {
        Int(handle.pointee.ndim)
    }

    /// Shape of tensor
    public var shape: [Int64] {
        guard let shapePtr = handle.pointee.shape else { return [] }
        return (0..<ndim).map { shapePtr[$0] }
    }

    // MARK: - Device Information

    /// Device type (CPU, CUDA, etc.)
    public var deviceType: DLDeviceType {
        handle.pointee.device.device_type
    }

    /// Device ID (e.g., GPU index)
    public var deviceID: Int32 {
        handle.pointee.device.device_id
    }

    /// Check if tensor is on CPU
    public var isCPU: Bool {
        deviceType == kDLCPU
    }

    /// Check if tensor is on CUDA GPU
    public var isGPU: Bool {
        deviceType == kDLCUDA || deviceType == kDLCUDAManaged
    }

    /// Check if tensor is in CUDA managed (unified) memory
    public var isCUDAManaged: Bool {
        deviceType == kDLCUDAManaged
    }

    // MARK: - Data Type Information

    /// Data type code (int, uint, float, etc.)
    public var typeCode: DLDataTypeCode {
        DLDataTypeCode(rawValue: UInt32(handle.pointee.dtype.code))
    }

    /// Number of bits per element
    public var bits: UInt8 {
        handle.pointee.dtype.bits
    }

    /// Number of lanes (for vector types)
    public var lanes: UInt16 {
        handle.pointee.dtype.lanes
    }

    /// Check if tensor is float64 (Double)
    public var isFloat64: Bool {
        typeCode == kDLFloat && bits == 64 && lanes == 1
    }

    /// Check if tensor is float32
    public var isFloat32: Bool {
        typeCode == kDLFloat && bits == 32 && lanes == 1
    }

    // MARK: - MPI Datatype Mapping

    /// Get corresponding MPI datatype
    public var mpiDatatype: MPIDatatype {
        switch (typeCode, bits) {
        case (kDLInt, 32):
            return .int32
        case (kDLInt, 64):
            return .int64
        case (kDLFloat, 32):
            return .float
        case (kDLFloat, 64):
            return .double
        default:
            // Default to double for unsupported types
            return .double
        }
    }

    // MARK: - Validation

    /// Validate tensor for MPI operations
    public func validate() throws {
        // Data pointer is always non-nil (UnsafeMutableRawPointer type)
        // but we check other properties for validity

        // Check if dimensions are reasonable
        guard ndim >= 0 && ndim <= 8 else {
            throw DLPackError.invalidDimensions(ndim)
        }

        // Check if element count is positive
        guard elementCount > 0 else {
            throw DLPackError.invalidElementCount(elementCount)
        }

        // Check if device type is supported
        if !isCPU && !isGPU {
            throw DLPackError.unsupportedDevice(deviceType)
        }
    }
}

// MARK: - DLPackManaged Tensor

/// Managed DLPack tensor with automatic cleanup
public struct DLPackManagedTensor {
    /// Pointer to DLManagedTensor structure
    let handle: UnsafeMutablePointer<DLManagedTensor>

    /// Initialize from DLManagedTensor pointer
    public init(_ dlManagedTensor: UnsafeMutablePointer<DLManagedTensor>) {
        self.handle = dlManagedTensor
    }

    /// Get the underlying DLTensor
    public var tensor: DLPackTensor {
        withUnsafeMutablePointer(to: &handle.pointee.dl_tensor) { ptr in
            DLPackTensor(ptr)
        }
    }

    /// Call deleter to free resources
    ///
    /// - Note: Caller is responsible for calling delete() explicitly
    ///         when done with the tensor to prevent memory leaks
    public func delete() {
        if let deleter = handle.pointee.deleter {
            deleter(handle)
        }
    }
}

// MARK: - Error Types

/// Errors related to DLPack tensor operations
public enum DLPackError: Error {
    case invalidPointer
    case invalidDimensions(Int)
    case invalidElementCount(Int)
    case unsupportedDevice(DLDeviceType)
    case unsupportedDataType(DLDataTypeCode)
    case mpiOperationFailed(String)

    public var description: String {
        switch self {
        case .invalidPointer:
            return "DLPack tensor has invalid data pointer"
        case .invalidDimensions(let ndim):
            return "Invalid number of dimensions: \(ndim)"
        case .invalidElementCount(let count):
            return "Invalid element count: \(count)"
        case .unsupportedDevice(let deviceType):
            return "Unsupported device type: \(deviceType)"
        case .unsupportedDataType(let typeCode):
            return "Unsupported data type code: \(typeCode)"
        case .mpiOperationFailed(let message):
            return "MPI operation failed: \(message)"
        }
    }
}

// MARK: - CustomStringConvertible

extension DLPackTensor: CustomStringConvertible {
    public var description: String {
        let deviceStr = isCPU ? "CPU" : (isGPU ? "CUDA:\(deviceID)" : "Device:\(deviceType)")
        let typeStr = isFloat64 ? "float64" : (isFloat32 ? "float32" : "type:\(bits)bit")
        return "DLPackTensor(shape: \(shape), dtype: \(typeStr), device: \(deviceStr))"
    }
}

// MARK: - Helper Functions

extension DLPackTensor {
    /// Create a contiguous view of the tensor
    /// Returns true if tensor is already contiguous
    public var isContiguous: Bool {
        guard let strides = handle.pointee.strides else {
            // NULL strides means compact row-major (contiguous)
            return true
        }

        // Check if strides match row-major layout
        var expectedStride = 1
        for i in (0..<ndim).reversed() {
            if strides[i] != expectedStride {
                return false
            }
            expectedStride *= Int(handle.pointee.shape[i])
        }
        return true
    }

    /// Get flat index from multi-dimensional index
    public func flatIndex(_ indices: [Int]) -> Int? {
        guard indices.count == ndim else { return nil }

        if let strides = handle.pointee.strides {
            // Use strides
            var index = 0
            for i in 0..<ndim {
                index += indices[i] * Int(strides[i])
            }
            return index
        } else {
            // Row-major compact
            var index = 0
            var mult = 1
            for i in (0..<ndim).reversed() {
                index += indices[i] * mult
                mult *= Int(handle.pointee.shape[i])
            }
            return index
        }
    }
}
