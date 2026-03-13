import CMPIBindings
import Foundation

/// MPI error codes and error handling
public enum MPIError: Error, CustomStringConvertible, Equatable {
    case success
    case invalidBuffer
    case invalidCount
    case invalidDatatype
    case invalidTag
    case invalidCommunicator
    case invalidGroup
    case invalidRank
    case invalidRoot
    case invalidRequest
    case invalidOperation
    case invalidTopology
    case dimensionError
    case invalidArgument
    case unknownError
    case truncatedMessage
    case otherError
    case internalError
    case inProgress
    case pending
    case invalidSession
    case invalidProcessSet
    case sessionError
    case unknown(Int32)

    /// Initialize from MPI error code
    public init(code: Int32) {
        switch code {
        case MPI_SUCCESS:
            self = .success
        case MPI_ERR_BUFFER:
            self = .invalidBuffer
        case MPI_ERR_COUNT:
            self = .invalidCount
        case MPI_ERR_TYPE:
            self = .invalidDatatype
        case MPI_ERR_TAG:
            self = .invalidTag
        case MPI_ERR_COMM:
            self = .invalidCommunicator
        case MPI_ERR_RANK:
            self = .invalidRank
        case MPI_ERR_ROOT:
            self = .invalidRoot
        case MPI_ERR_GROUP:
            self = .invalidGroup
        case MPI_ERR_REQUEST:
            self = .invalidRequest
        case MPI_ERR_OP:
            self = .invalidOperation
        case MPI_ERR_TOPOLOGY:
            self = .invalidTopology
        case MPI_ERR_DIMS:
            self = .dimensionError
        case MPI_ERR_ARG:
            self = .invalidArgument
        case MPI_ERR_UNKNOWN:
            self = .unknownError
        case MPI_ERR_TRUNCATE:
            self = .truncatedMessage
        case MPI_ERR_OTHER:
            self = .otherError
        case MPI_ERR_INTERN:
            self = .internalError
        case MPI_ERR_IN_STATUS:
            self = .inProgress
        case MPI_ERR_PENDING:
            self = .pending
        default:
            self = .unknown(code)
        }
    }

    public var description: String {
        switch self {
        case .success:
            return "MPI operation successful"
        case .invalidBuffer:
            return "Invalid buffer pointer"
        case .invalidCount:
            return "Invalid count argument"
        case .invalidDatatype:
            return "Invalid datatype"
        case .invalidTag:
            return "Invalid tag"
        case .invalidCommunicator:
            return "Invalid communicator"
        case .invalidGroup:
            return "Invalid group"
        case .invalidRank:
            return "Invalid rank"
        case .invalidRoot:
            return "Invalid root"
        case .invalidRequest:
            return "Invalid request"
        case .invalidOperation:
            return "Invalid operation"
        case .invalidTopology:
            return "Invalid topology"
        case .dimensionError:
            return "Invalid dimensions"
        case .invalidArgument:
            return "Invalid argument"
        case .unknownError:
            return "Unknown error"
        case .truncatedMessage:
            return "Message truncated"
        case .otherError:
            return "Other error"
        case .internalError:
            return "Internal MPI error"
        case .inProgress:
            return "Operation in progress"
        case .pending:
            return "Operation pending"
        case .invalidSession:
            return "Invalid MPI session"
        case .invalidProcessSet:
            return "Invalid process set"
        case .sessionError:
            return "Session error"
        case .unknown(let code):
            return "Unknown MPI error code: \(code)"
        }
    }
}

/// Check MPI error code and throw if not successful
@inline(__always)
public func checkMPIError(_ code: Int32) throws {
    guard code == MPI_SUCCESS else {
        throw MPIError(code: code)
    }
}

/// Get error string from MPI error code
public func mpiErrorString(_ errorCode: Int32) -> String {
    var errorString = [CChar](repeating: 0, count: Int(MPI_MAX_ERROR_STRING))
    var length: Int32 = 0

    MPI_Error_string(errorCode, &errorString, &length)

    return String(cString: errorString)
}
