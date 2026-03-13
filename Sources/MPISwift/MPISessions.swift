import CMPIBindings
import Foundation

/// MPI 5.0 Thread Level Support
public enum MPIThreadLevel: Int32 {
    case single = 0     // MPI_THREAD_SINGLE
    case funneled = 1   // MPI_THREAD_FUNNELED
    case serialized = 2 // MPI_THREAD_SERIALIZED
    case multiple = 3   // MPI_THREAD_MULTIPLE

    var cValue: Int32 {
        switch self {
        case .single: return Int32(MPI_THREAD_SINGLE_get())
        case .funneled: return Int32(MPI_THREAD_FUNNELED_get())
        case .serialized: return Int32(MPI_THREAD_SERIALIZED_get())
        case .multiple: return Int32(MPI_THREAD_MULTIPLE_get())
        }
    }
}

/// MPI 5.0 Session - Modern resource isolation model
public final class MPISession {
    private var session: MPI_Session
    private var isFinalized: Bool = false

    /// Initialize MPI session with thread support level
    /// - Parameter threadLevel: Desired thread support level
    /// - Parameter errorHandler: Optional error handler name
    /// - Throws: MPIError if initialization fails
    public init(threadLevel: MPIThreadLevel = .single, errorHandler: String? = nil) throws {
#if MPI5_SESSIONS_AVAILABLE
        var info = MPI_INFO_NULL_get()
        var errorCode: Int32

        // Create info object for session hints
        if let handler = errorHandler {
            errorCode = MPI_Info_create(&info)
            try checkMPIError(errorCode)

            errorCode = handler.withCString { handlerStr in
                MPI_Info_set(info, "mpi_error_handler", handlerStr)
            }
            try checkMPIError(errorCode)
        }

        // Initialize session
        var tempSession: MPI_Session?
        errorCode = MPI_Session_init(info, MPI_ERRORS_RETURN, &tempSession)

        // Clean up info
        if info != MPI_INFO_NULL_get() {
            var tempInfo = info
            MPI_Info_free(&tempInfo)
        }

        try checkMPIError(errorCode)
        guard let unwrappedSession = tempSession else {
            throw MPIError.sessionError
        }
        self.session = unwrappedSession
#else
        throw MPIError.sessionError
#endif
    }

    /// Get process set from session
    /// - Parameter name: Name of the process set (e.g., "mpi://WORLD")
    /// - Returns: Process set handle
    public func getProcessSet(name: String) throws -> MPIProcessSet {
#if MPI5_SESSIONS_AVAILABLE
        var pset = MPI_Session()
        let errorCode = name.withCString { nameStr in
            MPI_Session_get_pset_info(session, nameStr, &pset)
        }
        try checkMPIError(errorCode)
        return MPIProcessSet(handle: pset)
#else
        throw MPIError.sessionError
#endif
    }

    /// Create group from process set
    /// - Parameter processSet: Process set to create group from
    /// - Returns: MPI Group
    public func createGroup(from processSet: MPIProcessSet) throws -> MPIGroup {
#if MPI5_SESSIONS_AVAILABLE
        var group = MPI_GROUP_NULL_get()
        let errorCode = MPI_Group_from_session_pset(session, processSet.name.cString(using: .utf8), &group)
        try checkMPIError(errorCode)
        return MPIGroup(handle: group)
#else
        throw MPIError.sessionError
#endif
    }

    /// Create communicator from group
    /// - Parameters:
    ///   - group: Group to create communicator from
    ///   - tag: Tag for communicator creation
    /// - Returns: MPI Communicator
    public func createCommunicator(from group: MPIGroup, tag: Int32 = 0) throws -> MPICommunicator {
#if MPI5_SESSIONS_AVAILABLE
        var comm = MPI_COMM_NULL_get()
        let errorCode = MPI_Comm_create_from_group(group.handle, "session_comm", MPI_INFO_NULL_get(), MPI_ERRORS_RETURN, &comm)
        try checkMPIError(errorCode)
        return MPICommunicator(handle: comm)
#else
        throw MPIError.sessionError
#endif
    }

    /// Finalize the session
    public func finalize() throws {
        guard !isFinalized else { return }

#if MPI5_SESSIONS_AVAILABLE
        let errorCode = MPI_Session_finalize(&session)
        try checkMPIError(errorCode)
        isFinalized = true
#else
        throw MPIError.sessionError
#endif
    }

    deinit {
        if !isFinalized {
            try? finalize()
        }
    }
}

/// MPI Process Set - MPI 5.0 feature
public struct MPIProcessSet {
    let handle: MPI_Session?
    let name: String

    init(handle: MPI_Session?, name: String = "mpi://WORLD") {
        self.handle = handle
        self.name = name
    }

    /// Standard world process set
    public static var world: MPIProcessSet {
        // Placeholder - actual session handle should be provided by session
        return MPIProcessSet(handle: nil, name: "mpi://WORLD")
    }

    /// Self process set (only this process)
    public static var `self`: MPIProcessSet {
        // Placeholder - actual session handle should be provided by session
        return MPIProcessSet(handle: nil, name: "mpi://SELF")
    }
}

/// MPI Group wrapper
public struct MPIGroup {
    let handle: MPI_Group

    /// Get size of the group
    public var size: Int32 {
        var groupSize: Int32 = 0
        MPI_Group_size(handle, &groupSize)
        return groupSize
    }

    /// Get rank in the group
    public var rank: Int32 {
        var groupRank: Int32 = 0
        MPI_Group_rank(handle, &groupRank)
        return groupRank
    }
}

/// MPI Communicator wrapper
public struct MPICommunicator {
    let handle: MPI_Comm

    /// Get size of the communicator
    public var size: Int32 {
        var commSize: Int32 = 0
        MPI_Comm_size(handle, &commSize)
        return commSize
    }

    /// Get rank in the communicator
    public var rank: Int32 {
        var commRank: Int32 = 0
        MPI_Comm_rank(handle, &commRank)
        return commRank
    }

    /// MPI_COMM_WORLD
    public static var world: MPICommunicator {
        return MPICommunicator(handle: MPI_COMM_WORLD_get())
    }

    /// MPI_COMM_SELF
    public static var `self`: MPICommunicator {
        return MPICommunicator(handle: MPI_COMM_SELF_get())
    }
}
