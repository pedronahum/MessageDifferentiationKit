import CMPIBindings

/// MPI Constants and Special Values
public enum MPIConstants {
    /// Wildcard for receiving from any source
    public static let anySource: Int32 = Int32(MPI_ANY_SOURCE_get())

    /// Wildcard for receiving any tag
    public static let anyTag: Int32 = Int32(MPI_ANY_TAG_get())

    /// Null process (no-op destination/source)
    public static let procNull: Int32 = Int32(MPI_PROC_NULL_get())
}
