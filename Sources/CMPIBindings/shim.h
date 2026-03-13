#ifndef CMPI_BINDINGS_H
#define CMPI_BINDINGS_H

#include <mpi.h>

// MPI 5.0 specific feature detection
#if MPI_VERSION >= 5
#define MPI5_AVAILABLE 1
#else
#define MPI5_AVAILABLE 0
#warning "MPI 5.0 features not available. Some functionality may be limited."
#endif

// Ensure we have the large count interface
#if MPI_VERSION >= 4
#define MPI_LARGE_COUNT_AVAILABLE 1
#else
#define MPI_LARGE_COUNT_AVAILABLE 0
#endif

// Helper macros for MPI 5.0 features
#ifdef MPI5_AVAILABLE
#define MPI5_SESSIONS_AVAILABLE 1
#define MPI5_PARTITIONED_AVAILABLE 1
#define MPI5_PERSISTENT_COLLECTIVES_AVAILABLE 1
#else
#define MPI5_SESSIONS_AVAILABLE 0
#define MPI5_PARTITIONED_AVAILABLE 0
#define MPI5_PERSISTENT_COLLECTIVES_AVAILABLE 0
#endif

// Expose MPI predefined constants that Swift can't import from macros
// These are defined as inline functions to make them accessible to Swift

// NULL handles
static inline MPI_Comm MPI_COMM_WORLD_get(void) { return MPI_COMM_WORLD; }
static inline MPI_Comm MPI_COMM_SELF_get(void) { return MPI_COMM_SELF; }
static inline MPI_Comm MPI_COMM_NULL_get(void) { return MPI_COMM_NULL; }
static inline MPI_Group MPI_GROUP_NULL_get(void) { return MPI_GROUP_NULL; }
static inline MPI_Request MPI_REQUEST_NULL_get(void) { return MPI_REQUEST_NULL; }
static inline MPI_Info MPI_INFO_NULL_get(void) { return MPI_INFO_NULL; }

// Operations
static inline MPI_Op MPI_MAX_get(void) { return MPI_MAX; }
static inline MPI_Op MPI_MIN_get(void) { return MPI_MIN; }
static inline MPI_Op MPI_SUM_get(void) { return MPI_SUM; }
static inline MPI_Op MPI_PROD_get(void) { return MPI_PROD; }
static inline MPI_Op MPI_LAND_get(void) { return MPI_LAND; }
static inline MPI_Op MPI_LOR_get(void) { return MPI_LOR; }
static inline MPI_Op MPI_BAND_get(void) { return MPI_BAND; }
static inline MPI_Op MPI_BOR_get(void) { return MPI_BOR; }
static inline MPI_Op MPI_LXOR_get(void) { return MPI_LXOR; }
static inline MPI_Op MPI_BXOR_get(void) { return MPI_BXOR; }
static inline MPI_Op MPI_MAXLOC_get(void) { return MPI_MAXLOC; }
static inline MPI_Op MPI_MINLOC_get(void) { return MPI_MINLOC; }
static inline MPI_Op MPI_REPLACE_get(void) { return MPI_REPLACE; }

// Thread levels - return as constants
static inline int MPI_THREAD_SINGLE_get(void) { return MPI_THREAD_SINGLE; }
static inline int MPI_THREAD_FUNNELED_get(void) { return MPI_THREAD_FUNNELED; }
static inline int MPI_THREAD_SERIALIZED_get(void) { return MPI_THREAD_SERIALIZED; }
static inline int MPI_THREAD_MULTIPLE_get(void) { return MPI_THREAD_MULTIPLE; }

// Datatypes
static inline MPI_Datatype MPI_INT8_T_get(void) { return MPI_INT8_T; }
static inline MPI_Datatype MPI_INT16_T_get(void) { return MPI_INT16_T; }
static inline MPI_Datatype MPI_INT32_T_get(void) { return MPI_INT32_T; }
static inline MPI_Datatype MPI_INT64_T_get(void) { return MPI_INT64_T; }
static inline MPI_Datatype MPI_UINT8_T_get(void) { return MPI_UINT8_T; }
static inline MPI_Datatype MPI_UINT16_T_get(void) { return MPI_UINT16_T; }
static inline MPI_Datatype MPI_UINT32_T_get(void) { return MPI_UINT32_T; }
static inline MPI_Datatype MPI_UINT64_T_get(void) { return MPI_UINT64_T; }
static inline MPI_Datatype MPI_FLOAT_get(void) { return MPI_FLOAT; }
static inline MPI_Datatype MPI_DOUBLE_get(void) { return MPI_DOUBLE; }
static inline MPI_Datatype MPI_CHAR_get(void) { return MPI_CHAR; }
static inline MPI_Datatype MPI_BYTE_get(void) { return MPI_BYTE; }

// Special constants
static inline int MPI_ANY_SOURCE_get(void) { return MPI_ANY_SOURCE; }
static inline int MPI_ANY_TAG_get(void) { return MPI_ANY_TAG; }
static inline int MPI_PROC_NULL_get(void) { return MPI_PROC_NULL; }

#endif // CMPI_BINDINGS_H
