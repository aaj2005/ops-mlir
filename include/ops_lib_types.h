// //===- ops_lib_types.h - OPS Core Type Definitions ----------------*- C++ -*-===//
// //
// // Minimal type definitions extracted from OPS library.
// // Provides the core data structures needed for JIT capture without external deps.
// //
// //===----------------------------------------------------------------------===//

// #ifndef OPS_LIB_TYPES_H
// #define OPS_LIB_TYPES_H

// #include <cstdint>
// #include <cstring>

// //===----------------------------------------------------------------------===//
// // Constants
// //===----------------------------------------------------------------------===//

// #define OPS_MAX_DIM 5

// // Access type constants
// #define OPS_READ  0
// #define OPS_WRITE 1
// #define OPS_RW    2
// #define OPS_INC   3
// #define OPS_MIN   4
// #define OPS_MAX   5

// // Argument type constants
// #define OPS_ARG_GBL 0
// #define OPS_ARG_DAT 1
// #define OPS_ARG_IDX 2

// // Memory space constants
// #define OPS_HOST   1
// #define OPS_DEVICE 2

// typedef int ops_access;
// typedef int ops_arg_type;
// typedef int ops_memspace;

// //===----------------------------------------------------------------------===//
// // Forward Declarations
// //===----------------------------------------------------------------------===//

// class OPS_instance;
// class ops_block_core;
// class ops_dat_core;
// class ops_stencil_core;
// struct ops_reduction_core;
// struct ops_arg;

// //===----------------------------------------------------------------------===//
// // OPS Block (Mesh Block)
// //===----------------------------------------------------------------------===//

// class ops_block_core {
// public:
//   int index;              ///< Unique block index
//   int dims;               ///< Dimensionality (2, 3, etc.)
//   const char *name;       ///< Block name for diagnostics
//   OPS_instance *instance; ///< Pointer to OPS instance

//   ops_block_core() : index(0), dims(0), name(nullptr), instance(nullptr) {}
// };

// typedef ops_block_core *ops_block;

// //===----------------------------------------------------------------------===//
// // OPS Dataset (Structured Mesh Data)
// //===----------------------------------------------------------------------===//

// class ops_dat_core {
// public:
//   int index;                   ///< Unique dat index
//   ops_block block;             ///< Block this dat lives on
//   int dim;                     ///< Elements per grid point
//   int type_size;               ///< Bytes per element type
//   int elem_size;               ///< Total bytes per grid point (dim * type_size)
//   int size[OPS_MAX_DIM];       ///< Size in each dimension (including halo)
//   int base[OPS_MAX_DIM];       ///< Base offset to (0,0,...) from array start
//   int d_m[OPS_MAX_DIM];        ///< Halo depth in negative direction
//   int d_p[OPS_MAX_DIM];        ///< Halo depth in positive direction
//   int x_pad;                   ///< Padding for aligned memory
//   char *data;                  ///< Host data pointer
//   char *data_d;                ///< Device data pointer
//   const char *name;            ///< Dataset name for diagnostics
//   const char *type;            ///< Type string ("double", "float", etc.)
//   int dirty_hd;                ///< Host/device dirty flag
//   int locked_hd;               ///< Raw pointer lock flag
//   int user_managed;            ///< User manages memory flag
//   int is_hdf5;                 ///< HDF5 file flag
//   const char *hdf5_file;       ///< HDF5 file name
//   int e_dat;                   ///< Edge dat flag
//   size_t mem;                  ///< Memory allocated (bytes)
//   size_t base_offset;          ///< Offset to base index
//   int stride[OPS_MAX_DIM];     ///< Stride for multigrid

//   ops_dat_core() { memset(this, 0, sizeof(ops_dat_core)); }
// };

// typedef ops_dat_core *ops_dat;

// //===----------------------------------------------------------------------===//
// // OPS Stencil
// //===----------------------------------------------------------------------===//

// class ops_stencil_core {
// public:
//   int index;          ///< Unique stencil index
//   int dims;           ///< Stencil dimensionality
//   int points;         ///< Number of stencil points
//   const char *name;   ///< Stencil name for diagnostics
//   int *stencil;       ///< Stencil offsets: [dims * points] array
//   int *stride;        ///< Stride in each dimension
//   int *mgrid_stride;  ///< Stride under multigrid
//   int type;           ///< Type (0=regular, 1=prolongate, 2=restrict)

//   ops_stencil_core()
//       : index(0), dims(0), points(0), name(nullptr), stencil(nullptr),
//         stride(nullptr), mgrid_stride(nullptr), type(0) {}
// };

// typedef ops_stencil_core *ops_stencil;

// //===----------------------------------------------------------------------===//
// // OPS Reduction Handle
// //===----------------------------------------------------------------------===//

// struct ops_reduction_core {
//   char *data;          ///< Reduction result data
//   int size;            ///< Size in bytes
//   int initialized;     ///< Initialization flag
//   int index;           ///< Unique reduction index
//   ops_access acc;      ///< Last access type (MAX, MIN, INC)
//   char *type;          ///< Type string
//   char *name;          ///< Name for diagnostics
//   OPS_instance *instance;
// };

// typedef ops_reduction_core *ops_reduction;

// //===----------------------------------------------------------------------===//
// // OPS Loop Argument
// //===----------------------------------------------------------------------===//

// struct ops_arg {
//   ops_dat dat;          ///< Dataset (nullptr for globals/indices)
//   ops_stencil stencil;  ///< Stencil pattern
//   int dim;              ///< Dimensionality
//   int elem_size;        ///< Bytes per element
//   char *data;           ///< Host data pointer
//   char *data_d;         ///< Device data pointer
//   ops_access acc;       ///< Access type (READ, WRITE, MAX, etc.)
//   ops_arg_type argtype; ///< Argument type (DAT, GBL, IDX)
//   int opt;              ///< Optional flag (0=optional, 1=required)
// };

// //===----------------------------------------------------------------------===//
// // OPS Halo
// //===----------------------------------------------------------------------===//

// struct ops_halo_core {
//   ops_dat from;               ///< Source dataset
//   ops_dat to;                 ///< Destination dataset
//   int iter_size[OPS_MAX_DIM]; ///< Halo region iteration size
//   int from_base[OPS_MAX_DIM]; ///< Starting point in source
//   int to_base[OPS_MAX_DIM];   ///< Starting point in destination
//   int from_dir[OPS_MAX_DIM];  ///< Source increment direction
//   int to_dir[OPS_MAX_DIM];    ///< Destination increment direction
//   int index;                  ///< Unique halo index
// };

// typedef ops_halo_core *ops_halo;

// //===----------------------------------------------------------------------===//
// // OPS Halo Group
// //===----------------------------------------------------------------------===//

// typedef struct ops_halo_group_core {
//   int nhalos;           ///< Number of halos in group
//   ops_halo *halos;      ///< Array of halos
//   int index;            ///< Unique halo group index
//   OPS_instance *instance;
// } ops_halo_group_core;

// typedef ops_halo_group_core *ops_halo_group;

// #endif // OPS_LIB_TYPES_H
