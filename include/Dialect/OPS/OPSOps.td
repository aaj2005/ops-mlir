//===- OPSOps.td - OPS Dialect Operation Definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPS_OPS
#define OPS_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// OPS Dialect Definition
//===----------------------------------------------------------------------===//

def OPS_Dialect : Dialect {
  let name = "ops";
  let summary = "OPS DSL dialect for structured mesh computations";
  let description = [{
    The OPS dialect represents structured mesh parallel loops as captured from
    the OPS API. It models the loop structure, data dependencies, and stencil
    operations. The dialect is designed to be lowered to stencil dialect and
    then to concrete execution targets.
  }];
  let cppNamespace = "::ops_mlir::ops";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class OPS_Op<string mnemonic, list<Trait> traits = []>
    : Op<OPS_Dialect, mnemonic, traits> {
  let cppNamespace = "::ops_mlir::ops";
}

//===----------------------------------------------------------------------===//
// OPS Attributes
//===----------------------------------------------------------------------===//

// Argument descriptor attribute: holds metadata for a single loop argument
def OPS_ArgAttr : AttrDef<OPS_Dialect, "Arg"> {
  let mnemonic = "arg";
  let summary = "OPS loop argument descriptor";
  let description = [{
    Carries metadata for a single argument in an ops.par_loop operation.
    Includes the argument kind (dat, global, index, reduction), access type,
    dimensionality, element size, and stencil information if applicable.
  }];
  let parameters = (ins
    "int32_t":$kind,                        // ArgKind enum value
    "int32_t":$access,                      // OPS_READ, OPS_WRITE, OPS_MAX, etc.
    "int32_t":$dim,                         // dimensionality
    "int32_t":$elem_size,                   // bytes per element
    "uint64_t":$dat_ptr,                    // pointer to ops_dat_core
    "uint64_t":$host_ptr,                   // data pointer
    "mlir::DenseI64ArrayAttr":$stencil_offsets,  // flattened stencil offsets
    "mlir::DenseI64ArrayAttr":$dat_shape,   // [size0, size1, ..., d_m0, ..., d_p0, ...]
    "int32_t":$ndim                         // number of spatial dimensions
  );
}

// Dataset descriptor attribute
def OPS_DatAttr : AttrDef<OPS_Dialect, "Dat"> {
  let mnemonic = "dat";
  let summary = "OPS dataset descriptor";
  let parameters = (ins
    "uint64_t":$handle,
    "int32_t":$dim,
    "int32_t":$elem_size,
    "mlir::DenseI64ArrayAttr":$size,
    "mlir::DenseI64ArrayAttr":$base,
    "mlir::DenseI64ArrayAttr":$d_m,
    "mlir::DenseI64ArrayAttr":$d_p
  );
}

// Stencil descriptor attribute
def OPS_StencilAttr : AttrDef<OPS_Dialect, "Stencil"> {
  let mnemonic = "stencil";
  let summary = "OPS stencil descriptor";
  let parameters = (ins
    "uint64_t":$handle,
    "int32_t":$dims,
    "int32_t":$points,
    "mlir::DenseI64ArrayAttr":$offsets
  );
}

//===----------------------------------------------------------------------===//
// OPS Operations
//===----------------------------------------------------------------------===//

def OPS_ParLoopOp : OPS_Op<"par_loop", []> {
  let summary = "OPS parallel loop over a structured mesh block";
  let description = [{
    Represents a single OPS parallel loop capture. This operation encodes:
    - The kernel function pointer and metadata
    - The iteration range and dimensions
    - The arguments (datasets, stencils, reductions, globals, indices)
    - Access patterns for each argument

    Example:
    ```mlir
    ops.par_loop {
      kernel_ptr = 0x...,
      kernel_name = "apply_stencil",
      dims = 2,
      range = array<i64: 0, 4094, 0, 4094>,
      args = [
        #ops.arg<kind=1, access=0, dim=1, elem_size=8, ...>,
        #ops.arg<kind=1, access=1, dim=1, elem_size=8, ...>
      ]
    }
    ```
  }];

  let arguments = (ins
    I64Attr:$kernel_ptr,                   // C++ function pointer
    StrAttr:$kernel_name,                  // kernel function name
    I32Attr:$dims,                         // iteration space dimensionality
    DenseI64ArrayAttr:$range,              // loop bounds: [lo0, hi0, lo1, hi1, ...]
    ArrayAttr:$args,                       // argument descriptors
    OptionalAttr<I32Attr>:$block_dims      // block dimensionality (optional)
  );

  let results = (outs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // OPS_OPS
