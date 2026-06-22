//===- OPSOps.td - OPS Dialect Operation Definitions ----*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
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
    operations. The dialect is designed to be lowered to control flow and
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

// Structured descriptor for a single ops.par_loop argument, mirroring
// ops_mlir::ArgDesc (runtime/Core.h). Replaces the previous positional
// ArrayAttr encoding produced by IRBuilder::buildArgAttr.
def OPS_ArgAttr : AttrDef<OPS_Dialect, "Arg"> {
  let mnemonic = "arg";
  let summary = "OPS dataset argument descriptor";
  let description = [{
    Represents an OPS dataset argument with its kind, access pattern,
    dataset shape, and stencil offsets.
  }];
  let parameters = (ins
    "int32_t":$kind,
    "int32_t":$access,
    "int32_t":$dim,
    "int32_t":$elem_size,
    "int64_t":$dat_handle,
    "int64_t":$host_ptr,
    "mlir::DenseI64ArrayAttr":$stencil_offsets,
    "mlir::DenseI64ArrayAttr":$dat_shape,
    "int32_t":$ndim
  );
  let assemblyFormat = "`<` $kind `,` $access `,` $dim `,` $elem_size `,`"
                       " $dat_handle `,` $host_ptr `,` $stencil_offsets `,`"
                       " $dat_shape `,` $ndim `>`";
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

    The operation carries all loop metadata as attributes and is designed
    for lowering to control flow and memory operations during compilation.
  }];

  let arguments = (ins
    I64Attr:$kernel_ptr,
    StrAttr:$kernel_name,
    I32Attr:$dims,
    DenseI64ArrayAttr:$range,
    TypedArrayAttrBase<OPS_ArgAttr, "array of OPS dataset argument descriptors">:$args,
    OptionalAttr<I32Attr>:$block_dims
  );

  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

#endif // OPS_OPS
