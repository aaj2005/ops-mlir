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
    ArrayAttr:$args,
    OptionalAttr<I32Attr>:$block_dims
  );

  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

#endif // OPS_OPS
