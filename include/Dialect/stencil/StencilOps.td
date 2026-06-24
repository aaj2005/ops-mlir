//===- StencilOps.td - Stencil Dialect Operation Definitions ---*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===//
//
// First, minimal cut of the Stencil dialect: just enough ops
// (apply/access/load/store/return) to support lib/passes/OPSToStencil.cpp
// lowering ops.par_loop into a func.func with memref parameters and a
// stencil.apply body. Deliberately leaves out alloc/cast/index/reduce/yield
// and any verifiers/interfaces for now -- add back as real need arises,
// rather than carrying unused/unverified surface area.
//===----------------------------------------------------------------------===//

#ifndef STENCIL_OPS
#define STENCIL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "StencilBase.td"
include "StencilAttr.td"
include "StencilTypes.td"

// stencil.load/store's field operand accepts either a real
// !stencil.field<...> or a plain memref<...> directly -- the latter is
// what lib/passes/OPSToStencil.cpp produces (per the func.func-with-memref-
// parameters design), so load/store can be "trivial" pass-throughs once a
// field is already backed by a genuine memref, with no extra
// external_load/cast bridging op needed for that case.
def Stencil_FieldOrMemRef : AnyTypeOf<[Stencil_FieldType, AnyMemRef]>;

// stencil.access -- relative-offset read inside stencil.apply.
def Stencil_AccessOp : Stencil_Op<"access", [Pure, HasAncestor<"ApplyOp">]> {
  let summary = "Access a stencil temp at a static relative offset";
  let arguments = (ins
    Stencil_TempType:$temp,
    Stencil_IndexAttr:$offset
  );
  let results = (outs AnyType:$res);
  let assemblyFormat = "$temp `at` $offset attr-dict-with-keyword"
                       " `:` type($temp) `->` type($res)";
}

// stencil.apply -- core compute op; IsolatedFromAbove.
def Stencil_ApplyOp : Stencil_Op<"apply", [IsolatedFromAbove]> {
  let summary = "Apply a stencil computation over a domain";
  let arguments = (ins
    Variadic<Stencil_TempType>:$args,
    Stencil_BoundsAttr:$bounds
  );
  let results = (outs Variadic<Stencil_TempType>:$res);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "`(` $args `)` $bounds $region attr-dict-with-keyword"
                       " `:` functional-type($args, $res)";
}

// stencil.load -- reads a field into a temp.
def Stencil_LoadOp : Stencil_Op<"load"> {
  let summary = "Load a stencil.field into a stencil.temp";
  let arguments = (ins Stencil_FieldOrMemRef:$field);
  let results  = (outs Stencil_TempType:$res);
  let assemblyFormat = "$field attr-dict-with-keyword `:` type($field) `->` type($res)";
}

// stencil.index -- the current iteration's coordinate along one dimension,
// inside a stencil.apply body. Used for kernels that take an ops_arg_idx()
// argument (e.g. left_bndcon/right_bndcon's `const int *idx`).
def Stencil_IndexOp : Stencil_Op<"index", [Pure, HasAncestor<"ApplyOp">]> {
  let summary = "Return the current iteration index for a given dimension";
  let arguments = (ins I64Attr:$dim);
  let results = (outs Index:$idx);
  let assemblyFormat = "$dim attr-dict-with-keyword `:` type($idx)";
}

// stencil.return -- terminator of stencil.apply; matches result types.
def Stencil_ReturnOp : Stencil_Op<"return", [Terminator, Pure, HasParent<"ApplyOp">]> {
  let summary = "Terminate stencil.apply and yield computed values as its results";
  let arguments = (ins Variadic<AnyType>:$arg);
  let assemblyFormat = "$arg attr-dict-with-keyword `:` type($arg)";
}

// stencil.store -- writes a temp to a field over a sub-range.
def Stencil_StoreOp : Stencil_Op<"store"> {
  let summary = "Write a stencil.temp to a stencil.field over a sub-range";
  let arguments = (ins
    Stencil_TempType:$temp,
    Stencil_FieldOrMemRef:$field,
    Stencil_BoundsAttr:$bounds
  );
  let assemblyFormat = "$temp `to` $field `(` $bounds `)` attr-dict-with-keyword"
                       " `:` type($temp) `to` type($field)";
}

#endif // STENCIL_OPS
