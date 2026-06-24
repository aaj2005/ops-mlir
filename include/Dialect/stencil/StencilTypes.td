//===- StencilTypes.td - Stencil Dialect Type Definitions ------*- tablegen -*-===//
//
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//===----------------------------------------------------------------------===//

#ifndef STENCIL_TYPES
#define STENCIL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "StencilBase.td"
include "StencilAttr.td"

// Common parameters shared by field/temp: per-dimension bounds + element
// type, e.g. !stencil.field<#stencil.bounds<[-1,-1],[4097,4097]>, f64>.
// (Note: this is a deliberately simpler textual form than e.g. xDSL's
// `!stencil.field<[-1,4097]x[-1,4097]xf64>` -- kept to plain ODS-generated
// assemblyFormat for this first cut rather than a hand-rolled C++ parser.)
class Stencil_BoundedType<string name> : TypeDef<Stencil_Dialect, name> {
  let parameters = (ins Stencil_BoundsAttr:$bounds, "mlir::Type":$element_type);
  let assemblyFormat = "`<` $bounds `,` $element_type `>`";
}

def Stencil_FieldType : Stencil_BoundedType<"Field"> {
  let mnemonic = "field";
  let summary = "A stencil field: storage for a structured mesh dataset";
}

def Stencil_TempType : Stencil_BoundedType<"Temp"> {
  let mnemonic = "temp";
  let summary = "A stencil temp: an intermediate, register-like view of a field";
}

#endif // STENCIL_TYPES
