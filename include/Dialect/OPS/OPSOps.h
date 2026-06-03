#ifndef OPS_MLIR_DIALECT_OPS_OPSOPS_H
#define OPS_MLIR_DIALECT_OPS_OPSOPS_H

#include "mlir/IR/Op.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace ops_mlir::ops {

// Include generated files
#define GET_ATTRDEF_CLASSES
#include "Dialect/OPS/OPSAttr.h.inc"

#define GET_OP_CLASSES
#include "Dialect/OPS/OPSOps.h.inc"

} // namespace ops_mlir::ops

#endif // OPS_MLIR_DIALECT_OPS_OPSOPS_H
