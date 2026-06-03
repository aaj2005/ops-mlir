#ifndef OPS_MLIR_DIALECT_OPS_OPS_DIALECT_H
#define OPS_MLIR_DIALECT_OPS_OPS_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace ops_mlir::ops {

// Forward declarations
class OPSDialect;

// Include generated dialect class definition
#define GET_DIALECT_CLASSES
#include "Dialect/OPS/OPSDialect.h.inc"

} // namespace ops_mlir::ops

#endif // OPS_MLIR_DIALECT_OPS_OPS_DIALECT_H
