#include "Dialect/stencil/StencilDialect.h"
#include "Dialect/stencil/StencilOps.h"
#include "Dialect/stencil/StencilAttr.h"
#include "Dialect/stencil/StencilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::stencil;

#include "Dialect/stencil/StencilDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/stencil/StencilOpsAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/stencil/StencilOpsTypes.cpp.inc"

namespace mlir::stencil {

LogicalResult IndexAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 DenseI64ArrayAttr array) {
  if (array.size() < 1 || array.size() > 3)
    return emitError() << "expected 1 to 3 indices, got " << array.size();
  return success();
}

LogicalResult
StencilBoundsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           DenseI64ArrayAttr lb, DenseI64ArrayAttr ub) {
  if (lb.size() != ub.size())
    return emitError() << "lb (" << lb.size() << " dims) and ub ("
                        << ub.size() << " dims) rank mismatch";
  for (auto [l, u] : llvm::zip(lb.asArrayRef(), ub.asArrayRef()))
    if (l > u)
      return emitError() << "lb must be <= ub in every dimension";
  return success();
}

void StencilDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/stencil/StencilOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/stencil/StencilOpsAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/stencil/StencilOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::stencil
