#include "Dialect/OPS/OPSOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ops_mlir::ops;

namespace ops_mlir::ops {

void ParLoopOp::print(OpAsmPrinter &p) {
  p << "ops.par_loop";
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult ParLoopOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }
  return success();
}

} // namespace ops_mlir::ops
