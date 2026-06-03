#include "Dialect/OPS/OPSOps.h"
#include "Dialect/OPS/OPSDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace ops_mlir::ops;

// Include generated op implementations
#define GET_ATTRDEF_CLASSES
#include "Dialect/OPS/OPSAttr.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/OPS/OPSOps.cpp.inc"

//===----------------------------------------------------------------------===//
// OPS Dialect Initialization
//===----------------------------------------------------------------------===//

namespace ops_mlir::ops {

void OPSDialect::initialize() {
  addOperations<ParLoopOp>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/OPS/OPSAttr.cpp.inc"
  >();
}

} // namespace ops_mlir::ops

//===----------------------------------------------------------------------===//
// ParLoopOp Implementation
//===----------------------------------------------------------------------===//

namespace ops_mlir::ops {

LogicalResult ParLoopOp::verify() {
  auto dims = getDims();
  auto range = getRange();

  // Range should have 2*dims elements
  if (static_cast<int64_t>(range.size()) != 2 * dims) {
    return emitOpError("range must have 2*dims elements, got ")
           << range.size() << " but dims=" << dims;
  }

  // Check that all range values are non-negative and lo < hi
  for (int i = 0; i < dims; ++i) {
    int64_t lo = range[2*i];
    int64_t hi = range[2*i+1];
    if (lo >= hi) {
      return emitOpError("range [") << lo << ", " << hi
             << "] invalid: lower bound >= upper bound";
    }
  }

  return success();
}

void ParLoopOp::print(OpAsmPrinter &printer) {
  printer << "ops.par_loop {\n";
  printer.printAttributeWithoutType(getKernelPtrAttr());
  printer << ",\n";
  printer << "  kernel_name = " << getKernelNameAttr() << ",\n";
  printer << "  dims = " << getDimsAttr() << ",\n";
  printer << "  range = " << getRangeAttr() << ",\n";
  printer << "  args = [";
  interleaveComma(getArgs(), printer);
  printer << "]\n}\n";
}

ParseResult ParLoopOp::parse(OpAsmParser &parser, OperationState &result) {
  // For now, use generic format
  return failure();
}

} // namespace ops_mlir::ops
