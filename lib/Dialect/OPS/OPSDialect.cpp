#include "Dialect/OPS/OPSDialect.h"
#include "Dialect/OPS/OPSOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace ops_mlir::ops;

#include "Dialect/OPS/OPSDialect.cpp.inc"

namespace ops_mlir::ops {

void OPSDialect::initialize() {
  addOperations<ParLoopOp>();
}

mlir::Attribute OPSDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                          mlir::Type type) const {
  return nullptr;
}

void OPSDialect::printAttribute(mlir::Attribute attr,
                                mlir::DialectAsmPrinter &os) const {
  os << attr;
}

mlir::Type OPSDialect::parseType(mlir::DialectAsmParser &parser) const {
  return nullptr;
}

void OPSDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &os) const {
  os << type;
}

} // namespace ops_mlir::ops
