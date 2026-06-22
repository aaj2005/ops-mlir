#include "runtime/IRBuilder.h"
#include "Dialect/OPS/OPSDialect.h"
#include "Dialect/OPS/OPSOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"


// TODO: Casting pointers to int64_t is not portable. We should use a better way to represent pointers in MLIR attributes.

namespace ops_mlir {

IRBuilder::IRBuilder(mlir::MLIRContext *ctx) : ctx_(ctx) {
  // Register OPS dialect
  ctx_->getOrLoadDialect<ops_mlir::ops::OPSDialect>();
}

mlir::ModuleOp IRBuilder::buildModule(const std::vector<LoopDesc> &loops) {
  if (loops.empty()) {
    auto loc = mlir::UnknownLoc::get(ctx_);
    return mlir::ModuleOp::create(loc);
  }

  auto loc = mlir::UnknownLoc::get(ctx_);
  auto module = mlir::ModuleOp::create(loc);

  for (const auto &loop : loops) {
    auto *op = buildParLoopOp(loop);
    if (!op) {
      llvm::errs() << "Failed to build ops.par_loop for: " << loop.kernel_name
                   << "\n";
      return nullptr;
    }
    module.getBody()->push_back(op);
  }

  return module;
}

std::string IRBuilder::moduleToString(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  return result;
}

mlir::Operation *IRBuilder::buildParLoopOp(const LoopDesc &loop) {
  auto loc = mlir::UnknownLoc::get(ctx_);
  mlir::OpBuilder builder(ctx_);

  // Infer block dims from dat arguments
  int ndim = loop.dims;
  for (const auto &arg : loop.args) {
    if (arg.kind == ArgKind::Dat && !arg.dat.size.empty()) {
      ndim = std::max(ndim, static_cast<int>(arg.dat.size.size()));
    }
  }

  // Build range attribute
  std::vector<int64_t> rangeVec(loop.range.begin(), loop.range.end());
  auto rangeAttr = mlir::DenseI64ArrayAttr::get(ctx_, rangeVec);

  // Build kernel ptr attribute
  auto kernelPtrAttr =
      builder.getI64IntegerAttr(static_cast<int64_t>(loop.kernel_token));

  // Build kernel name attribute
  auto kernelNameAttr = builder.getStringAttr(loop.kernel_name);

  // Build dims attribute
  auto dimsAttr = builder.getI32IntegerAttr(loop.dims);

  // Build argument descriptors
  mlir::SmallVector<mlir::Attribute> argAttrs;
  for (const auto &arg : loop.args) {
    auto argAttr = buildArgAttr(arg, ndim);
    if (!argAttr) {
      llvm::errs() << "Failed to build arg attribute\n";
      return nullptr;
    }
    argAttrs.push_back(argAttr);
  }
  auto argsAttr = builder.getArrayAttr(argAttrs);

  // Build the ops.par_loop operation
  mlir::SmallVector<mlir::Value> operands; // No operands in Phase 1
  auto opState =
      mlir::OperationState(loc, ops_mlir::ops::ParLoopOp::getOperationName());
  opState.attributes.push_back(
      {mlir::StringAttr::get(ctx_, "kernel_ptr"), kernelPtrAttr});
  opState.attributes.push_back(
      {mlir::StringAttr::get(ctx_, "kernel_name"), kernelNameAttr});
  opState.attributes.push_back({mlir::StringAttr::get(ctx_, "dims"), dimsAttr});
  opState.attributes.push_back(
      {mlir::StringAttr::get(ctx_, "range"), rangeAttr});
  opState.attributes.push_back({mlir::StringAttr::get(ctx_, "args"), argsAttr});

  if (ndim > 0) {
    opState.attributes.push_back({mlir::StringAttr::get(ctx_, "block_dims"),
                                  builder.getI32IntegerAttr(ndim)});
  }

  return builder.create(opState);
}

mlir::Attribute IRBuilder::buildArgAttr(const ArgDesc &arg, int ndim) {
  auto loc = mlir::UnknownLoc::get(ctx_);
  mlir::OpBuilder builder(ctx_);

  // Build stencil offsets
  std::vector<int64_t> stencilOffsets;
  if (!arg.stencil.offsets.empty()) {
    stencilOffsets.assign(arg.stencil.offsets.begin(),
                          arg.stencil.offsets.end());
  }
  auto stencilOffsetsAttr = mlir::DenseI64ArrayAttr::get(ctx_, stencilOffsets);

  // Build dat shape: [size[0], size[1], ..., d_m[0], d_m[1], ..., d_p[0],
  // d_p[1], ...]
  std::vector<int64_t> datShape;
  for (int64_t s : arg.dat.size)
    datShape.push_back(s);
  for (int64_t d : arg.dat.d_m)
    datShape.push_back(d);
  for (int64_t d : arg.dat.d_p)
    datShape.push_back(d);
  auto datShapeAttr = mlir::DenseI64ArrayAttr::get(ctx_, datShape);

  return ops_mlir::ops::ArgAttr::get(
      ctx_, static_cast<int32_t>(arg.kind), arg.access, arg.dim,
      arg.elem_size, static_cast<int64_t>(arg.dat_handle),
      static_cast<int64_t>(arg.host_ptr), stencilOffsetsAttr, datShapeAttr,
      ndim);
}

mlir::DenseI64ArrayAttr
IRBuilder::buildStencilOffsets(const StencilDesc &stencil) {
  std::vector<int64_t> offsets(stencil.offsets.begin(), stencil.offsets.end());
  return mlir::DenseI64ArrayAttr::get(ctx_, offsets);
}

mlir::DenseI64ArrayAttr IRBuilder::buildDatShape(const DatDesc &dat) {
  std::vector<int64_t> shape;
  for (int64_t s : dat.size)
    shape.push_back(s);
  for (int64_t d : dat.d_m)
    shape.push_back(d);
  for (int64_t d : dat.d_p)
    shape.push_back(d);
  return mlir::DenseI64ArrayAttr::get(ctx_, shape);
}

} // namespace ops_mlir
