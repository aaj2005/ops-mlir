//===- OPSToStencil.cpp - Lower ops.par_loop to func.func + stencil.* ---===//
//
// Phase 0/1 of the pure-LLVM lowering plan: each ops.par_loop becomes a
// callable func.func with one memref<...> parameter per captured Dat
// argument (not a pointer literal -- see the design discussion in
// xdsl_impl/ops_to_xdsl.py's history), containing a stencil.apply built the
// same way the xDSL prototype did. stencil.load/store consume those memref
// parameters directly (Stencil_FieldOrMemRef), so there is no
// external_load/inttoptr bridging needed at all -- that whole problem
// class only existed because the earlier prototype tried to embed a raw
// pointer as a literal in the IR; real OPS buffers are supplied by the
// caller (JITEngine, eventually via mlir::ExecutionEngine) as genuine
// function arguments instead.
//
// Idx arguments (ops_arg_idx(), e.g. left_bndcon/right_bndcon's
// `const int *idx`) are modeled via stencil.index: one per dimension,
// appended to the kernel call after the dat-derived accesses, in the same
// order ops_arg_idx() args appear among the loop's arguments.
//
// Known simplifications (first cut): Gbl/Reduce arguments are still
// skipped (same as the xDSL prototype); the stencil access pattern
// defaults to a single (0, ..., 0) point per dat, since recovering the
// real per-point offsets requires dereferencing
// OPS_StencilAttr::getStencil()'s raw pointer, which this pass does not
// yet do (the xDSL prototype's --unsafe-dereference-pointers path is the
// reference for adding that here).
//
//===----------------------------------------------------------------------===//

#include "passes/OPSToStencil.h"

#include "Dialect/OPS/OPSOps.h"
#include "Dialect/stencil/StencilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace ops_mlir::ops;

namespace {

/// Per-dimension (lb, ub) bounds of a dat's full allocated (halo-included)
/// shape, matching xdsl_impl/ops_to_xdsl.py::field_bounds.
SmallVector<int64_t> datShape(DatAttr dat, int64_t ndim) {
  SmallVector<int64_t> shape;
  auto size = dat.getSize().asArrayRef();
  for (int64_t i = 0; i < ndim; ++i)
    shape.push_back(size[i]);
  return shape;
}

stencil::StencilBoundsAttr datBounds(OpBuilder &b, DatAttr dat, int64_t ndim) {
  auto size = dat.getSize().asArrayRef();
  auto d_m = dat.getDM().asArrayRef();
  auto d_p = dat.getDP().asArrayRef();
  SmallVector<int64_t> lb, ub;
  for (int64_t i = 0; i < ndim; ++i) {
    lb.push_back(d_m[i]);
    ub.push_back(size[i] + d_p[i]);
  }
  return stencil::StencilBoundsAttr::get(
      b.getContext(), DenseI64ArrayAttr::get(b.getContext(), lb),
      DenseI64ArrayAttr::get(b.getContext(), ub));
}

stencil::StencilBoundsAttr rangeBounds(OpBuilder &b, ArrayRef<int64_t> range,
                                        int64_t ndim) {
  SmallVector<int64_t> lb, ub;
  for (int64_t i = 0; i < ndim; ++i) {
    lb.push_back(range[2 * i]);
    ub.push_back(range[2 * i + 1]);
  }
  return stencil::StencilBoundsAttr::get(
      b.getContext(), DenseI64ArrayAttr::get(b.getContext(), lb),
      DenseI64ArrayAttr::get(b.getContext(), ub));
}

/// Lower a single ops.par_loop into a standalone func.func, inserted right
/// before it in the module. Returns the new function.
func::FuncOp convertParLoop(ParLoopOp loop, unsigned index, ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(loop);
  Location loc = loop.getLoc();

  int64_t ndim =
      loop->getAttrOfType<IntegerAttr>("dims").getInt();
  auto argsAttr = loop->getAttrOfType<ArrayAttr>("args");

  SmallVector<ArgAttr> datArgs;
  unsigned numIdxArgs = 0;
  for (Attribute a : argsAttr)
    if (auto arg = dyn_cast<ArgAttr>(a)) {
      if (arg.getArgtype() == /*OPS_ARG_DAT=*/1)
        datArgs.push_back(arg);
      else if (arg.getArgtype() == /*OPS_ARG_IDX=*/2)
        ++numIdxArgs;
    }

  // Build the function signature: one memref<...> param per dat arg.
  SmallVector<Type> paramTypes;
  for (ArgAttr arg : datArgs)
    paramTypes.push_back(
        MemRefType::get(datShape(arg.getDat(), ndim), builder.getF64Type()));

  StringRef kernelName = loop.getKernelNameAttr().getValue();

  auto funcType = builder.getFunctionType(paramTypes, {});
  std::string fnName =
      ("ops_par_loop_" + kernelName + "_" + Twine(index)).str();
  auto func = builder.create<func::FuncOp>(loc, fnName, funcType);
  func.setPrivate();
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto rangeAttr = loop->getAttrOfType<DenseI64ArrayAttr>("range");
  auto applyBounds = rangeBounds(builder, rangeAttr.asArrayRef(), ndim);

  SmallVector<std::pair<ArgAttr, Value>> loads; // (arg, stencil.temp)
  SmallVector<std::pair<ArgAttr, Value>> writeFields; // (arg, memref param)

  for (auto [i, arg] : llvm::enumerate(datArgs)) {
    Value field = entry->getArgument(i);
    int32_t acc = arg.getAcc();
    if (acc == /*READ=*/0 || acc == /*RW=*/2) {
      auto tempTy = stencil::TempType::get(ctx, datBounds(builder, arg.getDat(), ndim),
                                            builder.getF64Type());
      auto load = builder.create<stencil::LoadOp>(loc, tempTy, field);
      loads.emplace_back(arg, load.getRes());
    }
    if (acc == /*WRITE=*/1 || acc == /*RW=*/2 || acc == /*INC=*/3)
      writeFields.emplace_back(arg, field);
  }

  // stencil.apply body: one stencil.access per load (defaulting to a
  // single (0,...,0) point -- see file header), then a placeholder call
  // to the kernel, then stencil.return.
  SmallVector<Value> applyArgs;
  for (auto &[arg, temp] : loads)
    applyArgs.push_back(temp);

  SmallVector<Type> resultTypes;
  for (auto &[arg, field] : writeFields)
    resultTypes.push_back(stencil::TempType::get(
        ctx, datBounds(builder, arg.getDat(), ndim), builder.getF64Type()));
  if (resultTypes.empty())
    resultTypes.push_back(
        stencil::TempType::get(ctx, applyBounds, builder.getF64Type()));

  auto apply = builder.create<stencil::ApplyOp>(loc, resultTypes, applyArgs,
                                                 applyBounds);
  Block *applyBody = builder.createBlock(&apply.getRegion());
  for (Value v : applyArgs)
    applyBody->addArgument(v.getType(), loc);

  OpBuilder bodyBuilder(applyBody, applyBody->end());
  SmallVector<int64_t> zeroOffset(ndim, 0);
  auto offsetAttr = stencil::IndexAttr::get(
      ctx, DenseI64ArrayAttr::get(ctx, zeroOffset));

  SmallVector<Value> accessResults;
  for (BlockArgument blockArg : applyBody->getArguments())
    accessResults.push_back(bodyBuilder.create<stencil::AccessOp>(
        loc, builder.getF64Type(), blockArg, offsetAttr));

  // ops_arg_idx(): one stencil.index per dimension, per idx argument,
  // appended after the dat-derived accesses (see file header).
  SmallVector<Value> idxResults;
  for (unsigned i = 0; i < numIdxArgs; ++i)
    for (int64_t d = 0; d < ndim; ++d)
      idxResults.push_back(bodyBuilder.create<stencil::IndexOp>(
          loc, bodyBuilder.getIndexType(), bodyBuilder.getI64IntegerAttr(d)));

  SmallVector<Value> callArgs = accessResults;
  callArgs.append(idxResults.begin(), idxResults.end());

  // Declare (if not already present) an external func for the kernel body
  // -- the captured ops.par_loop only carries the kernel's name/pointer,
  // not its source, so this is a placeholder for whatever eventually
  // supplies the real kernel (see Phase 3 of the lowering plan: link in
  // separately-compiled LLVM IR for the real kernel function).
  if (!module.lookupSymbol<func::FuncOp>(kernelName)) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    SmallVector<Type> paramTypes(accessResults.size(), builder.getF64Type());
    paramTypes.append(idxResults.size(), builder.getIndexType());
    auto kernelType = builder.getFunctionType(
        paramTypes, SmallVector<Type>(std::max<size_t>(1, writeFields.size()),
                                       builder.getF64Type()));
    auto decl =
        moduleBuilder.create<func::FuncOp>(loc, kernelName, kernelType);
    decl.setPrivate();
  }

  size_t numResults = std::max<size_t>(1, writeFields.size());
  auto call = bodyBuilder.create<func::CallOp>(
      loc, SmallVector<Type>(numResults, builder.getF64Type()), kernelName,
      callArgs);
  bodyBuilder.create<stencil::ReturnOp>(loc, call.getResults());

  // createBlock() above left `builder`'s insertion point inside the new
  // apply body block; move it back to right after the apply op (i.e.
  // back in the function's entry block) before emitting the store/return.
  builder.setInsertionPointAfter(apply);

  for (auto [i, fieldPair] : llvm::enumerate(writeFields)) {
    auto &[arg, field] = fieldPair;
    builder.create<stencil::StoreOp>(loc, apply.getResult(i), field,
                                      datBounds(builder, arg.getDat(), ndim));
  }

  builder.create<func::ReturnOp>(loc);
  return func;
}

} // namespace

namespace ops_mlir {

/// Lower every ops.par_loop in `module` into its own func.func, in place.
void convertOPSParLoopsToStencil(ModuleOp module) {
  SmallVector<ParLoopOp> loops;
  module.walk([&](ParLoopOp op) { loops.push_back(op); });

  for (auto [i, loop] : llvm::enumerate(loops)) {
    convertParLoop(loop, i, module);
    loop.erase();
  }
}

} // namespace ops_mlir
