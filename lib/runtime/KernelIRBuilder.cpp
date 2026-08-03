//===- KernelIRBuilder.cpp - C++ kernel body -> MLIR ------------*- C++ -*-===//
// Tree walks a kernel's C++ AST to translate it into MLIR for the GPU backend, where
// the kernel body must become actual device code rather than a symbol resolved
// against a compiled host binary.
//
//===----------------------------------------------------------------------===//

#include "runtime/KernelIRBuilder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/MemoryBuffer.h"

#include <map>

namespace ops_mlir {

namespace {

class KernelFunctionFinder
    : public clang::RecursiveASTVisitor<KernelFunctionFinder> {
public:
  explicit KernelFunctionFinder(llvm::StringRef kernelName)
      : kernelName_(kernelName) {}

  bool VisitFunctionDecl(clang::FunctionDecl *decl) {
    // getName() asserts on non-identifier names (operators, constructors,
    // deduction guides, ...) -- <math.h> pulls in plenty of those via
    // <cmath> once parsed in C++ mode, so skip them explicitly.
    if (decl->getDeclName().isIdentifier() && decl->hasBody() &&
        decl->getName() == kernelName_)
      found_ = decl;
    // Keep walking in case of duplicate/shadowed declarations; the last
    // one with a body wins, matching normal C++ redeclaration semantics.
    return true;
  }

  clang::FunctionDecl *result() const { return found_; }

private:
  llvm::StringRef kernelName_;
  clang::FunctionDecl *found_ = nullptr;
};


/// Maps a <math.h> function name to the math dialect op that computes it.
/// Returns null for names outside the supported set.
using MathOpBuilder = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                      llvm::ArrayRef<mlir::Value>);

template <typename OpTy>
static mlir::Value buildUnaryMathOp(mlir::OpBuilder &b, mlir::Location loc,
                                    llvm::ArrayRef<mlir::Value> args) {
  return b.create<OpTy>(loc, args[0]);
}

template <typename OpTy>
static mlir::Value buildBinaryMathOp(mlir::OpBuilder &b, mlir::Location loc,
                                     llvm::ArrayRef<mlir::Value> args) {
  return b.create<OpTy>(loc, args[0], args[1]);
}

static MathOpBuilder lookupMathFunction(llvm::StringRef name) {
  static const llvm::StringMap<MathOpBuilder> kTable = {
      {"sin", &buildUnaryMathOp<mlir::math::SinOp>},
      {"cos", &buildUnaryMathOp<mlir::math::CosOp>},
      {"tan", &buildUnaryMathOp<mlir::math::TanOp>},
      {"asin", &buildUnaryMathOp<mlir::math::AsinOp>},
      {"acos", &buildUnaryMathOp<mlir::math::AcosOp>},
      {"atan", &buildUnaryMathOp<mlir::math::AtanOp>},
      {"sinh", &buildUnaryMathOp<mlir::math::SinhOp>},
      {"cosh", &buildUnaryMathOp<mlir::math::CoshOp>},
      {"tanh", &buildUnaryMathOp<mlir::math::TanhOp>},
      {"exp", &buildUnaryMathOp<mlir::math::ExpOp>},
      {"exp2", &buildUnaryMathOp<mlir::math::Exp2Op>},
      {"log", &buildUnaryMathOp<mlir::math::LogOp>},
      {"log2", &buildUnaryMathOp<mlir::math::Log2Op>},
      {"log10", &buildUnaryMathOp<mlir::math::Log10Op>},
      {"sqrt", &buildUnaryMathOp<mlir::math::SqrtOp>},
      {"fabs", &buildUnaryMathOp<mlir::math::AbsFOp>},
      {"floor", &buildUnaryMathOp<mlir::math::FloorOp>},
      {"ceil", &buildUnaryMathOp<mlir::math::CeilOp>},
      {"atan2", &buildBinaryMathOp<mlir::math::Atan2Op>},
      {"pow", &buildBinaryMathOp<mlir::math::PowFOp>},
  };
  auto it = kTable.find(name);
  return it == kTable.end() ? nullptr : it->second;
}

class ExprEmitter : public clang::ConstStmtVisitor<ExprEmitter, mlir::Value> {
public:
  ExprEmitter(mlir::OpBuilder &builder, mlir::Location loc,
              const std::map<const clang::ParmVarDecl *, mlir::Value> &params,
              const std::map<std::string, const void *> &constants,
              llvm::raw_ostream &errs)
      : builder_(builder), loc_(loc), params_(params),
        constants_(constants), errs_(errs) {}

  bool ok() const { return ok_; }

  mlir::Value VisitParenExpr(const clang::ParenExpr *expr) {
    return Visit(expr->getSubExpr());
  }

  mlir::Value VisitImplicitCastExpr(const clang::ImplicitCastExpr *expr) {
    mlir::Value sub = Visit(expr->getSubExpr());
    if (!ok_)
      return {};
    switch (expr->getCastKind()) {
    case clang::CK_LValueToRValue:
    case clang::CK_NoOp:
      return sub;
    case clang::CK_IntegralToFloating:
      return builder_.create<mlir::arith::SIToFPOp>(
          loc_, builder_.getF64Type(), sub);
    case clang::CK_FloatingCast:
      // KernelIRBuilder always materializes floating literals/results
      // as f64 directly (see VisitFloatingLiteral), so a float->double
      // promotion node has nothing left to do here.
      return sub;
    default:
      return fail(expr, "unsupported implicit cast");
    }
  }

  mlir::Value VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
    if (const auto *param =
            llvm::dyn_cast<clang::ParmVarDecl>(expr->getDecl())) {
      auto it = params_.find(param);
      if (it != params_.end())
        return it->second;
    }

    // If the kernel body references an extern global constant (e.g. `pi` or
    // `jmax`), look it up in the map of live addresses registered by the
    // app via JITEngine::registerKernelConstant. This mirrors enqueueParLoop
    // capturing a kernel's function pointer directly rather than resolving it
    // by symbol name later.
    if (const auto *var = llvm::dyn_cast<clang::VarDecl>(expr->getDecl())) {
      if (mlir::Value baked = bakeGlobalConstant(var))
        return baked;
    }

    return fail(expr, "reference to unsupported symbol '" +
                          expr->getDecl()->getNameAsString() +
                          "' (kernel bodies may only use their own "
                          "parameters, <math.h> calls, and extern globals "
                          "registered via JITEngine::registerKernelConstant)");
  }

  mlir::Value VisitFloatingLiteral(const clang::FloatingLiteral *expr) {
    double value = expr->getValueAsApproximateDouble();
    return builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getF64FloatAttr(value));
  }

  mlir::Value VisitIntegerLiteral(const clang::IntegerLiteral *expr) {
    int64_t value = expr->getValue().getSExtValue();
    return builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getI32IntegerAttr(static_cast<int32_t>(value)));
  }

  mlir::Value VisitUnaryMinus(const clang::UnaryOperator *expr) {
    mlir::Value sub = Visit(expr->getSubExpr());
    if (!ok_)
      return {};
    if (mlir::isa<mlir::FloatType>(sub.getType()))
      return builder_.create<mlir::arith::NegFOp>(loc_, sub);
    mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getI32IntegerAttr(0));
    return builder_.create<mlir::arith::SubIOp>(loc_, zero, sub);
  }

  mlir::Value VisitBinAdd(const clang::BinaryOperator *e) { return binOp(e); }
  mlir::Value VisitBinSub(const clang::BinaryOperator *e) { return binOp(e); }
  mlir::Value VisitBinMul(const clang::BinaryOperator *e) { return binOp(e); }
  mlir::Value VisitBinDiv(const clang::BinaryOperator *e) { return binOp(e); }

  mlir::Value VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr) {
    const auto *base = llvm::dyn_cast<clang::DeclRefExpr>(
        expr->getBase()->IgnoreParenImpCasts());
    const clang::ParmVarDecl *param =
        base ? llvm::dyn_cast<clang::ParmVarDecl>(base->getDecl()) : nullptr;
    if (!param || !params_.count(param))
      return fail(expr, "array subscript base must be a kernel parameter");

    mlir::Value indexVal = Visit(expr->getIdx());
    if (!ok_)
      return {};
    mlir::Value index = builder_.create<mlir::arith::IndexCastOp>(
        loc_, builder_.getIndexType(), indexVal);
    return builder_.create<mlir::memref::LoadOp>(loc_, params_.at(param),
                                                 mlir::ValueRange{index});
  }

  mlir::Value VisitCallExpr(const clang::CallExpr *expr) {
    const clang::FunctionDecl *callee = expr->getDirectCallee();
    bool hasSimpleName =
        callee && callee->getDeclName().isIdentifier();
    MathOpBuilder mathOp =
        hasSimpleName ? lookupMathFunction(callee->getName()) : nullptr;
    if (!mathOp) {
      return fail(expr, "call to unsupported function '" +
                            (callee ? callee->getNameAsString() : "<unknown>") +
                            "' (only <math.h> calls are supported)");
    }
    llvm::SmallVector<mlir::Value, 2> args;
    for (const clang::Expr *arg : expr->arguments()) {
      mlir::Value v = Visit(arg);
      if (!ok_)
        return {};
      args.push_back(v);
    }
    return mathOp(builder_, loc_, args);
  }

  mlir::Value VisitStmt(const clang::Stmt *stmt) {
    return fail(stmt, "unsupported expression construct");
  }

private:
  mlir::Value binOp(const clang::BinaryOperator *expr) {
    mlir::Value lhs = Visit(expr->getLHS());
    if (!ok_)
      return {};
    mlir::Value rhs = Visit(expr->getRHS());
    if (!ok_)
      return {};

    bool isFloat = mlir::isa<mlir::FloatType>(lhs.getType());
    switch (expr->getOpcode()) {
    case clang::BO_Add:
      return isFloat ? builder_.create<mlir::arith::AddFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::AddIOp>(loc_, lhs, rhs)
                             .getResult();
    case clang::BO_Sub:
      return isFloat ? builder_.create<mlir::arith::SubFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::SubIOp>(loc_, lhs, rhs)
                             .getResult();
    case clang::BO_Mul:
      return isFloat ? builder_.create<mlir::arith::MulFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::MulIOp>(loc_, lhs, rhs)
                             .getResult();
    case clang::BO_Div:
      return isFloat ? builder_.create<mlir::arith::DivFOp>(loc_, lhs, rhs)
                             .getResult()
                     : builder_.create<mlir::arith::DivSIOp>(loc_, lhs, rhs)
                             .getResult();
    default:
      return fail(expr, "unsupported binary operator");
    }
  }

  // If the kernel body references an extern global constant (e.g. `pi` or
  // `jmax`), look it up in the map of live addresses registered by the
  // app via JITEngine::registerKernelConstant. This mirrors enqueueParLoop
  // capturing a kernel's function pointer directly rather than resolving it
  // by symbol name later.
  mlir::Value bakeGlobalConstant(const clang::VarDecl *var) {
    if (!var->hasGlobalStorage())
      return {};

    auto it = constants_.find(var->getNameAsString());
    if (it == constants_.end() || !it->second)
      return {};
    const void *addr = it->second;

    clang::QualType type = var->getType();
    if (type->isSpecificBuiltinType(clang::BuiltinType::Double)) {
      double value = *reinterpret_cast<const double *>(addr);
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getF64FloatAttr(value));
    }
    if (type->isSpecificBuiltinType(clang::BuiltinType::Int)) {
      int32_t value = *reinterpret_cast<const int32_t *>(addr);
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getI32IntegerAttr(value));
    }
    return fail(nullptr, "global '" + var->getNameAsString() +
                             "' has an unsupported type (only extern "
                             "double/int globals can be baked in)");
  }

  mlir::Value fail(const clang::Stmt *at, const llvm::Twine &message) {
    if (ok_) {
      ok_ = false;
      errs_ << "KernelIRBuilder: " << message << "\n";
    }
    (void)at;
    return {};
  }

  mlir::OpBuilder &builder_;
  mlir::Location loc_;
  const std::map<const clang::ParmVarDecl *, mlir::Value> &params_;
  const std::map<std::string, const void *> &constants_;
  llvm::raw_ostream &errs_;
  bool ok_ = true;
};

} // namespace

//===----------------------------------------------------------------------===//
// KernelIRBuilder
//===----------------------------------------------------------------------===//

mlir::func::FuncOp KernelIRBuilder::generate(
    const std::string &sourceFile, const std::string &kernelName,
    int indexRank, const std::map<std::string, const void *> &constants,
    llvm::raw_ostream &errs) {
  auto codeOrErr = llvm::MemoryBuffer::getFile(sourceFile);
  if (!codeOrErr) {
    errs << "KernelIRBuilder: could not read '" << sourceFile
        << "': " << codeOrErr.getError().message() << "\n";
    return nullptr;
  }

  // -x c++: force C++ mode explicitly (buildASTFromCodeWithArgs infers the
  // language from sourceFile's extension -- .h means C -- which conflicts
  // with -std=c++17). -resource-dir: needed since building the AST via
  // this library API bypasses the driver logic that normally locates
  // Clang's own bundled headers (stddef.h etc.) relative to itself.
  std::unique_ptr<clang::ASTUnit> unit = clang::tooling::buildASTFromCodeWithArgs(
      (*codeOrErr)->getBuffer(),
      /*Args=*/{"-x", "c++", "-std=c++17",
               "-resource-dir=" OPS_CLANG_RESOURCE_DIR},
      sourceFile);
  if (!unit) {
    errs << "KernelIRBuilder: failed to parse '" << sourceFile << "'\n";
    return nullptr;
  }

  KernelFunctionFinder finder(kernelName);
  finder.TraverseDecl(unit->getASTContext().getTranslationUnitDecl());
  clang::FunctionDecl *decl = finder.result();
  if (!decl) {
    errs << "KernelIRBuilder: no definition of '" << kernelName << "' in "
        << sourceFile << "\n";
    return nullptr;
  }

  if (!decl->getReturnType()->isSpecificBuiltinType(clang::BuiltinType::Double)) {
    errs << "KernelIRBuilder: '" << kernelName
        << "' must return double\n";
    return nullptr;
  }

  mlir::OpBuilder builder(&context_);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 4> paramTypes;
  for (const clang::ParmVarDecl *param : decl->parameters()) {
    clang::QualType type = param->getType();
    if (type->isSpecificBuiltinType(clang::BuiltinType::Double)) {
      paramTypes.push_back(builder.getF64Type());
    } else if (type->isPointerType() &&
              type->getPointeeType()->isSpecificBuiltinType(
                  clang::BuiltinType::Int)) {
      paramTypes.push_back(
          mlir::MemRefType::get({indexRank}, builder.getI32Type()));
    } else {
      errs << "KernelIRBuilder: unsupported parameter type for '"
          << kernelName << "' (only double and const int* are supported)\n";
      return nullptr;
    }
  }

  auto funcType = builder.getFunctionType(paramTypes, {builder.getF64Type()});
  auto funcOp = mlir::func::FuncOp::create(loc, kernelName, funcType);
  funcOp.setPrivate();
  mlir::Block *entry = funcOp.addEntryBlock();

  std::map<const clang::ParmVarDecl *, mlir::Value> paramValues;
  for (auto [param, arg] : llvm::zip(decl->parameters(), entry->getArguments()))
    paramValues[param] = arg;

  builder.setInsertionPointToStart(entry);

  // Kernels in this codebase are a single `return <expr>;` -- anything
  // richer (locals, control flow) is out of scope for now.
  const auto *body = llvm::dyn_cast<clang::CompoundStmt>(decl->getBody());
  const clang::ReturnStmt *ret =
      (body && body->size() == 1)
          ? llvm::dyn_cast<clang::ReturnStmt>(*body->body_begin())
          : nullptr;
  if (!ret || !ret->getRetValue()) {
    errs << "KernelIRBuilder: '" << kernelName
        << "' must be a single `return <expr>;` statement\n";
    funcOp.erase();
    return nullptr;
  }

  ExprEmitter emitter(builder, loc, paramValues, constants, errs);
  mlir::Value result = emitter.Visit(ret->getRetValue());
  if (!emitter.ok()) {
    funcOp.erase();
    return nullptr;
  }

  builder.create<mlir::func::ReturnOp>(loc, result);
  return funcOp;
}

} // namespace ops_mlir
