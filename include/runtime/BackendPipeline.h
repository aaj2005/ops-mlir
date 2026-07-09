#ifndef OPS_BACKEND
#define OPS_BACKEND
#include <string>
#include <vector>
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace ops_mlir {

class BackendPipeline {
public:
  virtual ~BackendPipeline() = default;
  virtual std::vector<std::string> passes() const = 0;

  // Joins passes() into MLIR's textual pipeline syntax and runs it
  mlir::LogicalResult run(mlir::ModuleOp module, mlir::MLIRContext &ctx) const {
    mlir::PassManager pm(&ctx);
    std::string pipelineStr = join(passes());
    if (mlir::failed(mlir::parsePassPipeline(pipelineStr, pm))) {
      llvm::errs() << "failed to parse pass pipeline:\n" << pipelineStr << "\n";
      return mlir::failure();
    }
    return pm.run(module);
  }

private:
  static std::string join(const std::vector<std::string> &passes) {
    std::string result;
    for (size_t i = 0; i < passes.size(); ++i) {
      if (i) result += ",";
      result += passes[i];
    }
    return result;
  }
};

class CPUSequentialPipeline : public BackendPipeline {
public:
  std::vector<std::string> passes() const override {
    return {
      "convert-bufferization-to-memref",
      "convert-scf-to-cf",
      "convert-cf-to-llvm",
      "canonicalize",
      "cse",
      "lower-affine",
      "convert-math-to-llvm",
      "convert-arith-to-llvm",
      "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
      "expand-strided-metadata",
      "finalize-memref-to-llvm",
      "reconcile-unrealized-casts",
      "canonicalize",
      "cse",
    };
  }
};

class OpenMPPipeline : public BackendPipeline {
public:
  std::vector<std::string> passes() const override {
    return {
      "convert-bufferization-to-memref",
      "convert-scf-to-openmp",
      "canonicalize",
      "cse",
      "convert-openmp-to-llvm",
      "canonicalize",
      "lower-affine",
      "convert-math-to-llvm",
      "expand-strided-metadata",
      "finalize-memref-to-llvm",
      "canonicalize",
      "convert-scf-to-cf",
      "convert-cf-to-llvm",
      "lower-affine",
      "convert-arith-to-llvm",
      "convert-math-to-llvm",
      "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
      "reconcile-unrealized-casts",
    };
  }
};

class CudaPipeline : public BackendPipeline {
public:
  explicit CudaPipeline(std::string gpuSm) : gpuSm_(std::move(gpuSm)) {}

  std::vector<std::string> passes() const override {
    std::string nvvmTarget =
        "nvvm-attach-target{O=3 ftz fast chip=sm_" + gpuSm_ +
        " triple=nvptx64-nvidia-cuda}";

    return {
      "convert-bufferization-to-memref",
      "canonicalize",
      "cse",
      "reconcile-unrealized-casts",
      "func.func(gpu-map-parallel-loops)",
      "func.func(convert-parallel-loops-to-gpu)",
      "canonicalize",
      "cse",
      "fold-memref-alias-ops",
      "gpu-kernel-outlining",
      "canonicalize",
      "cse",
      "fold-memref-alias-ops",
      "expand-strided-metadata",
      "lower-affine",
      "canonicalize",
      "cse",
      "func.func(gpu-async-region)",
      "canonicalize",
      "cse",
      "convert-arith-to-llvm",
      "convert-math-to-llvm",
      "convert-scf-to-cf",
      "convert-cf-to-llvm",
      "canonicalize",
      "cse",
      "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
      nvvmTarget,
      "gpu.module(convert-gpu-to-nvvm,canonicalize,cse)",
      "gpu-to-llvm",
      "gpu-module-to-binary",
      "canonicalize",
      "cse",
    };
  }

private:
  std::string gpuSm_;
};

} // namespace ops_mlir


#endif // OPS_BACKEND
