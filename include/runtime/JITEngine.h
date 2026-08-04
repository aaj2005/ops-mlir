#ifndef OPS_CAPTURE_H
#define OPS_CAPTURE_H

#include "IRBuilder.h"
#include "Core.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace ops_mlir {

enum class Backend { Sequential, OpenMP, CUDA };

std::optional<Backend> parseBackendName(const std::string &name);
constexpr Backend kDefaultBackend = Backend::CUDA;

static constexpr const char *kBackendFlagPrefix = "--backend=";
static constexpr const char *kBackendEnvVar = "OPS_BACKEND";

struct XdslResult {
  bool success = false;
  std::string ir;      // populated on success
  std::string error;   // populated on failure
};

class JITEngine {
public:
  using FlushCallback = std::function<void(const std::vector<LoopDesc> &)>;

  static JITEngine &instance();

  void setFlushCallback(FlushCallback callback);

  void enqueueParLoop(std::uintptr_t kernelToken, const char *kernelName,
                      ops_block block, int dims, const int *range,
                      const ops_arg *args, std::size_t nargs);

  void flush();

  void compile_and_execute();

  void setBackend(Backend backend) { backend_ = backend; }
  Backend backend() const { return backend_; }

  // Needed for GPU backend kernel translation (via KernelIRBuilder) to materialize real MLIR
  void setKernelSourceFile(std::string path) {
    kernelSourceFile_ = std::move(path);
  }

  // Register extern global kernel constants (e.g. pi, jmax) for translation into MLIR 
  void registerKernelConstant(const std::string &name, const void *ptr) {
    kernelConstants_[name] = ptr;
  }

  const std::map<std::string, const void *> &kernelConstants() const {
    return kernelConstants_;
  }

  // Note - resolveBackend is currently unused. This gives the option of a
  Backend resolveBackend(int argc, char **argv);

  const std::vector<LoopDesc> &queue() const { return queue_; }

private:
  JITEngine();
  ~JITEngine();

  mlir::MLIRContext ctx;
  mlir::ModuleOp module;
  mlir::OwningOpRef<mlir::ModuleOp> loweredModule_;

  IRBuilder builder{&ctx};

  LoopDesc buildLoopDesc(std::uintptr_t kernelToken, const char *kernelName,
                         ops_block block, int dims, const int *range,
                         const ops_arg *args, std::size_t nargs);

  ArgDesc buildArgDesc(const ops_arg &arg);

  DatDesc describeDat(ops_dat dat);
  StencilDesc describeStencil(ops_stencil stencil);

  XdslResult runXdslLowering(const std::string &ir);

  void runBackendLowering(mlir::ModuleOp module, Backend backend);
  std::string detectNVGpuSm();

  void compile();
  void execute(std::unique_ptr<mlir::ExecutionEngine> engine);
  void registerCpuKernelSymbols(mlir::ExecutionEngine &engine);

  // Translates a kernel body from C++ source into MLIR for the GPU backend, materializing
  void materializeGpuKernel(const std::string &kernelName, int indexRank);

  // Returns the device buffer mirroring the given host `ops_dat` buffer,
  // allocating it (via cuMemAlloc) on first use. Kernels compiled for the
  // CUDA backend operate on device memory, not the host pointers `dat`
  // args normally carry, so execute() copies host->device before each
  // launch and device->host after for any dat the kernel writes. A no-op
  // returning 0 when built without OPS_ENABLE_CUDA.
  //
  // Note: unconditionally declared (not #ifdef OPS_ENABLE_CUDA) even
  // though only meaningful for CUDA builds -- OPS_ENABLE_CUDA is only
  // defined PRIVATE for the OPSRuntime target, so consumer translation
  // units (e.g. apps linking against it) never see that macro; gating a
  // class member on it here would make JITEngine's layout depend on
  // which TU compiles it, an ODR violation.
  std::uintptr_t ensureDeviceBuffer(std::uintptr_t hostPtr, std::size_t bytes);

  void synchronizeBackend(Backend backend);

private:
  Backend backend_ = kDefaultBackend;
  std::mutex mutex_;
  std::vector<LoopDesc> queue_;
  FlushCallback flushCallback_;
  std::string kernelSourceFile_;
  std::map<std::string, const void *> kernelConstants_;
  // Host ops_dat pointer -> device pointer (stored as uintptr_t to keep
  // this header CUDA-toolkit-header-free; only populated/used when built
  // with OPS_ENABLE_CUDA).
  std::map<std::uintptr_t, std::uintptr_t> deviceBuffers_;
};

const char *accessToString(int access);
const char *argKindToString(ArgKind kind);

} // namespace ops_mlir

#endif // OPS_CAPTURE_H
