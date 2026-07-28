#ifndef OPS_CAPTURE_H
#define OPS_CAPTURE_H

#include "IRBuilder.h"
#include "Core.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace ops_mlir {

enum class Backend { Sequential, OpenMP, CUDA };

std::optional<Backend> parseBackendName(const std::string &name);
constexpr Backend kDefaultBackend = Backend::Sequential;

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

private:
  Backend backend_ = kDefaultBackend;
  std::mutex mutex_;
  std::vector<LoopDesc> queue_;
  FlushCallback flushCallback_;
};

const char *accessToString(int access);
const char *argKindToString(ArgKind kind);

} // namespace ops_mlir

#endif // OPS_CAPTURE_H
